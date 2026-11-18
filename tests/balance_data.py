"""Reproducible synthetic data generator for affine balance tests.

The generator creates a fixed non-constant 2-D physical field with values in
[10, 1000] across two bands, and renders at least four GeoTIFF sources whose
stored values are affine-distorted versions of that field with known gains
(between 0.5 and 2) and biases (between -20 and 20).  Sources form chain and
cycle overlaps at different resolutions with independent band masks and
per-file scales/offsets.

Everything is driven by explicit integer seeds so that results are fully
reproducible; expected values are computed analytically from the known
coefficients, never from the output of the fitting functions.
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin

# Fixed field: smooth and non-constant, values inside [10, 1000].
FIELD_MIN = 10.0
FIELD_MAX = 1000.0


def reference_fields(height=120, width=160):
    """Return the fixed two-band physical field used by every generator."""
    yy, xx = np.mgrid[0:height, 0:width]
    field1 = (
        FIELD_MIN
        + (FIELD_MAX - FIELD_MIN) * (0.6 * xx / width + 0.4 * yy / height)
    )
    field2 = 500.0 + 300.0 * np.sin(xx / 12.0) * np.cos(yy / 15.0)
    return np.stack([field1, field2]).astype(np.float64)


# (left, bottom, right, top) in north-up CRS coordinates.  A-B, A-C, B-D and
# C-D overlaps give chains plus a closed cycle A-B-D-C-A.
DEFAULT_TILES = [
    (0, 50, 90, 120),
    (70, 50, 160, 120),
    (0, 0, 90, 70),
    (70, 0, 160, 70),
]

DEFAULT_GAINS = (1.1, 0.7, 1.6, 0.55)
DEFAULT_BIASES = (3.0, -11.0, 7.5, -4.2)
DEFAULT_FILE_SCALES = (1.0, 2.0, 0.5, 1.0)
DEFAULT_FILE_OFFSETS = (0.0, 1.0, -2.0, 0.5)
DEFAULT_RESOLUTIONS = (1.0, 1.0, 0.5, 2.0)


def expected_relative_coefficients(gains=DEFAULT_GAINS, biases=DEFAULT_BIASES):
    """Coefficients every fit must recover with the first source as reference.

    With the reference anchored to gain 1 / bias 0, the recoverable
    transform for source *s* is relative to the reference *r*::

        gain_s = g_s / g_r
        bias_s = (b_s - b_r) / g_r
    """
    g_r, b_r = gains[0], biases[0]
    return [(g / g_r, (b - b_r) / g_r) for g, b in zip(gains, biases)]


def write_sources(
    directory,
    fields=None,
    tiles=DEFAULT_TILES,
    gains=DEFAULT_GAINS,
    biases=DEFAULT_BIASES,
    file_scales=DEFAULT_FILE_SCALES,
    file_offsets=DEFAULT_FILE_OFFSETS,
    resolutions=DEFAULT_RESOLUTIONS,
    crs="EPSG:32631",
    nodata=-9999.0,
    mask_corner=(2, 1, 8, 8),
    zero_cells=(),
):
    """Render the field into per-source GeoTIFF files and return their paths.

    Parameters
    ----------
    directory : path-like
        Existing directory to write into.
    fields : ndarray, optional
        ``(bands, height, width)`` physical field.  Defaults to
        :func:`reference_fields`.
    mask_corner : tuple or None
        ``(source_index, band_index, height, width)`` of an independent band
        mask knocked out of one source, or ``None`` for no masking.
    zero_cells : sequence of (source, band, row, col)
        Explicit cells forced to the legal value zero (after applying the
        source transform), exercising the "zero is valid" rule.

    Returns
    -------
    list of str
        File paths in source order.
    """
    if fields is None:
        fields = reference_fields()
    n_bands, master_h, master_w = fields.shape
    paths = []
    for k, ((left, bottom, right, top), g, b, fs, fo, res) in enumerate(
        zip(tiles, gains, biases, file_scales, file_offsets, resolutions)
    ):
        th = int(round((top - bottom) / res))
        tw = int(round((right - left) / res))
        cx = left + (np.arange(tw) + 0.5) * res
        cy = top - (np.arange(th) + 0.5) * res
        gx, gy = np.meshgrid(cx, cy)
        gxi = np.clip(np.floor(gx).astype(int), 0, master_w - 1)
        gyi = np.clip(np.floor(master_h - gy).astype(int), 0, master_h - 1)
        phys = fields[:, gyi, gxi]
        stored = (phys - b) / g
        raw = (stored - fo) / fs
        mask = np.ones_like(raw, dtype=bool)
        if mask_corner is not None:
            src_idx, band_idx, ch, cw = mask_corner
            if src_idx == k:
                mask[band_idx, :ch, :cw] = False
        for src_idx, band_idx, row, col in zero_cells:
            if src_idx == k:
                # Force the *physical* value to zero and derive the raw
                # value through the inverse transform.
                raw[band_idx, row, col] = (0.0 - b - g * fo) / (g * fs)
                mask[band_idx, row, col] = True
        raw = np.where(mask, raw, nodata).astype("float32")
        path = f"{directory}/src{k}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=th,
            width=tw,
            count=n_bands,
            dtype="float32",
            nodata=nodata,
            crs=crs,
            transform=from_origin(left, top, res, res),
        ) as ds:
            ds.write(raw)
            ds.scales = [fs] * n_bands
            ds.offsets = [fo] * n_bands
        paths.append(path)
    return paths


def add_outliers(paths, fraction=0.05, dynamic_range=10.0, seed=1234567):
    """Inject fixed-seed outliers into valid cells of each source file.

    At most ``fraction`` of each source's valid cells per band are replaced
    with values outside the field range by at least
    ``dynamic_range`` field spans.  Positions use a fixed-seed uniform draw.
    Files are rewritten in place; the returned array describes every
    contaminated ``(source, band, row, col)`` so tests can exclude them from
    clean-cell checks.
    """
    rng = np.random.default_rng(seed)
    span = FIELD_MAX - FIELD_MIN
    contaminated = []
    for k, path in enumerate(paths):
        with rasterio.open(path) as ds:
            data = ds.read()
            nodata = ds.nodata
            scales = ds.scales
            offsets = ds.offsets
        for band in range(data.shape[0]):
            valid = np.argwhere(data[band] != nodata)
            n_bad = int(np.floor(fraction * len(valid)))
            chosen = valid[
                rng.choice(len(valid), size=n_bad, replace=False)
            ]
            for row, col in chosen:
                sign = 1.0 if rng.random() < 0.5 else -1.0
                # Outlier in raw units: far beyond any plausible physical
                # reading of this source.
                data[band, row, col] = (
                    sign * dynamic_range * span
                    + rng.uniform(-span, span)
                    - offsets[band]
                ) / scales[band]
                contaminated.append((k, band, int(row), int(col)))
        with rasterio.open(path, "r+") as ds:
            ds.write(data)
    return np.array(contaminated, dtype=np.int64)
