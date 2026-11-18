"""Joint affine calibration (gain/offset balancing) of raster sources.

This module implements the machinery behind ``merge(balance="affine")`` and
the ``balance_model`` / ``balance_source`` arguments of
:func:`rasterio.warp.reproject`.

The physical value of a pixel in band *b* of source *s* is obtained exactly
once from the stored (raw) value using the dataset's scale and offset::

    physical = raw * scales[b] + offsets[b]

The calibration model then expresses every source in the reference source's
physical frame with a per-band affine transform::

    reference_frame_value = gain[s, b] * physical[s, b] + bias[s, b]

The reference source has ``gain = 1.0`` and ``bias = 0.0`` on every selected
band.  Gains are constrained to be positive.  All overlaps (chains and
cycles) determine the single reference frame jointly through one global
robust least-squares problem per band -- nothing is propagated pairwise along
a tree.

Sampling is deterministic and independent of source ordering, file block
layout and the merge ``mem_limit`` setting.  Every overlapping source pair and
band owns a fixed sample quota, filled with a key-based uniform reservoir
driven by a pair/band specific ``numpy.random.Generator`` seeded from
stable identifiers.

Only :mod:`numpy` is required to apply or validate a model; SciPy is used
solely for sparse normal-equation assembly in :func:`fit_affine_balance`.
"""

import json
import math

import numpy as np

from rasterio import windows
from rasterio.enums import Resampling
from rasterio.transform import Affine

#: Model format version written by this implementation.
MODEL_VERSION = "1.0"

#: Only this major version can be read back.
SUPPORTED_MAJOR_VERSION = 1

# Numeric tolerances.  They are deliberately loose enough to accommodate
# float32 on-disk storage but tight enough to catch real metadata drift.
TRANSFORM_ABS_TOL = 1e-6
TRANSFORM_REL_TOL = 1e-9
SCALE_REL_TOL = 1e-6
OFFSET_ABS_TOL = 1e-6

# Fitting tolerances.
IRLS_MAX_ITER = 25
IRLS_COEF_TOL = 1e-10
IRLS_RIDGE_EPS = 1e-12
HUBER_K = 1.345  # 95% efficiency under Gaussian errors
RANK_REL_TOL = 1e-10

# Minimum number of valid paired samples for an individual overlap to carry
# any information, and minimum relative standard deviation of the sampled
# physical values (constant overlaps cannot identify a gain).
MIN_PAIR_SAMPLES = 30
MIN_REL_STD = 1e-12

# Streaming chunk size (pixels per overlap tile).  Bounds temporary memory
# regardless of raster dimensions; does not affect the sampled result.
CHUNK_PIXELS = 512 * 512

# Upper bound on GDAL's per-context block cache while fitting.  Without a
# bound the cache accumulates one decompressed block per touched file and can
# grow to hundreds of MB on many-source, large-raster jobs; 64 MB keeps the
# working set flat while tiled reads remain efficient.
FIT_GDAL_CACHEMAX_MB = 64


class BalanceError(ValueError):
    """Raised when an affine balance model cannot be identified or applied.

    It is a :class:`ValueError` so that callers can keep catching
    ``ValueError`` for invalid calibration inputs.
    """


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _crs_to_json(crs):
    if crs is None:
        return None
    return crs.to_wkt()


def _json_to_crs(value):
    if value is None:
        return None
    from rasterio.crs import CRS

    return CRS.from_user_input(value)


def _finite(value):
    """Return True for a finite real number (booleans accepted as numbers)."""
    return isinstance(value, (int, float, np.floating, np.integer)) and math.isfinite(
        float(value)
    )


def collect_source_meta(dataset, indexes):
    """Read the calibration-relevant metadata of one open dataset.

    Scales and offsets are read here, once.  GDAL reports missing metadata as
    scale 1.0 / offset 0.0, which is the identity in
    ``physical = raw * scale + offset``.
    """
    indexes = list(indexes)
    scales = [float(dataset.scales[b - 1]) for b in indexes]
    offsets = [float(dataset.offsets[b - 1]) for b in indexes]
    return {
        "crs": _crs_to_json(dataset.crs),
        "transform": [float(v) for v in dataset.transform.to_gdal()],
        "width": int(dataset.width),
        "height": int(dataset.height),
        "dtypes": [str(dataset.dtypes[b - 1]) for b in indexes],
        "scales": scales,
        "offsets": offsets,
        "band_indexes": [int(b) for b in indexes],
        "nodata": [
            None if dataset.nodatavals[b - 1] is None else float(dataset.nodatavals[b - 1])
            for b in indexes
        ],
    }


def _meta_matches(meta, dataset, band_index, tol_transform=TRANSFORM_ABS_TOL):
    """Check that an open dataset still agrees with recorded metadata."""
    from rasterio.crs import CRS

    pos = meta["band_indexes"].index(band_index)
    if dataset.width != meta["width"] or dataset.height != meta["height"]:
        return False, "dimensions"
    if str(dataset.dtypes[band_index - 1]) != meta["dtypes"][pos]:
        return False, "dtype"
    rec_transform = Affine.from_gdal(*meta["transform"])
    if not dataset.transform.almost_equals(rec_transform, TRANSFORM_ABS_TOL):
        return False, "transform"
    rec_crs = _json_to_crs(meta["crs"])
    if CRS.from_user_input(dataset.crs) != rec_crs:
        return False, "crs"
    scale = float(dataset.scales[band_index - 1])
    offset = float(dataset.offsets[band_index - 1])
    if not _close_rel(scale, meta["scales"][pos], SCALE_REL_TOL, 0.0):
        return False, "scale"
    if abs(offset - meta["offsets"][pos]) > OFFSET_ABS_TOL:
        return False, "offset"
    return True, None


def _close_rel(actual, expected, rel_tol, abs_tol):
    actual = float(actual)
    expected = float(expected)
    return abs(actual - expected) <= max(abs_tol, rel_tol * max(abs(actual), abs(expected)))


# ---------------------------------------------------------------------------
# Model container
# ---------------------------------------------------------------------------


class BalanceModel:
    """A validated per-source, per-band affine calibration model.

    Instances are plain data containers; use :func:`fit_affine_balance` to
    estimate one, :meth:`to_dict` / :func:`balance_model_from_dict` for JSON
    persistence, and :meth:`coefficients` / :meth:`apply` at read time.
    """

    def __init__(self, model_dict):
        d = balance_model_from_dict(model_dict)
        self._d = d

    # -- constructors ------------------------------------------------------

    @classmethod
    def _from_fit(cls, *, version, reference, sources, bands, diagnostics, tolerances):
        return cls(
            {
                "version": version,
                "reference": reference,
                "sources": sources,
                "bands": bands,
                "diagnostics": diagnostics,
                "tolerances": tolerances,
            }
        )

    # -- accessors ---------------------------------------------------------

    @property
    def version(self):
        return self._d["version"]

    @property
    def reference(self):
        return self._d["reference"]

    @property
    def source_ids(self):
        return list(self._d["sources"].keys())

    @property
    def band_indexes(self):
        """Selected raw (1-based) band numbers, shared by all sources."""
        any_id = next(iter(self._d["sources"]))
        return list(self._d["sources"][any_id]["band_indexes"])

    def source_meta(self, source_id):
        try:
            return self._d["sources"][source_id]
        except KeyError:
            raise BalanceError(f"balance model has no source {source_id!r}")

    def coefficients(self, source_id, band_index):
        """Return ``(gain, bias)`` for one source and raw band number."""
        self.source_meta(source_id)
        key = str(band_index)
        try:
            entry = self._d["bands"][key][source_id]
        except KeyError:
            raise BalanceError(
                f"balance model has no coefficients for source {source_id!r}, band {band_index}"
            )
        return float(entry["gain"]), float(entry["bias"])

    def validate_band(self, dataset, source_id, band_index):
        """Validate an open dataset's metadata against the model.

        Raises :class:`BalanceError` on mismatch.  Returns the recorded band
        position otherwise.
        """
        meta = self.source_meta(source_id)
        if band_index not in meta["band_indexes"]:
            raise BalanceError(
                f"band {band_index} of source {source_id!r} is not part of the balance model"
            )
        ok, reason = _meta_matches(meta, dataset, band_index)
        if not ok:
            raise BalanceError(
                f"source {source_id!r} band {band_index} {reason} does not match the "
                "balance model; refit before reuse"
            )

    # -- (de)serialisation -------------------------------------------------

    def to_dict(self):
        """Return a ``json.dumps``-ready deep copy of the model."""
        return json.loads(json.dumps(self._d))

    def to_json(self, path_or_buf=None, indent=2):
        text = json.dumps(self._d, indent=indent, sort_keys=True)
        if path_or_buf is None:
            return text
        if hasattr(path_or_buf, "write"):
            path_or_buf.write(text)
        else:
            with open(path_or_buf, "w", encoding="utf-8") as fh:
                fh.write(text)
        return None


def balance_model_from_dict(model_dict):
    """Validate a plain mapping (e.g. the result of ``json.load``)."""
    if not isinstance(model_dict, dict):
        raise BalanceError("balance model must be a JSON object (mapping)")
    version = model_dict.get("version")
    if not isinstance(version, str):
        raise BalanceError("balance model is missing a string 'version'")
    try:
        major = int(version.split(".")[0])
    except (ValueError, IndexError):
        raise BalanceError(f"unsupported balance model version {version!r}")
    if major != SUPPORTED_MAJOR_VERSION:
        raise BalanceError(
            f"unsupported balance model version {version!r}; "
            f"this rasterio supports major version {SUPPORTED_MAJOR_VERSION}"
        )
    reference = model_dict.get("reference")
    sources = model_dict.get("sources")
    bands = model_dict.get("bands")
    if not isinstance(sources, dict) or not sources:
        raise BalanceError("balance model must contain a non-empty 'sources' object")
    if reference not in sources:
        raise BalanceError(f"balance model reference {reference!r} is not among its sources")
    if not isinstance(bands, dict) or not bands:
        raise BalanceError("balance model must contain a non-empty 'bands' object")
    expected_ids = set(sources)
    for band_key, entries in bands.items():
        if not isinstance(entries, dict) or set(entries) != expected_ids:
            raise BalanceError(
                f"balance model band {band_key!r} must list every source exactly once"
            )
        for source_id, entry in entries.items():
            gain = entry.get("gain")
            bias = entry.get("bias")
            if not _finite(gain) or float(gain) <= 0.0:
                raise BalanceError(
                    f"balance model gain for {source_id!r} band {band_key} must be positive"
                )
            if not _finite(bias):
                raise BalanceError(
                    f"balance model bias for {source_id!r} band {band_key} must be finite"
                )
    for source_id, meta in sources.items():
        for key in ("crs", "transform", "width", "height", "dtypes", "scales", "offsets", "band_indexes"):
            if key not in meta:
                raise BalanceError(
                    f"balance model metadata for source {source_id!r} is missing {key!r}"
                )
    return model_dict


def load_balance_model(path_or_file):
    """Load a model from a path or an open text file (``json.load`` input)."""
    if hasattr(path_or_file, "read"):
        model_dict = json.load(path_or_file)
    else:
        with open(path_or_file, "r", encoding="utf-8") as fh:
            model_dict = json.load(fh)
    return BalanceModel(model_dict)


# ---------------------------------------------------------------------------
# Pixel-level semantics
# ---------------------------------------------------------------------------


def read_physical_window(dataset, band_index, window, out_shape=None,
                         resampling=Resampling.nearest):
    """Read one band window as float64 physical values plus a validity mask.

    Validity combines the dataset mask (nodata, mask band, open edges) with
    explicit NaN/infinity rejection.  Zero is always a valid value.  The
    band's scale/offset are applied here, once.  Nearest-neighbour
    resampling is used when ``out_shape`` differs from the native window
    shape: the read windows are aligned so that output pixel centers fall on
    source pixel centers, which keeps an exact (uninterpolated) value at
    every sampled location and avoids interpolation bias at coverage edges.
    """
    raw = dataset.read(
        band_index,
        window=window,
        out_shape=out_shape,
        masked=True,
        boundless=False,
        resampling=resampling,
    )
    mask_band = dataset.read_masks(
        indexes=band_index, window=window, out_shape=out_shape,
        resampling=resampling,
    )
    valid = (mask_band != 0) & (~np.ma.getmaskarray(raw))
    values = np.asarray(raw, dtype=np.float64)
    scale = float(dataset.scales[band_index - 1])
    offset = float(dataset.offsets[band_index - 1])
    physical = values * scale + offset
    valid &= np.isfinite(physical)
    return physical, valid


def _overlap_geometry(src_a, src_b):
    """Geometric descriptor of the overlap between two north-up datasets.

    Returns a dict with each dataset's native window covering the shared
    rectangle and the coarse-grid lattice shape, or ``None`` when the
    datasets do not overlap.

    Sampling points are the *coarse* source's pixel centers inside the
    overlap: every point is a native pixel center of both datasets once the
    finer source is mapped with nearest neighbour, so no interpolation bias
    is introduced at any resolution ratio.  Which source is coarse is purely
    geometric, keeping the result independent of argument order.
    """
    left = max(src_a.bounds[0], src_b.bounds[0])
    right = min(src_a.bounds[2], src_b.bounds[2])
    bottom = max(src_a.bounds[1], src_b.bounds[1])
    top = min(src_a.bounds[3], src_b.bounds[3])
    if left >= right or bottom >= top:
        return None
    res_a = math.hypot(src_a.transform.a, src_a.transform.e)
    res_b = math.hypot(src_b.transform.a, src_b.transform.e)
    if res_a <= res_b:
        fine, coarse = src_a, src_b
        swapped = False
    else:
        fine, coarse = src_b, src_a
        swapped = True

    def _window(ds):
        inv = ~ds.transform
        c0, r0 = inv * (left, top)
        c1, r1 = inv * (right, bottom)
        return windows.Window(
            int(round(min(c0, c1))),
            int(round(min(r0, r1))),
            max(1, int(round(abs(c1 - c0)))),
            max(1, int(round(abs(r1 - r0)))),
        ).crop(ds.height, ds.width)

    coarse_win = _window(coarse)
    if coarse_win.width <= 0 or coarse_win.height <= 0:
        return None
    coarse_bounds = windows.bounds(coarse_win, coarse.transform)
    fine_win = _window(fine)
    shape = (int(coarse_win.height), int(coarse_win.width))
    result = {
        "coarse_win": coarse_win,
        "fine_win": fine_win,
        "lattice_bounds": coarse_bounds,
        "shape": shape,
    }
    if swapped:
        result["win_a"] = coarse_win
        result["win_b"] = fine_win
    else:
        result["win_a"] = fine_win
        result["win_b"] = coarse_win
    return result


def _pair_seed(id_a, id_b, band_index):
    base = "|".join(sorted((str(id_a), str(id_b)))) + f"|band{int(band_index)}"
    # Stable 64-bit FNV-1a hash, independent of Python's hash seed.
    h = np.uint64(1469598103934665603)
    with np.errstate(over="ignore"):
        for byte in base.encode("utf-8"):
            h = np.uint64((h ^ np.uint64(byte)) * np.uint64(1099511628211))
    return int(h)


class _PairSampler:
    """Key-based uniform reservoir for one (pair, band) overlap.

    Every valid paired pixel receives an independent uniform key from a
    deterministic, pair/band-seeded ``numpy.random.Generator``; the sampler
    retains the ``quota`` smallest keys.  Retained values are therefore a
    uniform random sample of the full valid-pair population, and -- because
    keys are generated in streaming order from a fixed seed -- the result
    does not depend on how the overlap is tiled.
    """

    def __init__(self, id_a, id_b, band_index, quota):
        self.id_a = id_a
        self.id_b = id_b
        self.band_index = band_index
        self.quota = int(quota)
        self.rng = np.random.default_rng(_pair_seed(id_a, id_b, band_index))
        self._keys = np.empty(self.quota, dtype=np.float64)
        self.values_a = np.empty(self.quota, dtype=np.float64)
        self.values_b = np.empty(self.quota, dtype=np.float64)
        self.seen = 0
        self.kept = 0
        # Running statistics over every valid pair (not just kept ones).
        self.min_a = np.inf
        self.max_a = -np.inf
        self.min_b = np.inf
        self.max_b = -np.inf
        self.sum_a = 0.0
        self.sum_b = 0.0
        self.sumsq_a = 0.0
        self.sumsq_b = 0.0

    def add(self, phys_a, valid_a, phys_b, valid_b):
        valid = valid_a & valid_b
        count = int(valid.sum())
        if count == 0:
            return
        va = phys_a[valid]
        vb = phys_b[valid]
        # Statistics on the full valid population of the overlap.
        self.seen += count
        self.min_a = min(self.min_a, float(va.min()))
        self.max_a = max(self.max_a, float(va.max()))
        self.min_b = min(self.min_b, float(vb.min()))
        self.max_b = max(self.max_b, float(vb.max()))
        self.sum_a += float(va.sum())
        self.sum_b += float(vb.sum())
        self.sumsq_a += float(np.square(va, dtype=np.float64).sum())
        self.sumsq_b += float(np.square(vb, dtype=np.float64).sum())

        keys = self.rng.random(count, dtype=np.float64)
        if self.kept < self.quota:
            take = min(count, self.quota - self.kept)
            slot = slice(self.kept, self.kept + take)
            self._keys[slot] = keys[:take]
            self.values_a[slot] = va[:take]
            self.values_b[slot] = vb[:take]
            self.kept += take
            rest = slice(take, count)
            va, vb, keys = va[rest], vb[rest], keys[rest]
            count -= take
            if count == 0:
                return

        # Tank full: replace the largest retained key when candidates are
        # smaller.  Vectorised via argsort of the merged candidate batch.
        worst = self._keys.max()
        better = keys < worst
        if not better.any():
            return
        cand_keys = keys[better]
        cand_a = va[better]
        cand_b = vb[better]
        merged_keys = np.concatenate((self._keys, cand_keys))
        order = np.argsort(merged_keys, kind="stable")[: self.quota]
        merged_a = np.concatenate((self.values_a, cand_a))[order]
        merged_b = np.concatenate((self.values_b, cand_b))[order]
        self._keys = merged_keys[order]
        self.values_a = merged_a
        self.values_b = merged_b

    def summary(self):
        if self.seen == 0:
            return None
        mean_a = self.sum_a / self.seen
        mean_b = self.sum_b / self.seen
        var_a = max(self.sumsq_a / self.seen - mean_a * mean_a, 0.0)
        var_b = max(self.sumsq_b / self.seen - mean_b * mean_b, 0.0)
        return {
            "id_a": self.id_a,
            "id_b": self.id_b,
            "band": self.band_index,
            "valid_pairs": int(self.seen),
            "sampled_pairs": int(self.kept),
            "min_a": self.min_a,
            "max_a": self.max_a,
            "min_b": self.min_b,
            "max_b": self.max_b,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "std_a": math.sqrt(var_a),
            "std_b": math.sqrt(var_b),
            "values_a": self.values_a[: self.kept],
            "values_b": self.values_b[: self.kept],
        }


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _huber_weights(residuals, scale):
    if scale <= 0.0:
        scale = float(np.std(residuals)) or 1.0
    standardized = np.abs(residuals) / scale
    weights = np.ones_like(standardized)
    big = standardized > HUBER_K
    weights[big] = HUBER_K / standardized[big]
    return weights


def _solve_band(samplers, source_ids, reference_id, band_index):
    """Global robust solve for one band; returns coefficients + diagnostics."""
    summaries = [s.summary() for s in samplers]
    summaries = [s for s in summaries if s is not None]
    pair_diag = []
    failures = []

    connected = {reference_id}
    adjacency = {sid: set() for sid in source_ids}
    for summary in summaries:
        a, b = summary["id_a"], summary["id_b"]
        adjacency[a].add(b)
        adjacency[b].add(a)
        rel_std = max(
            summary["std_a"] / max(abs(summary["mean_a"]), 1e-300),
            summary["std_b"] / max(abs(summary["mean_b"]), 1e-300),
        )
        entry = {
            "sources": [a, b],
            "band": band_index,
            "valid_pairs": summary["valid_pairs"],
            "sampled_pairs": summary["sampled_pairs"],
            "relative_std": float(rel_std),
        }
        if summary["valid_pairs"] < MIN_PAIR_SAMPLES:
            entry["status"] = "insufficient_samples"
            failures.append(
                f"overlap {a!r}/{b!r} band {band_index}: only "
                f"{summary['valid_pairs']} valid paired pixels "
                f"(minimum {MIN_PAIR_SAMPLES})"
            )
        elif rel_std < MIN_REL_STD:
            entry["status"] = "constant_overlap"
            failures.append(
                f"overlap {a!r}/{b!r} band {band_index}: overlapping values are "
                "effectively constant; gain is not identifiable"
            )
        else:
            entry["status"] = "ok"
            connected.add(a)
            connected.add(b)
        pair_diag.append(entry)

    # Reachability to the reference through informative overlaps.
    informative = {reference_id}
    for entry in pair_diag:
        if entry["status"] == "ok":
            informative.update(entry["sources"])
    reachable = set()
    frontier = {reference_id}
    while frontier:
        node = frontier.pop()
        if node in reachable:
            continue
        reachable.add(node)
        frontier |= (adjacency[node] & informative) - reachable
    unreachable = [sid for sid in source_ids if sid not in reachable]
    for sid in unreachable:
        failures.append(
            f"source {sid!r} band {band_index}: no informative overlap chain "
            f"connects it to reference {reference_id!r}"
        )

    if failures:
        raise BalanceError(
            "affine balance is not identifiable for band "
            f"{band_index}: " + "; ".join(failures)
        )

    # Build the global design matrix.  Each non-reference source has two
    # free parameters (gain, bias); the reference frame is fixed by gain 1,
    # bias 0 on the reference source.  For every sampled pair (a, b) the
    # equality ``gain_a*x_a + bias_a = gain_b*x_b + bias_b`` contributes one
    # row, with reference-side terms moved to the right-hand side.  All
    # overlaps, including cycles, are thus solved jointly.
    free_ids = [sid for sid in source_ids if sid != reference_id]
    col_index = {sid: 2 * i for i, sid in enumerate(free_ids)}
    n_col = 2 * len(free_ids)
    design_parts = []
    rhs_parts = []
    pair_residuals = {}
    pair_row_ranges = []
    row_cursor = 0
    for summary in summaries:
        a, b = summary["id_a"], summary["id_b"]
        va, vb = summary["values_a"], summary["values_b"]
        n = va.shape[0]
        pair_residuals[(a, b)] = (va, vb)
        pair_row_ranges.append((a, b, row_cursor, row_cursor + n))
        row_cursor += n
        # Equality of reference-frame values:
        #   gain_a*x_a + bias_a = gain_b*x_b + bias_b
        # Reference-side parameters are fixed (gain 1, bias 0) and move to
        # the right-hand side.
        row = np.zeros((n, n_col), dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)
        if a == reference_id:
            j = col_index[b]
            row[:, j] = vb
            row[:, j + 1] = 1.0
            rhs = va
        elif b == reference_id:
            j = col_index[a]
            row[:, j] = va
            row[:, j + 1] = 1.0
            rhs = vb
        else:
            ja, jb = col_index[a], col_index[b]
            row[:, ja] = va
            row[:, ja + 1] = 1.0
            row[:, jb] = -vb
            row[:, jb + 1] = -1.0
        design_parts.append(row)
        rhs_parts.append(rhs)

    design = np.vstack(design_parts)
    rhs = np.concatenate(rhs_parts)
    del design_parts, rhs_parts

    # Column scaling so gains and biases have comparable magnitudes.
    scales_vec = np.maximum(np.abs(design).max(axis=0), 1e-300)
    design_scaled = design / scales_vec

    # Rank check on the unscaled informative matrix.
    singular = np.linalg.svd(design_scaled, compute_uv=False)
    rank = int((singular > RANK_REL_TOL * singular[0]).sum())
    if rank < n_col:
        raise BalanceError(
            f"affine balance is rank-deficient for band {band_index} "
            f"(rank {rank} of {n_col} free parameters); at least one gain/bias "
            "pair is not identifiable from the overlaps"
        )

    def _robust_scale(residuals):
        """Huber scale: MAD with a high-quantile fallback.

        The MAD can collapse to zero when more than half the equations of a
        small overlap are gross outliers; the 90th percentile of absolute
        residuals then supplies a finite, conservative scale instead.
        """
        mad = float(np.median(np.abs(residuals - np.median(residuals))))
        scale = 1.4826 * mad
        if not (scale > 0.0) or not math.isfinite(scale):
            scale = float(np.quantile(np.abs(residuals), 0.9))
        return scale if scale > 0.0 else (float(np.std(residuals)) or 1.0)

    def _pair_robust_relation(values_first, values_second):
        """Robust affine relation ``first ~= slope*second + intercept``.

        A three-stage high-breakdown estimator:

        1. A decile slope seed (median level in the top vs. bottom decile of
           the predictor) has a ~20% breakdown point, so gross outliers that
           corrupt ordinary least squares cannot capture it.
        2. Repeated least-trimmed-squares refinement (80% smallest residuals)
           converges to the clean-sample solution.
        3. A final Huber polish with hard rejection beyond four robust
           standard deviations down-weights any remaining borderline points.
        """
        x = values_second
        y = values_first
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        k = max(2, len(xs) // 10)
        x_lo, x_hi = np.median(xs[:k]), np.median(xs[-k:])
        y_lo, y_hi = np.median(ys[:k]), np.median(ys[-k:])
        slope = (y_hi - y_lo) / (x_hi - x_lo)
        if not math.isfinite(slope) or slope == 0.0:
            slope = 1.0
        intercept = float(np.median(y - slope * x))
        theta = np.array([slope, intercept])

        design_pair = np.vstack([x, np.ones_like(x)]).T
        keep_count = max(3, int(0.8 * len(x)))
        for _ in range(6):
            residual = np.abs(y - design_pair @ theta)
            keep = np.argsort(residual)[:keep_count]
            theta, *_ = np.linalg.lstsq(
                design_pair[keep], y[keep], rcond=None
            )
        residual = y - design_pair @ theta
        scale = _robust_scale(residual)
        weights = _huber_weights(residual, scale)
        weights[np.abs(residual) / scale > 4.0] = 0.0
        if weights.sum() > 0:
            sw = np.sqrt(weights)
            lhs = (design_pair * sw[:, None]).T @ (design_pair * sw[:, None])
            rhs_w = (design_pair * sw[:, None]).T @ (y * sw)
            theta = np.linalg.solve(lhs, rhs_w)
        return float(theta[0]), float(theta[1])

    def _solve_weighted(weight_vector):
        sw = np.sqrt(weight_vector)
        lhs = (design * sw[:, None]).T @ (design * sw[:, None])
        rhs_weighted = (design * sw[:, None]).T @ (rhs * sw)
        # Scale-relative ridge keeps regularization negligible compared with
        # each column's own energy regardless of physical magnitudes.
        diag = np.diag(lhs).copy()
        np.fill_diagonal(lhs, diag + IRLS_RIDGE_EPS * np.maximum(diag, 1e-300))
        return np.linalg.solve(lhs, rhs_weighted)

    # Stage 1: seed every free source with a robust per-pair estimate.
    # A breadth-first walk from the reference through informative overlaps
    # gives each source exactly one seeding relation against an already
    # anchored neighbour.  Every estimate derives from a trimmed,
    # outlier-free pair regression, so the seed cannot be the constant
    # solution that global OLS falls into.
    seed_gain = {reference_id: 1.0}
    seed_bias = {reference_id: 0.0}
    # Index summaries by endpoint for quick neighbour lookup.
    neighbours = {sid: [] for sid in source_ids}
    for sm in summaries:
        neighbours[sm["id_a"]].append(sm)
        neighbours[sm["id_b"]].append(sm)
    frontier = [reference_id]
    while frontier:
        anchor = frontier.pop(0)
        for sm in neighbours[anchor]:
            a, b = sm["id_a"], sm["id_b"]
            if a == anchor:
                other = b
                # anchor_stored = slope*other_stored + intercept
                slope, intercept = _pair_robust_relation(
                    sm["values_a"], sm["values_b"]
                )
            else:
                other = a
                slope, intercept = _pair_robust_relation(
                    sm["values_b"], sm["values_a"]
                )
            if other in seed_gain:
                continue
            g_anchor, b_anchor = seed_gain[anchor], seed_bias[anchor]
            seed_gain[other] = g_anchor * slope
            seed_bias[other] = g_anchor * intercept + b_anchor
            frontier.append(other)
    for sid in free_ids:
        if sid not in seed_gain:
            seed_gain[sid] = 1.0
            seed_bias[sid] = 0.0
    theta = np.array(
        [value for sid in free_ids for value in (seed_gain[sid], seed_bias[sid])],
        dtype=np.float64,
    )

    # Stage 2: joint Huber IRLS from the robust seed, with hard rejection of
    # gross outliers.  Robust scales are estimated per overlap (each pair has
    # its own clean residual variance), and each pair's weights are
    # normalized to sum one so overlaps with many pixels cannot drown out
    # small ones.  This also stops a contaminated pair inflating any global
    # scale and rescuing its own outliers.
    weights = np.ones(design.shape[0], dtype=np.float64)
    for iteration in range(1, IRLS_MAX_ITER + 1):
        residuals = rhs - design @ theta
        weights = np.ones(design.shape[0], dtype=np.float64)
        for _a, _b, r0, r1 in pair_row_ranges:
            pair_resid = residuals[r0:r1]
            scale = _robust_scale(pair_resid)
            pair_weights = _huber_weights(pair_resid, scale)
            pair_weights[np.abs(pair_resid) / scale > 4.0] = 0.0
            total = pair_weights.sum()
            if total > 0:
                pair_weights = pair_weights / total
            else:
                # Whole pair rejected: fall back to equal weights so the
                # overlap still contributes its geometric constraint.
                pair_weights = np.full(r1 - r0, 1.0 / (r1 - r0))
            weights[r0:r1] = pair_weights
        if weights.sum() <= 0:
            break
        new_theta = _solve_weighted(weights)
        scale_ref = max(np.abs(theta).max(), 1e-300)
        if np.abs(new_theta - theta).max() / scale_ref < IRLS_COEF_TOL:
            theta = new_theta
            break
        theta = new_theta
    robust_scale = _robust_scale(rhs - design @ theta)

    # Final residual diagnostics per overlap pair.
    coef = {reference_id: (1.0, 0.0)}
    for sid in free_ids:
        j = col_index[sid]
        gain = float(theta[j])
        bias = float(theta[j + 1])
        if not math.isfinite(gain) or gain <= 0.0 or not math.isfinite(bias):
            raise BalanceError(
                f"fit for source {sid!r} band {band_index} produced non-positive "
                f"gain or non-finite coefficients (gain={gain}, bias={bias})"
            )
        coef[sid] = (gain, bias)

    residual_diag = []
    for (a, b), (va, vb) in pair_residuals.items():
        ga, ba = coef[a]
        gb, bb = coef[b]
        resid = ga * va + ba - (gb * vb + bb)
        resid_sorted = np.sort(resid)
        q = lambda p: float(
            resid_sorted[min(len(resid_sorted) - 1, int(round(p * (len(resid_sorted) - 1))))]
        )
        residual_diag.append(
            {
                "sources": [a, b],
                "band": band_index,
                "rmse": float(np.sqrt(np.mean(resid ** 2))),
                "mean_abs_residual": float(np.mean(np.abs(resid))),
                "median_abs_residual": float(np.median(np.abs(resid))),
                "max_abs_residual": float(np.abs(resid).max()),
                "q05_residual": q(0.05),
                "q95_residual": q(0.95),
            }
        )

    global_resid = rhs - design @ theta
    band_diag = {
        "band": band_index,
        "iterations": iteration,
        "rank": rank,
        "n_equations": int(design.shape[0]),
        "robust_scale": float(robust_scale),
        "rmse": float(np.sqrt(np.mean(global_resid ** 2))),
        "mean_abs_residual": float(np.mean(np.abs(global_resid))),
        "overlaps": pair_diag,
        "residuals": residual_diag,
    }
    return coef, band_diag


def fit_affine_balance(open_sources, source_ids, reference_id, indexes, sample_limit):
    """Fit the joint affine model from open datasets.

    Parameters
    ----------
    open_sources : sequence of open rasterio datasets
        Same length as ``source_ids``.
    source_ids : sequence of str
        Non-empty unique identifiers.
    reference_id : str
        ID of the reference source (gain 1, bias 0).
    indexes : sequence of int
        Raw 1-based band numbers to fit.
    sample_limit : int
        Maximum total retained paired samples across the whole fit.

    Returns
    -------
    BalanceModel
    """
    indexes = [int(b) for b in indexes]

    import rasterio

    # Bound GDAL's block cache for the duration of the fit so that touching
    # many large tiled sources cannot accumulate unbounded decompressed
    # blocks.  An explicitly user-configured cache is left untouched.
    from rasterio.errors import EnvError

    try:
        user_cache = rasterio.env.getenv().get("GDAL_CACHEMAX")
    except EnvError:
        user_cache = None
    cache_kwds = {} if user_cache is not None else {
        "GDAL_CACHEMAX": FIT_GDAL_CACHEMAX_MB
    }
    with rasterio.Env(**cache_kwds):
        return _fit_affine_balance_impl(
            open_sources, source_ids, reference_id, indexes, sample_limit
        )


def _fit_affine_balance_impl(open_sources, source_ids, reference_id, indexes, sample_limit):
    n_sources = len(open_sources)
    indexes = [int(b) for b in indexes]

    # Quotas: equal share per informative (overlapping) pair/band.  Pair
    # membership is geometric and order-independent, so rearrangement of
    # sources cannot change any pair's quota, RNG stream or samples.
    pair_bands = []
    geometry_cache = {}
    for i in range(n_sources):
        for j in range(i + 1, n_sources):
            geometry = _overlap_geometry(open_sources[i], open_sources[j])
            if geometry is not None:
                geometry_cache[(i, j)] = geometry
                for band in indexes:
                    pair_bands.append((i, j, band))
    if not pair_bands:
        raise BalanceError(
            "no two sources overlap; an affine balance model cannot be identified"
        )
    quota = max(1, int(sample_limit // len(pair_bands)))

    samplers = {
        (i, j, band): _PairSampler(
            source_ids[i], source_ids[j], band, quota
        )
        for (i, j, band) in pair_bands
    }

    # Streaming pass over every overlap, tiled in two dimensions so that
    # whole rasters are never resident.  Lattice points are coarse pixel
    # centers; the coarse source is read at native resolution and the fine
    # source at the matching shape with nearest resampling (an exact point
    # lookup at aligned windows).  Tiles split native windows at whole pixel
    # rows and columns, so tiling never changes which stored pixel a lattice
    # point hits.  Tiles are visited in fixed row-major order, which is the
    # only order the pair-seeded reservoir RNG depends on.
    tile_target = int(math.sqrt(CHUNK_PIXELS))
    for (i, j, band), sampler in samplers.items():
        geometry = geometry_cache[(i, j)]
        win_i, win_j, shape = geometry["win_a"], geometry["win_b"], geometry["shape"]
        ds_i, ds_j = open_sources[i], open_sources[j]
        total_h, total_w = shape
        tile_h = max(1, min(total_h, tile_target))
        tile_w = max(1, min(total_w, tile_target))

        def _sub(window, row0, rows, col0, cols):
            # Map lattice spans to native pixel spans with the same nearest
            # mapping as a full-window read.
            row_scale = window.height / total_h
            col_scale = window.width / total_w
            r0 = min(window.height, int(math.floor(row0 * row_scale + 0.5)))
            r1 = min(
                window.height,
                int(math.floor((row0 + rows) * row_scale + 0.5)),
            )
            c0 = min(window.width, int(math.floor(col0 * col_scale + 0.5)))
            c1 = min(
                window.width,
                int(math.floor((col0 + cols) * col_scale + 0.5)),
            )
            return windows.Window(
                window.col_off + c0,
                window.row_off + r0,
                max(1, c1 - c0),
                max(1, r1 - r0),
            )

        for row0 in range(0, total_h, tile_h):
            rows = min(tile_h, total_h - row0)
            for col0 in range(0, total_w, tile_w):
                cols = min(tile_w, total_w - col0)
                phys_i, valid_i = read_physical_window(
                    ds_i, band,
                    _sub(win_i, row0, rows, col0, cols),
                    out_shape=(rows, cols),
                )
                phys_j, valid_j = read_physical_window(
                    ds_j, band,
                    _sub(win_j, row0, rows, col0, cols),
                    out_shape=(rows, cols),
                )
                sampler.add(phys_i, valid_i, phys_j, valid_j)

    sources_meta = {}
    for sid, ds in zip(source_ids, open_sources):
        meta = collect_source_meta(ds, indexes)
        sources_meta[sid] = meta

    bands_out = {}
    diagnostics = {"bands": [], "sample_limit": int(sample_limit), "pair_quota": quota}
    total_sampled = 0
    total_valid = 0
    for band in indexes:
        band_samplers = [
            samplers[(i, j, band)]
            for (i, j, b) in pair_bands
            if b == band
        ]
        coef, band_diag = _solve_band(
            band_samplers, list(source_ids), reference_id, band
        )
        bands_out[str(band)] = {
            sid: {"gain": float(g), "bias": float(b)}
            for sid, (g, b) in coef.items()
        }
        diagnostics["bands"].append(band_diag)
        for ov in band_diag["overlaps"]:
            total_valid += ov["valid_pairs"]
        for ov in band_diag["overlaps"]:
            total_sampled += ov["sampled_pairs"]
    diagnostics["total_valid_pairs"] = total_valid
    diagnostics["total_sampled_pairs"] = total_sampled

    tolerances = {
        "transform_abs_tol": TRANSFORM_ABS_TOL,
        "transform_rel_tol": TRANSFORM_REL_TOL,
        "scale_rel_tol": SCALE_REL_TOL,
        "offset_abs_tol": OFFSET_ABS_TOL,
        "irls_coef_tol": IRLS_COEF_TOL,
        "rank_rel_tol": RANK_REL_TOL,
        "huber_k": HUBER_K,
        "min_pair_samples": MIN_PAIR_SAMPLES,
        "min_relative_std": MIN_REL_STD,
    }
    return BalanceModel._from_fit(
        version=MODEL_VERSION,
        reference=reference_id,
        sources=sources_meta,
        bands=bands_out,
        diagnostics=diagnostics,
        tolerances=tolerances,
    )


# ---------------------------------------------------------------------------
# Application of a model to values read from disk
# ---------------------------------------------------------------------------


def apply_affine(physical, gain, bias):
    """Apply ``gain * physical + bias`` (float64, no in-place writes)."""
    return np.asarray(physical, dtype=np.float64) * float(gain) + float(bias)


def calibrated_vrt_xml(dataset, source_id, model, band_indexes):
    """Build a float64 VRT document applying the model lazily per band.

    The VRT preserves the source nodata/mask semantics and adds per-band
    ``ScaleRatio`` / ``ScaleOffset`` so that GDAL computes
    ``gain * (raw * file_scale + file_offset) + bias`` while warping.
    Nothing is materialised and the source file is opened, not modified.
    """
    import xml.etree.ElementTree as ET

    from rasterio._path import _parse_path
    from rasterio.dtypes import _gdal_typename

    meta = model.source_meta(source_id)
    transform = Affine.from_gdal(*meta["transform"])
    root = ET.Element("VRTDataset")
    root.attrib["rasterXSize"] = str(meta["width"])
    root.attrib["rasterYSize"] = str(meta["height"])
    srs = ET.SubElement(root, "SRS")
    srs.text = meta["crs"] or ""
    gt = ET.SubElement(root, "GeoTransform")
    gt.text = ",".join(str(v) for v in transform.to_gdal())

    # GDAL's warper takes a single scalar src_nodata, so every VRT band must
    # advertise the same calibrated nodata.  Use the smallest calibrated
    # sentinel magnitude (a value no clean measurement reaches) consistently;
    # the source-level NODATA element on each ComplexSource still filters raw
    # invalid pixels before scaling.
    calibrated_sentinels = []
    for raw_bidx in band_indexes:
        pos = meta["band_indexes"].index(raw_bidx)
        gain, bias = model.coefficients(source_id, raw_bidx)
        raw_nodata = meta["nodata"][pos]
        if raw_nodata is not None:
            ratio = gain * meta["scales"][pos]
            offset = gain * meta["offsets"][pos] + bias
            calibrated_sentinels.append(float(raw_nodata) * ratio + offset)
    shared_nodata = (
        min(calibrated_sentinels, key=abs) if calibrated_sentinels else None
    )

    for out_bidx, raw_bidx in enumerate(band_indexes, start=1):
        pos = meta["band_indexes"].index(raw_bidx)
        gain, bias = model.coefficients(source_id, raw_bidx)
        file_scale = meta["scales"][pos]
        file_offset = meta["offsets"][pos]
        ratio = gain * file_scale
        offset = gain * file_offset + bias

        band_el = ET.SubElement(root, "VRTRasterBand")
        band_el.attrib["dataType"] = "Float64"
        band_el.attrib["band"] = str(out_bidx)
        nodata = meta["nodata"][pos]
        if shared_nodata is not None:
            nd = ET.SubElement(band_el, "NoDataValue")
            nd.text = repr(float(shared_nodata))
        ci = ET.SubElement(band_el, "ColorInterp")
        ci.text = "Gray" if out_bidx == 1 else "Undefined"

        source_el = ET.SubElement(band_el, "ComplexSource")
        fname = ET.SubElement(source_el, "SourceFilename")
        fname.attrib["relativeToVRT"] = "0"
        fname.attrib["shared"] = "0"
        fname.text = _parse_path(dataset.name).as_vsi()
        sband = ET.SubElement(source_el, "SourceBand")
        sband.text = str(raw_bidx)
        props = ET.SubElement(source_el, "SourceProperties")
        props.attrib["RasterXSize"] = str(meta["width"])
        props.attrib["RasterYSize"] = str(meta["height"])
        props.attrib["dataType"] = _gdal_typename(meta["dtypes"][pos])
        srcrect = ET.SubElement(source_el, "SrcRect")
        srcrect.attrib.update(xOff="0", yOff="0", xSize=str(meta["width"]), ySize=str(meta["height"]))
        dstrect = ET.SubElement(source_el, "DstRect")
        dstrect.attrib.update(xOff="0", yOff="0", xSize=str(meta["width"]), ySize=str(meta["height"]))
        if nodata is not None:
            ndsrc = ET.SubElement(source_el, "NODATA")
            ndsrc.text = repr(float(nodata))
        ratio_el = ET.SubElement(source_el, "ScaleRatio")
        ratio_el.text = repr(float(ratio))
        off_el = ET.SubElement(source_el, "ScaleOffset")
        off_el.text = repr(float(offset))

    # Preserve validity semantics (nodata / mask band / open edges).
    # The VRT inherits per-band NODATA declarations, which GDAL's warper
    # honors directly; an explicit mask band is added when present.
    mask_flags = dataset.mask_flag_enums
    from rasterio.enums import MaskFlags

    if any(MaskFlags.per_dataset in flags for flags in mask_flags):
        maskband = ET.SubElement(root, "MaskBand")
        mband = ET.SubElement(maskband, "VRTRasterBand")
        mband.attrib["dataType"] = "Byte"
        simple = ET.SubElement(mband, "SimpleSource")
        fname = ET.SubElement(simple, "SourceFilename")
        fname.attrib["relativeToVRT"] = "0"
        fname.attrib["shared"] = "0"
        fname.text = _parse_path(dataset.name).as_vsi()
        sband = ET.SubElement(simple, "SourceBand")
        sband.text = "mask,1"
        props = ET.SubElement(simple, "SourceProperties")
        props.attrib["RasterXSize"] = str(meta["width"])
        props.attrib["RasterYSize"] = str(meta["height"])
        props.attrib["dataType"] = "Byte"
        srcrect = ET.SubElement(simple, "SrcRect")
        srcrect.attrib.update(xOff="0", yOff="0", xSize=str(meta["width"]), ySize=str(meta["height"]))
        dstrect = ET.SubElement(simple, "DstRect")
        dstrect.attrib.update(xOff="0", yOff="0", xSize=str(meta["width"]), ySize=str(meta["height"]))

    return ET.tostring(root, encoding="utf-8")
