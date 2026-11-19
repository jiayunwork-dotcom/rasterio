"""Regression tests for per-band validity through calibration.

These tests cover the failure mode where a legal finite value on one band
(including zero, even when it equals another band's calibrated nodata) was
erased by a shared multi-band nodata sentinel, and where a non-finite
calibrated result (overflow to +/- infinity or NaN) leaked into the output.

The validity rule exercised throughout:

* source nodata, explicit dataset mask bands and NaN/+/-infinity are all
  recognized as invalid;
* after file scale/offset and the model's gain/bias are applied, only values
  still representable as finite floating point numbers are valid
  observations;
* a band's legal finite result never disappears merely because it matches a
  different band's calibrated nodata;
* merge ``first``/``last`` never let a calibrated-invalid candidate occupy a
  cell or overwrite an existing valid observation; cells without any source
  receive the output nodata.
"""

import json
import math
import sys

import numpy as np
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject

SRC_NODATA = -9999.0
DST_NODATA = -7777.0


def _write_two_band(path, band1, band2, *, nodata=SRC_NODATA, mask=None,
                    tiled=False, block=16):
    """Write a two-band float64 GeoTIFF and an optional explicit mask."""
    height, width = band1.shape
    kwds = {}
    if tiled:
        kwds.update(tiled=True, blockxsize=block, blockysize=block)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=2,
        dtype="float64", nodata=nodata, crs="EPSG:32631",
        transform=from_origin(0.0, float(height), 1.0, 1.0), **kwds,
    ) as ds:
        ds.write(np.stack([band1, band2]))
        if mask is not None:
            ds.write_mask(mask)


def _model(path, height, width, biases, scales=(1.0, 1.0),
           offsets=(0.0, 0.0), nodata=SRC_NODATA):
    with rasterio.open(path) as ds:
        crs_wkt = ds.crs.to_wkt()
        transform = list(ds.transform.to_gdal())
    return {
        "version": "1.0",
        "reference": "S",
        "sources": {
            "S": {
                "crs": crs_wkt,
                "transform": transform,
                "width": width,
                "height": height,
                "dtypes": ["float64", "float64"],
                "scales": list(scales),
                "offsets": list(offsets),
                "band_indexes": [1, 2],
                "nodata": [nodata, nodata],
            }
        },
        "bands": {
            "1": {"S": {"gain": 1.0, "bias": float(biases[0])}},
            "2": {"S": {"gain": 1.0, "bias": float(biases[1])}},
        },
        "diagnostics": {},
        "tolerances": {},
    }


# ---------------------------------------------------------------------------
# reproject: the zero-vs-nodata collision from the bug report
# ---------------------------------------------------------------------------


def test_reproject_zero_survives_band_nodata_collision(tmp_path):
    """Band 1 adds 9999 (its nodata calibrates to 0); band 2 keeps a legal 0.

    The legal zero of band 2 must survive the multi-band warp even though
    band 1's calibrated nodata is exactly zero.
    """
    height = width = 8
    band1 = np.full((height, width), 10.0)
    band2 = np.full((height, width), 10.0)
    band1[2, 3] = SRC_NODATA          # true hole on band 1
    band2[5, 4] = 0.0                 # legal zero on band 2
    band2[6, 1] = SRC_NODATA          # true hole on band 2
    path = str(tmp_path / "src.tif")
    _write_two_band(path, band1, band2)
    model = _model(path, height, width, biases=(9999.0, 0.0))

    with rasterio.open(path) as src:
        dst = np.full((2, height, width), DST_NODATA, dtype="float64")
        reproject(
            source=rasterio.band(src, [1, 2]),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=DST_NODATA,
            resampling=Resampling.nearest,
            balance_model=model,
            balance_source="S",
        )

    # Legal zero preserved; true holes remain empty.
    assert dst[1, 5, 4] == 0.0
    assert dst[0, 2, 3] == DST_NODATA
    assert dst[1, 6, 1] == DST_NODATA
    # Ordinary cells carry the calibration.
    assert dst[0, 0, 0] == pytest.approx(10009.0)
    assert dst[1, 0, 0] == pytest.approx(10.0)
    # No non-finite values leak.
    assert np.isfinite(dst[dst != DST_NODATA]).all()


def test_reproject_band_selection_order_independent(tmp_path):
    """Band 2 alone, with band 1, and in reverse order give the same data."""
    height = width = 8
    band1 = np.full((height, width), 10.0)
    band2 = np.full((height, width), 10.0)
    band1[2, 3] = SRC_NODATA
    band2[5, 4] = 0.0
    band2[6, 1] = SRC_NODATA
    path = str(tmp_path / "src.tif")
    _write_two_band(path, band1, band2)
    model = _model(path, height, width, biases=(9999.0, 0.0))

    def warp(selection):
        with rasterio.open(path) as src:
            n = len(selection)
            dst = np.full((n, height, width), DST_NODATA, dtype="float64")
            reproject(
                source=rasterio.band(src, selection),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=src.transform,
                dst_crs=src.crs,
                dst_nodata=DST_NODATA,
                resampling=Resampling.nearest,
                balance_model=model,
                balance_source="S",
            )
        return dst

    both = warp([1, 2])
    single = warp([2])
    reversed_sel = warp([2, 1])

    # Band 2's values and validity are identical however it is selected.
    np.testing.assert_array_equal(
        np.isfinite(both[1]), np.isfinite(single[0])
    )
    np.testing.assert_allclose(
        both[1][np.isfinite(both[1])],
        single[0][np.isfinite(single[0])],
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        np.isfinite(reversed_sel[0]), np.isfinite(both[1])
    )
    np.testing.assert_allclose(
        reversed_sel[0][np.isfinite(reversed_sel[0])],
        both[1][np.isfinite(both[1])],
        rtol=1e-12,
    )
    # The single-band selection keeps the zero too.
    assert single[0, 5, 4] == 0.0


def test_reproject_file_destination_masks_reopen_per_band(tmp_path):
    """Validity written to a file reopens with independent per-band masks."""
    height = width = 16
    band1 = np.full((height, width), 7.0)
    band2 = np.full((height, width), 7.0)
    band1[2, 3] = SRC_NODATA
    band2[6, 1] = SRC_NODATA
    band1[4, 4] = 0.0
    band2[5, 5] = 0.0
    path = str(tmp_path / "src.tif")
    out = str(tmp_path / "dst.tif")
    _write_two_band(path, band1, band2, tiled=True)
    model = _model(path, height, width, biases=(9999.0, 0.0))

    with rasterio.open(path) as src, rasterio.open(
        out, "w", driver="GTiff", height=height, width=width, count=2,
        dtype="float64", nodata=DST_NODATA, crs="EPSG:32631",
        transform=from_origin(0.0, float(height), 1.0, 1.0),
        tiled=True, blockxsize=16, blockysize=16,
    ) as dst:
        reproject(
            source=rasterio.band(src, [1, 2]),
            destination=rasterio.band(dst, [1, 2]),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst.transform,
            dst_crs=dst.crs,
            dst_nodata=DST_NODATA,
            resampling=Resampling.nearest,
            balance_model=model,
            balance_source="S",
        )

    with rasterio.open(out) as check:
        values = check.read()
        masks = check.read_masks()

    # Holes are invalid independently per band.
    assert masks[0, 2, 3] == 0
    assert masks[1, 6, 1] == 0
    # A hole on one band does not invalidate the same cell on the other.
    assert masks[1, 2, 3] == 255
    assert masks[0, 6, 1] == 255
    # Legal zeros are valid (band 1's zero calibrates to 9999, band 2's
    # stays zero).
    assert values[0, 4, 4] == pytest.approx(9999.0)
    assert values[1, 5, 5] == 0.0
    assert masks[0, 4, 4] == 255 and masks[1, 5, 5] == 255


def test_reproject_rejects_nan_and_infinity(tmp_path):
    """Explicit NaN and +/- infinity inputs are invalid after calibration."""
    height = width = 16
    band1 = np.full((height, width), 7.0)
    band2 = np.full((height, width), 7.0)
    band1[7, 7] = float("nan")
    band2[8, 8] = float("inf")
    band1[9, 9] = float("-inf")
    path = str(tmp_path / "src.tif")
    _write_two_band(path, band1, band2, tiled=True)
    model = _model(path, height, width, biases=(1.0, 2.0))

    with rasterio.open(path) as src:
        dst = np.full((2, height, width), DST_NODATA, dtype="float64")
        reproject(
            source=rasterio.band(src, [1, 2]),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=DST_NODATA,
            resampling=Resampling.nearest,
            balance_model=model,
            balance_source="S",
        )
    assert dst[0, 7, 7] == DST_NODATA
    assert dst[1, 8, 8] == DST_NODATA
    assert dst[0, 9, 9] == DST_NODATA
    assert not (np.isnan(dst).any() or np.isinf(dst).any())


def test_reproject_explicit_mask_band_invalidates(tmp_path):
    """A cell invalid only through an explicit mask band stays invalid."""
    height = width = 16
    band1 = np.full((height, width), 5.0)
    band2 = np.full((height, width), 5.0)
    band2[3, 3] = 0.0
    mask = np.full((height, width), 255, dtype=np.uint8)
    mask[:4, :4] = 0
    path = str(tmp_path / "src.tif")
    _write_two_band(path, band1, band2, mask=mask)
    model = _model(path, height, width, biases=(0.0, 0.0))

    with rasterio.open(path) as src:
        assert any(
            rasterio.enums.MaskFlags.per_dataset in flags
            for flags in src.mask_flag_enums
        )
        dst = np.full((2, height, width), DST_NODATA, dtype="float64")
        reproject(
            source=rasterio.band(src, [1, 2]),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=DST_NODATA,
            resampling=Resampling.nearest,
            balance_model=model,
            balance_source="S",
        )
    # The whole masked region is invalid even though values are finite.
    assert np.all(dst[:, :4, :4] == DST_NODATA)
    # The legal zero inside the masked region is not resurrected.
    assert dst[1, 3, 3] == DST_NODATA
    # Unmasked cells are valid.
    assert np.allclose(dst[:, 4:, 4:], 5.0)


@pytest.mark.parametrize(
    "resampling", [Resampling.nearest, Resampling.bilinear]
)
def test_reproject_constant_region_survives_resampling(tmp_path, resampling):
    """A fully valid constant region yields that constant under resampling."""
    height = width = 16
    band1 = np.full((height, width), 7.0)
    band2 = np.full((height, width), 7.0)
    # Holes/non-finite cells confined away from the checked constant corner.
    band1[15, 15] = SRC_NODATA
    band2[15, 14] = float("nan")
    path = str(tmp_path / "src.tif")
    _write_two_band(path, band1, band2)
    model = _model(path, height, width, biases=(9999.0, 0.0))

    with rasterio.open(path) as src:
        dst = np.full((2, height, width), DST_NODATA, dtype="float64")
        reproject(
            source=rasterio.band(src, [1, 2]),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=DST_NODATA,
            resampling=resampling,
            balance_model=model,
            balance_source="S",
        )
    # The interior constant corner is far from every invalid cell.
    np.testing.assert_allclose(dst[0, :3, :3], 10006.0, rtol=1e-12)
    np.testing.assert_allclose(dst[1, :3, :3], 7.0, rtol=1e-12)
    assert dst[0, 15, 15] == DST_NODATA
    assert dst[1, 15, 14] == DST_NODATA


def test_reproject_block_size_invariance(tmp_path):
    """Changing the destination block layout must not change the result."""
    height = width = 32
    rng = np.random.default_rng(7)
    band1 = rng.uniform(1, 100, (height, width))
    band2 = rng.uniform(1, 100, (height, width))
    band1[3, 4] = SRC_NODATA
    band2[5, 6] = 0.0
    path = str(tmp_path / "src.tif")
    _write_two_band(path, band1, band2)
    model = _model(path, height, width, biases=(50.0, -25.0))
    transform = from_origin(0.0, float(height), 1.0, 1.0)

    results = []
    for block in (16, 32):
        out = str(tmp_path / f"dst_{block}.tif")
        with rasterio.open(path) as src, rasterio.open(
            out, "w", driver="GTiff", height=height, width=width, count=2,
            dtype="float64", nodata=DST_NODATA, crs="EPSG:32631",
            transform=transform, tiled=True,
            blockxsize=block, blockysize=block,
        ) as dst:
            reproject(
                source=rasterio.band(src, [1, 2]),
                destination=rasterio.band(dst, [1, 2]),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst.crs,
                dst_nodata=DST_NODATA,
                resampling=Resampling.nearest,
                balance_model=model,
                balance_source="S",
            )
        with rasterio.open(out) as check:
            results.append((check.read(), check.read_masks()))

    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])


# ---------------------------------------------------------------------------
# merge: overflow / invalid candidates and first/last compositing
# ---------------------------------------------------------------------------


def _write_single(path, values, *, width=None, nodata=SRC_NODATA, left=0.0):
    height, w = values.shape
    width = w if width is None else width
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float64", nodata=nodata, crs="EPSG:32631",
        transform=from_origin(left, float(height), 1.0, 1.0),
    ) as ds:
        ds.write(values[None, ...])


def test_merge_overflow_cell_is_invalid_first_and_last(tmp_path):
    """The 1e308 cell calibrates past float64 and must become a hole."""
    a = np.arange(1, 65, dtype="float64").reshape(8, 8)
    pa = str(tmp_path / "a.tif")
    _write_single(pa, a)

    b = np.empty((8, 16))
    b[:, :8] = a / 2.0
    b[:, 8:] = 5.0
    b[0, 8] = 1e308
    pb = str(tmp_path / "b.tif")
    _write_single(pb, b, width=16)

    for method in ("first", "last"):
        out = str(tmp_path / f"merged_{method}.tif")
        with rasterio.open(pa) as da, rasterio.open(pb) as db:
            merge(
                [da, db],
                balance="affine",
                source_ids=["A", "B"],
                balance_reference="A",
                balance_report=str(tmp_path / f"report_{method}.json"),
                dst_path=out,
                method=method,
                nodata=DST_NODATA,
            )
        with rasterio.open(out) as ds:
            merged = ds.read(1)
            masks = ds.read_masks(1)
            report = json.load(open(str(tmp_path / f"report_{method}.json")))
        gain = report["bands"]["1"]["B"]["gain"]
        assert gain == pytest.approx(2.0, rel=1e-6)
        # Overflow cell is a finite nodata hole, never +/- infinity.
        assert np.isfinite(merged).all()
        assert merged[0, 8] == DST_NODATA
        assert masks[0, 8] == 0
        # Overlap keeps the reference field.
        np.testing.assert_allclose(merged[:, :8], a, rtol=1e-8)
        # The remaining right-side constant cells calibrate and stay valid.
        expected = gain * 5.0
        np.testing.assert_allclose(
            merged[:, 9:], expected, rtol=1e-8
        )
        assert masks[:, 9:].min() == 255


def test_merge_invalid_candidate_never_overwrites_valid(tmp_path):
    """first/last: an invalid candidate cannot displace an existing value."""
    # A and C fully cover an 8x8 grid; B shares the grid but carries an
    # overflow and a raw nodata hole at cells where A/C hold valid values.
    a = (np.arange(64, dtype="float64") + 10).reshape(8, 8)
    pa = str(tmp_path / "a.tif")
    _write_single(pa, a)
    c = a / 2.0
    pc = str(tmp_path / "c.tif")
    _write_single(pc, c)
    b = a / 2.0
    b[0, 0] = 1e308
    b[1, 1] = SRC_NODATA
    pb = str(tmp_path / "b.tif")
    _write_single(pb, b)

    for method in ("first", "last"):
        out = str(tmp_path / f"ov_{method}.tif")
        with rasterio.open(pa) as da, rasterio.open(pc) as dc, \
                rasterio.open(pb) as db:
            merge(
                [da, dc, db],
                balance="affine",
                source_ids=["A", "C", "B"],
                balance_reference="A",
                balance_report=str(tmp_path / f"ov_report_{method}.json"),
                dst_path=out,
                method=method,
                nodata=DST_NODATA,
            )
        with rasterio.open(out) as ds:
            merged = ds.read(1)
            masks = ds.read_masks(1)
        # Both problematic cells remain valid (A/C cover them) and finite.
        assert np.isfinite(merged[0, 0]) and np.isfinite(merged[1, 1])
        assert masks[0, 0] == 255 and masks[1, 1] == 255
        assert merged[0, 0] == pytest.approx(a[0, 0])
        assert merged[1, 1] == pytest.approx(a[1, 1])


def test_merge_no_source_cell_gets_nodata(tmp_path):
    """A cell every source marks invalid receives the output nodata."""
    # A spans 8 columns; B spans 12.  The 8-column overlap is a finite,
    # identifiable A = 2*B relation.  B's unique columns 8..11 hold overflow
    # cells that no other source covers.
    rng = np.random.default_rng(11)
    a = rng.uniform(20, 200, (8, 8))
    pa = str(tmp_path / "a.tif")
    with rasterio.open(
        pa, "w", driver="GTiff", height=8, width=8, count=1,
        dtype="float64", nodata=SRC_NODATA, crs="EPSG:32631",
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
    ) as ds:
        ds.write(a[None, ...])

    b = np.empty((8, 12))
    b[:, :8] = a / 2.0
    b[:, 8:] = 1e308            # unique region: every cell overflows
    pb = str(tmp_path / "b.tif")
    with rasterio.open(
        pb, "w", driver="GTiff", height=8, width=12, count=1,
        dtype="float64", nodata=SRC_NODATA, crs="EPSG:32631",
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
    ) as ds:
        ds.write(b[None, ...])

    out = str(tmp_path / "out.tif")
    with rasterio.open(pa) as da, rasterio.open(pb) as db:
        merge(
            [da, db],
            balance="affine",
            source_ids=["A", "B"],
            balance_reference="A",
            balance_report=str(tmp_path / "ns.json"),
            dst_path=out,
            method="last",
            nodata=DST_NODATA,
        )
    with rasterio.open(out) as ds:
        merged = ds.read(1)
        masks = ds.read_masks(1)
    # Unique columns have only overflowing B candidates -> nodata holes.
    assert np.all(merged[:, 8:] == DST_NODATA)
    assert np.all(masks[:, 8:] == 0)
    # The overlap columns are covered by finite observations.
    assert np.isfinite(merged[:, :8]).all()


def test_merge_default_nan_nodata_for_invalid(tmp_path):
    """Without an explicit nodata, invalid cells become NaN and mask out."""
    a = np.arange(1, 65, dtype="float64").reshape(8, 8)
    pa = str(tmp_path / "a.tif")
    _write_single(pa, a)
    b = np.empty((8, 16))
    b[:, :8] = a / 2.0
    b[:, 8:] = 5.0
    b[0, 8] = 1e308
    pb = str(tmp_path / "b.tif")
    _write_single(pb, b, width=16)

    with rasterio.open(pa) as da, rasterio.open(pb) as db:
        arr, _ = merge(
            [da, db],
            balance="affine",
            source_ids=["A", "B"],
            balance_reference="A",
            balance_report=str(tmp_path / "nan_report.json"),
            bounds=(0.0, 0.0, 16.0, 8.0),
            res=(1.0, 1.0),
        )
    assert arr.shape == (1, 8, 16)
    assert math.isnan(arr[0, 0, 8])
    finite = arr[np.isfinite(arr)]
    assert not np.isinf(finite).any()


# ---------------------------------------------------------------------------
# Nearest-neighbour cross-check: reproject separately, composite in order
# ---------------------------------------------------------------------------


def test_nearest_matches_separate_reproject_then_composite(tmp_path):
    """merge nearest output equals separately warped, order-composited data."""
    height = width = 16
    yy, xx = np.mgrid[0:height, 0:width]
    field = 10.0 + 3.0 * xx + 0.5 * yy
    paths = []
    for k, left in enumerate((0.0, 6.0)):
        vals = np.stack([field, field * 2.0]).astype("float64")
        vals[0, 2, 3] = SRC_NODATA
        vals[1, 5 + k, 4] = 0.0 if k == 0 else SRC_NODATA
        p = str(tmp_path / f"src_{k}.tif")
        with rasterio.open(
            p, "w", driver="GTiff", height=height, width=width, count=2,
            dtype="float64", nodata=SRC_NODATA, crs="EPSG:32631",
            transform=from_origin(left, float(height), 1.0, 1.0),
        ) as ds:
            ds.write(vals)
        paths.append(p)

    report = str(tmp_path / "fit.json")
    out = str(tmp_path / "merged.tif")
    with rasterio.open(paths[0]) as d0, rasterio.open(paths[1]) as d1:
        merge(
            [d0, d1],
            balance="affine",
            source_ids=["S0", "S1"],
            balance_reference="S0",
            balance_report=report,
            dst_path=out,
            method="first",
            nodata=DST_NODATA,
        )
    fitted = json.load(open(report))
    target_transform = from_origin(0.0, float(height), 1.0, 1.0)
    layers = []
    for k, p in enumerate(paths):
        with rasterio.open(p) as src:
            dst = np.full((2, height, 22), DST_NODATA, dtype="float64")
            reproject(
                source=rasterio.band(src, [1, 2]),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=src.crs,
                dst_nodata=DST_NODATA,
                resampling=Resampling.nearest,
                balance_model=fitted,
                balance_source=f"S{k}",
            )
            layers.append(dst)

    control = np.full_like(layers[0], DST_NODATA)
    for layer in layers:
        valid = np.isfinite(layer) & (layer != DST_NODATA)
        empty = ~(np.isfinite(control) & (control != DST_NODATA))
        np.copyto(control, layer, where=valid & empty)

    with rasterio.open(out) as ds:
        merged = ds.read()
    np.testing.assert_array_equal(
        np.isfinite(control) & (control != DST_NODATA),
        np.isfinite(merged) & (merged != DST_NODATA),
    )
    ctrl_valid = np.isfinite(control) & (control != DST_NODATA)
    np.testing.assert_allclose(
        control[ctrl_valid], merged[ctrl_valid], rtol=1e-12
    )


# ---------------------------------------------------------------------------
# Scale/offset handling and input immutability
# ---------------------------------------------------------------------------


def test_file_scale_offset_applied_once(tmp_path):
    """File scale/offset plus model bias compose correctly and are read once."""
    height = width = 8
    raw = np.full((height, width), 4.0)
    path = str(tmp_path / "scaled.tif")
    file_scale, file_offset = 2.0, 1.0
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float64", nodata=SRC_NODATA, crs="EPSG:32631",
        transform=from_origin(0.0, float(height), 1.0, 1.0),
    ) as ds:
        ds.write(raw[None, ...])
        ds.scales = [file_scale]
        ds.offsets = [file_offset]
    with rasterio.open(path) as ds:
        crs_wkt = ds.crs.to_wkt()
        gt = list(ds.transform.to_gdal())
    model = {
        "version": "1.0", "reference": "S",
        "sources": {"S": {
            "crs": crs_wkt, "transform": gt, "width": width, "height": height,
            "dtypes": ["float64"], "scales": [file_scale],
            "offsets": [file_offset], "band_indexes": [1],
            "nodata": [SRC_NODATA]}},
        "bands": {"1": {"S": {"gain": 3.0, "bias": 5.0}}},
        "diagnostics": {}, "tolerances": {},
    }
    # physical = raw*2 + 1 = 9; calibrated = 3*9 + 5 = 32
    with rasterio.open(path) as src:
        dst = np.full((1, height, width), DST_NODATA)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=DST_NODATA,
            resampling=Resampling.nearest,
            balance_model=model,
            balance_source="S",
        )
    np.testing.assert_allclose(dst[0], 32.0, rtol=1e-12)


def test_balance_reproject_does_not_mutate_inputs(tmp_path):
    """Source file, open dataset and passed model are left untouched."""
    import hashlib
    import copy

    height = width = 8
    band1 = np.full((height, width), 6.0)
    band2 = np.full((height, width), 6.0)
    band1[2, 3] = SRC_NODATA
    band2[5, 4] = 0.0
    path = str(tmp_path / "src.tif")
    _write_two_band(path, band1, band2)
    model = _model(path, height, width, biases=(9999.0, 0.0))
    model_snapshot = copy.deepcopy(model)

    def digest(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()

    before = digest(path)
    with rasterio.open(path) as src:
        raw_before = src.read().copy()
        scales_before = list(src.scales)
        offsets_before = list(src.offsets)
        dst = np.full((2, height, width), DST_NODATA)
        reproject(
            source=rasterio.band(src, [1, 2]),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=DST_NODATA,
            resampling=Resampling.nearest,
            balance_model=model,
            balance_source="S",
        )
        np.testing.assert_array_equal(src.read(), raw_before)
        assert list(src.scales) == scales_before
        assert list(src.offsets) == offsets_before
    assert digest(path) == before
    assert model == model_snapshot


@pytest.mark.slow
@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RSS via getrusage"
)
def test_balanced_reproject_streaming_memory(tmp_path):
    """File-to-file balanced reproject stays within a tight RSS budget.

    Two 4096x4096 two-band float64 tiled sources are warped to files in a
    clean process with a 16 MiB GDAL block cache; the extra peak RSS over
    the post-import idle baseline must stay under 128 MiB.  Returning a full
    ndarray is deliberately avoided (that path is exempt from the budget).
    """
    import resource
    import subprocess
    import textwrap

    size = 4096
    block = 256
    work = tmp_path / "mem"
    work.mkdir()
    helper = work / "helper.py"
    helper.write_text(
        textwrap.dedent(f"""
            import os
            import resource
            import numpy as np
            import rasterio
            import rasterio.windows as windows_mod
            from rasterio.enums import Resampling
            from rasterio.transform import from_origin
            from rasterio.warp import reproject

            size = {size}
            block = {block}
            work = {str(work)!r}
            transform = from_origin(0.0, float(size), 1.0, 1.0)
            model = {{
                "version": "1.0", "reference": "S",
                "sources": {{"S": {{
                    "crs": None,
                    "transform": list(transform.to_gdal()),
                    "width": size, "height": size,
                    "dtypes": ["float64", "float64"],
                    "scales": [1.0, 1.0], "offsets": [0.0, 0.0],
                    "band_indexes": [1, 2],
                    "nodata": [{SRC_NODATA}, {SRC_NODATA}]}}}},
                "bands": {{
                    "1": {{"S": {{"gain": 1.0, "bias": 1.0}}}},
                    "2": {{"S": {{"gain": 1.0, "bias": 0.0}}}}}},
                "diagnostics": {{}}, "tolerances": {{}},
            }}
            # Baseline after import; input generation that follows is
            # excluded from the budget, which measures the warp's own extra
            # working set.
            idle = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            for k in range(2):
                src_path = os.path.join(work, f"big_src{{k}}.tif")
                dst_path = os.path.join(work, f"big_dst{{k}}.tif")
                with rasterio.open(
                    src_path, "w", driver="GTiff", height=size, width=size,
                    count=2, dtype="float64", nodata={SRC_NODATA},
                    crs="EPSG:32631", transform=transform,
                    tiled=True, blockxsize=block, blockysize=block,
                ) as ds:
                    # Write in row strips to keep the writer's footprint low.
                    ones = np.ones((2, block, size), dtype="float64")
                    for r in range(0, size, block):
                        ds.write(
                            ones * (1.0 + r),
                            window=windows_mod.Window(0, r, size, block),
                        )
                with rasterio.Env(GDAL_CACHEMAX=16):
                    with rasterio.open(src_path) as src:
                        model["sources"]["S"]["crs"] = src.crs.to_wkt()
                        with rasterio.open(
                            dst_path, "w", driver="GTiff", height=size,
                            width=size, count=2, dtype="float64",
                            nodata={DST_NODATA}, crs="EPSG:32631",
                            transform=transform, tiled=True,
                            blockxsize=block, blockysize=block,
                        ) as dst:
                            reproject(
                                source=rasterio.band(src, [1, 2]),
                                destination=rasterio.band(dst, [1, 2]),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=dst.crs,
                                dst_nodata={DST_NODATA},
                                resampling=Resampling.nearest,
                                balance_model=model,
                                balance_source="S",
                            )
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            print(f"IDLE={{idle:.1f}} PEAK={{peak:.1f}} EXTRA={{peak-idle:.1f}}")
        """)
    )
    env = dict(**__import__("os").environ, GDAL_CACHEMAX="16")
    proc = subprocess.run(
        [sys.executable, str(helper)], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("IDLE=")][0]
    extra = float(line.split("EXTRA=")[1])
    assert extra <= 128.0, line


@pytest.mark.slow
@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RSS via getrusage"
)
def test_balanced_merge_streaming_memory(tmp_path):
    """File-to-file balanced merge stays within a 128 MiB RSS budget.

    Two 4096x4096 two-band float64 tiled sources are merged to a file in a
    clean process with a 16 MiB GDAL block cache; extra peak RSS over the
    post-import idle baseline must stay under 128 MiB.  The merge result is
    written to disk; no full ndarray is returned.
    """
    import subprocess
    import textwrap

    size = 4096
    block = 256
    work = tmp_path / "merge_mem"
    work.mkdir()
    helper = work / "helper.py"
    helper.write_text(
        textwrap.dedent(f"""
            import os
            import resource
            import numpy as np
            import rasterio
            import rasterio.windows as windows_mod
            from rasterio.merge import merge
            from rasterio.transform import from_origin

            size = {size}
            block = {block}
            work = {str(work)!r}
            transform = from_origin(0.0, float(size), 1.0, 1.0)
            paths = []
            # Two overlapping sources: B is shifted right by half a raster,
            # and its stored values are A/2 so the fitted gain is ~2.
            # Inputs are generated one strip at a time so the preparation
            # stage never holds a full raster; the memory budget covers the
            # merge operation itself.
            rng = np.random.default_rng(20261119)
            for k, left in enumerate((0.0, size / 2.0)):
                path = os.path.join(work, f"merge_src{{k}}.tif")
                with rasterio.open(
                    path, "w", driver="GTiff", height=size, width=size,
                    count=2, dtype="float64", nodata={SRC_NODATA},
                    crs="EPSG:32631",
                    transform=from_origin(left, float(size), 1.0, 1.0),
                    tiled=True, blockxsize=block, blockysize=block,
                ) as ds:
                    strip = np.empty((2, block, size), dtype="float64")
                    for r in range(0, size, block):
                        base = rng.uniform(20.0, 200.0, (block, size))
                        strip[0] = base if k == 0 else base / 2.0
                        strip[1] = strip[0] * 0.5 + 1.0
                        ds.write(
                            strip,
                            window=windows_mod.Window(0, r, size, block),
                        )
                paths.append(path)

            # Baseline after import and input preparation; the budget applies
            # to the merge's additional working set.
            idle = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            with rasterio.Env(GDAL_CACHEMAX=16):
                datasets = [rasterio.open(p) for p in paths]
                try:
                    merge(
                        datasets,
                        balance="affine",
                        source_ids=["S0", "S1"],
                        balance_reference="S0",
                        balance_report=os.path.join(work, "report.json"),
                        balance_sample_limit=100000,
                        dst_path=os.path.join(work, "merged.tif"),
                        nodata={DST_NODATA},
                        mem_limit=16,
                    )
                finally:
                    for ds in datasets:
                        ds.close()
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            print(f"IDLE={{idle:.1f}} PEAK={{peak:.1f}} EXTRA={{peak-idle:.1f}}")
        """)
    )
    env = dict(**__import__("os").environ, GDAL_CACHEMAX="16")
    proc = subprocess.run(
        [sys.executable, str(helper)], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0, proc.stderr
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("IDLE=")][0]
    extra = float(line.split("EXTRA=")[1])
    assert extra <= 128.0, line
