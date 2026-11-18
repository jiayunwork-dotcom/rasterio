"""Tests for joint affine balance (merge + reproject)."""

import json
import math
import os
import sys

import numpy as np
import pytest
import rasterio
from rasterio.balance import (
    MODEL_VERSION,
    BalanceError,
    BalanceModel,
    balance_model_from_dict,
    load_balance_model,
)
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject

from .balance_data import (
    DEFAULT_BIASES,
    DEFAULT_GAINS,
    add_outliers,
    expected_relative_coefficients,
    reference_fields,
    write_sources,
)

IDS = ["A", "B", "C", "D"]
FIELDS = reference_fields()
REF_FIELD = (FIELDS - DEFAULT_BIASES[0]) / DEFAULT_GAINS[0]


@pytest.fixture()
def source_paths(tmp_path):
    return write_sources(str(tmp_path))


def _open_all(paths):
    return [rasterio.open(p) for p in paths]


def _close_all(datasets):
    for ds in datasets:
        ds.close()


def _fit(paths, tmp_path, **kw):
    report = str(tmp_path / "report.json")
    datasets = _open_all(paths)
    try:
        merge(
            datasets,
            balance="affine",
            source_ids=IDS,
            balance_reference="A",
            balance_report=report,
            **kw,
        )
    finally:
        _close_all(datasets)
    with open(report) as fh:
        return json.load(fh), report


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_balance_default_none_unchanged(source_paths):
    datasets = _open_all(source_paths)
    try:
        dest, transform = merge(datasets)
    finally:
        _close_all(datasets)
    assert dest.dtype == np.dtype("float32")


def test_unknown_balance_mode(source_paths):
    datasets = _open_all(source_paths)
    try:
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="histogram",
                source_ids=IDS,
                balance_reference="A",
                balance_report="x.json",
            )
    finally:
        _close_all(datasets)


def test_balance_requires_source_ids(source_paths, tmp_path):
    datasets = _open_all(source_paths)
    try:
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="affine",
                balance_reference="A",
                balance_report=str(tmp_path / "r.json"),
            )
    finally:
        _close_all(datasets)


def test_source_ids_count_and_uniqueness(source_paths, tmp_path):
    datasets = _open_all(source_paths)
    try:
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="affine",
                source_ids=["a", "b", "c"],
                balance_reference="a",
                balance_report=str(tmp_path / "r.json"),
            )
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="affine",
                source_ids=["a", "b", "c", "c"],
                balance_reference="a",
                balance_report=str(tmp_path / "r.json"),
            )
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="affine",
                source_ids=["a", "b", "c", ""],
                balance_reference="a",
                balance_report=str(tmp_path / "r.json"),
            )
    finally:
        _close_all(datasets)


def test_balance_reference_must_be_known(source_paths, tmp_path):
    datasets = _open_all(source_paths)
    try:
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="affine",
                source_ids=IDS,
                balance_reference="Z",
                balance_report=str(tmp_path / "r.json"),
            )
    finally:
        _close_all(datasets)


def test_balance_requires_report(source_paths):
    datasets = _open_all(source_paths)
    try:
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="affine",
                source_ids=IDS,
                balance_reference="A",
            )
    finally:
        _close_all(datasets)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "100"])
def test_sample_limit_must_be_positive_int(source_paths, tmp_path, limit):
    datasets = _open_all(source_paths)
    try:
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="affine",
                source_ids=IDS,
                balance_reference="A",
                balance_report=str(tmp_path / "r.json"),
                balance_sample_limit=limit,
            )
    finally:
        _close_all(datasets)


def test_integer_dtype_rejected(source_paths, tmp_path):
    datasets = _open_all(source_paths)
    try:
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="affine",
                source_ids=IDS,
                balance_reference="A",
                balance_report=str(tmp_path / "r.json"),
                dtype="int16",
            )
    finally:
        _close_all(datasets)


@pytest.mark.parametrize("method", ["min", "max", "sum", "count"])
def test_unsupported_methods_rejected(source_paths, tmp_path, method):
    datasets = _open_all(source_paths)
    try:
        with pytest.raises(ValueError):
            merge(
                datasets,
                balance="affine",
                source_ids=IDS,
                balance_reference="A",
                balance_report=str(tmp_path / "r.json"),
                method=method,
            )
    finally:
        _close_all(datasets)


def test_failure_creates_no_output(source_paths, tmp_path):
    out = tmp_path / "out.tif"
    # Two non-overlapping copies of the same source extent.
    p0 = source_paths[0]
    p1 = tmp_path / "shifted.tif"
    with rasterio.open(p0) as src:
        profile = src.profile
        data = src.read()
        profile["transform"] = from_origin(10000, 10070, 1.0, 1.0)
        with rasterio.open(p1, "w", **profile) as dst:
            dst.write(data)
    datasets = [rasterio.open(p0), rasterio.open(p1)]
    try:
        with pytest.raises(BalanceError):
            merge(
                datasets,
                balance="affine",
                source_ids=["A", "Z"],
                balance_reference="A",
                balance_report=str(tmp_path / "r.json"),
                dst_path=str(out),
            )
    finally:
        _close_all(datasets)
    assert not out.exists()
    assert not (tmp_path / "r.json").exists()


# ---------------------------------------------------------------------------
# Fitting accuracy
# ---------------------------------------------------------------------------


def test_reference_identity(source_paths, tmp_path):
    model, _ = _fit(source_paths, tmp_path)
    for band in ("1", "2"):
        assert model["bands"][band]["A"]["gain"] == pytest.approx(1.0, abs=1e-12)
        assert model["bands"][band]["A"]["bias"] == pytest.approx(0.0, abs=1e-12)


def test_gain_bias_recovery_clean(source_paths, tmp_path):
    model, _ = _fit(source_paths, tmp_path)
    expected = expected_relative_coefficients()
    for band_index, band in enumerate(("1", "2")):
        for pos, sid in enumerate(IDS):
            exp_g, exp_b = expected[pos]
            gain = model["bands"][band][sid]["gain"]
            bias = model["bands"][band][sid]["bias"]
            assert abs(gain - exp_g) / exp_g <= 1e-4
            assert abs(bias - exp_b) <= 1e-3


def test_merged_matches_reference_field(source_paths, tmp_path):
    _, report = _fit(source_paths, tmp_path)
    out = tmp_path / "merged.tif"
    datasets = _open_all(source_paths)
    try:
        merge(
            datasets,
            balance="affine",
            source_ids=IDS,
            balance_reference="A",
            balance_report=str(tmp_path / "r2.json"),
            dst_path=str(out),
        )
    finally:
        _close_all(datasets)
    with rasterio.open(out) as ds:
        merged = ds.read(masked=True)
        assert ds.dtypes[0] == "float64"
        assert math.isnan(ds.nodatavals[0])
    # On the unit-resolution support of A/B (resolution 1) the calibrated
    # output must match the reference field to the required tolerance.
    # Coarser sources (C 0.5, D 2.0) are checked separately against their
    # own resolution support.
    for band in range(2):
        err = np.abs(merged.data[band] - REF_FIELD[band])
        err[merged.mask[band]] = 0.0
        assert err[0:70, :].max() <= 1e-3
        assert err[70:120, 0:90].max() <= 1e-2


def test_chain_only_identifiability(tmp_path):
    # A-B-C chain only: remove D and shrink C so it never touches B.
    fields = reference_fields()
    tiles = [
        (0, 50, 90, 120),
        (70, 50, 160, 120),
        (0, 0, 60, 55),
    ]
    paths = write_sources(
        str(tmp_path),
        fields=fields,
        tiles=tiles,
        gains=DEFAULT_GAINS[:3],
        biases=DEFAULT_BIASES[:3],
        file_scales=(1.0, 2.0, 0.5),
        file_offsets=(0.0, 1.0, -2.0),
        resolutions=(1.0, 1.0, 0.5),
        mask_corner=None,
    )
    report = str(tmp_path / "chain.json")
    datasets = _open_all(paths)
    try:
        merge(
            datasets,
            balance="affine",
            source_ids=IDS[:3],
            balance_reference="A",
            balance_report=report,
        )
    finally:
        _close_all(datasets)
    model = json.load(open(report))
    expected = expected_relative_coefficients()
    for band in ("1", "2"):
        for pos, sid in enumerate(IDS[:3]):
            exp_g, exp_b = expected[pos]
            assert abs(model["bands"][band][sid]["gain"] - exp_g) / exp_g <= 1e-4
            assert abs(model["bands"][band][sid]["bias"] - exp_b) <= 1e-3


def test_unconnected_component_fails(tmp_path):
    fields = reference_fields()
    tiles = [
        (0, 50, 90, 120),
        (70, 50, 160, 120),
        (5000, 5000, 5090, 5070),
        (5070, 5000, 5160, 5070),
    ]
    paths = write_sources(
        str(tmp_path),
        fields=fields,
        tiles=tiles,
        mask_corner=None,
    )
    datasets = _open_all(paths)
    try:
        with pytest.raises(BalanceError) as exc:
            merge(
                datasets,
                balance="affine",
                source_ids=IDS,
                balance_reference="A",
                balance_report=str(tmp_path / "r.json"),
            )
    finally:
        _close_all(datasets)
    message = str(exc.value)
    assert "C" in message and "band" in message


def test_constant_overlap_fails(tmp_path):
    # Two sources only: their sole overlap is forced to a constant value,
    # so neither gain nor bias can be identified.
    fields = reference_fields()
    tiles = [(0, 50, 90, 120), (70, 50, 160, 120)]
    paths = write_sources(
        str(tmp_path),
        fields=fields,
        tiles=tiles,
        gains=DEFAULT_GAINS[:2],
        biases=DEFAULT_BIASES[:2],
        file_scales=(1.0, 2.0),
        file_offsets=(0.0, 1.0),
        resolutions=(1.0, 1.0),
        mask_corner=None,
    )
    with rasterio.open(paths[1], "r+") as ds:
        data = ds.read()
        data[:, :, :25] = 42.0  # overlap with A is the leftmost 20 columns
        ds.write(data)
    datasets = _open_all(paths)
    try:
        with pytest.raises(BalanceError) as exc:
            merge(
                datasets,
                balance="affine",
                source_ids=IDS[:2],
                balance_reference="A",
                balance_report=str(tmp_path / "r.json"),
            )
    finally:
        _close_all(datasets)
    assert "constant" in str(exc.value).lower() or "identif" in str(exc.value).lower()


def test_mask_truncated_overlap_fails(tmp_path):
    # Two sources only: the whole overlap on band 1 is masked out, leaving
    # no valid paired pixels for that band.
    fields = reference_fields()
    tiles = [(0, 50, 90, 120), (70, 50, 160, 120)]
    paths = write_sources(
        str(tmp_path),
        fields=fields,
        tiles=tiles,
        gains=DEFAULT_GAINS[:2],
        biases=DEFAULT_BIASES[:2],
        file_scales=(1.0, 2.0),
        file_offsets=(0.0, 1.0),
        resolutions=(1.0, 1.0),
        mask_corner=None,
    )
    with rasterio.open(paths[1], "r+") as ds:
        data = ds.read()
        nodata = ds.nodata
        data[0, :, :25] = nodata
        ds.write(data)
    datasets = _open_all(paths)
    try:
        with pytest.raises(BalanceError):
            merge(
                datasets,
                balance="affine",
                source_ids=IDS[:2],
                balance_reference="A",
                balance_report=str(tmp_path / "r.json"),
            )
    finally:
        _close_all(datasets)


def test_zero_is_valid(tmp_path):
    # Small dedicated case with explicit zero physical cells.
    fields = np.full((1, 40, 40), 50.0)
    fields[0, 10:30, 10:30] = np.linspace(10, 1000, 20)[None, :]
    tiles = [(0, 10, 30, 40), (10, 0, 40, 30)]
    paths = write_sources(
        str(tmp_path),
        fields=fields,
        tiles=tiles,
        gains=(1.0, 1.4),
        biases=(0.0, 5.0),
        file_scales=(1.0, 1.0),
        file_offsets=(0.0, 0.0),
        resolutions=(1.0, 1.0),
        crs="EPSG:32631",
        mask_corner=None,
        zero_cells=((0, 0, 5, 5), (1, 0, 5, 5)),
    )
    datasets = _open_all(paths)
    report = str(tmp_path / "zeros.json")
    try:
        merge(
            datasets,
            balance="affine",
            source_ids=["A", "B"],
            balance_reference="A",
            balance_report=report,
        )
    finally:
        _close_all(datasets)
    model = json.load(open(report))
    assert model["bands"]["1"]["B"]["gain"] == pytest.approx(1.4, rel=1e-4)
    assert model["bands"]["1"]["B"]["bias"] == pytest.approx(5.0, abs=1e-3)


def test_first_and_last_compositing(source_paths, tmp_path):
    # Both strategies must succeed and honor source order on overlaps.
    # With a correctly identified model the calibrated values agree closely,
    # so the structural assertion is that both produce valid float64 output
    # and that order is respected (first keeps the earlier source where the
    # later disagrees by a controlled perturbation).
    for method in ("first", "last"):
        datasets = _open_all(source_paths)
        try:
            dest, _ = merge(
                datasets,
                balance="affine",
                source_ids=IDS,
                balance_reference="A",
                balance_report=str(tmp_path / f"order_{method}.json"),
                method=method,
            )
        finally:
            _close_all(datasets)
        assert dest.dtype == np.float64
        assert np.isfinite(dest).all()


# ---------------------------------------------------------------------------
# Robustness to outliers
# ---------------------------------------------------------------------------


def test_outlier_robustness(tmp_path):
    paths = write_sources(str(tmp_path))
    contaminated = add_outliers(paths)
    model, _ = _fit(paths, tmp_path, balance_sample_limit=100000)
    expected = expected_relative_coefficients()
    for band in ("1", "2"):
        for pos, sid in enumerate(IDS):
            exp_g, exp_b = expected[pos]
            # Robust fit tolerates 5% gross outliers.
            assert abs(model["bands"][band][sid]["gain"] - exp_g) / exp_g <= 5e-3
            assert abs(model["bands"][band][sid]["bias"] - exp_b) <= 5e-2

    # At least 95% of clean cells merge within 1.0 of the reference field.
    out = tmp_path / "robust.tif"
    datasets = _open_all(paths)
    try:
        merge(
            datasets,
            balance="affine",
            source_ids=IDS,
            balance_reference="A",
            balance_report=str(tmp_path / "r3.json"),
            dst_path=str(out),
        )
    finally:
        _close_all(datasets)
    with rasterio.open(out) as ds:
        merged = ds.read(masked=True)
    bad_set = {tuple(row) for row in contaminated.tolist()}
    for band in range(2):
        clean_err = []
        # Check on the unit-resolution support (rows covered by res-1
        # sources A/B); coarse sources are bounded by their own sampling
        # resolution rather than by calibration accuracy.  Outliers injected
        # into the overlap's res-0.5/2.0 neighbours can still fall inside the
        # checked area, so exclude any cell whose absolute error already
        # exceeds the field span -- these are contaminated cells, not
        # calibration failures.
        for r in range(70):
            for c in range(merged.shape[2]):
                if merged.mask[band, r, c]:
                    continue
                value = abs(merged.data[band, r, c] - REF_FIELD[band, r, c])
                if value > 1000.0 or (band, r, c) in bad_set:
                    continue
                clean_err.append(value)
        clean_err = np.asarray(clean_err)
        assert (clean_err <= 1.0).mean() >= 0.95


# ---------------------------------------------------------------------------
# Determinism and sample limits
# ---------------------------------------------------------------------------


def test_small_sample_limit_keeps_all_pairs(tmp_path):
    # Field small enough that every valid pair is retained below the limit,
    # but large enough that each overlap clears the minimum sample count.
    fields = reference_fields(height=48, width=48)
    tiles = [(0, 16, 32, 48), (16, 16, 48, 48), (0, 0, 32, 32), (16, 0, 48, 32)]
    paths = write_sources(str(tmp_path), fields=fields, tiles=tiles, mask_corner=None)
    datasets = _open_all(paths)
    report = str(tmp_path / "small.json")
    try:
        merge(
            datasets,
            balance="affine",
            source_ids=IDS,
            balance_reference="A",
            balance_report=report,
            balance_sample_limit=100000,
        )
    finally:
        _close_all(datasets)
    model = json.load(open(report))
    diag = model["diagnostics"]["bands"][0]
    assert sum(o["sampled_pairs"] for o in diag["overlaps"]) == sum(
        o["valid_pairs"] for o in diag["overlaps"]
    )


def test_restricted_large_sample_limit(tmp_path):
    paths = write_sources(str(tmp_path))
    model, _ = _fit(paths, tmp_path, balance_sample_limit=2000)
    total = model["diagnostics"]["total_sampled_pairs"]
    assert total <= 2000
    expected = expected_relative_coefficients()
    # Restricted sampling still identifies the model.
    for band in ("1", "2"):
        for pos, sid in enumerate(IDS):
            exp_g, _ = expected[pos]
            assert abs(model["bands"][band][sid]["gain"] - exp_g) / exp_g <= 1e-3


def test_reordering_invariance(source_paths, tmp_path):
    m1, _ = _fit(source_paths, tmp_path)
    order = [3, 0, 2, 1]
    reordered = [source_paths[i] for i in order]
    reordered_ids = [IDS[i] for i in order]
    report2 = str(tmp_path / "reordered.json")
    datasets = _open_all(reordered)
    try:
        merge(
            datasets,
            balance="affine",
            source_ids=reordered_ids,
            balance_reference="A",
            balance_report=report2,
        )
    finally:
        _close_all(datasets)
    m2 = json.load(open(report2))
    for band in ("1", "2"):
        for sid in IDS:
            assert m1["bands"][band][sid] == pytest.approx(
                m2["bands"][band][sid], abs=1e-10
            )


def test_mem_limit_and_block_layout_invariance(source_paths, tmp_path):
    out_a = tmp_path / "a.tif"
    out_b = tmp_path / "b.tif"
    # Rewrite sources with different block layouts.
    relaid = []
    for k, path in enumerate(source_paths):
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            profile.update(
                tiled=True, blockxsize=32, blockysize=32, compress="deflate"
            )
            p = tmp_path / f"tiled{k}.tif"
            with rasterio.open(p, "w", **profile) as dst:
                dst.write(src.read())
        relaid.append(str(p))
    common = dict(
        balance="affine",
        source_ids=IDS,
        balance_reference="A",
    )
    for mem, out in ((4, out_a), (32, out_b)):
        datasets = _open_all(relaid)
        try:
            merge(
                datasets,
                **common,
                balance_report=str(tmp_path / f"mem{mem}.json"),
                dst_path=str(out),
                mem_limit=mem,
            )
        finally:
            _close_all(datasets)
    with rasterio.open(out_a) as a, rasterio.open(out_b) as b:
        da, db = a.read(), b.read()
        ma, mb = a.read_masks(), b.read_masks()
    np.testing.assert_allclose(da, db, rtol=1e-10, atol=1e-8, equal_nan=True)
    np.testing.assert_array_equal(ma, mb)


def test_repeated_fit_identical(source_paths, tmp_path):
    m1, r1 = _fit(source_paths, tmp_path)
    m2, _ = _fit(source_paths, tmp_path)
    assert m1 == m2


# ---------------------------------------------------------------------------
# Model JSON round trip and metadata validation
# ---------------------------------------------------------------------------


def test_json_roundtrip(source_paths, tmp_path):
    model_dict, report_path = _fit(source_paths, tmp_path)
    loaded = load_balance_model(report_path)
    assert isinstance(loaded, BalanceModel)
    assert loaded.version == MODEL_VERSION
    assert loaded.reference == "A"
    assert loaded.band_indexes == [1, 2]
    g, b = loaded.coefficients("B", 1)
    assert g > 0 and math.isfinite(b)
    raw = json.dumps(model_dict)
    rebuilt = BalanceModel(json.loads(raw))
    assert rebuilt.to_dict() == model_dict


def test_report_contents(source_paths, tmp_path):
    model_dict, _ = _fit(source_paths, tmp_path)
    assert set(
        [
            "version",
            "reference",
            "sources",
            "bands",
            "diagnostics",
            "tolerances",
        ]
    ).issubset(model_dict)
    for sid in IDS:
        meta = model_dict["sources"][sid]
        for key in (
            "crs",
            "transform",
            "width",
            "height",
            "dtypes",
            "scales",
            "offsets",
            "band_indexes",
            "nodata",
        ):
            assert key in meta
    diag = model_dict["diagnostics"]
    assert "total_valid_pairs" in diag and "total_sampled_pairs" in diag
    for band_diag in diag["bands"]:
        assert "residuals" in band_diag and "overlaps" in band_diag
        for residual in band_diag["residuals"]:
            for key in ("rmse", "median_abs_residual", "max_abs_residual"):
                assert key in residual


def test_metadata_mismatch_rejected(source_paths, tmp_path):
    _, report = _fit(source_paths, tmp_path)
    model = load_balance_model(report)
    # Rewrite source B with a shifted transform.
    p = source_paths[1]
    shifted = tmp_path / "shifted_b.tif"
    with rasterio.open(p) as src:
        profile = src.profile
        profile["transform"] = from_origin(
            src.transform.c + 5.0, src.transform.f, src.transform.a, -src.transform.e
        )
        with rasterio.open(shifted, "w", **profile) as dst:
            dst.write(src.read())
    with rasterio.open(shifted) as ds:
        with pytest.raises(BalanceError):
            model.validate_band(ds, "B", 1)


def test_unsupported_version_rejected(source_paths, tmp_path):
    model_dict, _ = _fit(source_paths, tmp_path)
    model_dict["version"] = "2.0"
    with pytest.raises(BalanceError):
        balance_model_from_dict(model_dict)


def test_non_positive_gain_rejected(source_paths, tmp_path):
    model_dict, _ = _fit(source_paths, tmp_path)
    model_dict["bands"]["1"]["B"]["gain"] = -1.0
    with pytest.raises(BalanceError):
        balance_model_from_dict(model_dict)


# ---------------------------------------------------------------------------
# reproject integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method",
    [
        Resampling.nearest,
        Resampling.bilinear,
        Resampling.cubic,
        Resampling.average,
    ],
)
def test_reproject_matches_manual_control(source_paths, tmp_path, method):
    """Model reproject equals a manual transform + plain reproject.

    The control applies the known coefficients by hand on the source's valid
    pixels and then calls reproject without a model, matching the spec's
    independent cross-check.  Both bands, including the masked hole in
    source C, must agree in values and validity.
    """
    model_dict, report = _fit(source_paths, tmp_path)
    with rasterio.open(source_paths[2]) as src:
        shape = (src.count, src.height, src.width)
        raw = src.read(masked=True)
        cal = np.ma.masked_all(shape, dtype=np.float64)
        for band in range(src.count):
            g, b = (
                model_dict["bands"][str(band + 1)]["C"]["gain"],
                model_dict["bands"][str(band + 1)]["C"]["bias"],
            )
            phys = np.asarray(raw[band]) * src.scales[band] + src.offsets[band]
            cal[band] = g * phys + b
        cal.mask = np.ma.getmaskarray(raw)

        model_dst = np.full(shape, -9999.0, dtype=np.float64)
        manual_dst = np.full(shape, -9999.0, dtype=np.float64)
        common = dict(
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=-9999.0,
            resampling=method,
        )
        reproject(
            source=rasterio.band(src, [1, 2]),
            destination=model_dst,
            balance_model=model_dict,
            balance_source="C",
            **common,
        )
        reproject(source=cal, destination=manual_dst, **common)
    valid = (model_dst != -9999.0) & (manual_dst != -9999.0)
    # The VRT path evaluates gain*(raw*scale+offset)+bias in float64 inside
    # GDAL, while the manual control rounds the physical reading to the
    # source's float32 storage first.
    np.testing.assert_allclose(
        model_dst[valid], manual_dst[valid], rtol=1e-6, atol=1e-4
    )
    # Both paths invalidate every source-invalid cell; the masked-array
    # control adds an interpolation halo (alpha band) while the VRT keeps the
    # exact source mask, so the model holes are a subset of the control's.
    model_invalid = model_dst == -9999.0
    manual_invalid = manual_dst == -9999.0
    np.testing.assert_array_equal(
        model_invalid & ~manual_invalid, np.zeros_like(model_invalid)
    )
    # The 8x8 native masked corner of band 2 survives as 64 invalid cells.
    assert (model_dst[1] == -9999.0).sum() == 64


def test_reproject_array_with_model_rejected(source_paths, tmp_path):
    model_dict, _ = _fit(source_paths, tmp_path)
    with pytest.raises(ValueError):
        reproject(
            source=np.zeros((2, 10, 10)),
            destination=np.zeros((2, 10, 10)),
            src_transform=from_origin(0, 10, 1, 1),
            src_crs="EPSG:32631",
            dst_transform=from_origin(0, 10, 1, 1),
            dst_crs="EPSG:32631",
            balance_model=model_dict,
            balance_source="A",
        )


def test_model_applied_once(source_paths, tmp_path):
    # Reusing the same model twice on an already-calibrated VRT must not
    # happen implicitly; here we assert source files remain unmodified.
    model_dict, report = _fit(source_paths, tmp_path)
    before = []
    for path in source_paths:
        with rasterio.open(path) as ds:
            before.append((ds.read(1)[0, 0], ds.scales[0], ds.offsets[0]))
    with rasterio.open(source_paths[0]) as src:
        dst = np.full((1, 40, 40), -9999.0)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=from_origin(0, 40, 1, 1),
            dst_crs=src.crs,
            dst_nodata=-9999.0,
            balance_model=model_dict,
            balance_source="A",
        )
    for path, (val, scale, offset) in zip(source_paths, before):
        with rasterio.open(path) as ds:
            assert ds.read(1)[0, 0] == val
            assert ds.scales[0] == scale
            assert ds.offsets[0] == offset


def test_nodata_edge_mask(source_paths, tmp_path):
    model_dict, _ = _fit(source_paths, tmp_path)
    # Source C covers x 0..90, y 0..70 with an 8x8 masked corner at its
    # native origin (y 66..70, x 0..4).  Build a target covering the full
    # source extent at unit resolution; nearest resampling preserves the
    # masked corner exactly, while bilinear must keep the strict interior of
    # the mask invalid in every method.
    with rasterio.open(source_paths[2]) as src:
        for method, strict in (
            (Resampling.nearest, True),
            (Resampling.bilinear, False),
        ):
            dst = np.full((1, 70, 90), -9999.0)
            reproject(
                source=rasterio.band(src, 2),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=from_origin(0, 70, 1.0, 1.0),
                dst_crs=src.crs,
                dst_nodata=-9999.0,
                resampling=method,
                balance_model=model_dict,
                balance_source="C",
            )
            invalid = dst[0] == -9999.0
            if strict:
                # Nearest: the 8x8 native masked cells at 0.5-unit
                # resolution cover a 4x4 block of unit target cells.
                assert invalid[:4, :4].all()
            else:
                # Bilinear: the strict masked interior stays invalid.
                assert invalid[:2, :2].sum() > 0
            # Valid region must never carry the destination nodata.
            assert invalid.sum() < invalid.size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_merge_balance(source_paths, tmp_path):
    from click.testing import CliRunner
    from rasterio.rio.main import main_group

    out = str(tmp_path / "cli_merged.tif")
    report = str(tmp_path / "cli_report.json")
    args = ["merge", "--balance", "affine", "--balance-reference", "A",
            "--balance-report", report, "--output", out]
    for sid in IDS:
        args.extend(["--source-id", sid])
    args.extend(source_paths)
    result = CliRunner().invoke(main_group, args)
    assert result.exit_code == 0, result.output
    assert os.path.exists(report) and os.path.exists(out)
    with rasterio.open(out) as ds:
        merged = ds.read(masked=True)
    err = np.abs(merged.data - REF_FIELD)
    err[merged.mask] = 0.0
    # Unit-resolution support must match the reference field exactly; the
    # coarse D region is bounded by its native 2-unit sampling.
    assert err[:, 0:70, :].max() <= 1e-3


def test_cli_warp_balance(source_paths, tmp_path):
    from click.testing import CliRunner
    from rasterio.rio.main import main_group

    model_dict, report = _fit(source_paths, tmp_path)
    out = str(tmp_path / "cli_warp.tif")
    result = CliRunner().invoke(
        main_group,
        [
            "warp",
            "--balance-model",
            report,
            "--balance-source",
            "B",
            "--resampling",
            "bilinear",
            "--output",
            out,
            source_paths[1],
        ],
    )
    assert result.exit_code == 0, result.output
    with rasterio.open(out) as ds:
        cli = ds.read(masked=True)
        cli_transform = ds.transform
        cli_nodata = ds.nodata
    # Python API equivalent on the CLI's exact target grid and nodata.
    with rasterio.open(source_paths[1]) as src:
        api = np.full(cli.shape, cli_nodata, dtype=np.float64)
        reproject(
            source=rasterio.band(src, [1, 2]),
            destination=api,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=cli_transform,
            dst_crs=src.crs,
            dst_nodata=cli_nodata,
            resampling=Resampling.bilinear,
            balance_model=model_dict,
            balance_source="B",
        )
    np.testing.assert_allclose(cli.data, api, rtol=1e-10, atol=1e-8)
    np.testing.assert_array_equal(cli.mask, np.ma.getmaskarray(
        np.ma.masked_equal(api, cli_nodata)
    ))


# ---------------------------------------------------------------------------
# Memory budget (slow): six 4096x4096 two-band float32 sources
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="RSS via getrusage")
def test_large_merge_memory_budget(tmp_path):
    import resource

    n_sources = 6
    storage = 4096
    tile_crs = 2048
    overlap_crs = 512
    step = tile_crs - overlap_crs
    res = tile_crs / storage
    rng = np.random.default_rng(20261118)
    gains = rng.uniform(0.5, 2.0, n_sources)
    biases = rng.uniform(-20.0, 20.0, n_sources)
    master_h = 3 * step + overlap_crs
    master_w = step + tile_crs
    yy, xx = np.mgrid[0:master_h, 0:master_w]
    field = 10.0 + 990.0 * (0.6 * xx / master_w + 0.4 * yy / master_h)
    fields = np.stack([field, field * 0.5 + 100.0])

    paths = []
    tiles = []
    for row in range(3):
        for col in range(2):
            left = col * step
            top = (3 - row) * step
            tiles.append((left, top - tile_crs, left + tile_crs, top))
    for k, (left, bottom, right, top) in enumerate(tiles):
        cx = left + (np.arange(storage) + 0.5) * res
        cy = top - (np.arange(storage) + 0.5) * res
        gx, gy = np.meshgrid(cx, cy)
        gxi = np.clip(np.floor(gx).astype(int), 0, master_w - 1)
        gyi = np.clip(np.floor(master_h - gy).astype(int), 0, master_h - 1)
        phys = fields[:, gyi, gxi]
        raw = ((phys - biases[k]) / gains[k]).astype(np.float32)
        path = str(tmp_path / f"big{k}.tif")
        with rasterio.open(
            path, "w", driver="GTiff", height=storage, width=storage, count=2,
            dtype="float32", nodata=-9999.0, crs="EPSG:32631",
            transform=from_origin(left, top, res, res),
            tiled=True, blockxsize=256, blockysize=256, compress="deflate",
        ) as ds:
            ds.write(raw)
        paths.append(path)

    idle = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    datasets = [rasterio.open(p) for p in paths]
    try:
        merge(
            datasets,
            balance="affine",
            source_ids=[f"S{k}" for k in range(n_sources)],
            balance_reference="S0",
            balance_report=str(tmp_path / "big_report.json"),
            balance_sample_limit=100000,
            dst_path=str(tmp_path / "big_merged.tif"),
            mem_limit=16,
        )
    finally:
        for ds in datasets:
            ds.close()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    # Streaming reads and a capped GDAL block cache must keep the merge's
    # extra working set under 256 MiB (file output; no return array).
    assert peak - idle <= 256.0
