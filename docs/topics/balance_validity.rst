# Per-band validity through affine calibration: fix notes

## The two defects

1. **`reproject(balance_model=...)` erased a legal finite value.**
   `calibrated_vrt_xml` forced every VRT band to advertise one *shared*
   calibrated nodata (the smallest-magnitude sentinel), because the warp
   options only carried a single scalar `src_nodata`.  When band 1's raw
   nodata (-9999) calibrates to `0` (model bias +9999), the shared sentinel
   became `0`, so band 2's legitimate physical zero was masked in any
   multi-band warp even though band 2's own nodata stayed -9999.  The scalar
   path also relied on `UNIFIED_SRC_NODATA`, which cannot express a hole that
   exists on only one band.

2. **`merge(balance="affine")` leaked non-finite calibrated values.**
   A stored value such as `1e308` with a fitted gain near 2 overflows to
   `+inf` in float64.  The compositing mask only considered the source's
   dataset mask and nodata, so the infinity was treated as a valid
   observation and written into the output.

## Where validity is decided now

The rule is applied at every stage data passes through:

| Stage | Validity authority |
|---|---|
| Read (`read_physical_window`) | dataset mask band **and** masked-array mask **and** `isfinite(physical)`; file scale/offset applied once |
| Calibration (`apply_affine`) | result must stay finite; merge ANDs `isfinite(calibrated)` into the per-band valid mask |
| Calibrated VRT | one calibrated `NoDataValue` **per band**, source-level raw `NODATA`, forwarded explicit mask band |
| GDAL warp | per-band `padfSrcNoDataReal` sequence with `UNIFIED_SRC_NODATA=NO` |
| Post-warp window | `NaN`/`±inf` replaced with the destination nodata; file validity recovered from finite-vs-nodata |
| Composite (merge first/last) | only finite, non-nodata candidates may occupy or overwrite a cell |

A band's legal finite result can never be dropped because it equals another
band's calibrated nodata: each band is matched only against its own sentinel.

## Why these representations (the tradeoffs)

- **Precision: float64 throughout.**  The calibrated VRT bands are `Float64`
  and GDAL evaluates `gain * (raw * file_scale + file_offset) + bias` in
  double precision inside the warper, matching the NumPy merge path.  The
  destination for a balanced warp is always float64; integer output is
  rejected up front because a calibrated physical value may not be
  representable as an integer.

- **Per-band masking without materialising rasters.**  GDAL's warp options
  accept an array of source nodata values, one per band
  (`padfSrcNoDataReal`).  The Cython layer (`_warp.pyx`) now accepts a
  Python sequence for `src_nodata`; a scalar keeps the historical broadcast.
  This makes per-band validity a property of the warp itself, so no full
  validity array or pre-warp pass over the raster is needed.  A `None`
  entry becomes an unreachable NaN sentinel, leaving that band's mask band /
  NaN / infinity handling solely in charge (needed when a band has no
  nodata, or its calibrated sentinel is itself non-finite).

- **Streaming memory instead of an in-memory warp.**  The balanced warp is
  driven destination-window by destination-window over the destination
  file's native block lattice (`_destination_windows`).  Only one block's
  float64 arrays (a few MB at 256x256) are resident; GDAL's block cache is
  bounded by the caller's environment (16 MiB in the memory tests).  An
  ndarray destination is a single window because the caller already holds
  the whole array — streaming there would only add copies.  Window
  partitions are disjoint, so block size changes memory use but never the
  result (asserted by `test_reproject_block_size_invariance`).

- **Extra reads/writes.**  Each destination block issues one GDAL warp call.
  Relative to one whole-raster warp this adds call overhead per block, not
  extra pixel I/O: source blocks are read through the same GDAL cache and
  destination blocks are written once.  The merge path already worked this
  way and its memory profile is unchanged in shape.

- **File validity on reopen.**  A GeoTIFF per-dataset mask band is a single
  band shared by all raster bands, so writing one would *collapse* the
  independent per-band validity rather than express it.  Instead, invalid
  cells hold the finite destination nodata and valid cells hold finite
  measurements; `read_masks()` then derives the correct per-band mask from
  the nodata marker, including a hole on one band beside a finite value on
  the other.  With the default NaN nodata the same distinction falls out of
  `isfinite`.

- **Non-finite outputs.**  A VRT cannot declare "infinity is invalid", and
  GDAL may emit `inf`/`NaN` after scaling or interpolation.  After each
  window warp those cells are explicitly replaced with the destination
  nodata.  This is the one O(band × window) extra pass, performed on data
  already in memory, so it adds no I/O and no whole-raster buffer.

- **Bilinear and other interpolating resamplers.**  Because source holes
  reach GDAL as per-band nodata and mask-band zeros, the warper excludes
  them from the interpolation kernel; a fully valid constant neighborhood
  still evaluates to that constant, and a non-finite input can never seed a
  neighborhood.

## Compatibility

- Calling `reproject`/`merge` without balance parameters is untouched; the
  scalar `src_nodata` behavior is byte-for-byte the old path.
- Existing balance JSON reports load and validate unchanged.
- Source files, the passed model mapping and open datasets are never
  modified (asserted by `test_balance_reproject_does_not_mutate_inputs`).
- `src_nodata`/`src_alpha`/`dst_alpha` combined with a balance model now
  raise a clear `ValueError` rather than being silently misapplied to
  calibrated values.

## Verification

- Two small rerunnable regression rasters are exercised by
  `tests/test_balance_validity.py`: the dual-band zero/sentinel collision
  (single-band, multi-band and reversed band selection) and the
  overflow-into-infinity merge, covering explicit masks, per-band-distinct
  holes, legal zeros, first/last compositing, tiled file delivery, block
  invariance, bilinear/cubic/average/lanczos resampling, scale/offset and
  input immutability.
- File-to-file memory checks (`pytest -m slow`): two 4096x4096 two-band
  float64 tiled sources, separate clean processes, `GDAL_CACHEMAX=16`,
  extra peak RSS measured from the post-import/pre-work baseline.
  Observed: balanced reproject ≈ 51-58 MiB, balanced merge ≈ 114-117 MiB,
  both inside the 128 MiB budget.
