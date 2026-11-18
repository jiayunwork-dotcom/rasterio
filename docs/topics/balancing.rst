.. _topics-balance:

=============================================
Balancing gains and biases across data takes
=============================================

Repeated quantitative observations of one area are often collected in
separate takes whose stored values live on different scales.  Rasterio's
affine balance mode estimates a per-source, per-band calibration from the
valid overlapping pixels, expresses every source in one common reference
frame, and applies that calibration while merging -- or later, while
reprojecting a single source.

The calibration target is the *physical* value of a pixel, obtained from
the stored (raw) value using the dataset's own metadata::

    physical = raw * scale + offset

Each source then receives an affine transform into the reference frame::

    reference_value = gain * physical + bias

The reference source is anchored at ``gain = 1`` and ``bias = 0`` on every
selected band.  Every gain is constrained to be positive and every bias to
be finite.  Zero is always treated as a legal measured value; nodata, the
dataset validity mask, NaN and infinity never participate in the fit and
never pollute the output.

Enabling balance mode in a merge
---------------------------------

Pass ``balance="affine"`` to :func:`rasterio.merge.merge`, together with a
unique ``source_ids`` list, a ``balance_reference`` ID, a ``balance_report``
JSON path, and optionally a ``balance_sample_limit`` (default 100000)::

    from rasterio.merge import merge

    dest, transform = merge(
        datasets,
        balance="affine",
        source_ids=["A", "B", "C", "D"],
        balance_reference="A",
        balance_report="balance.json",
        balance_sample_limit=100000,
    )

With ``balance=None`` (the default) the historical behaviour, return
values and dtypes are unchanged.  In balance mode:

* the output is floating point (``float64`` unless another float dtype is
  requested explicitly; integer dtypes raise ``ValueError``);
* only the ``first`` and ``last`` compositing strategies are supported;
* all parameter validation and the identifiability checks run *before* any
  output file or report is created, so a rejected calibration leaves no
  artifacts behind;
* the source files are opened for reading only and are never rewritten.

How the joint model is estimated
--------------------------------

No information is propagated pairwise down a single tree.  For every
sampled overlap pair ``(a, b)`` and band, the equality of reference-frame
values contributes a linear equation::

    gain_a * physical_a + bias_a = gain_b * physical_b + bias_b

Chains (A overlaps B, B overlaps C) and closed cycles (A-B-D-C-A) therefore
determine one global, jointly consistent reference frame.  The reference
source's fixed identity parameters anchor the system, and the stacked
weighted normal equations are solved per band with NumPy's dense linear
algebra.  SciPy is **not** required: the design matrix is assembled directly
from the retained samples, which are capped by ``balance_sample_limit``.

Robustness to local contamination
---------------------------------

The estimator is deliberately robust rather than a plain ordinary least
squares fit or a min/max stretch:

1. Each overlap gets a high-breakdown seed from the median level of the top
   and bottom deciles of its predictor values, followed by repeated
   least-trimmed-squares refinement on the 80% smallest residuals.
2. A global Huber IRLS polish then reconciles every overlap jointly.  The
   Huber scale is estimated per overlap from the median absolute deviation
   (with a 90th-percentile fallback), each overlap's weights are normalized
   to sum one so small overlaps are not drowned out by large ones, and
   residuals beyond four robust standard deviations receive zero weight.

This tolerates a handful of gross outliers (the test suite injects up to 5%
contaminated pairs per overlap at ten times the field's dynamic range) while
recovering gains and biases to high precision on clean data.  It costs
roughly one small dense solve per band plus repeated streamed reads; unlike
min/max stretching it is not hijacked by a single extreme pixel.

Deterministic, memory-bounded sampling
---------------------------------------

A source pair's sample points are the coarser source's pixel centers inside
the shared extent; the finer source is read at the matching shape with
nearest-neighbour resampling, which is an exact point lookup at aligned
windows.  Each pair/band owns an equal sample quota filled by a key-based
uniform reservoir driven by a ``numpy.random.Generator`` seeded from a
stable hash of the sorted source IDs and band number.  Consequently:

* reordering ``sources``/``source_ids`` cannot change the reference frame;
* changing file block layout or ``mem_limit`` only changes how reads are
  tiled, never which pixels are retained or the fitted coefficients;
* at most ``balance_sample_limit`` paired samples are retained in total,
  and overlaps are streamed in fixed-size tiles so neither whole sources
  nor whole rasters are read into memory.

GDAL's block cache is capped at 64 MB during a balance merge (unless the
user has configured ``GDAL_CACHEMAX`` explicitly) to keep the extra working
set flat on very large collections.

Identifiability and failure modes
---------------------------------

A fit is rejected with a :class:`rasterio.balance.BalanceError` (a
``ValueError``) -- never silently replaced by an identity transform -- when:

* a source has no chain of informative overlaps connecting it to the
  reference;
* an overlap contains fewer than 30 valid paired pixels;
* the overlapping physical values are effectively constant (relative
  standard deviation below 1e-12), so a gain cannot be identified;
* the global design matrix is rank-deficient for a band;
* a fitted gain is non-positive or a coefficient is non-finite.

The JSON report records, per failure, the source, band and reason.

The balance report (JSON model)
-------------------------------

``balance_report`` is a plain JSON object readable with the standard
library's ``json.load``.  It contains:

``version``
    Model format version (currently ``"1.0"``).  A different major version
    is rejected on read.

``reference``
    The ID of the reference source.

``sources``
    Per-source metadata keyed by source ID: CRS (WKT), GDAL affine
    transform, width/height, per-band dtypes, scales, offsets, raw band
    numbers and nodata values.

``bands``
    Per raw band number, per source ``{"gain": ..., "bias": ...}``
    coefficients.  The reference source always reports ``1.0`` / ``0.0``.

``diagnostics``
    The configured sample limit and per-pair quota, total valid and sampled
    pair counts, and per-band overlap statistics (valid/sampled pair counts,
    relative standard deviation, status) plus residual summaries (RMSE,
    mean/median/max absolute residual, 5th/95th residual quantiles).

``tolerances``
    The numeric tolerances used for metadata validation and the rank/IRLS
    convergence thresholds, so a report documents its own precision claims.

Reusing a model with ``reproject`` and ``rio warp``
---------------------------------------------------

A saved model can be applied to a single source during an independent
reprojection::

    from rasterio.warp import reproject

    reproject(
        source=rasterio.band(src, [1, 2]),
        destination=dst_band,
        balance_model=json.load(open("balance.json")),
        balance_source="B",
        resampling=Resampling.bilinear,
    )

The model is implemented as a lazy in-memory VRT whose per-band
``ScaleRatio``/``ScaleOffset`` apply
``gain * (raw * file_scale + file_offset) + bias`` while GDAL warps, so the
same scales/offsets, coefficients and nodata/mask semantics are used as in
the merge path and nothing is materialized.  Only file
:class:`rasterio.Band` inputs are accepted; an ndarray source with a model
raises ``ValueError``, and the model is applied exactly once.  On read-back
the source ID, CRS/transform, dimensions, dtype and per-band scales/offsets
are validated against the report; any mismatch or unsupported model version
raises before warping.

The model intentionally cannot prove that a source file's pixels were not
modified externally after the report was written: reuse assumes the files
are the same physical observations that were calibrated (same grid, band
order and calibration metadata).  Content changes that leave all metadata
intact are outside what metadata validation can detect.

Command line
------------

``rio merge`` accepts ``--balance affine``, repeatable ``--source-id``
(one per input, in order), ``--balance-reference``, ``--balance-report``
and ``--balance-sample-limit`` and writes the same JSON report.  ``rio
warp`` accepts ``--balance-model`` (a report path) and ``--balance-source``.
The CLI outputs and valid regions match the Python API.
