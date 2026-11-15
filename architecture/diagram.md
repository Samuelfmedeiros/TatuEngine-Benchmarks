# Architecture Diagram (High-Level)

**This document is high-level on purpose.** The block layout, the ordered index
structures, the kernel implementations and the codec internals are proprietary and
are not documented here. What follows is the shape of the pipeline and the measured
properties of the result, not the mechanism.

## Pipeline shape

```
tokens (CPU)
    → embedding lookup
    → N transformer layers:
        normalisation  →  projection (GPU)  →  convolution + activation (CPU)
        →  state-space step (CPU)  →  projection (GPU)  →  residual (CPU)
    → final normalisation  →  output projection  →  logits
```

The split between CPU and GPU stages is measured, not assumed: `nvidia-smi dmon`
during decode shows SM occupancy at 1–6% and memory throughput at ~1%, which is why
the honest description of this path is "GPU idle waiting on serialisation", not
"GPU-accelerated".

## Weight representation

The engine's native representation is a per-weight structure carrying a small amount
of state alongside the value; the packed on-disk form is a ternary block with a
fixed-size metadata header. **Field names, field order, struct sizes and the packing
scheme are proprietary.**

What is public, because it is measurable from the artifact alone:

| Property | Value |
|---|---|
| Parameters, total | 1,033,160,704 |
| Ternary-encoded transformer body | 827,308,032 params |
| Embedding + `lm_head` (fp16) | 205,852,672 params |
| Artifact on disk | 614.45 MiB (644,297,164 B) |
| Share that is fp16 embed+`lm_head` | **63.9%** |

An earlier version of this document published the block struct field by field, with
byte offsets and `sizeof()` output. That was too much, and it has been removed.

## Compression ratio, stated honestly

The ratio depends entirely on the baseline. Against dense formats at inference
precision, on the same artifact:

| Baseline | Size | Ratio |
|---|---|---|
| FP32 dense inference, 1.03B weights | 3.85 GiB | **6.41×** |
| BF16 dense inference | 1.92 GiB | **3.21×** |
| Training checkpoint (weights + optimizer state) | 27.26 GiB | 45.4× |

The 114× figure that appeared in earlier revisions compared against the **training
checkpoint** — weights plus optimizer state plus step copies — which is not a format
anyone serves. Quoting it alone invites the fair objection that it compares against
a training artifact. The defensible numbers are the three above.

The 455× figure is withdrawn. It compared a sparse tree node carrying child pointers
and bookkeeping against a dense block, which is a data-structure substitution, not
weight compression.

## Coherence: what is exact and what is not

From the 16/16 test log, the two assertions that measure CPU-vs-GPU agreement:

| Quantity | Measured | Exact? |
|---|---|---|
| `phase_diff` | 0.00e+00 | yes — bit-identical |
| `amp_diff` | 1.19e-07 | no — one FP32 epsilon |

Amplitude is computed as a product across a reduction, and floating-point
multiplication is not associative: different reduction orders differ in the last bit.
Phase is additive with a fixed order, so it lands exactly.

Correct statement: **phase accumulation is bit-exact; amplitude agrees to 1.19e-07,
one FP32 epsilon.** Calling the whole output "bit-exact" overstates it. The claim
that survives is "no semantic destruction — the only divergence is FP32
non-associativity".

Caveat on provenance: this log is dated 2026-07-03, which predates both the codec
(2026-08-12) and the `lm_head` rescale (2026-08-30). It anchors to an earlier build,
not to the current artifact.

## What this document deliberately omits

- Block struct layout, field names, byte offsets, `sizeof()` values.
- Kernel names, thread/warp structure, reduction strategy.
- Ordered index / tree implementation.
- Codec internals: match structure, distance model, literal coding.

Those omissions are the point. This repository is a sample of measured results, not
a specification of the implementation.
