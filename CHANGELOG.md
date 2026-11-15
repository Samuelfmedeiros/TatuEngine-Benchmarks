# 📋 CHANGELOG — TatuEngine-Benchmarks

## [2026-11-15] — External baseline measured, and the model lineage corrected

Two things changed: the Results table's model was **misidentified**, and an
external baseline was finally measured on the same machine.

### The artifact behind the Results table is a pruned `mamba2-1.3b`

`logs/coherence-test.txt` labelled the model "BitMamba-2-1B". That label does not
match the measured checkpoint:

| File | `ssm.A_log` | `in_proj` | Distinct values |
|---|---|---|---|
| `bitmamba_1b.pt` (HF Zhayr1/BitMamba-2) | `(32,)` | `(8288, 2048)` | 3 (ternary) |
| `bitmamba_fundacao_fase3_1.02B.pt` (Results) | `(64,)` | `(8512, 2048)` | 4,978 |

`scripts/prune_mamba.py:40` declares `REPO_ID = "state-spaces/mamba2-1.3b"`; the
local parent is 1,446,747,136 params, 48 layers, vocab 50,288. So the "1.02B" is a
**depth-pruned Microsoft Apache-2.0 Mamba2** (48 → 32 layers, vocab 50,288 →
50,257). It is not ternary in storage, and it is not Zhayr1's model. Attribution
is pending; the README now states this.

### External baseline: `jserv/bitmamba.c` on the same machine

`bitmamba.c` is a portable C11 inference engine for BitMamba-2 (MIT). Built here
with `-O3 -mavx2 -mfma` on an **i5-12400** (Alder Lake, same generation as the
`i3-12100F` the upstream README benchmarks on), same `bitmamba_1b.bin`
(644,297,164 bytes, 32 heads):

| Threads | tok/s (200 tokens, seed 42) |
|---|---|
| 1 | 0.73 |
| 4 | 7.14 |
| 6 | 55.63 / 70.80 / 63.70 |
| **8** | **69.97 / 75.56 / 73.72** |
| 12 | 20.20 |

**~73 tok/s at 8 threads**, versus the upstream README's 52.86 tok/s on an
i3-12100F — consistent given the i5-12400's higher clocks.

**This is 4.3× the TatuEngine GPU figure** (17 tok/s on an RTX 3060), measured on
a model 245 MiB in size that fits entirely in VRAM. That is the honest
comparison, and it is in the README now.

TatuEngine's own paired run on the same `.bin` is in progress; when it lands it
replaces the "17 tok/s" with a matched-profile measurement.

---

## [2026-11-14] — Paired held-out: the generalization gap measured on both sides

The previous entry measured held-out on one side only and with sampling bias. This
entry replaces that measurement: both **sides** are now measured on the same
checkpoint, with the same random sampling and the same split.

**Sampling:** row-groups drawn at random across **all** parquets (the previous run
took the front of the file order, which is systematically more contaminated). A
32-token signature was searched against the training tensor (2,396,160,000 tokens),
checkpoint `step_902000`:

| Corpus | Scanned | Clean | Contamination |
|---|---|---|---|
| Legal PT-BR | 200 | 40 | **64%** |
| Wiki PT | 120 | 40 | **44%** |

The contamination figures in the previous entry (83% legal, 92% wiki) were an
**artifact of sampling bias** and are wrong. Corrected here.

**Paired result** — 40 documents per side, 1024 tokens per document:

| Corpus | Metric | In-distribution | Held-out | Delta |
|---|---|---|---|---|
| Legal | loss | 1.9547 | **2.5505** | +0.596 (+30.5%) |
| Legal | ppl | 7.06 | **12.81** | +5.75 |
| Legal | top-1 | 64.85% | **57.92%** | −6.93 pp |
| Legal | bits/tok | 2.820 | **3.680** | +0.860 |
| Wiki | loss | 2.9257 | **4.1302** | +1.205 (+41.2%) |
| Wiki | ppl | 18.65 | **62.19** | +43.54 |
| Wiki | top-1 | 49.75% | **40.65%** | −9.10 pp |
| Wiki | bits/tok | 4.221 | **5.959** | +1.738 |

**Three conclusions the numbers support:**

1. **No legal specialization.** Held-out legal (2.5505) sits below held-out wiki
   (4.1302), but in-distribution legal sits below in-distribution wiki too — the
   difference is text difficulty, not domain skill.

2. **The generalization gap is larger on wiki** (41.2% vs 30.5%). If legal had
   learned domain structure, its gap would be the smallest. It is the smallest —
   but because that corpus is more templated and repetitive, which also explains
   its higher contamination (64% vs 44%).

3. **The in-distribution number is memory.** The 30–41% loss drop when leaving
   training is the direct measurement.

**Scripts:** `scripts/eval_heldout.py` (filters contaminated documents by random
sampling, evaluates the clean ones) and `scripts/eval_indist.py` (reads the cache
from that split and evaluates the complement — the contaminated ones). The second
exists because comparing absolute loss across different corpora is invalid: each
corpus has its own entropy.

**Caveat:** 40 documents per side is a defensible sample, not a large one. The
numbers establish the direction and order of magnitude of the gap, not the exact
loss value in production.

### Environment used (for reproducibility)

```
python 3.11.17 · torch 2.14.1+cu130 · transformers 5.22.1 · numpy 2.4.6
GPU: NVIDIA GeForce RTX 3060 (12 GB)
training seed: 42 (torch.manual_seed) · loader: 1234 · eval seed: 42
checkpoint step_902000: pytorch_model.bin md5 707c7b101de071bd20c8b4e6ffed488f
train_ids.pt: 8.93 GB · 2,396,160,000 tokens
```

---

## [2026-11-13] — Held-out measured: the "legal specialization" was memorization

The README said the numbers were in-distribution but did not know the size of the
problem. It is now measured, and the result overturns one of the repo's own
conclusions.

**Scan:** 60 legal documents, signature of the first 32 tokens searched in the
training tensor (2,396,160,000 tokens, via mmap). **Only 9 of the 60 (15%) were
absent from training.** The 9 clean ones (9,035 tokens) were evaluated:

| Metric | Contaminated | Held-out (clean) | Difference |
|---|---|---|---|
| loss | 1.6141 | **2.9878** | +1.37 (1.85×) |
| ppl | 5.02 | **19.84** | +14.82 (3.95×) |
| top-1 | 69.3% | **53.6%** | −15.75 pp |
| bits/tok | 2.33 | **4.31** | +1.98 |

**Consequence:** held-out legal (2.9878) is statistically equal to wiki (2.9815).
The reading that "the model specialized in the dominant legal domain" was wrong —
there is no specialization, there is memorization. The model is uniformly mediocre
in both domains.

New script: `scripts/eval_limpo.py` (in the main TatuEngine repo) does the scan,
filters by signature and evaluates only the absent documents. It caches the indices
so the 40-minute scan is not repeated.

**Caveat recorded in the README:** 9 documents is a small sample and the scan takes
the front of the parquet order. It refutes the specialization claim; it does not
establish an absolute generalization number. **Superseded by the 2026-11-14 entry
above**, which corrects the sampling bias and measures both sides.

### Environment used (for reproducibility)

```
python 3.11.17 · torch 2.14.1+cu130 · transformers 5.22.1 · numpy 2.4.6
GPU: NVIDIA GeForce RTX 3060 (12 GB)
training seed: 42 (torch.manual_seed) · loader: 1234
checkpoint step_896000: pytorch_model.bin md5 b10efe19299afeb07a73c43457c1d0ba
                        training_state.pt   md5 17bb4c91ff0aecc95e506b7ba2a1d464
train_ids.pt: 8.93 GB · 2,396,160,000 tokens
legalpt: 40 parquets, 9.70 GB · wiki: 6 parquets, 1.47 GB
```

---

## [2026-11-13] — Methodology corrections after external review

A technical review raised three real problems; all were confirmed by measurement
and corrected:

- **114× compression was compared incorrectly.** 27.26 GiB / 914,432,000 weights =
  32 B/weight, which is a training checkpoint (weights + optimizer), not an
  inference representation. Re-derived against the real artifact (614.45 MiB,
  1,033,160,704 params): **6.41× vs FP32 dense · 3.21× vs BF16 · 45.4× vs the
  training checkpoint**. The 455× figure is withdrawn — it compared a sparse tree
  node against a dense block, which is a data-structure substitution, not weight
  compression.
- **"Bit-exact" was too strong a claim.** The log shows `phase_diff 0.00e+00` but
  `amp_diff 1.19e-07`. Phase is bit-exact; amplitude is 1 FP32 epsilon because
  multiplication is not associative. Corrected to "phase bit-exact; amplitude to
  1 FP32 epsilon · no semantic destruction".
- **No train/validation separation.** Training consumed `train_ids.pt` (9.58 GB)
  built from the same parquets used for evaluation. The loss/ppl numbers are
  **in-distribution**, not generalization. Documented as a caveat in the README and
  in `logs/prediction-120m.txt`.

Added:

- **"What You Can and Cannot Verify"** section in three layers (reproducible by
  third parties / verifiable in principle / not established).
- **"Hybrid CPU/GPU Split"** section with what is and is not measured — explicitly
  without inventing per-operation latency that nobody collected.
- Statement that no checkpoint, tokenized data or corpus content is published —
  only aggregate metrics.

## [2026-11-13] — Next-token prediction + lossless compression benchmarks

- **Next-token prediction (120M model):** loss, perplexity, top-1, top-5 and
  bits/token on two corpora (legal PT-BR and wiki PT), across 3 checkpoints
  (884K / 886K / 888K). Deterministic argmax evaluation.
- **Lossless compression (.tlz):** comparison against `xz -9e` and `zstd -19` on
  real text (wiki PT) and real source code, with roundtrip verified by md5.
- **Documented methodology correction:** the previous measurement reported a 14.2%
  gap on text; it was an artifact of a synthetic corpus with repeated documents.
  Current numbers come from real corpora: 1.97% (text) and 3.99% (source code).
- New file `logs/prediction-120m.txt` with the raw measurement output and the
  generation samples.

## [2026-10-31] — Initial release

- Paper in Markdown (numbers, architecture, links)
- Coherence logs (16/16 passed, RTX 3060)
- Architecture diagram (pipeline shape, measured properties only)
- LICENSE (MIT), .gitignore, README.pt-BR, CONTRIBUTING, SECURITY
- Description and layout aligned with Samuel's other repos
