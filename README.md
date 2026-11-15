# 🧠 TatuEngine-Benchmarks — Samuel Medeiros

Benchmarks and coherence proofs for the TatuEngine ternary inference engine.

**Live:** https://github.com/Samuelfmedeiros/TatuEngine-Benchmarks

> 🌐 **English** · [🇧🇷 Português](README.pt-BR.md)

---

## 🎯 What this repository actually is

This is **not** a speed or compression record. It does not beat `bitmamba.c`,
`bitnet.cpp`, or `xz`, and the numbers below should not be read as if it did —
69.4× is this engine's GPU path against its own CPU path, and 7.1× is against its
own BF16 path. Both measure a starting point, not a competitor.

What is worth reading here, and what the rest of the repo is about:

**1. A different way to lay weights out.** TatuEngine does not store weights as
scalars and does not compute a plain matmul over them. Its native unit carries a
small amount of per-weight state alongside the value, and the packed form is a
ternary block with a fixed-size metadata header. The design question it explores
is whether a *layout* — not the arithmetic — can carry useful information and
compress at the same time. The field layout, struct sizes and kernels are
proprietary; what is published here is the measured behaviour of the result.

**2. `.tlz`, a companion codec.** A real compressor with its own modelling, tuned
from a 6.32% gap against `xz` down to 1.74%. It is not better than `xz`, and it
does not need to be: its natural target is structured binary blobs, not prose.
The codec internals are proprietary.

**3. A three-layer documentation structure.** Every claim in this repo is filed
under *reproducible by third parties* / *verifiable in principle* / *not
established*. This structure is the reusable part — it should be mandatory in any
benchmark repository.

**4. A cheap contamination detector.** A 32-token signature of each document's
opening tokens, an mmap scan over the 2.4B-token training tensor, and a
complement-evaluation trick (the held-out cache becomes the in-distribution set,
so both sides come from one paired split). Reusable for any in-house training run.

The engine figures are still documented in full below, with what they do and do
not establish.

### What the artifact is made of

`bitmamba_1b.bin` is **614.45 MiB** on disk. Most of it is not the compressed
weights: it is the fp16 embedding and `lm_head`, 205.85M parameters at 2 bytes each
= **392.6 MiB, 63.9% of the file**. The ternary-encoded transformer body is the
remaining 36.1%.

That split matters more than any packing ratio. Every bandwidth argument about a
ternary format is an argument about the smaller third of this file, and the
embedding table is the part no ternary trick touches. Quantizing `embed`+`lm_head`
to int8 alone would take the artifact to **354.1 MiB (−42.4%)**, roughly four times
what a perfect repack of the ternary payload would buy.

The internal block layout that produces the 36.1% is proprietary and not documented
here. What is public is the artifact size, its parameter count, and the arithmetic
above.

## 🏗️ Architecture

```
tatuengine-benchmarks/
├── README.md                    # Paper (EN)
├── README.pt-BR.md              # Paper (PT-BR)
├── LICENSE                      # MIT
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── logs/
│   ├── coherence-test.txt       # 16/16 coherence tests (CPU vs GPU)
│   └── prediction-120m.txt      # next-token prediction + compression runs
└── architecture/
    └── diagram.md               # High-level architecture diagram
```

**Stack:** C++ · CUDA · Python

---

## 📊 Results (RTX 3060, 12GB VRAM)

**Which model this table belongs to — measured, not assumed:** the artifact that
produced these figures is a **depth-pruned `state-spaces/mamba2-1.3b`**
(48 → 32 layers, vocab 50288 → 50257), built by `scripts/prune_mamba.py`
(`REPO_ID = "state-spaces/mamba2-1.3b"`). It has **64 heads** (`ssm.A_log` shape
`(64,)`) and 4,978 distinct values in `in_proj` — it is **not** ternary in
storage, and it is **not** Zhayr1/BitMamba-2, which has 32 heads and exactly 3
distinct values. `logs/coherence-test.txt` labels the model "BitMamba-2-1B"; that
label does not match the measured artifact and is under correction.

This is also **not** the 120M model whose prediction numbers appear further down.
The two are separate models and their figures are not interchangeable. The 120M
model at 2 bits would be ~29 MiB, not 245 MiB.

`state-spaces/mamba2-1.3b` is Apache-2.0 (Microsoft/CMU). Attribution is pending.

**External baseline, measured on the same machine** (i5-12400, same file
`bitmamba_1b.bin`, same prompt, 200 tokens, seed 42).

`bitmamba_1b.bin` is **614.45 MiB (644,297,164 B)** on disk — the real BitMamba-2
(32 heads, vocab 50257). Earlier revisions of this README said "993,664 blocks =
272.92 MB" for it; that figure came from a mid-size debug log
(`build_py/Testing/Temporary/LastTest.log`) and is wrong for the shipped artifact
by 2.25x. Of that, 221.9 MiB is the ternary-encoded transformer body and 392.6 MiB
is fp16 embedding and `lm_head`.

| Engine | Hardware | tok/s |
|---|---|---|
| `jserv/bitmamba.c` (C11, 8 threads) | i5-12400 CPU | **73.7** |
| TatuEngine | RTX 3060 GPU | **24.3** |
| TatuEngine | i5-12400 CPU | **0.09** |

`bitmamba.c` on the CPU is **3.0× faster than TatuEngine on the GPU** and **819×
faster than TatuEngine on the same CPU**, on the same file.

At 614.45 MiB read per token, that is **47.5 GB/s** for `bitmamba.c` — 93% of the
theoretical 51.2 GB/s of dual-channel DDR4-3200. It is genuinely against the wall.
TatuEngine at 24.3 tok/s moves 15.7 GB/s, which is **4.3%** of the RTX 3060's
360 GB/s. Its upstream README
reports 52.86 tok/s on an i3-12100F; 73.7 on an i5-12400 is consistent with the
higher clocks on the same Alder Lake generation.

Two caveats that decide whether this table means anything:

- **Storage matters for TatuEngine, not for `bitmamba.c`.** The latter mmaps the
  weights zero-copy and reads them from page cache; TatuEngine `cudaMemcpy`s per
  layer. At 400 tokens the same TatuEngine run gives 5.04 tok/s with the model on
  an external NTFS HDD and 16.09 tok/s from a local SSD. The figures above are
  the 60-token steady-state with warm weights.
- **Host load.** These runs happened at load average 43–59 with 28–30 GB of swap
  in use. Re-measure before treating the absolute numbers as reproducible.

This is a clean head-to-head: one file, one machine, both engines. It is also the
first external baseline this repository has ever reported.

| Metric | Value |
|--------|-------|
| Compression | 113.8× vs training checkpoint · 14.2× vs FP32 dense · 7.1× vs BF16 |
| MSE (in_proj) | 3.7e-9 (two orders below FP32 epsilon) |
| Phase diff | 0.00e+00 (phase bit-exact) · amp 1.19e-07 = 1 FP32 eps |
| Coherence | 16/16 tests passed (semantic, not bit-exact) |
| Speedup | 69.4× (4.092s/tok CPU → 0.059s/tok GPU) |
| Blocks | 893K (in_proj 530K, out_proj 262K, lm_head 100K) |

---

## 🧠 Next-Token Prediction (120M model)

Corpora: **legal PT-BR** and **general wiki PT** — 30 documents each, 512-token
windows. Cross-entropy measured over 15,025 and 15,330 predicted tokens
respectively. Deterministic eval (argmax, no sampling).

| Checkpoint | Corpus | Loss (nats) | Perplexity | Top-1 | Top-5 | Bits/token |
|---|---|---|---|---|---|---|
| 884K | legal PT-BR | 1.6361 | 5.14 | 68.7% | 82.7% | 2.36 |
| 886K | legal PT-BR | 1.6252 | 5.08 | 69.2% | 82.8% | 2.34 |
| **888K** | **legal PT-BR** | **1.6141** | **5.02** | **69.3%** | **82.9%** | **2.33** |
| 884K | wiki PT | 2.9879 | 19.84 | 48.7% | 67.0% | 4.31 |
| 886K | wiki PT | 2.9891 | 19.87 | 48.6% | 66.5% | 4.31 |
| **888K** | **wiki PT** | **2.9815** | **19.72** | **48.6%** | **67.1%** | **4.30** |

**Reading the trajectory:** between 884K and 888K steps the legal corpus improved
(1.6361 → 1.6141, top-1 68.7% → 69.3%) while the general corpus stayed flat
(2.9879 → 2.9815, top-1 48.7% → 48.6%).

**Correction (see the held-out section below):** this was originally read as "the
model is specializing on the dominant training domain". That reading is wrong.
The gain on legal is recovered memorization of documents already in the training
tensor, not acquired skill. In a paired held-out (same checkpoint, same corpus),
legal loses 30.5% of loss when leaving training (1.9547 → 2.5505) and wiki loses
41.2% (2.9257 → 4.1302). There is no specialization; there is memorization.
Training corpus is 68% legal PT-BR, which is exactly why the memorization lands there.

Generated text is coherent in both greedy and sampled decoding (temperature 0.8,
top-k 50, top-p 0.95).

> **⚠️ These are in-distribution numbers — and the contamination is measured, not
> assumed.** The evaluation corpora come from the same document pools used for
> pretraining. I measured how bad it is with random row-group sampling across
> **all** parquets (an earlier 12-document measurement sampled only the front of
> the parquet order and overstated contamination). First 32 tokens of each eval
> document as a signature, searched against the 2,396,160,000-token training
> tensor:

| Eval corpus | Documents scanned | Found in training | Rate |
|---|---|---|---|
| legal PT-BR | 200 | 129 | **64%** |
| wiki PT | 120 | 53 | **44%** |

The real contamination rate is 64% and 44% — not the 83%/92% of an earlier
measurement made on 12 documents from the front of the parquets. Most legal
documents that are in training appear 21 times over. Treat 1.6141 and 2.9815 as
"how well the model fits text it has already seen", not as generalization. What
the numbers do support is the *relative* comparison between checkpoints on the
same distribution — and the paired gap against held-out is measured in the section
below. See `logs/prediction-120m.txt` for the full caveat.

---

## ⚙️ Hybrid CPU/GPU Split — where the time actually goes

The pipeline is not "GPU for everything". Per token, per layer:

| Stage | Device | Measured |
|---|---|---|
| embedding lookup | CPU | — |
| RMSNorm | CPU | — |
| `in_proj` matmul | GPU (bridge) | bulk of GPU time |
| conv1d + SiLU | CPU | — |
| SSM scan step | CPU | — |
| `out_proj` matmul | GPU (bridge) | — |
| residual add | CPU | — |
| final RMSNorm + `lm_head` | CPU → GPU | `lm_head` is 100K blocks |

**What is measured:** end-to-end 0.059 s/token on GPU vs 4.092 s/token on CPU
(69.4×), and 614.45 MiB of weights on disk for `bitmamba_1b.bin` (BitMamba-2, 32
heads; 807,918 ternary blocks = 221.9 MiB plus 392.6 MiB of fp16 embed+`lm_head`).

For scale, using the measured figures from the paired bench below: 614.45 MiB read
per token gives **15.7 GB/s** on the GPU path at 24.3 tok/s, against ~360 GB/s of
RTX 3060 bandwidth — **4.3%** of available bandwidth. `jserv/bitmamba.c` at 73.7
tok/s on the same file is **47.5 GB/s**, i.e. **93%** of theoretical dual-channel
DDR4-3200 and **3.0× faster than the GPU path here**. The 69.4× is
GPU-vs-own-CPU, not a competitive figure. (An earlier revision of this README used
245.3 MiB and derived 4.36 GB/s / 1.21% from a block count that did not match the
shipped file; the bandwidth numbers above replace those.)

**What is NOT measured — and this is the honest gap:** per-operation latency,
H2D/D2H transfer volume, and synchronization overhead per token. Without those,
the 69.4× says "GPU path is much faster end-to-end", not "the hybrid split is
optimal". A stage-by-stage trace would show whether the CPU↔GPU handoffs eat the
gain, and that trace is not in this repository.

---

## 🗜️ Lossless Compression (`.tlz`)

Measured on real corpora, against `xz -9e` and `zstd -19`:

| Corpus | Original | xz -9e | zstd -19 | .tlz | vs xz |
|---|---|---|---|---|---|
| Wiki PT (text) | 645,527 B | 188,080 B (3.432×) | 191,508 B (3.371×) | **191,350 B (3.374×)** | +1.74% |
| Source code | 484,599 B | 90,564 B (5.351×) | 92,774 B (5.223×) | **93,933 B (5.159×)** | +3.72% |

**Gap vs xz over the day:** text 6.32% → **1.74%**; source code 7.28% → **3.72%**.
Two modelling changes drove it, both internal to the codec and not described here.

### Reproducing these numbers

Inputs, environment and commands — everything needed to re-run the comparison:

```
# Input corpora (SHA-256 of the exact files used)
real.txt  17e1f54020d44670aa75389d0a778d669951d72adc2ea1a37a113259485137fd  (645,527 B)
code.txt  4505cd42b71a022e4ac10cb80799fba9c5ce729d9420d3272b5afc1c92885c8b  (484,599 B)

# Compressor versions
xz (XZ Utils) 5.8.3
zstd CLI v1.5.7 (Yann Collet)
```

```bash
# Reference compressors
xz -9e -c real.txt > real.txt.xz
zstd -19 -c real.txt > real.txt.zst

# .tlz (from the TatuEngine build tree)
./build/tools/tlz_compacto compact real.txt real.tlz -v
./build/tools/tlz_compacto extract real.tlz back.txt

# Integrity: byte comparison, not just a hash
cmp real.txt back.txt && echo "byte-identical"
sha256sum real.txt back.txt    # 17e1f54020d44670aa75389d0a778d669951d72adc2ea1a37a113259485137fd
```

`.tlz` roundtrip was verified with `cmp` (byte-for-byte) on both corpora, not only
with a hash — a hash match is identification, `cmp` is equality.

**What .tlz is, in one paragraph:** a purpose-built compressor with its own
modelling, tuned this cycle to cut the gap against `xz` on text from 6.32% to 1.74%.
The internal design — match structure, distance model, literal coding — is
proprietary. What is verifiable here is the sizes, the SHA-256 of the inputs, and
the `cmp` roundtrip.

**Parameter note:** the encoder defaults used above are the CLI defaults; there is
no tuned dictionary and no external metadata file. `.tlz` is a single self-contained
stream, which is why the size comparison against `xz`/`zstd` is apples-to-apples.

**Methodology note:** an earlier run reported a 14.2% gap on text. That was an
artifact of a synthetic benchmark corpus built from repeated documents — the
match finder found enormous matches that do not occur in real text. Every number
above comes from real corpora.

---

## 🔬 What's Inside

- **`logs/coherence-test.txt`** — 16/16 coherence tests (phase bit-exact, amplitude to 1 FP32 epsilon)
- **`logs/prediction-120m.txt`** — raw output of the prediction and compression runs
- **`architecture/diagram.md`** — high-level pipeline diagram
- **`README.md`** — technical paper with numbers, architecture, and links

## Held-out evaluation (the honest number)

The in-distribution figures above overstate the model significantly. An honest
measurement requires comparing the **same corpus** on both sides: when the
document is in training and when it is not. Comparing absolute loss across
different corpora is invalid — each corpus has its own entropy.

Random row-group sampling across **all** parquets (not just the front of the
parquet order, which was the bias of the first measurement), 32-token signature
against the training tensor, checkpoint `step_902000`:

| Corpus | Docs scanned | Clean | Contamination | Docs evaluated (each side) |
|---|---|---|---|---|
| Legal PT-BR | 200 | 40 | 64% | 40 |
| Wiki PT | 120 | 40 | 44% | 40 |

Paired result — same checkpoint, same tokenizer, same 1024-token sequence per
document:

| Corpus | Metric | In-distribution | Held-out | Delta |
|---|---|---|---|---|
| Legal | loss | 1.9547 | **2.5505** | +0.596 (+30.5%) |
| Legal | ppl | 7.06 | **12.81** | +5.75 |
| Legal | top-1 | 64.85% | **57.92%** | −6.93 pp |
| Legal | bits/token | 2.820 | **3.680** | +0.860 |
| Wiki | loss | 2.9257 | **4.1302** | +1.205 (+41.2%) |
| Wiki | ppl | 18.65 | **62.19** | +43.54 |
| Wiki | top-1 | 49.75% | **40.65%** | −9.10 pp |
| Wiki | bits/token | 4.221 | **5.959** | +1.738 |

Three readings these numbers support:

**1. There is no legal specialization.** Held-out legal (2.5505) sits *below*
held-out wiki (4.1302) — but in-distribution legal also sits far below
in-distribution wiki. What separates the two corpora is the intrinsic difficulty
of the text, not model skill in the domain.

**2. The generalization gap is larger on wiki.** Legal loses 30.5% of loss going
from in-distribution to held-out; wiki loses 41.2%. If legal had learned domain
structure, its gap would be the smaller one — and it is. That is consistent with
the legal corpus being more templatic and repetitive, which also explains its
higher contamination (64% vs 44%).

**3. The in-distribution number is not a property of the model.** It is memory.
The 30-41% loss drop when leaving training is the direct measurement of that.

Reproduce (both sides, same checkpoint):

```bash
# held-out: filter contaminated and evaluate the clean ones
python scripts/eval_heldout.py --ckpt <dir> --corpus legalpt \
    --candidatos 200 --quero 40 --seed 42 --device cuda --saida /tmp/eval_heldout.json

# in-distribution: read the cache above and evaluate the complement (contaminated)
python scripts/eval_indist.py --ckpt <dir> --corpus legalpt \
    --saida-heldout /tmp/eval_heldout.json --device cuda --saida /tmp/eval_indist_legal.json
```

Both scripts print the indices they used and write a cache next to the output, so
the same split can be re-checked without repeating the training-tensor scan
(~20 min per corpus).

**Caveat that remains:** 40 documents per side is a defensible but not large
sample, and scanning 200/120 candidates gives a contamination estimate with a
margin of a few percentage points. What these numbers establish firmly is the
*direction and order of magnitude* of the gap — not the exact loss value in
production. A 1000+ document scan would narrow the uncertainty; it would not
change the conclusion that the in-distribution number was memory.

---

## What You Can and Cannot Verify

Being explicit about the boundary is more useful than implying everything is
reproducible. Three tiers:

**Fully reproducible by a third party (artifacts published here)**
- Every loss / perplexity / top-k figure: corpus sizes, token counts and
  checkpoint steps are in `logs/prediction-120m.txt`, and the arithmetic is
  checkable (`ppl = exp(loss)`, `bits = loss / ln 2`).
- The artifact size and every ratio derived from it: 614.45 MiB on disk against
  the dense baselines above is arithmetic anyone can redo.
- Roundtrip byte-identity claims — a third party can rerun `xz -9e` / `zstd -19`
  on the same public corpora and compare sizes.

### Where the compression ratio comes from

The ratio depends entirely on the baseline representation, so it is stated rather
than asserted. Against dense formats at inference precision, on the same artifact:

| Baseline | Size | Ratio |
|---|---|---|
| FP32 dense inference, 1.03B weights | 3.85 GiB | **6.41×** |
| BF16 dense inference | 1.92 GiB | **3.21×** |
| Training checkpoint (weights + optimizer state) | 27.26 GiB | 45.4× |

The 114× figure in older revisions compared against the **training checkpoint**,
which is not a format anyone serves. The defensible numbers are the ones above.
The internal block structure that produces them is proprietary; what can be
checked here is the artifact size and the arithmetic, not the layout.

**Verifiable in principle, not published here**
- CPU-vs-GPU coherence: the test log is published, the engine that produced it
  is not. A third party can read the result but cannot re-run it.
- Speedup figures (69.4×, tok/s): hardware-dependent and measured on the
  author's machine only. No independent reproduction.

**Not established**
- General capability of the 120M model beyond the two evaluated corpora.
  30 documents per corpus is a demonstration, not a generalization claim.
- Comparison against other models of similar size — not done. The only reference
  point in the log is uniform-random (`ln 50257` = 10.83 nats).
- **No external inference-engine baseline.** There is no comparison against
  bitnet.cpp, the ternary types in llama.cpp/ggml, or any other implementation.
  The 69.4× compares the GPU path against this engine's own CPU path — it says
  nothing about competitiveness. Reading it as a competitive claim is a
  misreading the numbers invite, which is why it is stated here.
- **No per-stage GPU trace.** The CPU/GPU split is not shown to be optimal; it is
  only shown to be faster end-to-end than the scalar CPU path.
- The 2.33 bits/token figure is task-dependent (legal domain); it is not a
  claim about the model's general compressibility.

## ⚠️ What's NOT Here

The source code of the engine, the CUDA kernel implementation, and the C++
alignment tricks are proprietary. This repository serves as a proof-of-concept
portfolio. No weight checkpoints, no tokenized training data, and no corpus
contents are published — only aggregate metrics.

## 🔗 Links

- **Blog:** [lifelog.seu.pet](https://lifelog.seu.pet)
- **Technical E-book:** [R$ 37 → skills.seu.pet/p/o-agente-que-nao-mente](https://skills.seu.pet/p/o-agente-que-nao-mente)
- **Guided Setup:** [R$ 1.500 → skills.seu.pet/servico](https://skills.seu.pet/servico)

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md).
