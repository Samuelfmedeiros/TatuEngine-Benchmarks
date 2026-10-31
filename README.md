# 🧠 TatuEngine-Benchmarks — Samuel Medeiros

Benchmarks and coherence proofs for the TatuEngine ternary inference engine.

**Live:** https://github.com/Samuelfmedeiros/TatuEngine-Benchmarks

> 🌐 **English** · [🇧🇷 Português](README.pt-BR.md)

---

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
│   └── coherence-test.txt       # 16/16 coherence tests (CPU vs GPU)
└── architecture/
    └── diagram.md               # High-level architecture diagram
```

**Stack:** C++ · CUDA · Python · Astro (blog) · FastAPI (skills.seu.pet)

---

## 📊 Results (RTX 3060, 12GB VRAM)

| Metric | Value |
|--------|-------|
| Compression | 27.26 GB → 245 MB (114×) |
| MSE (in_proj) | 3.7e-9 (two orders below FP32 epsilon) |
| Phase diff | 0.00e+00 (GPU bit-exact vs CPU) |
| Coherence | 16/16 tests passed |
| Speedup | 69.4× (4.092s/tok CPU → 0.059s/tok GPU) |
| Blocks | 893K (in_proj 530K, out_proj 262K, lm_head 100K) |

## 🔬 What's Inside

- **`logs/coherence-test.txt`** — 16/16 coherence tests proving GPU traversal is bit-exact against CPU ground truth
- **`architecture/diagram.md`** — high-level BlockLens, BVH, and hybrid pipeline diagrams
- **`README.md`** — technical paper with numbers, architecture, and links

## ⚠️ What's NOT Here

The source code of the engine, the CUDA kernel implementation, and the C++ alignment tricks are proprietary. This repository serves as a proof-of-concept portfolio.

## 🔗 Links

- **Blog:** [lifelog.seu.pet](https://lifelog.seu.pet)
- **E-book Técnico:** [R$ 37 → skills.seu.pet/p/o-agente-que-nao-mente]
- **Setup Guiado:** [R$ 1.500 → skills.seu.pet/servico]

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md).
