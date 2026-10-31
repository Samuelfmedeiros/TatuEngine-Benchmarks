# 🧠 TatuEngine-Benchmarks — Samuel Medeiros

Benchmarks e provas de coerência do motor de inferência ternária TatuEngine.

**Live:** https://github.com/Samuelfmedeiros/TatuEngine-Benchmarks

> 🌐 [English](README.md) · 🇧🇷 **Português**

---

## 🏗️ Arquitetura

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
│   └── coherence-test.txt       # 16/16 testes de coerência (CPU vs GPU)
└── architecture/
    └── diagram.md               # Diagrama de arquitetura de alto nível
```

**Stack:** C++ · CUDA · Python · Astro (blog) · FastAPI (skills.seu.pet)

---

## 📊 Resultados (RTX 3060, 12GB VRAM)

| Métrica | Valor |
|---------|-------|
| Compressão | 27.26 GB → 245 MB (114×) |
| MSE (in_proj) | 3.7e-9 (duas ordens abaixo do epsilon do FP32) |
| Phase diff | 0.00e+00 (GPU bit-exact vs CPU) |
| Coerência | 16/16 testes passados |
| Speedup | 69.4× (4.092s/tok CPU → 0.059s/tok GPU) |
| Blocos | 893K (in_proj 530K, out_proj 262K, lm_head 100K) |

## 🔬 O que está aqui

- **`logs/coherence-test.txt`** — 16/16 testes de coerência provando que a travessia GPU é bit-exact contra o CPU ground truth
- **`architecture/diagram.md`** — diagramas de alto nível do BlockLens, BVH e pipeline híbrido
- **`README.md`** — paper técnico com números, arquitetura e links

## ⚠️ O que NÃO está aqui

O código-fonte do motor, o kernel CUDA real e a lógica de alinhamento em C++ são proprietários. Este repositório serve como portfólio de provas técnicas.

## 🔗 Links

- **Blog:** [lifelog.seu.pet](https://lifelog.seu.pet)
- **E-book Técnico:** [R$ 37 → skills.seu.pet/p/o-agente-que-nao-mente]
- **Setup Guiado:** [R$ 1.500 → skills.seu.pet/servico]

---

## 📋 Changelog

Ver [CHANGELOG.md](CHANGELOG.md).
