# 🧠 TatuEngine-Benchmarks — Samuel Medeiros

Benchmarks e provas de coerência do motor de inferência ternária TatuEngine.

**Live:** https://github.com/Samuelfmedeiros/TatuEngine-Benchmarks

---

## 📊 Resultados Medidos (RTX 3060)

- **Compressão:** 27.26 GB → 245 MB (114×) via block encoding (blocos ternários 32×32 em BlockLens de 288 bytes)
- **MSE:** 3.7e-9 (duas ordens abaixo do epsilon do FP32)
- **Phase diff:** 0.00e+00 (travessia GPU bit-a-bit idêntica ao CPU ground truth)
- **Coerência:** 16/16 testes passados
- **Speedup:** 69.4× (4.092s/tok CPU → 0.059s/tok GPU)
- **Hardware:** RTX 3060 (12GB VRAM)

## 🏗️ Arquitetura (alto nível)

- **BlockLens:** bloco de 32×32 pesos ternários {-1, 0, +1} empacotado em 288 bytes (2 bits/peso + 32 bytes de metadados)
- **BVH:** Block-encoded BVH com 893K blocos para 1B params (in_proj, out_proj, lm_head)
- **Pipeline:** Híbrido CPU (estado SSM, conv1d, RMSNorm) + GPU (bridge_matmul para in_proj/out_proj/lm_head)
- **Paradigma:** Substituir multiplicações de ponto flutuante por operações de adição em tensores ternários

## 📁 Estrutura

```
tatuengine-benchmarks/
├── README.md                    # Paper em Markdown (EN)
├── README.pt-BR.md              # Este arquivo
├── LICENSE                      # MIT
├── .gitignore
├── logs/
│   └── coherence-test.txt       # 16/16 testes de coerência (CPU vs GPU)
└── architecture/
    └── diagram.md               # Diagrama de alto nível (BlockLens, BVH, pipeline)
```

## 🔗 Links

- **Blog:** [lifelog.seu.pet](https://lifelog.seu.pet)
- **E-book Técnico:** [R$ 37 → skills.seu.pet/p/o-agente-que-nao-mente]
- **Setup Guiado:** [R$ 1.500 → skills.seu.pet/servico]

## ⚠️ O que NÃO está aqui

O código-fonte do motor, o kernel CUDA real e a lógica de alinhamento em C++ são proprietários. Este repositório serve como portfólio de provas técnicas.
