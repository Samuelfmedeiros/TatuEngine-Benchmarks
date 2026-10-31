# TatuEngine-Benchmarks

Benchmarks and coherence proofs for the TatuEngine block-encoded ternary inference engine.

## Summary

- **Compression:** 27.26 GB → 245 MB (114×) via block encoding (32×32 ternary blocks in 288-byte BlockLens)
- **MSE:** 3.7e-9 (two orders below FP32 epsilon)
- **Phase diff:** 0.00e+00 (GPU traversal bit-a-bit identical to CPU ground truth)
- **Coherence:** 16/16 tests passed
- **Speedup:** 69.4× (4.092s/tok CPU → 0.059s/tok GPU)
- **Hardware:** RTX 3060 (12GB VRAM)

## Architecture (high-level)

- **BlockLens:** 32×32 ternary weight block {-1, 0, +1} packed into 288 bytes (2 bits/weight + 32 bytes metadata)
- **BVH:** Block-encoded BVH with 893K blocks for 1B params (in_proj, out_proj, lm_head)
- **Pipeline:** Hybrid CPU (SSM state, conv1d, RMSNorm) + GPU (bridge_matmul for in_proj/out_proj/lm_head)
- **Paradigm:** Replace floating-point multiplications with addition operations on ternary tensors

## What's NOT here

The source code of the engine, the CUDA kernel implementation, and the C++ alignment tricks are proprietary. This repository serves as a proof-of-concept portfolio.

## Links

- Blog: [link]
- E-book: [R$ 37 → skills.seu.pet/p/o-agente-que-nao-mente]
- Consulting: [R$ 1.500 → skills.seu.pet/servico]
