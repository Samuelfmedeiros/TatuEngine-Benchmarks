# Architecture Diagram (High-Level)

## Block Encoding

```
TriangleLens (CPU): ~120+ bytes per weight
    ↓ block_codec
BlockLens (GPU): 288 bytes per 32×32 block (1024 weights)
    = 0.28125 bytes/weight (~2.25 bits)
    = 114× compression
```

## BVH Pipeline

```
BVH (CPU) → upload_lenses() → CompactLens (52 bytes) → VRAM
    ↓
traverse_rays_kernel<<<blocks, 256, shared>>>
    ├── 1 thread / lens: Δφ = η·0.5 + bias, A = 1/(1+|Δφ|·attn)
    ├── warp shuffle: __shfl_xor_sync reduction (phase sum, amp product)
    └── block reduce: shared memory → single result
    ↓
EngineResult (phase, amplitude, depth)
```

## Hybrid Pipeline

```
Token (CPU): embedding lookup
    → 32× layers:
        RMSNorm (CPU) → in_proj (bridge_matmul/GPU) → conv1d+SiLU (CPU)
        → SSM step (CPU) → out_proj (bridge_matmul/GPU) → residual (CPU)
    → Final RMSNorm (CPU) → lm_head (bridge_matmul/GPU) → logits
```

## Memory Layout

```
in_proj:  530K blocks × 288 bytes = ~152 MB
out_proj: 262K blocks × 288 bytes = ~75 MB
lm_head:  100K blocks × 288 bytes = ~29 MB
Total:    893K blocks = 245 MB (vs 27.26 GB original)
```
