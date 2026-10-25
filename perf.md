# GPT-2 124M — throughput log

Setup: B=16, T=1024 (16,384 tokens/step), AdamW lr 3e-4, tiny-Shakespeare via DataLoaderLite.
Timing: wall clock per step with device synchronize; step 0 excluded (warm-up).
Achieved FLOPs ≈ 6 · 124M · tok/s (ignores attention's extra ~15% at T=1024).

| Date | Hardware | Change | ms/step (steady) | tok/s | ≈ TFLOPS | Notes |
|---|---|---|---|---|---|---|
| 2026-10-17 | Mac Studio M2 Ultra (MPS) | FP32 baseline | ~1,590–1,670 | ~10,000 | ~7.4 | dt drifts up ~5% over 45 steps |
| 2026-10-17 | Mac Studio M2 Ultra (MPS) | + bf16 autocast | ~2,116 | ~7,750 | ~5.8 | ~22% lower throughput (steps ~30% longer): no fast bf16 matmul path on MPS, autocast adds casts. Loss matches FP32 to ~3 decimals. |
| 2026-10-17 | Mac Studio M2 Ultra (MPS) | FP32 + `torch.compile` (no autocast) | ~1,463–1,485 | ~11,100 | ~8.3 | ~8% faster than FP32 over the same steps. Step 0 = 9,168 ms (compile time). Loss identical to FP32 run. |
| 2026-10-17 | Mac Studio M2 Ultra (MPS) | FP32 + `torch.compile`, manual attention (re-run) | ~1,462–1,467 (steps 1–13) | ~11,190 | 9.56 (exact count) | Step 0 = 2,986 ms: compile likely reused its on-disk cache from the previous run. |
| 2026-10-17 | Mac Studio M2 Ultra (MPS) | FP32 + `torch.compile` + SDPA `is_causal=True` | ~1,464–1,471 (steps 1–13) | ~11,180 | 9.55 (exact count) | No gain over compiled manual attention. FlashAttention backend is CUDA-only; on MPS, compile likely already fused mask/softmax. Step 0 = 10,255 ms (new graph → recompile). |
| | A100 / H100 (rented) | FP32 baseline | | | | |
| | | + TF32 (`set_float32_matmul_precision("high")`) | | | | |
| | | + bf16 autocast | | | | |
| | | + `torch.compile` | | | | |
| | | + flash attention | | | | |
| | | + vocab 50304 | | | | |

## Width vs throughput (Mac, FP32 + torch.compile, B=16, T=1024)
Training FLOPs/token counted exactly: 3 × (2·P_matmul + L·4·T·d), P_matmul = L·12d² + d·V.

| Config | FLOPs/token (train) | tok/s | Achieved TFLOPS |
|---|---|---|---|
| d=768, 12 heads (GPT-2 124M) | 854.4M | ~10,666 | 9.1 |
| d=1024, 16 heads | 1,365.7M | ~7,958 | 10.9 (+19%) |

Wider matmuls → higher arithmetic intensity → better utilization. Prediction (wider ↑, deeper ≈ flat) held for width; depth not yet tested.

## Ablation: what slowed the Mac from ~1,465 to ~1,747 ms/step (FP32 + compile + SDPA, B=16, T=1024)

| Run | vocab | clip_grad_norm_ | AdamW | ms/step (steps 1–5) | TFLOPS |
|---|---|---|---|---|---|
| Before LR/clip/vocab changes | 50257 | off | plain | ~1,463–1,471 | 9.55 |
| A: clip on, unfused | 50304 | on | plain | ~1,734–1,786 (drifts up) | ~7.9 |
| A': clip on, fused | 50304 | on | fused | ~1,746–1,749 (flat through step 16) | 8.01 |
| C: clip off, fused | 50304 | **off** | fused | ~1,479–1,485 | 9.45 |

Result: `clip_grad_norm_` costs ~265 ms/step on MPS (~18% slower). Its memory traffic (read ~0.5 GB of FP32 grads, scale them) is only ~1–2 ms at ~800 GB/s, so the cost is overhead/implementation on MPS, not bandwidth. `foreach=True` is not supported on MPS (RuntimeError: "can't use the foreach API on mps tensors"), so MPS takes the per-tensor path: ~148 tensors × several small kernels per step. Fused AdamW: ~1% faster, and removed the upward step-time drift (hypothesis: fewer temporaries; unconfirmed).

## Finding: MPS gives a wrong loss when the logits tensor exceeds 2^31 elements (2026-10-25)

Same model, same seed (step 0, untrained), FineWeb-Edu val shard, M2 Ultra (MPS), FP32:

| B | logits elements (B·1024·50304) | > 2^31? | step-0 val loss | step-0 train loss | HellaSwag |
|---|---|---|---|---|---|
| 64 | 3.30e9 | yes | 7.1403 | 7.1359 | 2496/10042 |
| 4 | 2.06e8 | no | 10.9553 | 10.9550 | 2496/10042 |

Expected untrained loss ≈ ln(50304) = 10.83 (plus a little for init noise), so B=4 is correct and B=64 is silently wrong.
HellaSwag is identical in both runs: it runs 4-row batches (far below 2^31), and the seed is the same — which isolates the problem to the big-batch path.
Hypothesis: an MPS kernel uses 32-bit indexing and silently overflows past 2^31 elements. Predicted threshold: B ≤ 41 correct, B ≥ 42 wrong (2^31 / (1024·50304) ≈ 41.7). Not yet tested.
B=64 was also ~7x slower per token (likely swapping: FP32 activations + 13 GB of logits exceed free unified memory).
Lesson: always check the step-0 loss against ln(V) on a new machine/config.
