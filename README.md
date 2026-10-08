# gpt2 — "Let's reproduce GPT-2 (124M)"

Phase 1, item 4. Following Karpathy's 4-hour video, typed by hand. Reference code: github.com/karpathy/build-nanogpt (one commit per step of the video) — look only when stuck, same rules as nanoGPT.

**Done when:** followed along on a rented GPU. The output item after it needs a measured worklog entry: tokens/sec, MFU, memory, and where the time goes.

## Slices
1. **Model + pretrained weights (Mac).** Write the GPT-2 `nn.Module` with HF-compatible names, load OpenAI's 124M weights, sample text. Then train from scratch on tiny-Shakespeare to check the loop. File: `train_gpt2.py`.
2. **Speedups (1x rented Nvidia GPU).** TF32, bf16 autocast, `torch.compile`, flash attention, nice numbers (vocab 50304). Measure tokens/sec after each change.
3. **Training recipe.** AdamW betas + weight decay, grad clipping, cosine LR with warmup, gradient accumulation to a 0.5M-token batch, DDP.
4. **Full run (8x rented GPUs).** FineWeb-Edu 10B sample, val loss + HellaSwag eval, checkpoints. Then the measured writeup.

## Rules
- Mac (MPS) for slice 1 and debugging. Anything that goes in the worklog is measured on Nvidia.
- Every performance change gets a before/after tokens/sec number.
- Kill GPU instances the moment a run ends.

## Setup
    source ../.venv/bin/activate
    pip install transformers tiktoken numpy tqdm datasets
