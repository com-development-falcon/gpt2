import inspect
import math
import os
import torch
import time
import torch.nn as nn
from dataclasses import dataclass
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hellaswag import iterate_examples, render_example

# -------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # q, k, v in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = config.flash

        # mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, nh, T, hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, nh, T, hs
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, nh, T, hs
        # attention (materializes the large T, T matrix for all queries and keys)

        if self.flash:
            # fused attention kernel (FlashAttention on CUDA); never materializes the T x T matrix
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # B, nh, T, hs
        else:
            # manual attention: materializes the large T x T matrix
            att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # B, nh, T, T
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v  # B, nh, T, hs

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head output side by side
        # output proj
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50257  # number of tokens, 50K BPE merges, 256 bytes tokens, + 1 <|endoftext|> token
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    flash: bool = True  # True: F.scaled_dot_product_attention; False: manual attention


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing sheme
        self.transformer.wte.weight = self.lm_head.weight

        # init param
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def flops_per_token(self, T):
        """Training FLOPs per token: 3x forward (forward + 2x backward)."""
        cfg = self.config
        L, d, V = cfg.n_layer, cfg.n_embd, cfg.vocab_size
        P_mat = L * 12 * d * d + d * V  # matmul weights: 12d^2 per block + lm_head
        attn = L * 4 * T * d  # q@k^T and att@v, full TxT matrix
        return 3 * (2 * P_mat + attn)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T, )
        pos_emb = self.transformer.wpe(pos)  # T, n_embd
        tok_emb = self.transformer.wte(idx)  # B, T, n_embd
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # B, T, vocab_size

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weights from hugginface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("Loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M param
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M param
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M param
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M param
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device in ("cuda", "mps")
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

def get_most_likely_row(tokens, mask, logits):
    # per-token loss, with the usual shift: predict token t+1 from position t
    shift_logits = (logits[..., :-1, :]).contiguous()          # (4, N-1, V)
    shift_tokens = (tokens[..., 1:]).contiguous()              # (4, N-1)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)      # (4, N-1)
    # average the loss over the ending tokens only (mask == 1)
    shift_mask = (mask[..., 1:]).contiguous()                  # shift the mask like the tokens
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)                # (4,)
    # the ending with the lowest average loss is the model's answer
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------
import tiktoken
import numpy as np

enc = tiktoken.get_encoding("gpt2")


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
        # ------- for testing --------
        # # at init load tokens from disk and store them in memory
        # with open("input.txt", "r", encoding="utf-8") as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# ----------------------------------------------------------
# run the training loop
# simple launch
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
from torch.distributed import init_process_group, destroy_process_group

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropreiately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, check point etc
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetech device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
device_type = "cuda" if device.startswith("cuda") else device

# ======================= run configuration (toggle here) =======================
# perf switches -- flip one at a time to fill perf.md
use_tf32 = True  # TF32 matmuls on Ampere+ GPUs (no effect on MPS)
use_bf16 = True  # bf16 autocast on CUDA (ignored on MPS / CPU)
use_compile = False  # torch.compile for training; evals run on the uncompiled raw_model
use_flash = True  # fused attention (SDPA) vs manual attention
vocab_size = 50304  # 50257 = original GPT-2 vocab, 50304 = "nice number" padding

# batch / schedule
total_batch_size = 4 * 1024 * 2  # tokens per optimizer step; 524288 (2**19) for real runs
B = 4  # micro batch size; 64 on an 80 GB GPU, <= 32 on MPS
T = 1024  # sequence length
max_steps = 20  # 19073 = one pass over 10B tokens; ~50 for perf runs
warmup_steps = 715  # 715 for the full run; ~10 for perf runs

# evals / logging
run_evals = True  # val loss, HellaSwag and sampling every eval_interval steps
eval_interval = 250
save_checkpoints = True  # model checkpoint every 5000 steps and at the last step
perf_skip_steps = 3  # warm-up steps excluded from the perf summary
# ===============================================================================

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
if device_type == "mps" and torch.mps.is_available():
    torch.mps.manual_seed(1337)


# get a data batch
# total_batch_size = 524288  # 2**19 ~ 0.5M in number of tokens
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total_batch_size is dicisible by B*T*ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
val_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
)

if use_tf32:
    torch.set_float32_matmul_precision("high")

# create model
model = GPT(
    GPTConfig(
        vocab_size=vocab_size,
        flash=use_flash,
        # n_embd=1024,
        # n_head=16
    )
)
model.to(device)
raw_model = model  # keep a handle on the uncompiled model
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


max_lr = 6e-4
min_lr = max_lr * 0.1


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# optimization
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device=device_type
)

# create the log directory we will write checkpoints to and logs to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass
perf_file = os.path.join(log_dir, "perf.csv")
run_tag = (
    f"tf32={use_tf32} bf16={use_bf16} compile={use_compile} flash={use_flash} "
    f"vocab={vocab_size} B={B} T={T} total={total_batch_size} gpus={ddp_world_size} device={device_type}"
)
if master_process:
    print(f"run config: {run_tag}")
    with open(perf_file, "w") as f:
        f.write(f"# {run_tag}\n")
        f.write("step,dt_ms,tok_per_sec,tflops,peak_mem_gb\n")
perf_rows = []  # (dt_ms, tok_per_sec, tflops, peak_mem_gb) after warm-up


def peak_mem_gb():
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device_type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    return 0.0



flops_per_tok = raw_model.flops_per_token(train_loader.T)
print(f"training FLOPs per token: {flops_per_tok:,}")
for step in range(max_steps):
    last_step = (step == max_steps - 1)

    # once in a while evalutate our validation loss
    if (run_evals and step % eval_interval == 0) or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                if device_type == "cuda" and use_bf16:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if save_checkpoints and step > 0 and (step % 5000 == 0 or last_step):
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if run_evals and (step % eval_interval == 0 or last_step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples('val')):
            # only process examples where i % ddp_world_size == ddp
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                if device_type == "cuda" and use_bf16:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = raw_model(tokens)
                else:
                    logits, loss = raw_model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)

        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")


    # once in a while generate from the model
    if run_evals and ((step > 0 and step % eval_interval == 0) or last_step):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, _ = raw_model(xgen)
                logits = logits[:, -1, :]  # B, vocab_size
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50
                # topk_probs bere becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # B, 1
                xcol = torch.gather(topk_indices, -1, ix)  # B, 1
                xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} same {i}: {decoded}")

    # training loop
    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if device_type == "cuda" and use_bf16:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        # import code; code.interact(local=locals())
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == (grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = (
        (train_loader.B * train_loader.T) * grad_accum_steps * ddp_world_size
    )
    tokens_per_sec = tokens_processed / dt
    tflops = flops_per_tok * tokens_per_sec / 1e12
    if master_process:
        mem = peak_mem_gb()
        print(
            f"step {step:4d}| loss: {loss_accum.item():.6f}| lr: {lr:.4e}| norm: {norm:.4f}| dt: {dt * 1000:.2f}ms| tok/sec: {tokens_per_sec:.0f}| TFLOPS: {tflops:.2f}| mem: {mem:.1f}GB"
        )
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")
        with open(perf_file, "a") as f:
            f.write(f"{step},{dt * 1000:.2f},{tokens_per_sec:.0f},{tflops:.3f},{mem:.2f}\n")
        if step >= perf_skip_steps:
            perf_rows.append((dt * 1000, tokens_per_sec, tflops, mem))

if master_process and perf_rows:
    n = len(perf_rows)
    avg = lambda i: sum(r[i] for r in perf_rows) / n
    print("=" * 80)
    print(f"perf summary ({run_tag})")
    print(
        f"steps {perf_skip_steps}..{max_steps - 1} (n={n}): "
        f"dt {avg(0):.1f} ms | {avg(1):.0f} tok/s | {avg(2):.2f} TFLOPS | "
        f"peak mem {max(r[3] for r in perf_rows):.1f} GB"
    )
    print("=" * 80)

if ddp:
    destroy_process_group()

exit(0)
# -----------------------------
