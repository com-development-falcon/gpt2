"""
Generate text from a trained checkpoint.

  python sample.py                                   # final model, default prompt
  python sample.py --prompt "The meaning of life is" --n 3 --max_tokens 80
  python sample.py --ckpt log_full_run/model_05000.pt   # an earlier checkpoint
  python sample.py --compare                          # also sample OpenAI's GPT-2 124M

train_gpt2.py has no __main__ guard (importing it would start training), so we
exec only the model part of it: everything above the DDP setup line.
"""
import argparse
import torch
import tiktoken
from torch.nn import functional as F

# --- load the model classes (GPTConfig, GPT, ...) from train_gpt2.py without training ---
src = open("train_gpt2.py").read()
src = src.split("ddp = int(os.environ")[0]                      # stop before DDP / training setup
src = src.replace("from hellaswag import iterate_examples, render_example", "")  # not needed here
exec(src, globals())  # defines GPTConfig, GPT in __main__, which torch.load needs to unpickle 'config'

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="log_full_run/model_19072.pt")
p.add_argument("--prompt", default="Hello, I'm a language model,")
p.add_argument("--n", type=int, default=4, help="number of samples")
p.add_argument("--max_tokens", type=int, default=64)
p.add_argument("--temp", type=float, default=1.0)
p.add_argument("--top_k", type=int, default=50)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--compare", action="store_true", help="also sample OpenAI GPT-2 124M")
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
enc = tiktoken.get_encoding("gpt2")


def load_mine(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    print(f"loaded {path}: step {ckpt['step']}, val loss {float(ckpt['val_loss']):.4f}")
    return model


@torch.no_grad()
def generate(model, prompt, n, max_tokens, temp, top_k, seed):
    model.to(device).eval()
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)
    x = tokens.unsqueeze(0).repeat(n, 1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(max_tokens):
        out = model(x[:, -1024:])
        logits = out[0] if isinstance(out, tuple) else out.logits   # mine returns (logits, loss); HF returns an object
        logits = logits[:, -1, :50257] / temp                        # drop the 47 padded vocab slots
        probs = F.softmax(logits, dim=-1)
        topk_p, topk_i = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_p.cpu(), 1, generator=g).to(device)
        x = torch.cat((x, torch.gather(topk_i, -1, ix)), dim=1)
    return [enc.decode(row.tolist()) for row in x]


def show(title, samples):
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)
    for i, s in enumerate(samples):
        print(f"\n[{i}] {s}")


mine = load_mine(args.ckpt)
show(f"MY GPT-2 ({args.ckpt})", generate(mine, args.prompt, args.n, args.max_tokens, args.temp, args.top_k, args.seed))

if args.compare:
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained("gpt2")
    show("OPENAI GPT-2 124M", generate(hf, args.prompt, args.n, args.max_tokens, args.temp, args.top_k, args.seed))
