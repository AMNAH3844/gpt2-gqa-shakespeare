"""Train the GPT model (with grouped-query attention) on Tiny Shakespeare.

Needs input.txt in the current folder, plus model.py and bpe.py from this repo.
    python train.py                       # default run: 3000 steps, 2 key/value heads
    python train.py --n_kv_head 6         # plain multi-head attention, for comparing with GQA
    python train.py --max_steps 1000      # shorter run

What it does:
    1. Loads the BPE tokenizer from merges.pkl (trains and saves it if the file is missing).
    2. Encodes the text once and caches the token IDs in data.pt. Delete data.pt after changing merges.pkl.
    3. Trains with AdamW, learning rate warmup then cosine decay, and gradient clipping.
    4. Every --eval_interval steps prints train and validation loss, and saves the weights whenever the
       validation loss improves. The saved file is the best checkpoint, not the last one.
    5. Saves a plot of the loss curves.

Careful: the checkpoint (--out) and the plot (--plot) are overwritten if they already exist.
"""

import argparse
import math
import os
import time

import torch

from bpe import encode, load_merges, save_merges, train_bpe
from model import GPT, GPTConfig, count_parameters


def parse_args():
    p = argparse.ArgumentParser(description="Train a GPT-2 style model with GQA on Shakespeare")
    p.add_argument("--input", default="input.txt")
    p.add_argument("--merges", default="merges.pkl")
    p.add_argument("--data_cache", default="data.pt")
    p.add_argument("--out", default="gpt_gqa_best.pt", help="where the best checkpoint is saved")
    p.add_argument("--plot", default="loss_curve.png", help="where the loss plot is saved")
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--n_kv_head", type=int, default=2, help="key/value heads; 6 means plain multi-head attention")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--patience", type=int, default=0,
                   help="stop after this many evaluations without a better validation loss (0 = never stop early)")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def prepare_data(input_path, merges_path, cache_path, vocab_size):
    """Return (train_data, val_data) as 1D tensors of token IDs, split 90% / 10%."""
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if os.path.exists(merges_path):
        merges = load_merges(merges_path)
        print(f"loaded {len(merges)} merges from {merges_path}")
    else:
        print("no merges file found, training the BPE tokenizer (about a minute)")
        merges, _ = train_bpe(text, vocab_size)
        save_merges(merges, merges_path)

    if os.path.exists(cache_path):
        data = torch.load(cache_path)
        print(f"loaded cached token IDs from {cache_path}")
    else:
        print("encoding the dataset (about a minute)")
        data = torch.tensor(encode(text, merges), dtype=torch.long)
        torch.save(data, cache_path)

    assert int(data.max()) < vocab_size, "token IDs do not fit in vocab_size; delete data.pt and try again"
    n = int(0.9 * len(data))
    print(f"tokens: {len(data)} | train: {n} | val: {len(data) - n}")
    return data[:n], data[n:]


def get_lr(step, max_lr, warmup_steps, max_steps):
    """Linear warmup, then cosine decay down to one tenth of max_lr."""
    min_lr = max_lr / 10
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def configure_optimizer(model, lr, weight_decay):
    """AdamW with weight decay on the weight matrices only (not on layer norm parameters)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


def get_batch(data, batch_size, block_size, device):
    """Random chunks of block_size tokens (x) and the same chunks shifted by one token (y)."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, batch_size, block_size, device, eval_iters):
    """Average loss over several random batches from each split."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window needed
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(history["step"], history["train"], label="train")
    plt.plot(history["step"], history["val"], label="validation")
    best_i = history["val"].index(min(history["val"]))
    plt.axvline(history["step"][best_i], color="gray", linestyle="--", label="best checkpoint")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("GPT-2 style model with GQA on Shakespeare")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("saved", path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cpu":
        print("warning: no GPU found, training will be slow (use Google Colab with a T4 GPU)")

    train_data, val_data = prepare_data(args.input, args.merges, args.data_cache, args.vocab_size)
    splits = {"train": train_data, "val": val_data}

    cfg = GPTConfig(vocab_size=args.vocab_size, n_kv_head=args.n_kv_head)
    model = GPT(cfg).to(device)
    print(cfg)
    print("parameters:", count_parameters(model))

    optimizer = configure_optimizer(model, args.max_lr, args.weight_decay)

    history = {"step": [], "train": [], "val": []}
    best_val = float("inf")
    evals_without_gain = 0
    t0 = time.time()

    for step in range(args.max_steps):
        lr = get_lr(step, args.max_lr, args.warmup_steps, args.max_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, cfg.block_size, device)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(model, splits, args.batch_size, cfg.block_size, device, args.eval_iters)
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(f"step {step:4d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.5f} | grad norm {norm:.2f} | {time.time() - t0:.0f}s")

            if losses["val"] < best_val:
                best_val = losses["val"]
                evals_without_gain = 0
                torch.save(model.state_dict(), args.out)
            else:
                evals_without_gain += 1
                if args.patience and evals_without_gain >= args.patience:
                    print(f"validation loss has not improved for {args.patience} evaluations, stopping early")
                    break

    best_step = history["step"][history["val"].index(best_val)]
    print(f"best val loss: {best_val:.4f} at step {best_step} (saved to {args.out})")
    plot_history(history, args.plot)


if __name__ == "__main__":
    main()
