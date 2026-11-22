"""Generate Shakespeare-like text from a trained checkpoint.

Needs merges.pkl and a checkpoint (default gpt_gqa_best.pt) in the current folder, plus model.py and bpe.py.
    python generate.py                                   # three default prompts, printed on screen
    python generate.py --prompt "ROMEO:" --prompt "KING HENRY:\\n"
    python generate.py --temperature 1.0 --top_k 100     # more random text
    python generate.py --out samples.txt                 # also save the text to a file

How the text is made: the prompt is turned into token IDs, then the model predicts the next token over and
over. At each step it takes the scores of the last position, divides them by the temperature, keeps only the
top_k most likely tokens, turns the scores into probabilities and draws one token. That token is appended and
the loop repeats. At the end the token IDs are decoded back to text.

The output depends on the seed AND on the device: the same seed on a CPU and on a GPU gives different text.
"""

import argparse

import torch
import torch.nn.functional as F

from bpe import build_vocab, decode, encode, load_merges
from model import GPT, GPTConfig

DEFAULT_PROMPTS = ["ROMEO:", "First Citizen:\n", "KING HENRY:\n"]


def parse_args():
    p = argparse.ArgumentParser(description="Generate text with the trained GPT model")
    p.add_argument("--checkpoint", default="gpt_gqa_best.pt")
    p.add_argument("--merges", default="merges.pkl")
    p.add_argument("--prompt", action="append",
                   help="starting text; can be given several times; write \\n for a new line")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8, help="lower = safer text, higher = more random")
    p.add_argument("--top_k", type=int, default=40, help="only sample from the k most likely tokens")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--vocab_size", type=int, default=512)
    p.add_argument("--n_kv_head", type=int, default=2, help="must match the checkpoint (2 = GQA, 6 = multi-head)")
    p.add_argument("--out", default=None, help="also save the generated text to this file (it is overwritten)")
    return p.parse_args()


@torch.no_grad()
def generate(model, merges, prompt, max_new_tokens=300, temperature=0.8, top_k=40, device="cpu"):
    """Continue `prompt` by max_new_tokens tokens and return the whole text."""
    assert temperature > 0, "temperature must be above 0"
    ids = encode(prompt, merges)
    assert len(ids) > 0, "the prompt must not be empty"
    idx = torch.tensor([ids], dtype=torch.long, device=device)  # (1, T)
    block_size = model.cfg.block_size

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]                  # the model sees at most block_size tokens
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature          # scores for the next token only
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[:, [-1]]] = float("-inf")      # drop everything outside the top k
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx[0].tolist()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    torch.manual_seed(args.seed)

    merges = load_merges(args.merges)
    vocab = build_vocab(merges)
    assert len(vocab) == args.vocab_size, (
        f"merges.pkl gives a vocabulary of {len(vocab)} tokens but --vocab_size is {args.vocab_size}")

    cfg = GPTConfig(vocab_size=args.vocab_size, n_kv_head=args.n_kv_head)
    model = GPT(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("loaded", args.checkpoint)

    prompts = [p.replace("\\n", "\n") for p in args.prompt] if args.prompt else DEFAULT_PROMPTS

    results = []
    for prompt in prompts:
        ids = generate(model, merges, prompt, args.max_new_tokens, args.temperature, args.top_k, device)
        text = decode(ids, vocab)
        results.append(text)
        print("=" * 60)
        print(text)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for text in results:
                f.write("=" * 60 + "\n" + text + "\n\n")
        print("saved", args.out)


if __name__ == "__main__":
    main()
