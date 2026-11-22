"""Byte-pair encoding (BPE) tokenizer, written from scratch.

How it works:
    1. Start from raw UTF-8 bytes. Token IDs 0..255 are the 256 possible byte values.
    2. Count how often each pair of neighbouring tokens appears.
    3. Merge the most frequent pair into one new token (ID 256, then 257, ...) and remember the rule.
    4. Repeat until the vocabulary has the wanted size.

The learned rules are stored in `merges`, a dict {(token_a, token_b): new_token_id}, in the order they
were learned. That dict is the whole trained tokenizer.

Other files use it like this:
    from bpe import train_bpe, build_vocab, encode, decode, save_merges, load_merges

Run `python bpe.py` to train on input.txt, save merges.pkl and run the checks.
"""

import pickle


def get_stats(ids):
    """Count how often each pair of neighbouring tokens appears in `ids`."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with the single token `new_id`."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def train_bpe(text, vocab_size=512, verbose=True):
    """Learn `vocab_size - 256` merges from `text`.

    Returns (merges, ids): the merge rules and the text as token IDs after all merges.
    """
    assert vocab_size >= 256, "vocab_size must be at least 256 (the raw bytes)"
    num_merges = vocab_size - 256
    ids = list(text.encode("utf-8"))
    n_bytes = len(ids)
    merges = {}
    for i in range(num_merges):
        stats = get_stats(ids)
        if not stats:  # text is too short to merge any further
            break
        pair = max(stats, key=stats.get)
        new_id = 256 + i
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        if verbose and (i % 32 == 0 or i == num_merges - 1):
            print(f"merge {i + 1}/{num_merges}: {pair} -> {new_id} ({stats[pair]} times)")
    if verbose:
        print("tokens before:", n_bytes, "| after:", len(ids), "| compression ratio:", round(n_bytes / len(ids), 3))
    return merges, ids


def build_vocab(merges):
    """Map every token ID to the bytes it stands for (e.g. 273 -> b' th')."""
    vocab = {i: bytes([i]) for i in range(256)}
    for (p0, p1), idx in merges.items():  # merges are in learned order, so p0 and p1 already exist
        vocab[idx] = vocab[p0] + vocab[p1]
    return vocab


def encode(text, merges):
    """Turn text into token IDs by applying the learned merges, earliest-learned first."""
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:  # none of the remaining pairs was ever merged
            break
        ids = merge(ids, pair, merges[pair])
    return ids


def decode(ids, vocab):
    """Turn token IDs back into text."""
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def save_merges(merges, path="merges.pkl"):
    with open(path, "wb") as f:
        pickle.dump(merges, f)


def load_merges(path="merges.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    print("characters:", len(text), "| bytes:", len(text.encode("utf-8")))

    merges, ids = train_bpe(text, vocab_size=512)
    save_merges(merges, "merges.pkl")
    print("saved merges.pkl with", len(merges), "merges")

    vocab = build_vocab(merges)

    # the saved file must load back to the same rules
    assert load_merges("merges.pkl") == merges
    print("save and load: OK")

    # decode(encode(text)) must give the original text back
    sample = "the king shall speak, and ROMEO will answer."
    enc = encode(sample, merges)
    print(enc)
    print([vocab[i] for i in enc])
    assert decode(enc, vocab) == sample
    assert decode(encode(text[:20000], merges), vocab) == text[:20000]
    print("round trip: OK")

    # text with non-ASCII characters still works, because we start from bytes
    odd = "café — naïve"
    assert decode(encode(odd, merges), vocab) == odd
    print("non-ASCII round trip: OK")

    # the whole text, merged during training, must decode back as well
    assert decode(ids, vocab) == text
    print("full text round trip: OK")
    print("all checks passed")
