# GPT-2 style language model with Grouped-Query Attention and a from-scratch BPE tokenizer

A small GPT-2 style language model built in PyTorch and trained on the Tiny Shakespeare dataset. It follows the structure of Andrej Karpathy's GPT-2 code-along, with two changes:

1. **Grouped-query attention (GQA)** instead of standard multi-head attention.
2. **A byte-pair encoding (BPE) tokenizer written from scratch** and trained on the Shakespeare text. No `tiktoken` and no pre-trained tokenizer is used.

All the code was written by us, step by step, as part of an AI/ML internship task.

## Repository layout

| File | What it is |
| --- | --- |
| `bpe.py` | BPE tokenizer: pair counting, merging, training, encode, decode, save and load |
| `model.py` | The model: config, GQA attention, MLP, transformer block, full GPT |
| `train.py` | Training script: AdamW, warmup and cosine decay, gradient clipping, best-checkpoint saving, loss plot |
| `generate.py` | Text generation with temperature and top-k sampling |
| `gpt2_gqa.ipynb` | The notebook the code was developed in, with the outputs of the run |
| `input.txt` | Tiny Shakespeare dataset (about 1.1 MB) |
| `merges.pkl` | The learned BPE merge rules (the trained tokenizer) |
| `gpt_gqa_best.pt` | Weights of the best checkpoint (lowest validation loss) |
| `loss_curve.png` | Train and validation loss during training |
| `samples.txt` | Text generated from the best checkpoint |
| `requirements.txt` | Python packages (`torch`, `matplotlib`) |

## 1. BPE tokenizer (`bpe.py`)

- Starts from raw UTF-8 bytes (token IDs 0 to 255), so any text can be encoded.
- Repeatedly counts adjacent token pairs, merges the most frequent pair into a new token, and records the rule.
- 256 merges, giving a vocabulary of **512 tokens**.
- Compresses the dataset from 1,115,394 bytes to **568,210 tokens** (about 1.96 bytes per token).
- `decode(encode(text)) == text` is checked on the data, including non-ASCII text.
- The merges learn real structure, for example `" the "`, `", and "`, and the `":\n"` that ends a speaker label.

## 2. Model (`model.py`)

| Setting | Value |
| --- | --- |
| Layers | 6 |
| Embedding size | 384 |
| Query heads | 6 (head size 64) |
| Key/value heads | **2** (3 query heads share each KV head) |
| Context length | 256 tokens |
| Vocabulary | 512 |
| Parameters | 9,742,080 |

Architecture, as in GPT-2: learned token and position embeddings, pre-norm transformer blocks with residual connections, causal self-attention, an MLP with GELU (4x width), a final layer norm, and an output layer whose weights are tied to the token embedding. Weights are initialized with std 0.02, and the residual output projections are scaled by 1/sqrt(2 x layers). Biases are left out.

### What GQA changes

In standard multi-head attention every query head has its own key and value head. In GQA, the 6 query heads are split into 2 groups of 3, and each group shares one key/value head.

- The query projection stays 384 wide, but the key and value projections shrink from 384 to 2 x 64 = **128**.
- Before the attention scores are computed, each KV head is repeated 3 times so it lines up with the query heads.
- Attention parameters per layer: **393,216**, against 589,824 for standard multi-head attention at this size, a reduction of one third.
- The key/value cache needed during generation is also 3 times smaller.

`python model.py` runs a self-test: the output shape matches the input, changing only the last token leaves all earlier outputs unchanged (the causal mask works), the parameter counts above are correct, the weights are tied, and the start loss is close to ln(512) = 6.24.

## 3. Training (`train.py`)

| Setting | Value |
| --- | --- |
| Data | Tiny Shakespeare, 568,210 tokens: 90% train (511,389), 10% validation (56,821) |
| Batch | 32 sequences x 256 tokens |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 on weight matrices only |
| Learning rate | 1e-3 peak, 100 warmup steps, cosine decay to 1e-4 |
| Gradient clipping | 1.0 |
| Steps | 3,000 |
| Hardware | One Tesla T4 on Google Colab, about 0.18 s per step (about 10 minutes in total) |

Before the full run, the model was checked by overfitting a single batch (loss from 6.3 to 0.002) and by confirming that the starting loss is close to ln(512) = 6.24.

The checkpoint in this repository comes from the notebook run on Colab. `train.py` is the script version of the same training loop.

## 4. Results

![Loss curves](loss_curve.png)

| Step | Train loss | Validation loss |
| --- | --- | --- |
| 0 | 6.26 | 6.26 |
| 250 | 3.36 | 3.69 |
| 500 | 2.64 | 3.20 |
| **750** | 2.18 | **3.08** |
| 1000 | 1.68 | 3.36 |
| 1500 | 0.63 | 4.40 |
| 2000 | 0.25 | 5.35 |
| 2999 | 0.12 | 6.09 |

**Overfitting.** Validation loss is lowest at step 750 (3.085) and then rises while the training loss keeps falling to 0.12. The model has 9.7M parameters but sees only about 511k training tokens, and 3,000 steps are about 48 passes over the data, so after roughly 12 passes it starts memorizing. The checkpoint with the best validation loss is saved during training and used for all generated text. Possible fixes are dropout, a smaller model, or more data.

## 5. Generated text (`generate.py`)

Generated from `gpt_gqa_best.pt` on a CPU with temperature 0.8, top-k 40 and seed 1337. Full output is in `samples.txt`.

```
PASTE THE FIRST 8 TO 10 LINES OF samples.txt HERE
```

The model has clearly learned the format of a play: capitalised speaker names, colons, short speeches and blank lines between them. The text often drifts in meaning and contains invented words, because the model is small, the vocabulary is only 512 tokens and the dataset is about 1 MB.

The same seed gives different text on a CPU and on a GPU, so `samples.txt` was made on a CPU to be reproducible.

## How to run

```bash
pip install -r requirements.txt

python model.py          # self-test of the model
python bpe.py            # trains the tokenizer on input.txt and rewrites merges.pkl (same result every time)
python train.py          # trains the model; use a GPU (Google Colab T4), a CPU needs about 8 s per step
python generate.py       # generates text from gpt_gqa_best.pt
```

Useful options:

```bash
python train.py --max_steps 1000 --patience 3        # shorter run, stops when validation loss stops improving
python train.py --n_kv_head 6 --out mha.pt --plot mha.png   # plain multi-head attention, to compare with GQA
python generate.py --prompt "ROMEO:" --temperature 1.0 --out my_samples.txt
```

`train.py` overwrites `gpt_gqa_best.pt` and `loss_curve.png` unless `--out` and `--plot` are given. `input.txt` is included, or it can be downloaded from
`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`.

## Team workflow

Two contributors worked with GitHub flow: one feature branch per part, merged into `main` through pull requests that the other person reviewed.

| Branch | Contents |
| --- | --- |
| `feature/gqa-model` | `model.py` |
| `feature/bpe-tokenizer` | `bpe.py`, `merges.pkl` |
| `feature/training` | `train.py` |
| `feature/generation` | `generate.py`, checkpoint, loss plot, samples, notebook |

## Acknowledgements

Based on Andrej Karpathy's "Let's reproduce GPT-2 (124M)" and "Let's build the GPT Tokenizer" videos. Dataset: Tiny Shakespeare.
