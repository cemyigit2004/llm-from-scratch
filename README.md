# LLM From Scratch

A decoder-only Transformer language model implemented and pretrained from scratch using PyTorch — a GPT-style architecture built and trained end-to-end: data cleaning, tokenization, model, training, validation, checkpointing and text generation.

## Architecture

- Token + learned positional embeddings (`d_model = 384`)
- 6 stacked Transformer blocks (`n_layers = 6`), each with:
  - Pre-Norm `LayerNorm` → Multi-Head Causal Self-Attention (`n_heads = 6`) → residual connection
  - Pre-Norm `LayerNorm` → MLP (`384 → 1536 → 384`, GELU activation) → residual connection
- Causal masking (`torch.tril`) so a token can only attend to itself and earlier tokens
- Final `LayerNorm` + `lm_head` linear projection to vocabulary logits (`vocab_size = 50257`)
- Cross-entropy loss computed against next-token targets when training

## Dataset pipeline

```
data/raw/*.jsonl
      ↓  prepare_data.py (cleaning, dedup)
data/cleaned/data.jsonl
      ↓  LMDataset (dataset.py)
token stream, split into fixed context_length chunks
      ↓
(x, y) pairs — y is x shifted by one token (next-token prediction)
```

`train.py` splits the dataset into train/validation (90/10) using a seeded generator (`manual_seed(42)`) so the split is deterministic across runs.

## GPT-2 tokenizer

Tokenization uses the pretrained GPT-2 BPE tokenizer (`openai-community/gpt2`, via Hugging Face `transformers`) rather than training a tokenizer from scratch.

## Training

`train.py` runs the full training loop:

- Forward pass → cross-entropy loss
- Backpropagation
- Gradient clipping (`clip_grad_norm_`, `max_norm=1.0`)
- AdamW optimizer
- Cosine annealing learning rate scheduler (`CosineAnnealingLR`)

## Validation & perplexity

After each epoch, the model is evaluated on the held-out validation set (`model.eval()`, `torch.no_grad()`) and reports validation loss and perplexity (`exp(val_loss)`) alongside the training loss and current learning rate.

## Checkpoints

At the end of every epoch, `train.py` saves model, optimizer and scheduler state (plus the epoch number and losses) to `checkpoints/latest.pt`. If that file exists when training starts, it's loaded automatically and training resumes from the next epoch — so an interrupted run can continue without losing optimizer/scheduler state. Checkpoint files are git-ignored.

## Text generation

`generate.py` loads a trained checkpoint and autoregressively generates text from a prompt:

```
prompt
  ↓ tokenize
  ↓ model forward (per step)
  ↓ temperature scaling
  ↓ Top-K filtering
  ↓ Top-P (nucleus) filtering
  ↓ multinomial sampling
  ↓ EOS check (stop early if hit)
  ↓ append token, repeat
  ↓ decode
generated text
```

- **Temperature** — scales logits before softmax to control how sharp/flat the probability distribution is.
- **Top-K** — keeps only the K highest-logit candidates before sampling.
- **Top-P (nucleus)** — keeps the smallest set of candidates whose cumulative probability exceeds `top_p`.
- **EOS stopping** — generation stops as soon as the tokenizer's end-of-text token is sampled.

## How to run

```bash
# 1. Clean the raw dataset
python prepare_data.py

# 2. Sanity-check the dataset pipeline
python test_dataset.py

# 3. Sanity-check the model's forward pass
python test_model.py

# 4. Train (saves checkpoints/latest.pt each epoch; re-run to resume)
python train.py

# 5. Generate text from the trained checkpoint
python generate.py
```

## Hardware

Development environment:

- PyTorch (CUDA build)
- NVIDIA GPU

Model size (`d_model`, `n_layers`, `context_length`, batch size) can be adjusted in `GPTConfig` depending on available GPU memory.

## Project structure

```text
.
├── data/
│   ├── raw/
│   └── cleaned/
├── checkpoints/            # gitignored — created by train.py
├── model.py                 # GPTConfig, CausalSelfAttention, MLP, TransformerBlock, GPT
├── dataset.py                # LMDataset — tokenizes and chunks text into (x, y) pairs
├── prepare_data.py           # cleans/dedups raw data into data/cleaned/data.jsonl
├── train.py                  # training + validation + checkpoint save/resume
├── generate.py                # autoregressive text generation from a checkpoint
├── test_tokenizer.py          # GPT-2 tokenizer sanity check
├── test_dataset.py            # LMDataset / DataLoader sanity check
├── test_model.py              # GPT forward pass sanity check
├── gpu_test.py                # CUDA availability check
├── requirements.txt
└── README.md
```

## Status

The core pipeline is complete and verified end-to-end: data cleaning → tokenization → dataset → model → training → validation → checkpointing → text generation all run successfully together.

Possible next steps (not required for v1): larger real pretraining corpus, step-based LR scheduling with warmup, and instruction fine-tuning (SFT) on top of the pretrained base model.
