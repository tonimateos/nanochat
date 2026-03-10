# Custom Debugging Scripts

This directory (and fork) is aimed at adding custom debugging and visualization scripts to `nanochat`.

## Available Scripts

### 1. Token Inspection (`inspect_tokens.py`)
Visualizes the trained vocabulary, showing special tokens and samples of merged tokens.

**How to run:**
From the root directory, run:
```bash
source .venv/bin/activate
python -m scripts.custom.inspect_tokens
```

**Output:**
- Total vocabulary size.
- List of special tokens and their IDs (e.g., `<|bos|>`, `<|user_start|>`).
- Samples of "merges" (common character combinations).

---

## Tokenizer Internals

When you train a tokenizer using `scripts.tok_train`, it generates two key files in `~/.cache/nanochat/tokenizer/`:

### 1. `tokenizer.pkl` (The Dictionary)
This is a pickled `tiktoken` encoding object. It contains the mapping between text strings and their integer IDs. It is used during both training (to turn text into tensors) and inference (to turn the model's numbers back into readable text).

### 2. `token_bytes.pt` (The Metric Helper)
This is a PyTorch tensor that stores the raw byte length of every token in the vocabulary. Nanochat uses this to calculate **"Bits Per Byte" (bpb)**, a metric that allows you to compare models fairly even if they use different tokenizers or vocabulary sizes.
