# Custom Debugging Scripts (Mac M2 8GB RAM)

This directory (and fork) is aimed at adding custom debugging and visualization scripts to `nanochat`. All experiments and configurations here are optimized for a **Mac M2 with 8GB of RAM**.

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

## Pretraining (Mac M2 Optimized)

To train a "tiny" model that fits within 8GB of RAM, use the following configuration:

```bash
source .venv/bin/activate
python -m scripts.base_train \
    --depth=2 \
    --max-seq-len=256 \
    --device-batch-size=1 \
    --total-batch-size=256 \
    --window-pattern=L \
    --num-iterations=100 \
    --core-metric-every=-1 \
    --sample-every=20 \
    --run=my-first-run
```

---

## Tokenizer Internals
