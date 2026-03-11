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
- Samples of "merges" (common character combinations).

### 2. Checkpoint Inspection (`inspect_checkpoint.py`)
Loads a saved checkpoint (weights, metadata, and optimizer state) and prints summary statistics to ensure training is healthy (no NaNs, weights are changing).

**How to run:**
```bash
python -m scripts.custom.inspect_checkpoint
```

### 3. Model Sampling (`sample.py` or `base_eval.py`)
Generates text from a saved base model checkpoint. Note that extremely small models or those early in training (like 100 steps) will likely produce gibberish.

**Using `base_eval.py` (standard):**
```bash
python -m scripts.base_eval --model-tag d6 --step 800 --eval sample
```

**Using `sample.py` (custom prompt/tokens):**
```bash
python -m scripts.custom.sample --prompt "The capital of France is" --num-tokens 50
```

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
    --run=my-first-run \
    --wandb-log-every=1
```

## Training Log Explanation

When running `base_train.py`, you will see lines like this in your console:
`step 00499/04000 (12.47%) | loss: 7.457520 | lrm: 1.00 | dt: 19.05ms | tok/sec: 13,435 | bf16_mfu: 0.00 | epoch: 1 pq: 0 rg: 1 | total time: 0.16m | eta: 1.1m`

- **step**: Current iteration / Total planned iterations (Percentage complete).
- **loss**: The "Cross-Entropy Loss". Measures how poorly the model predicted the next token. (10.4 is random; < 4.0 is getting coherent).
- **lrm**: Learning Rate Multiplier. Starts at 0 (warmup), goes to 1.0, then decays (warmdown) to 0.05 at the end.
- **dt**: Delta Time. Time taken for one full training step (forward + backward pass).
- **tok/sec**: Throughput. Number of text tokens processed per second. Higher is better!
- **bf16_mfu**: Model FLOPs Utilization. Efficiency of the GPU usage (Note: This is usually 0.00 on Mac/MPS as it's optimized for NVIDIA).
- **epoch**: How many times we've looped through the entire dataset.
- **pq / rg**: Internal data loader indices. Useful for technical debugging of data resumes.
- **total time**: Clock time since training started.
- **eta**: Estimated time remaining until the run finishes.

---
