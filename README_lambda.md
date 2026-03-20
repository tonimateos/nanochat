# Training Nanochat on Lambda GPU Cloud

This guide outlines the steps to deploy and train Nanochat to full GPT-2 capability using [Lambda GPU Cloud](https://lambda.ai/service/gpu-cloud). According to the official documentation, training a GPT-2 quality model takes approximately 2-3 hours on an 8xH100 GPU node.

## 1. Sign In & Setup

1. **Create an Account:** Go to [Lambda Cloud](https://cloud.lambdalabs.com/) and sign up or log in.
2. **Add SSH Key:** Before launching an instance, go to your account settings and add your public SSH key. This is required to access your instances securely.
   - If you don't have an SSH key, you can generate one on your local machine by opening a terminal and running:
     ```bash
     ssh-keygen -t ed25519 -C "your_email@example.com"
     ```
   - Press Enter to accept the default file location, and optionally enter a passphrase.
   - Then, display your public key to copy it:
     ```bash
     cat ~/.ssh/id_ed25519.pub
     ```
   - Copy the **entire output string** (including the `ssh-ed25519` at the beginning and your `email@example.com` at the end) and paste it into the Lambda Cloud SSH Keys section.
3. **Add Payment Method:** Ensure your billing information is up to date, as you cannot launch GPU instances without a valid payment method.

## 2. Launching the Instance

1. Navigate to the **Instances** dashboard and click **Launch Instance**.
2. **Select Instance Type:** 
   - For the fastest training (reference speedrun), select an **8x H100 (80GB)** instance (approx. $24/hr). 
   - Alternatively, an **8x A100 (80GB)** instance is also fully capable but will take slightly longer (around 4-5 hours). **Note:** Using an A100 will *not* result in a less capable model. The script is configured to use the exact same model depth, dataset scale, and sequence length regardless of hardware. The only difference is training wall-clock time and the precision used under the hood (A100 will use bfloat16 instead of the explicit FP8 requested by the H100 run, but this doesn't degrade capability).
   - **Out of Capacity?** If standard on-demand instances are unavailable, you can use a **1-click cluster**. Lambda's H100 SXM5 clusters are typically deployed as 8-GPU nodes (with +208 vCPUs, +1800 GiB RAM). At ~$2.70/GPU/hr, the total cost for the 8-GPU node is ~$21.60/hr. This is actually a very powerful and cost-effective alternative for the speedrun.
3. Select an Ubuntu image (e.g., Ubuntu 22.04 or 24.04 with standard ML drivers).
4. Select your SSH key and launch the instance.

> [!WARNING]  
> **💸 WHEN YOU START PAYING:** You start paying the hourly rate **the moment the instance status changes to "Booting" or "Running"**. The billing continues as long as the instance exists, even if you are not actively running code.

## 3. Environment Setup & Training

Once the instance is running, go to the **Instances** dashboard on Lambda Cloud. You will see a column named **IP** next to your instance. Copy that IPv4 address and use it to SSH into the machine:

```bash
ssh ubuntu@<PUBLIC_IP>
```

Clone the repository and start the speedrun:

```bash
# Clone your fork (or Karpathy's)
git clone https://github.com/tonimateos/nanochat.git
cd nanochat

# Follow the standard setup using `uv`
# (If uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh)
uv venv
source .venv/bin/activate
uv pip install -r pyproject.toml

# Start the training
# You have two good options to prevent the training from dying if your SSH disconnects:

# Option A: using `screen` (Best for active monitoring)
# Creates a virtual terminal you can detach from and reattach to later.
screen -S nanochat
bash runs/speedrun.sh
# To detach: press Ctrl+A, then D
# To reattach later: run `screen -r nanochat`

# Option B: using `nohup` (Best for fire-and-forget & log keeping)
# Runs the process in the background and saves all output to a file (training.log)
# so you can easily debug or grep the logs later.
nohup bash runs/speedrun.sh > training.log 2>&1 &
# To monitor the logs in real-time:
# tail -f training.log
```
*Note: The script will take ~2-3 hours to complete on an 8xH100 node.*
## 4. Accessing the Model After Training

Once training is complete, you can talk to your model using the web UI!

1. In the same terminal, ensure your virtual environment is active:
   ```bash
   source .venv/bin/activate
   ```
2. Start the web server:
   ```bash
   python -m scripts.chat_web
   ```
3. Open your local browser and navigate to:
   ```
   http://<PUBLIC_IP>:8000
   ```
### Opening Port 8000 (Firewall)
By default, Lambda Cloud only opens port 22 (SSH). To access the web UI, you must open port 8000. **You can do this at any time (even before launching the instance):**

1. Go to the **Firewall** page on the left sidebar of the Lambda Cloud console.
2. Select **Global rules** (applies to all instances) OR create a specific **Ruleset**.
3. Click **Edit rules** -> **Add rule**.
4. Set the following:
   - **Type:** Custom TCP
   - **Port range:** 8000
   - **Source:** 0.0.0.0/0 (or your specific IP address for better security)
5. Save the rule.

*(If you cannot change the firewall, you can securely tunnel it over SSH to your local machine: `ssh -N -f -L 8000:localhost:8000 ubuntu@<PUBLIC_IP>` and visit `http://localhost:8000`)*

## 5. Saving Your Work and Terminating

> [!CAUTION]  
> **🛑 WHEN YOU STOP PAYING:** You only stop paying when you completely **Terminate / Delete** the instance. Simply stopping the training script or closing SSH **does not** stop the billing. On Lambda, if you restart or stop the machine, you might still be billed for the storage or the reservation of the node. **To stop the meter, you must delete the instance.**

**Before you terminate:**
Since terminating destroys all data on the node, make sure to download any checkpoints or edited files you want to keep. By default, the `speedrun.sh` script produces exactly two files: one final base model and one final SFT chat model (each ~3GB in size). From your local machine, run:

```bash
# Download the final chat-tuned models
scp -r ubuntu@<PUBLIC_IP>:/home/ubuntu/nanochat/chatsft_checkpoints/ /path/to/local/save/dir/

# (Optional) Download the base pre-training models if you want them
scp -r ubuntu@<PUBLIC_IP>:/home/ubuntu/nanochat/base_checkpoints/ /path/to/local/save/dir/
```

After downloading your models, go to the Lambda Cloud Dashboard, select your instance, and click **Terminate**. Double-check that it disappears from your active instances list to ensure billing has stopped.

## 6. Other Sources for Checkpoints

If you prefer to download pre-trained checkpoints (base or chat-tuned) rather than training them yourself, here are your best options:

### A. Hugging Face Hub (Recommended)
The **Hugging Face Hub** is the primary source for external `nanochat` models. Look for repositories with `chatsft_checkpoints` and a matching `tokenizer.pkl`.

**Top Recommendation:**
- **[pankajmathur/nanochat-d34-finetuned](https://huggingface.co/pankajmathur/nanochat-d34-finetuned)**: A complete, high-quality set including finetuned d34 checkpoints and the correct matching tokenizer.

| Source | Link | Description |
|--------|------|-------------|
| **Recommended (d34 SFT)** | [pankajmathur/nanochat-d34-finetuned](https://huggingface.co/pankajmathur/nanochat-d34-finetuned) | **Complete set (Model + Meta + Tokenizer).** <br> High-quality chat-tuned model (65k vocab). |
| **Official/Students** | [nanochat-students](https://huggingface.co/nanochat-students) | The official community hub. Hosts `d20` series models. |
| **Karpathy** | [karpathy](https://huggingface.co/karpathy) | Andrej Karpathy's HF profile. Check for larger `d32/d34` models. |
| **Search All** | [Hugging Face Search](https://huggingface.co/models?search=nanochat) | Search for "nanochat" to see all community-uploaded models. |

### B. Standard OpenAI GPT-2 Models
The `scripts/base_eval.py` script supports evaluating standard GPT-2 models via the Hugging Face `transformers` library.

```bash
# Example: Evaluate GPT-2 124M
torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2
```

> [!NOTE]  
> Standard GPT-2 models use the original OpenAI architecture. While `nanochat` can evaluate them, they differ slightly from native `nanochat` models (e.g., `nanochat` uses softcapping and specific weight initialization).

### C. Requirements for Loading
To load a native `.pt` checkpoint into `nanochat`, you **must** have the following files in the same directory:
1.  **Model Weights:** `model_XXXXXX.pt`
2.  **Metadata:** `meta_XXXXXX.json` (Contains `model_config` like depth and width).
3.  **Tokenizer:** The `tokenizer/` directory must contain the `tokenizer.pkl` or `tokenizer.model` used for that run.

Place downloaded files into a subfolder of `base_checkpoints/` or `chatsft_checkpoints/` and load them via the `--model-tag` argument.

## 7. Showcasing Your Model on Hugging Face Spaces

You can easily host your trained model for free using Hugging Face (HF) Spaces and Gradio.

### Step 1: Upload your Checkpoint to HF
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Create a new **Model Repository** (e.g., `my-nanochat-gpt2`).
3. **Download and Upload the Files**: We recommend downloading a complete set from [pankajmathur/nanochat-d34-finetuned](https://huggingface.co/pankajmathur/nanochat-d34-finetuned/tree/main) and uploading them to the **root** of your repository:
   - `model_000700.pt` (The weights)
   - `meta_000700.json` (The metadata)
   - `tokenizer.pkl` (The matching tokenizer)
    - **Model Checkpoint:** e.g., `model_000700.pt` (the one with the highest step number).
    - **Metadata File:** e.g., `meta_000700.json` (crucial for loading the architecture config).
    - **Tokenizer File:** e.g., `tokenizer.pkl`.
4.  Ensure the file names you upload match the `filename` arguments in your `app.py` script.

### Step 2: Create a Hugging Face Space
1. On Hugging Face, click on your profile and select **New Space**.
2. Give it a name (e.g., `Nanochat-Demo`).
3. Select **Gradio** as the Space SDK.
4. Set the Hardware to a basic CPU or a free-tier GPU. The 1.5B GPT-2 model generates tokens fine on a CPU, but a T4 GPU (often free or very cheap on HF) makes it much faster.

### Step 3: Configure your Space
Clone your Space to your local machine (or add files directly via the HF web interface). You need the following files inside your Space:

1. **Your codebase:** Clone or copy the `nanochat` code into the Space repository.
2. **`requirements.txt`:** 
   ```text
   torch>=2.4.0
   huggingface-hub
   transformers
   tqdm
   tiktoken
   ```
3. **`app.py`:** Create this python script to serve the web UI using Gradio:
   ```python
   import gradio as gr
   import torch
   import os
   from huggingface_hub import hf_hub_download
   from nanochat.checkpoint_manager import load_model_direct
   from nanochat.engine import Engine

   # 1. Download model, metadata, and tokenizer from YOUR HF model repo
   # Update 'tonideville/my-nanochat-gpt2' with your actual repo ID 
   repo_id = "tonideville/my-nanochat-gpt2"
   checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model_000700.pt")
   _ = hf_hub_download(repo_id=repo_id, filename="meta_000700.json") # Ensure metadata is in same cache folder
   tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.pkl")
   
   # 2. Load the model and engine
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   # Point load_model_direct to the directory containing tokenizer.pkl
   model, tokenizer, _ = load_model_direct(checkpoint_path, device, phase="eval", tokenizer_dir=os.path.dirname(tokenizer_path))
   engine = Engine(model, tokenizer)

   def chat_fn(message, history):
       bos = tokenizer.get_bos_token_id()
       user_start = tokenizer.encode_special("<|user_start|>")
       user_end = tokenizer.encode_special("<|user_end|>")
       assistant_start = tokenizer.encode_special("<|assistant_start|>")
       assistant_end = tokenizer.encode_special("<|assistant_end|>")

       tokens = [bos]
       for user_msg, bot_msg in history:
           tokens.extend([user_start] + tokenizer.encode(user_msg) + [user_end])
           tokens.extend([assistant_start] + tokenizer.encode(bot_msg) + [assistant_end])
       
       tokens.extend([user_start] + tokenizer.encode(message) + [user_end, assistant_start])

       response_tokens = []
       for token_col, _ in engine.generate(tokens, num_samples=1, max_tokens=512, temperature=0.8, top_k=50):
           token = token_col[0]
           if token == assistant_end or token == bos:
               break
           response_tokens.append(token)
           yield tokenizer.decode(response_tokens)

   demo = gr.ChatInterface(chat_fn, title="Nanochat GPT-2 Demo")
   demo.launch()
   ```

Once you commit and push these files to your Space on Hugging Face, it will automatically install the requirements and launch the Gradio web UI publicly for anyone to use!

## 8. Troubleshooting

### Gibberish or Nonsense Output
If your model produces gibberish when you talk to it:
1.  **Check your Checkpoint:** Ensure you uploaded a checkpoint from the `chatsft_checkpoints/` folder. If you accidentally uploaded a checkpoint from `base_checkpoints/`, your model is a "Base Model" (pretraining only) and doesn't understand the special chat tokens.
2.  **Tokenizer Dir:** Make sure `load_model_direct` is correctly pointed to the directory containing your `tokenizer.pkl`. If it uses the wrong tokenizer, the token IDs will be mismatched.
3.  **Special Tokens:** Check your `app.py` uses the same special tokens (e.g., `<|user_start|>`) that your tokenizer was trained with.
