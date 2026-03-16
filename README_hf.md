# Training Nanochat on Hugging Face Spaces

This guide details how to train the `nanochat` model from scratch using Hugging Face infrastructure. By utilizing **Hugging Face Spaces** configured with a custom Docker environment, you can access high-end GPUs (e.g., A10G, A100) exactly when you need them.

## Requirements

- **Hugging Face Account**: You need an account on huggingface.co.
- **Access Token**: An `HF_TOKEN` with **write** permissions to your account.
- **Model Repository**: A created Model repository on Hugging Face where your checkpoints will be stored (e.g., `your-username/nanochat-checkpoints`).

## Approach Overview

1. **High-End GPUs**: We use Hugging Face Docker Spaces which can be equipped with powerful GPUs and charge per minute.
2. **Full Pipeline From Scratch**: The Space is configured to run `scripts/hf_full_run.sh`, which downloads the dataset, trains the tokenizer, pre-trains a base model from scratch, and finally performs Supervised Fine-Tuning (SFT).
3. **Continuous Checkpointing**: Once SFT finishes, the script will automatically upload the final model checkpoint `.pt` files to your HF Model repository.
4. **Auto-Pause (Cost Savings)**: To stop paying immediately after the uploaded finishes (or if it crashes), the training script will use the `huggingface_hub` API to programmatically pause the Space.
5. **Live Debugging**: You can monitor training progress (loss, tokens/sec) in real-time through the Space's "Logs" tab.

## Step-by-Step Setup

### 1. Create a Checkpoint Repository

You will need a dedicated **Model Repository** on Hugging Face to store your model weights (checkpoints). This ensures your trained models are saved persistently, even if the Space where training occurs is paused or deleted.

1.  Go to [huggingface.co/new](https://huggingface.co/new).
2.  Set the **Owner** to your username
3.  Set the **Model name** to something descriptive (e.g., `nanochat-checkpoints`).
4.  Choose the **Visibility** (Public or Private). A Private repository is recommended if you don't want others to access your intermediate training checkpoints.
5.  Click **Create model**.

You will use the identifier `your-username/your-model-name` in your training script to specify where to upload the files.

### 2. Configure the Training Pipeline

The Dockerfile is configured to run `bash scripts/hf_full_run.sh`. This script will orchestrate downloading the data, pre-training the base model, and running the Supervised Fine-Tuning.

The final fine-tuning step has been updated to handle Hugging Face checkpoint uploads and auto-shutdown. It detects two optional environment variables to enable these features:

- `HF_REPO`: The identifier for your checkpoint repository (e.g., `your-username/nanochat-checkpoints`). If provided, the script will automatically upload `.pt` and `.json` checkpoint files here at the end of training.
- `HF_SPACE`: The identifier for your running Space (e.g., `your-username/nanochat-training-space`). If provided, the script will automatically pause the Space when the pipeline completely finishes or crashes.

**Authentication:** Both of these features require an `HF_TOKEN` environment variable with **write** permissions to your account in order to authenticate the uploads and pause actions.

### 3. Create a Custom Docker Space and Push Code

Hugging Face Spaces allow you to run any Docker container. 

1. Go to Hugging Face and create a new **Space**.
2. Select **Docker** as the SDK.
3. Choose the **Blank** template.
4. For **Space Hardware**, choose **CPU Basic (Free)**. You don't want to pay for an expensive GPU while you are still pushing your files and configuring your secrets. We will upgrade the hardware in the next section.
5. Clone the newly created Space repository to your local machine (Hugging Face will provide the `git clone` command on the empty Space page).
6. Copy your `nanochat` codebase into that cloned folder, making sure the custom `Dockerfile` we created is at the root. 
   - **Important:** Do NOT copy the `.git`, `wandb`, `runs`, or `.venv` folders, as these contain your local environment, git history, and large local logs which will severely slow down the Docker build and waste Space storage.
   - **Important:** Do NOT copy your local `README.md` over the one Hugging Face created for you. The Hugging Face `README.md` contains a special YAML configuration block at the very top (e.g. `title:`, `sdk: docker`) that is required for the Space to run!
7. Commit and push the files back to Hugging Face:
   ```bash
   git add .
   git commit -m "Initialize training space"
   git push
   ```

### 4. Deploy and Configure the Space

1. **Provide your Environment Variables**: The Space needs write access to your account and needs to know where to upload to. Scroll down to **Variables and secrets**.
    - Click **New variable**, name it `HF_REPO` and set the value to your model repository (e.g., `tonideville/nanochat-checkpoints`).
    - Click **New variable**, name it `HF_SPACE` and set the value to your space name (e.g., `tonideville/nanochat-training-space`).
2. **Provide the `HF_TOKEN`**: The Space needs write access to your account to upload checkpoints and auto-pause itself. 
    1. Go to your Hugging Face [Access Tokens page](https://huggingface.co/settings/tokens).
    2. Click **Create new token**.
    3. Name it (e.g., "Nanochat-Training"), select **Write** for the token type, and click Create.
    4. Copy the generated token.
    5. Back in your Space's **Settings** tab, scroll down to **Variables and secrets** and click **New secret**.
    6. Name the secret exactly `HF_TOKEN` and paste your copied token as the value.
3. **Assign Hardware**: Go to the **Settings** tab of your new Space and select your preferred GPU hardware (e.g., A10G Large, L4, or A100). **Note: You will start being billed per minute as soon as you assign the paid hardware and the Space starts.**
4. **Start Training**: Once the hardware is assigned, the Space will rebuild automatically. When the Docker image builds and starts running, your `CMD` will execute and training will begin!

### 5. Debugging and Accessing Checkpoints

- **Live Logs**: Click the **Logs** tab in your Space UI to stream `stdout` and `stderr` exactly like a local terminal. This is where you monitor your loss and training speed.
- **Accessing Weights**: Visit your `nanochat-checkpoints` Model repository on the Hub at any time to see and download the `.pt` files uploaded by your script.

### 6. Using Your Trained Model

Once the space auto-pauses, your model is fully trained and securely saved in your `nanochat-checkpoints` repository. The training Space turns off to save money. You have two primary ways to interact with your new AI:

#### Option A: Chat Locally (Free)
1. Download the `model_XXXXXX.pt` and `meta_XXXXXX.json` files from your `nanochat-checkpoints` repository.
2. Place these files on your local machine in the correct cache directory (e.g., `~/.cache/nanochat/chatsft_checkpoints/d24/`).
3. Chat via the **terminal**:
   ```bash
   python -m scripts.chat_cli -p "Why is the sky blue?"
   ```
   *(Leave out the `-p` string to start an interactive back-and-forth session)*
4. Chat via the **Web UI** (ChatGPT clone):
   ```bash
   python -m scripts.chat_web
   ```
   *(This hosts a local webpage at `http://127.0.0.1:8000/` you can open in your browser)*

#### Option B: Host it on Hugging Face (Public Web App)
If you want the model to be permanently hosted on the internet for others to use:
1. Create a **brand new**, separate Hugging Face Space.
2. Instead of `Docker`, select **Gradio** or **Streamlit** as the SDK.
3. Keep the hardware set to a cheap/free CPU or very small GPU.
4. Configure that new Space to download your `.pt` file from your checkpoints repository and run the UI!
