FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /app

# Copy the project files
COPY . .

# Install dependencies using uv
RUN pip install uv && uv pip install --system -r pyproject.toml

# We require HF_TOKEN to be exposed as an environment variable to authenticate with Hugging Face Hub (both for checkpoints and pausing the Space)
# The user will set this up via the Hugging Face Space Settings UI

# The command to execute when the container starts
# The script will automatically pick up HF_REPO and HF_SPACE environment variables
CMD ["python", "scripts/chat_sft.py"]
