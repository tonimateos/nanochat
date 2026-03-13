"""
Script to inspect SFT (Supervised Fine-Tuning) data.
Visualizes how conversations are packed into training batches and which tokens are masked.
"""
import os
import argparse
import torch
from nanochat.common import get_base_dir, print_banner
from nanochat.tokenizer import HuggingFaceTokenizer
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# Color codes for visualization
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"  # Assistant responses (trained on)
RED = "\033[31m"    # Prompts / Instructions (masked out)
BLUE = "\033[34m"   # Special tokens / BOS
YELLOW = "\033[33m" # Metadata / Padding

def get_sft_mixture(args, base_dir):
    identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
    
    train_tasks = []
    
    # Try loading SmolTalk if --quick is NOT used
    if not args.quick:
        try:
            print("Loading SmolTalk...")
            train_tasks.append(SmolTalk(split="train"))
        except Exception as e:
            print(f"Warning: Could not load SmolTalk: {e}")
            print("Proceeding without SmolTalk.")
    
    # Always try to load identity conversations if they exist
    if os.path.exists(identity_conversations_filepath):
        print(f"Loading Custom JSON from {identity_conversations_filepath}...")
        train_tasks.append(CustomJSON(filepath=identity_conversations_filepath))
    else:
        print(f"Warning: Identity conversations not found at {identity_conversations_filepath}")

    if not args.quick and not train_tasks:
        # Fallback if everything else fails for some reason
        try:
            train_tasks.extend([
                MMLU(subset="auxiliary_train", split="train"),
                GSM8K(subset="main", split="train"),
            ])
        except Exception as e:
            print(f"Warning: Could not load extra tasks: {e}")
        
    if not train_tasks:
        raise RuntimeError("No tasks could be loaded. Please ensure you have data or internet access.")
        
    return TaskMixture(train_tasks)

def inspect_batch(inputs, targets, tokenizer):
    B, T = inputs.shape
    bos_token = tokenizer.get_bos_token_id()
    
    for b in range(B):
        print(f"\n{BOLD}Row {b+1}:{RESET}")
        line_out = ""
        for t in range(T):
            token_id = inputs[b, t].item()
            target_id = targets[b, t].item()
            token_str = tokenizer.decode([token_id])
            
            # Escape newlines for compact view
            token_str = token_str.replace("\n", "\\n")
            
            if token_id == bos_token:
                line_out += f"{BLUE}[BOS]{RESET}"
            elif target_id == -1:
                # Masked out (usually user prompt or padding)
                if token_id == bos_token and t > 0:
                    line_out += f"{YELLOW}[PAD]{RESET}"
                else:
                    line_out += f"{RED}{token_str}{RESET}"
            else:
                # Not masked (usually assistant response)
                line_out += f"{GREEN}{token_str}{RESET}"
        
        print(line_out)
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Inspect SFT data batches")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of batches to inspect")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length for visualization")
    parser.add_argument("--quick", action="store_true", help="Only load SmolTalk for faster startup")
    args = parser.parse_args()

    print_banner()
    print(f"{BOLD}SFT Data Inspector{RESET}")
    print(f"Legend: {RED}Prompt (Masked){RESET} | {GREEN}Response (Trained){RESET} | {BLUE}[BOS]{RESET} | {YELLOW}[PAD]{RESET}")
    print("=" * 80)

    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    base_dir = get_base_dir()
    
    print("Loading data mixture...")
    dataset = get_sft_mixture(args, base_dir)
    print(f"Mixture loaded. Total rows: {len(dataset):,}")

    # Replicate the data generator logic from chat_sft.py
    def get_batch():
        row_capacity = args.max_seq_len + 1
        bos_token = tokenizer.get_bos_token_id()
        conv_buffer = []
        cursor = 0
        
        while True:
            rows = []
            mask_rows = []
            row_lengths = []
            
            for _ in range(args.batch_size):
                row = []
                mask_row = []
                padded = False
                while len(row) < row_capacity:
                    # Fetch more if buffer low
                    while len(conv_buffer) < 20: 
                        conversation = dataset[cursor % len(dataset)]
                        # Use the actual project's render_conversation logic
                        ids, mask = tokenizer.render_conversation(conversation, max_tokens=args.max_seq_len)
                        conv_buffer.append((ids, mask))
                        cursor += 1
                    
                    remaining = row_capacity - len(row)
                    # Simple greedy fit for inspection script
                    best_idx = -1
                    for i, (conv, _) in enumerate(conv_buffer):
                        if len(conv) <= remaining:
                            best_idx = i
                            break
                            
                    if best_idx >= 0:
                        conv, conv_mask = conv_buffer.pop(best_idx)
                        row.extend(conv)
                        mask_row.extend(conv_mask)
                    else:
                        content_len = len(row)
                        row.extend([bos_token] * remaining)
                        mask_row.extend([0] * remaining)
                        padded = True
                        break
                
                row_lengths.append(content_len if padded else row_capacity)
                rows.append(row[:row_capacity])
                mask_rows.append(mask_row[:row_capacity])
            
            batch_tensor = torch.tensor(rows, dtype=torch.long)
            inputs = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:].clone()
            
            mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
            mask_targets = mask_tensor[:, 1:]
            targets[mask_targets == 0] = -1
            
            for i, content_len in enumerate(row_lengths):
                if content_len < row_capacity:
                    targets[i, content_len-1:] = -1
            
            yield inputs, targets

    loader = get_batch()
    for i in range(args.num_batches):
        print(f"\n{BOLD}BATCH {i+1}{RESET}")
        inputs, targets = next(loader)
        inspect_batch(inputs, targets, tokenizer)

if __name__ == "__main__":
    main()
