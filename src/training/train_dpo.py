"""
DPO fine-tuning: train Qwen2.5-7B to prefer good search queries over bad ones.
Uses paired (chosen, rejected) search actions from WebShop trajectories.
"""
import json, os, sys, torch, argparse, random
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer


def load_all_trajectories(file_paths):
    """Load ALL trajectories (both success and failure)."""
    all_trajs = []
    for fp in file_paths:
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                if "history" in r and len(r["history"]) > 0:
                    all_trajs.append(r)
    return all_trajs


def build_dpo_pairs(trajectories):
    """Build DPO pairs: (instruction, chosen_search, rejected_search)."""
    # Group trajectories by similar instructions to find natural pairs
    good_first_searches = []  # reward > 0.7, first search was effective
    bad_first_searches = []   # reward = 0, first search failed

    for traj in trajectories:
        if len(traj.get("history", [])) == 0:
            continue
        first_action = traj["history"][0].get("action", "")
        if not first_action.startswith("search["):
            continue

        instruction = traj["instruction"]
        query = first_action[7:-1]  # extract keywords from search[...]
        reward = traj.get("reward", 0)

        entry = {
            "instruction": instruction,
            "query": query,
            "reward": reward,
        }

        if reward >= 0.7:
            good_first_searches.append(entry)
        elif reward == 0:
            bad_first_searches.append(entry)

    # Build pairs: match good searches with bad searches for similar instructions
    pairs = []
    random.shuffle(good_first_searches)

    for good in good_first_searches:
        # Find a bad search with similar instruction length (proxy for similar complexity)
        candidates = [b for b in bad_first_searches
                      if abs(len(b["instruction"].split()) - len(good["instruction"].split())) < 5]
        if not candidates:
            candidates = bad_first_searches

        bad = random.choice(candidates)

        PROMPT = (
            "You are a web shopping agent. Use SHORT keyword queries (3-6 words) for search.\n\n"
            f"Instruction: {good['instruction']}\n\n"
            "You are on the search page. What search query should you use?\n"
            "Format: search[keywords]"
        )

        pairs.append({
            "prompt": PROMPT,
            "chosen": f"search[{good['query']}]",
            "rejected": f"search[{bad['query']}]",
        })

        if len(pairs) >= 200:
            break

    print(f"Built {len(pairs)} DPO pairs ({len(good_first_searches)} good, {len(bad_first_searches)} bad searches)")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/dpo_adapters")
    parser.add_argument("--data_dir", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results")
    parser.add_argument("--num_epochs", type=int, default=2)
    args = parser.parse_args()

    # Load all trajectories
    files = [
        f"{args.data_dir}/baseline_50.jsonl",
        f"{args.data_dir}/baseline_150.jsonl",
        f"{args.data_dir}/keyword_batch1.jsonl",
        f"{args.data_dir}/keyword_batch2.jsonl",
    ]
    trajectories = load_all_trajectories(files)
    print(f"Loaded {len(trajectories)} trajectories")

    # Build DPO pairs
    pairs = build_dpo_pairs(trajectories)
    dataset = Dataset.from_list(pairs)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Load model FP16
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # DPO training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
    )

    print(f"Training DPO: {len(dataset['train'])} pairs × {args.num_epochs} epochs")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    dpo_trainer.train()
    print(f"End: {datetime.now().strftime('%H:%M:%S')}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"DPO adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
