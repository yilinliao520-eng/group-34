"""
FP16 LoRA fine-tune Qwen2.5-7B on WebShop success trajectories.
No bitsandbytes needed — 40GB GPU fits FP16 model + LoRA comfortably.
"""
import json, os, sys, torch, argparse
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


def load_trajectories(file_paths):
    trajectories = []
    for fp in file_paths:
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                if r.get("reward", 0) > 0.5:
                    trajectories.append(r)
    print(f"Loaded {len(trajectories)} success trajectories from {len(file_paths)} files")
    return trajectories


def format_trajectory(traj, tokenizer):
    PROMPT = (
        "You are a web shopping agent. Use SHORT keyword queries (3-6 words) for search. "
        "Available actions: search[keywords], click[ASIN], click[Buy Now], click[< Prev], click[Next >], "
        "click[Back to Search], click[option_value]. Format: Thought: <reasoning>\nAction: search[x] or click[x]"
    )

    messages = [{"role": "system", "content": PROMPT}]
    for h in traj.get("history", []):
        obs = h.get("observation", "")[:2000]
        thought = h.get("thought", "").strip()
        action = h.get("action", "").strip()
        messages.append({"role": "user", "content": f"Observation: {obs}"})
        messages.append({"role": "assistant", "content": f"Thought: {thought}\nAction: {action}"})

    return tokenizer.apply_chat_template(messages, tokenize=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/lora_adapters")
    parser.add_argument("--data_dir", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Load data
    files = [
        f"{args.data_dir}/baseline_50.jsonl",
        f"{args.data_dir}/baseline_150.jsonl",
        f"{args.data_dir}/keyword_batch1.jsonl",
        f"{args.data_dir}/keyword_batch2.jsonl",
    ]
    trajectories = load_trajectories(files)
    if len(trajectories) < 10:
        print("ERROR: Need at least 10 trajectories")
        sys.exit(1)

    # Load model FP16
    print("Loading model in FP16...")
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
    model.gradient_checkpointing_enable()

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format data
    texts = [format_trajectory(t, tokenizer) for t in trajectories]
    print(f"Formatted {len(texts)} training examples")
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=2048,
            return_tensors=None,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"Tokenized {len(dataset)} examples")

    # Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"Training: {args.num_epochs} epochs × {len(dataset)} examples, batch={args.batch_size}×4")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    trainer.train()
    print(f"End: {datetime.now().strftime('%H:%M:%S')}")

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
