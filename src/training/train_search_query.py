"""
Step 1: Train search query generation LoRA.
Uses (instruction → search query) pairs from goal_query_predict.json.
Small sample first (500 pairs) to validate the approach.
"""
import json, os, torch, argparse, random
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

DATA_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop/baseline_models/data/goal_query_predict.json"


def load_query_pairs(filepath, max_samples=500):
    with open(filepath) as f:
        data = json.load(f)
    pairs = []
    for instruction, queries in data.items():
        if queries and len(queries) > 0:
            pairs.append({"instruction": instruction, "query": queries[0]})
    if max_samples and len(pairs) > max_samples:
        pairs = random.Random(42).sample(pairs, max_samples)
    print(f"Loaded {len(pairs)} (instruction, query) pairs")
    return pairs


def format_for_training(pairs, tokenizer):
    SYSTEM = (
        "You are a search query generator for an e-commerce website. "
        "Given a shopping instruction, generate 3-6 concise keywords as a search query."
    )
    texts = []
    for p in pairs:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": p["instruction"]},
            {"role": "assistant", "content": f"search[{p['query']}]"},
        ]
        texts.append(tokenizer.apply_chat_template(messages, tokenize=False))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/search_query_adapters")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=3)
    args = parser.parse_args()

    # Load data
    pairs = load_query_pairs(DATA_PATH, args.max_samples)

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Format and tokenize
    texts = format_for_training(pairs, tokenizer)
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"Tokenized {len(dataset)} examples")

    # Train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    from transformers import DataCollatorForLanguageModeling
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print(f"Training: {args.num_epochs} epochs, {len(dataset)} examples")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    trainer.train()
    print(f"End: {datetime.now().strftime('%H:%M:%S')}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
