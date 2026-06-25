"""
Unified SFT: Train on cleaned ReAct-format data.
"""
import json, os, torch, argparse, random
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

DATA_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/data/unified_sft_data.jsonl"


def load_data(filepath, max_samples=None):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    if max_samples:
        data = data[:max_samples]
    print(f"Loaded {len(data)} examples")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/unified_sft_adapters")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--num_epochs", type=int, default=2)
    args = parser.parse_args()

    data = load_data(DATA_PATH, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    # Apply chat template
    texts = [tokenizer.apply_chat_template(item["messages"], tokenize=False) for item in data]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1536)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    print(f"Tokenized {len(dataset)} examples")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=20,
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
