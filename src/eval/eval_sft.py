"""Evaluate fine-tuned LoRA model on test goals."""
import json, sys, os, time, re, torch

sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/src/agent")
WEBSHOP_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop"
sys.path.insert(0, WEBSHOP_PATH)

import gym
import web_agent_site.envs.web_agent_text_env
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from react_agent import SEARCH_PATTERN, CLICK_PATTERN

MODEL_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/lora_adapters"
NUM_GOALS = 30
START_IDX = 280

KEYWORD_PROMPT = (
    "You are a web shopping agent. Use SHORT keyword queries (3-6 words) for search. "
    "Available actions: search[keywords], click[ASIN], click[Buy Now], click[< Prev], "
    "click[Next >], click[Back to Search], click[option_value]. "
    "Format: Thought: <reasoning>\nAction: search[x] or click[x]"
)


def build_prompt(instruction, obs, clickables, history):
    acts_str = ", ".join(clickables[:30])
    parts = [KEYWORD_PROMPT]
    for h in history[-2:]:
        parts.append(f"Instruction: {h['instruction']}")
        parts.append(f"Observation: {h['observation'][:1500]}")
        parts.append(f"Thought: {h['thought']}")
        parts.append(f"Action: {h['action']}")
    parts.append(f"Instruction: {instruction}")
    parts.append(f"Observation: {obs[:2000]}")
    parts.append(f"Available actions: {acts_str}")
    parts.append("Thought:")
    return "\n".join(parts)


def act(tokenizer, model, instruction, obs, clickables, history):
    prompt = build_prompt(instruction, obs, clickables, history)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=128, temperature=0.0,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    thought = ""
    tm = re.search(r"Thought:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if tm:
        thought = tm.group(1).strip()

    sm = SEARCH_PATTERN.search(response)
    cm = CLICK_PATTERN.search(response)
    if sm:
        action = f"search[{sm.group(1).strip()}]"
    elif cm:
        action = f"click[{cm.group(1).strip()}]"
    else:
        action = "search[ ]"
    return thought, action


def main():
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()
    model.eval()
    print("Model loaded and LoRA merged.")

    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=1000)
    goals = env.server.goals

    results = []
    for i in range(START_IDX, START_IDX + NUM_GOALS):
        goal = goals[i]
        instruction = goal["instruction_text"]
        obs, _ = env.reset(session=i)
        available = env.get_available_actions()
        clickables = available.get("clickables", [])
        history = []
        total_reward = 0
        start_t = time.time()

        for step in range(7):
            thought, action = act(tokenizer, model, instruction, obs, clickables, history)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            history.append({
                "instruction": instruction, "observation": obs,
                "thought": thought, "action": action,
            })
            available = env.get_available_actions()
            clickables = available.get("clickables", [])
            if done:
                break

        elapsed = time.time() - start_t
        success = total_reward > 0.5
        results.append({
            "goal": i, "reward": total_reward, "steps": len(history),
            "success": success, "time": elapsed,
        })
        print(f"[{i}] r={total_reward:.3f} s={len(history)} ok={success} "
              f"t={elapsed:.1f}s | {instruction[:50]}...")

    n = len(results)
    success_n = sum(1 for r in results if r["success"])
    avg_r = sum(r["reward"] for r in results) / n
    avg_s = sum(r["steps"] for r in results) / n
    print(f"\n=== SFT Eval ({n} goals) ===")
    print(f"Success: {success_n}/{n} = {success_n/n*100:.1f}%")
    print(f"Avg reward: {avg_r:.3f}")
    print(f"Avg steps: {avg_s:.1f}")

    out_path = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results/sft_eval.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
