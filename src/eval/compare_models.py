"""
Quick comparison: baseline vs search_query_SFT on search query quality.
Tests 3 models on the same 20 goals:
  - baseline: zero-shot Qwen2.5-7B
  - search_sft: Qwen + search query LoRA
  - expert_sft: Qwen + expert trajectory LoRA
"""
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
ADAPTERS = {
    "baseline": None,
    "expert_sft": "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/expert_sft_adapters",
    "unified_sft": "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/unified_sft_adapters",
}

KEYWORD_PROMPT = (
    "You are a web shopping agent. Use SHORT keyword queries (3-6 words) for search. "
    "Available actions: search[keywords], click[ASIN], click[Buy Now], click[< Prev], "
    "click[Next >], click[Back to Search], click[option_value]. "
    "Format: Thought: <reasoning>\nAction: search[x] or click[x]"
)

NUM_GOALS = 50
START_IDX = 0


def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )
    adapter_path = ADAPTERS[name]
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return tokenizer, model


def act(tokenizer, model, instruction, obs, clickables, history):
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
    prompt = "\n".join(parts)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.0, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    thought, action = "", "search[ ]"
    tm = re.search(r"Thought:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if tm: thought = tm.group(1).strip()
    sm = SEARCH_PATTERN.search(response)
    cm = CLICK_PATTERN.search(response)
    if sm: action = f"search[{sm.group(1).strip()}]"
    elif cm: action = f"click[{cm.group(1).strip()}]"
    return thought, action


def eval_model(name, tokenizer, model, env, goals):
    results = []
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    for i in range(START_IDX, START_IDX + NUM_GOALS):
        goal = goals[i]
        instruction = goal["instruction_text"]
        obs, _ = env.reset(session=i)
        available = env.get_available_actions()
        clickables = available.get("clickables", [])
        history = []
        total_reward = 0
        first_query = ""
        for step in range(7):
            thought, action = act(tokenizer, model, instruction, obs, clickables, history)
            if step == 0 and action.startswith("search["):
                first_query = action[7:-1]
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            history.append({"instruction": instruction, "observation": obs, "thought": thought, "action": action})
            available = env.get_available_actions()
            clickables = available.get("clickables", [])
            if done: break
        success = total_reward > 0.5
        results.append({
            "goal": i, "reward": total_reward, "steps": len(history),
            "success": success, "first_query": first_query,
            "instruction": instruction[:80],
        })
        print(f"[{i}] r={total_reward:.3f} s={len(history)} ok={success} q='{first_query[:60]}'")
    n = len(results)
    success_n = sum(1 for r in results if r["success"])
    avg_r = sum(r["reward"] for r in results) / n
    avg_query_len = sum(len(r["first_query"].split()) for r in results) / n
    print(f"\n{name}: success={success_n}/{n}={success_n/n*100:.0f}% avg_r={avg_r:.3f} avg_ql={avg_query_len:.1f} words")
    return results


def main():
    print("Loading environment...")
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=None)
    goals = env.server.goals
    print(f"{len(goals)} goals available")

    all_results = {}
    for name in ["baseline", "expert_sft", "unified_sft"]:
        print(f"\nLoading {name}...")
        tokenizer, model = load_model(name)
        print(f"Model {name} loaded.")
        results = eval_model(name, tokenizer, model, env, goals)
        all_results[name] = results
        del model, tokenizer
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY (20 goals each)")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Success':>8} {'Avg R':>8} {'Avg QLen':>10}")
    print("-" * 48)
    for name in ["baseline", "expert_sft", "unified_sft"]:
        r = all_results[name]
        n = len(r)
        sn = sum(1 for x in r if x["success"])
        ar = sum(x["reward"] for x in r) / n
        ql = sum(len(x["first_query"].split()) for x in r) / n
        print(f"{name:<20} {sn}/{n}={sn/n*100:>4.0f}% {ar:>8.3f} {ql:>8.1f} w")

    # Save
    out = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results/model_comparison.json"
    with open(out, "w") as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
