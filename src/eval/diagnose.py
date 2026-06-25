"""
Diagnostic evaluation: collect full per-step trajectories for failure analysis.
Runs baseline + expert_sft + unified_sft on the same 50 goals.
Saves detailed per-episode data for classification.
"""
import json, os, sys, time, re, torch

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
OUTDIR = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results/full_env_audit"
NUM_GOALS = 50
START_IDX = 0
MAX_STEPS = 7


def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )
    adapter = ADAPTERS.get(name)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tokenizer, model


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


def generate_action(tokenizer, model, instruction, obs, clickables, history):
    prompt = build_prompt(instruction, obs, clickables, history)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    thought = ""
    tm = re.search(r"Thought:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if tm:
        thought = tm.group(1).strip()
    action = "search[ ]"
    sm = SEARCH_PATTERN.search(response)
    cm = CLICK_PATTERN.search(response)
    if sm:
        action = f"search[{sm.group(1).strip()}]"
    elif cm:
        action = f"click[{cm.group(1).strip()}]"
    return thought, action, response


def diagnose_episode(tokenizer, model, env, goal_idx, goal):
    """Run one episode and collect detailed diagnostics."""
    instruction = goal["instruction_text"]
    obs, _ = env.reset(session=goal_idx)
    available = env.get_available_actions()
    clickables = available.get("clickables", [])
    history = []
    total_reward = 0
    done = False

    steps = []
    searches = []
    zero_result_count = 0
    repeat_search_count = 0
    prev_query = ""

    for step in range(MAX_STEPS):
        thought, action, raw = generate_action(tokenizer, model, instruction, obs, clickables, history)

        # Classify action
        action_type = "unknown"
        action_arg = ""
        sm = SEARCH_PATTERN.search(action)
        cm = CLICK_PATTERN.search(action)
        if sm:
            action_type = "search"
            action_arg = sm.group(1).strip()
            searches.append(action_arg)
            if action_arg == prev_query:
                repeat_search_count += 1
            prev_query = action_arg
        elif cm:
            action_arg = cm.group(1).strip()
            if action_arg.lower() == "buy now":
                action_type = "buy"
            else:
                action_type = "click"

        # Execute
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # Check if search returned results (approximate)
        if action_type == "search" and "Page 1 (Total results: 0)" in obs:
            zero_result_count += 1

        steps.append({
            "step": step + 1,
            "action_type": action_type,
            "action": action,
            "thought": thought[:200],
            "reward": reward,
            "done": done,
        })

        available = env.get_available_actions()
        clickables = available.get("clickables", [])
        history.append({
            "instruction": instruction,
            "observation": obs,
            "thought": thought,
            "action": action,
        })
        if done:
            break

    # Build diagnosis
    query_words = [len(q.split()) for q in searches if q.strip()]
    avg_query_len = sum(query_words) / len(query_words) if query_words else 0
    first_query = searches[0] if searches else ""
    first_query_len = len(first_query.split()) if first_query.strip() else 0
    buy_occurred = any(s["action_type"] == "buy" for s in steps)

    # Failure classification
    failure_type = "none"
    if total_reward <= 0.5:
        if zero_result_count >= 2:
            failure_type = "zero_result_long_query"
        elif not buy_occurred and len(steps) >= MAX_STEPS:
            failure_type = "max_steps_no_buy"
        elif buy_occurred and total_reward <= 0.1:
            failure_type = "wrong_product_bought"
        elif buy_occurred and total_reward < 0.5:
            failure_type = "partial_match_premature_buy"
        else:
            failure_type = "other_failure"

    return {
        "goal_idx": goal_idx,
        "instruction": instruction,
        "total_reward": round(total_reward, 3),
        "success": total_reward > 0.5,
        "failure_type": failure_type,
        "num_steps": len(steps),
        "num_searches": len(searches),
        "zero_result_count": zero_result_count,
        "repeat_search_count": repeat_search_count,
        "avg_query_words": round(avg_query_len, 1),
        "first_query": first_query,
        "first_query_words": first_query_len,
        "buy_occurred": buy_occurred,
        "steps": steps,
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("Loading environment (full)...")
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
    goals = env.server.goals
    print(f"{len(goals)} goals loaded")

    all_results = {}

    for name in ["baseline", "expert_sft", "unified_sft"]:
        print(f"\n{'='*50}\n  Diagnostic: {name}\n{'='*50}")
        tokenizer, model = load_model(name)
        print(f"Model loaded: {name}")

        diag_results = []
        for i in range(START_IDX, START_IDX + NUM_GOALS):
            goal = goals[i]
            result = diagnose_episode(tokenizer, model, env, i, goal)
            diag_results.append(result)
            print(f"[{i}] r={result['total_reward']:.3f} fail={result['failure_type']} "
                  f"ql={result['first_query_words']}w zero={result['zero_result_count']} "
                  f"buy={result['buy_occurred']} | {result['instruction'][:50]}...")

        # Save per-model diagnostics
        outpath = f"{OUTDIR}/diagnostic_{name}_synthetic_keyword_50.jsonl"
        with open(outpath, "w") as f:
            for r in diag_results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(diag_results)} episodes to {outpath}")

        all_results[name] = diag_results

        del model, tokenizer
        torch.cuda.empty_cache()

    # Cross-model failure analysis
    print(f"\n{'='*60}\nFAILURE TYPE DISTRIBUTION\n{'='*60}")
    for name in ["baseline", "expert_sft", "unified_sft"]:
        diag = all_results[name]
        failure_types = {}
        for d in diag:
            ft = d["failure_type"]
            failure_types[ft] = failure_types.get(ft, 0) + 1
        print(f"\n{name}:")
        for ft, count in sorted(failure_types.items(), key=lambda x: -x[1]):
            print(f"  {ft}: {count}")

    # Save summary
    summary_path = f"{OUTDIR}/diagnostic_summary.json"
    with open(summary_path, "w") as f:
        json.dump({name: {
            "success_rate": sum(1 for d in diag if d["success"]) / len(diag),
            "avg_reward": sum(d["total_reward"] for d in diag) / len(diag),
            "avg_first_query_words": sum(d["first_query_words"] for d in diag) / len(diag),
            "zero_result_rate": sum(d["zero_result_count"] for d in diag) / sum(d["num_searches"] for d in diag) if sum(d["num_searches"] for d in diag) > 0 else 0,
            "failure_types": {ft: sum(1 for d in diag if d["failure_type"] == ft) for ft in set(d["failure_type"] for d in diag)},
        } for name, diag in all_results.items()}, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
