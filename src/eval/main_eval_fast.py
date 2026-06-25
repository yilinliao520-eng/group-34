"""
Fast unified evaluator: loads model + environment once, runs multiple configs.
Supports resume (skips completed goal_idx).
Usage: python main_eval_fast.py
"""
import json, sys, os, re, time, torch
from datetime import datetime

sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/src/agent")
WEBSHOP_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop"
sys.path.insert(0, WEBSHOP_PATH)

import gym
import web_agent_site.envs.web_agent_text_env
from react_agent import ReActAgent, SYSTEM_PROMPT, SEARCH_PATTERN, CLICK_PATTERN
from anti_loop_controller import AntiLoopController
from peft import PeftModel

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
MAX_STEPS = 7


def get_completed_indices(jsonl_path):
    """Load completed goal_idx from JSONL for resume support."""
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                completed.add(r["goal_idx"])
            except Exception:
                pass
    return completed


def write_result(result, jsonl_path):
    """Stream single result to JSONL."""
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(result) + "\n")


def load_model(adapter_path=None):
    """Load Qwen2.5-7B with optional LoRA adapter."""
    agent = ReActAgent(MODEL_PATH, system_prompt=KEYWORD_PROMPT)
    if adapter_path:
        agent.model = PeftModel.from_pretrained(agent.model, adapter_path)
        agent.model = agent.model.merge_and_unload()
        agent.model.eval()
    return agent


def run_episode(agent, env, goal, goal_idx, controller=None):
    """Run one episode with optional controller."""
    instruction = goal["instruction_text"]
    history = []
    total_reward = 0
    interventions = []
    first_query = ""
    num_searches = 0
    buy_occurred = False

    obs, _ = env.reset(session=goal_idx)
    available = env.get_available_actions()
    clickables = available.get("clickables", [])

    if controller:
        controller.reset()

    for step in range(MAX_STEPS):
        if controller:
            thought, action, step_interventions = controller.act(instruction, obs, clickables, history)
            interventions.extend(step_interventions)
        else:
            thought, action = agent.act(instruction, obs, clickables, history)

        obs, reward, done, _ = env.step(action)
        total_reward += reward
        history.append({"instruction": instruction, "observation": obs, "thought": thought, "action": action})

        if action.startswith("search["):
            num_searches += 1
            if num_searches == 1:
                first_query = action[7:-1]
        if "buy now" in action.lower():
            buy_occurred = True

        available = env.get_available_actions()
        clickables = available.get("clickables", [])
        if done:
            break

    return {
        "total_reward": round(total_reward, 3),
        "success": total_reward > 0.5,
        "steps": len(history),
        "first_query": first_query,
        "first_query_words": len(first_query.split()) if first_query.strip() else 0,
        "num_searches": num_searches,
        "buy_occurred": buy_occurred,
        "interventions": interventions,
    }


def run_config(agent, env, name, goals, goal_indices, jsonl_path, use_controller=False):
    """Run a config on given goal indices, with resume support."""
    completed = get_completed_indices(jsonl_path)
    to_run = [i for i in goal_indices if i not in completed]
    print(f"  {name}: {len(completed)} done, {len(to_run)} remaining")

    controller = None
    if use_controller:
        controller = AntiLoopController(agent, max_steps=MAX_STEPS)

    for i in to_run:
        goal = goals[i]
        result = run_episode(agent, env, goal, i, controller=controller)
        result["goal_idx"] = i
        result["instruction"] = goal["instruction_text"]
        write_result(result, jsonl_path)
        ctrl_str = f" ctrl={result['interventions']}" if result['interventions'] else ""
        print(f"  [{name}] {i} r={result['total_reward']:.3f} ok={result['success']}{ctrl_str}")

    # Load all results for summary
    all_results = []
    with open(jsonl_path) as f:
        for line in f:
            all_results.append(json.loads(line))

    n = len(all_results)
    if n == 0:
        return {"config": name, "n": 0}
    sr = sum(1 for r in all_results if r["success"]) / n
    avg_r = sum(r["total_reward"] for r in all_results) / n
    interventions = sum(len(r.get("interventions", [])) for r in all_results)
    print(f"  {name}: {int(sr*n)}/{n} = {sr*100:.0f}% SR, avg_r={avg_r:.3f}")
    return {"config": name, "n": n, "success_rate": sr, "avg_reward": avg_r, "interventions": interventions}


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M')}] Loading environment...")
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
    goals = env.server.goals
    print(f"  {len(goals)} goals loaded")

    # Use goals 0-49 as fixed test set
    goal_indices = list(range(50))
    print(f"  Fixed test set: goals 0-49")

    summaries = []

    # === Config 1: baseline (no adapter) ===
    print(f"\n[{datetime.now().strftime('%H:%M')}] Loading model (no adapter)...")
    agent = load_model()
    agent._env = env
    name = "baseline_keyword"
    path = f"{OUTDIR}/{name}_full_50.jsonl"
    s = run_config(agent, env, name, goals, goal_indices, path, use_controller=False)
    summaries.append(s)

    # === Config 2: baseline + controller ===
    name = "baseline_keyword_ctrl"
    path = f"{OUTDIR}/{name}_full_50.jsonl"
    s = run_config(agent, env, name, goals, goal_indices, path, use_controller=True)
    summaries.append(s)
    del agent
    torch.cuda.empty_cache()

    # === Config 3: expert_sft ===
    print(f"\n[{datetime.now().strftime('%H:%M')}] Loading expert_sft adapter...")
    agent = load_model(ADAPTERS["expert_sft"])
    agent._env = env
    name = "expert_sft"
    path = f"{OUTDIR}/{name}_full_50.jsonl"
    s = run_config(agent, env, name, goals, goal_indices, path, use_controller=False)
    summaries.append(s)
    del agent
    torch.cuda.empty_cache()

    # === Config 4: unified_sft ===
    print(f"\n[{datetime.now().strftime('%H:%M')}] Loading unified_sft adapter...")
    agent = load_model(ADAPTERS["unified_sft"])
    agent._env = env
    name = "unified_sft"
    path = f"{OUTDIR}/{name}_full_50.jsonl"
    s = run_config(agent, env, name, goals, goal_indices, path, use_controller=False)
    summaries.append(s)

    # === Config 5: unified_sft + controller ===
    name = "unified_sft_ctrl"
    path = f"{OUTDIR}/{name}_full_50.jsonl"
    s = run_config(agent, env, name, goals, goal_indices, path, use_controller=True)
    summaries.append(s)
    del agent
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*65}")
    print(f"  UNIFIED EVAL SUMMARY (full env, goals 0-49, max_steps=7)")
    print(f"{'='*65}")
    print(f"{'Config':<30} {'SR':>8} {'Avg R':>8} {'Interventions':>12}")
    print("-" * 60)
    for s in summaries:
        if s["n"] > 0:
            print(f"{s['config']:<30} {s['success_rate']*100:>7.1f}% {s['avg_reward']:>8.3f} {s.get('interventions', 0):>12}")

    summary_path = f"{OUTDIR}/unified_eval_full_50_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
