"""Quick eval: raw ReAct baseline on 50 goals (no keyword prompt)."""
import json, sys, os, time, re, torch
sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/src/agent")
WEBSHOP_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop"
sys.path.insert(0, WEBSHOP_PATH)
import gym, web_agent_site.envs.web_agent_text_env
from transformers import AutoModelForCausalLM, AutoTokenizer
from react_agent import ReActAgent, SYSTEM_PROMPT, SEARCH_PATTERN, CLICK_PATTERN

MODEL_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct"
OUTDIR = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results/full_env_audit"
NUM_GOALS = 50
START_IDX = 0

def main():
    print("Loading model...")
    agent = ReActAgent(MODEL_PATH, system_prompt=None)
    print("Model loaded. Using original ReAct SYSTEM_PROMPT.")

    env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
    goals = env.server.goals
    print(f"{len(goals)} goals loaded")

    results = []
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
            thought, action = agent.act(instruction, obs, clickables, history)
            if step == 0 and action.startswith("search["):
                first_query = action[7:-1]
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            history.append({"instruction": instruction, "observation": obs, "thought": thought, "action": action})
            available = env.get_available_actions()
            clickables = available.get("clickables", [])
            if done:
                break

        result = {
            "goal_idx": i, "instruction": instruction,
            "steps": len(history), "reward": total_reward,
            "success": total_reward > 0.5,
            "first_query": first_query,
            "first_query_words": len(first_query.split()) if first_query.strip() else 0,
        }
        results.append(result)
        print(f"[{i}] r={total_reward:.3f} s={len(history)} ok={result['success']} q='{first_query[:60]}'")

    n = len(results)
    sn = sum(1 for r in results if r["success"])
    print(f"\nRaw ReAct Baseline: {sn}/{n} = {sn/n*100:.0f}% | avg_r={sum(r['reward'] for r in results)/n:.3f} | avg_ql={sum(r['first_query_words'] for r in results)/n:.1f}")

    outpath = f"{OUTDIR}/baseline_full_synthetic_raw_react_50.jsonl"
    with open(outpath, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved to {outpath}")

if __name__ == "__main__":
    main()
