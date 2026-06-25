"""
Run ReAct baseline agent on WebShop test goals.
Usage: python run_baseline.py --num_goals 100 --output ../results/baseline_results.jsonl
"""
import argparse
import json
import os
import sys
import time
import re
from datetime import datetime
from collections import defaultdict

# Add WebShop to path
WEBSHOP_PATH = os.environ.get(
    "WEBSHOP_PATH",
    "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop",
)
sys.path.insert(0, WEBSHOP_PATH)

import gym
sys.path.insert(0, WEBSHOP_PATH)
import web_agent_site.envs.web_agent_text_env

from react_agent import ReActAgent

MAX_STEPS = 7

KEYWORD_PROMPT = (
    "You are a web shopping agent. Your task is to find and buy a product "
    "matching the user's instruction.\n\n"
    "CRITICAL RULE: Use SHORT keyword queries with 3-6 words for search. "
    "Do NOT include all constraints at once.\n\n"
    "Available actions:\n"
    "- search[3-6 keywords]: Search using short keyword queries\n"
    "- click[ASIN]: View product details\n"
    "- click[Buy Now]: Purchase\n"
    "- click[< Prev] / click[Next >]: Navigate\n"
    "- click[Back to Search]: Return to results\n"
    "- click[option_value]: Select size/color\n\n"
    "Format: Thought: <reasoning>\nAction: search[keywords] or click[element]\n"
)

PROMPTS = {"baseline": None, "keyword_only": KEYWORD_PROMPT}


def extract_constraints(instruction):
    constraints = []
    keywords = ["slim", "fit", "french", "cuffs", "white", "light", "blue",
                "medium", "large", "small", "xl", "xxl", "red", "black",
                "cotton", "wool", "silk", "dress", "shirt", "pants", "shoes"]
    ins_lower = instruction.lower()
    for kw in keywords:
        if kw in ins_lower:
            constraints.append(kw)
    return constraints


def check_constraint_match(instruction, purchased_product):
    """Simple heuristic: check how many instruction keywords appear in product title/attrs."""
    ins_lower = instruction.lower()
    prod_lower = purchased_product.get("title", "").lower()
    prod_attrs = " ".join(
        str(v) for v in purchased_product.get("attributes", {}).values()
    ).lower()
    prod_text = prod_lower + " " + prod_attrs

    keywords = re.findall(r"[a-zA-Z]+", ins_lower)
    stopwords = {"a", "an", "the", "and", "or", "in", "on", "at", "to", "for",
                 "of", "with", "is", "are", "was", "be", "find", "me", "i", "you",
                 "that", "this", "it", "under", "over", "size", "color", "price",
                 "product", "item", "looking", "want", "need", "search", "buy"}
    keywords = [k for k in keywords if k not in stopwords and len(k) > 1]

    matched = sum(1 for kw in keywords if kw in prod_text)
    return matched, len(keywords)


def log_result(result, log_file):
    with open(log_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def run_episode(agent, env, goal_idx, goal):
    instruction = goal["instruction_text"]
    obs, _ = env.reset(session=goal_idx)
    available = env.get_available_actions()
    history = []
    total_reward = 0
    done = False
    steps = 0

    clickables = available.get("clickables", [])

    for step in range(MAX_STEPS):
        steps = step + 1
        thought, action = agent.act(instruction, obs, clickables, history)

        obs, reward, done, _ = env.step(action)
        total_reward += reward

        history.append({
            "instruction": instruction,
            "observation": obs,
            "thought": thought,
            "action": action,
        })

        available = env.get_available_actions()
        clickables = available.get("clickables", [])

        if done:
            break

    return {
        "goal_idx": goal_idx,
        "instruction": instruction,
        "steps": steps,
        "reward": total_reward,
        "success": total_reward > 0.5,
        "history": [
            {"thought": h["thought"], "action": h["action"]}
            for h in history
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num_goals", type=int, default=100)
    parser.add_argument("--output", type=str,
                        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results/baseline_results.jsonl")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--prompt_variant", type=str, default="baseline", choices=["baseline", "keyword_only"])
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    system_prompt = PROMPTS.get(args.prompt_variant)
    agent = ReActAgent(args.model_path, system_prompt=system_prompt)
    print(f"Prompt variant: {args.prompt_variant}")

    print("Creating WebShop environment...")
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=1000)

    goals = env.server.goals
    num_goals = min(args.num_goals, len(goals))
    print(f"Running {num_goals} goals (start={args.start_idx})...")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = []
    metrics = defaultdict(list)

    for i in range(args.start_idx, args.start_idx + num_goals):
        goal = goals[i]
        print(f"\n[{i+1}/{args.start_idx + num_goals}] {goal['instruction_text'][:80]}...")

        start_time = time.time()
        result = run_episode(agent, env, i, goal)
        elapsed = time.time() - start_time

        result["time"] = elapsed
        results.append(result)
        log_result(result, args.output)

        metrics["reward"].append(result["reward"])
        metrics["steps"].append(result["steps"])
        metrics["success"].append(1 if result["success"] else 0)
        metrics["time"].append(elapsed)

        print(f"  Reward: {result['reward']:.3f} | Steps: {result['steps']} | "
              f"Time: {elapsed:.1f}s | Success: {result['success']}")

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)
    n = len(results)
    print(f"Total episodes: {n}")
    print(f"Success rate: {sum(metrics['success'])/n*100:.1f}%")
    print(f"Avg reward:    {sum(metrics['reward'])/n:.3f}")
    print(f"Avg steps:     {sum(metrics['steps'])/n:.1f}")
    print(f"Avg time:      {sum(metrics['time'])/n:.1f}s")
    print(f"\nResults saved to: {args.output}")

    # Save summary
    summary_path = args.output.replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model_path,
            "num_episodes": n,
            "success_rate": sum(metrics["success"]) / n,
            "avg_reward": sum(metrics["reward"]) / n,
            "avg_steps": sum(metrics["steps"]) / n,
            "avg_time": sum(metrics["time"]) / n,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Failure case analysis
    failures = [r for r in results if not r["success"]]
    print(f"\nFailure cases: {len(failures)}/{n}")
    for f in failures[:5]:
        print(f"  Goal {f['goal_idx']}: {f['instruction'][:60]}... "
              f"(reward={f['reward']:.2f}, steps={f['steps']})")


if __name__ == "__main__":
    main()
