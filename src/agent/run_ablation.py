"""
Phase 2B: Prompt ablation experiments.
Tests three strategies vs baseline:
  - keyword_only: Force search queries to <= 6 keywords
  - backoff: If no results, automatically broaden query
  - constraint_check: List constraints before buying
"""
import json, sys, os, time, argparse, re
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/src/agent")
WEBSHOP_PATH = os.environ.get("WEBSHOP_PATH", "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop")
sys.path.insert(0, WEBSHOP_PATH)

import gym
import web_agent_site.envs.web_agent_text_env
from react_agent import ReActAgent

ABLATION_PROMPTS = {
    "baseline": None,  # uses default system prompt

    "keyword_only": (
        "You are a web shopping agent. Your task is to find and buy a product "
        "matching the user's instruction.\n\n"
        "CRITICAL RULE FOR SEARCH: Use SHORT keyword queries with 3-6 important words. "
        "Do NOT include all constraints at once. Separate complex constraints across "
        "multiple searches if needed.\n\n"
        "Available actions:\n"
        "- search[3-6 keywords]: Search using short keyword queries\n"
        "- click[ASIN]: View product details\n"
        "- click[Buy Now]: Purchase\n"
        "- click[< Prev] / click[Next >]: Navigate\n"
        "- click[Back to Search]: Return to results\n"
        "- click[option_value]: Select size/color\n\n"
        "Format: Thought: <reasoning>\nAction: search[keywords] or click[element]\n"
    ),

    "backoff": (
        "You are a web shopping agent. Your task is to find and buy a product "
        "matching the user's instruction.\n\n"
        "IMPORTANT BACKOFF RULE: If your search returns no results, immediately try "
        "a broader query with fewer, more general keywords. Do NOT repeat the same "
        "failed search.\n\n"
        "Available actions:\n"
        "- search[keywords]: Search for products\n"
        "- click[ASIN]: View details\n"
        "- click[Buy Now]: Purchase\n"
        "- click[< Prev] / click[Next >]: Navigate\n"
        "- click[Back to Search]: Return\n"
        "- click[option_value]: Select option\n\n"
        "Format: Thought: <reasoning>\nAction: search[keywords] or click[element]\n"
    ),

    "constraint_check": (
        "You are a web shopping agent. Your task is to find and buy a product "
        "matching the user's instruction.\n\n"
        "BEFORE BUYING: Thought must list each constraint and check if product matches. "
        "Only click Buy Now after ALL constraints are verified.\n\n"
        "Available actions:\n"
        "- search[keywords]: Search\n"
        "- click[ASIN]: View details\n"
        "- click[Buy Now]: Purchase (only after constraint check)\n"
        "- click[< Prev] / click[Next >]: Navigate\n"
        "- click[option_value]: Select option\n\n"
        "Format: Thought: [constraint check: color=OK, size=OK, ...]\n"
        "Action: search[keywords] or click[element]\n"
    ),

    "combined": (
        "You are a web shopping agent.\n\n"
        "RULES:\n"
        "1. For search: use 3-6 short keywords. Broaden if no results.\n"
        "2. Before buy: check ALL constraints explicitly.\n"
        "3. Compare at least 2 products if possible.\n\n"
        "Available actions:\n"
        "- search[short keywords]: Search\n"
        "- click[ASIN]: View details\n"
        "- click[Buy Now]: Purchase\n"
        "- click[< Prev] / click[Next >]: Navigate\n"
        "- click[option_value]: Select option\n\n"
        "Format: Thought: [check] Action: search/x or click/y\n"
    ),
}

MAX_STEPS = 7
NUM_TEST_GOALS = 30


class AblationAgent(ReActAgent):
    def __init__(self, model_path, variant="baseline"):
        super().__init__(model_path)
        self.variant = variant
        self.custom_system = ABLATION_PROMPTS.get(variant)

    def _build_prompt(self, instruction, observation, available_actions, history):
        acts_str = ", ".join(available_actions[:30])
        parts = []

        if self.custom_system:
            parts.append(self.custom_system)
        else:
            from react_agent import SYSTEM_PROMPT
            parts.append(SYSTEM_PROMPT)

        for h in history[-2:]:
            parts.append(f"Instruction: {h['instruction']}")
            parts.append(f"Observation: {h['observation'][:1500]}")
            parts.append(f"Thought: {h['thought']}")
            parts.append(f"Action: {h['action']}")

        parts.append(f"Instruction: {instruction}")
        parts.append(f"Observation: {observation[:2000]}")
        parts.append(f"Available actions: {acts_str}")
        parts.append("Thought:")

        prompt = "\n".join(parts)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return text


def run_ablation(agent, env, goals, start_idx, num_goals, output_file):
    results = []
    metrics = defaultdict(list)

    for i in range(start_idx, start_idx + num_goals):
        goal = goals[i]
        obs, _ = env.reset(session=i)
        available = env.get_available_actions()
        clickables = available.get("clickables", [])
        history = []
        done = False
        total_reward = 0
        steps = 0

        for step in range(MAX_STEPS):
            steps = step + 1
            thought, action = agent.act(goal["instruction_text"], obs, clickables, history)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            history.append({
                "instruction": goal["instruction_text"],
                "observation": obs, "thought": thought, "action": action,
            })
            available = env.get_available_actions()
            clickables = available.get("clickables", [])
            if done:
                break

        result = {
            "goal_idx": i, "instruction": goal["instruction_text"],
            "steps": steps, "reward": total_reward,
            "success": total_reward > 0.5,
            "history": [{"thought": h["thought"], "action": h["action"]} for h in history],
        }
        results.append(result)
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        metrics["reward"].append(result["reward"])
        metrics["steps"].append(result["steps"])
        metrics["success"].append(1 if result["success"] else 0)
        print(f"[{i+1}/{start_idx+num_goals}] r={total_reward:.3f} s={steps} ok={result['success']} | {goal['instruction_text'][:50]}...")

    n = len(results)
    return {
        "variant": agent.variant,
        "num_episodes": n,
        "success_rate": sum(metrics["success"]) / n,
        "avg_reward": sum(metrics["reward"]) / n,
        "avg_steps": sum(metrics["steps"]) / n,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--variants", type=str, default="baseline,keyword_only,backoff,constraint_check,combined")
    parser.add_argument("--num_goals", type=int, default=30)
    parser.add_argument("--start_idx", type=int, default=250)
    parser.add_argument("--output_dir", type=str,
        default="/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    print(f"Testing variants: {variants}")
    print(f"Goals: {args.num_goals} (start={args.start_idx})")

    agent = AblationAgent(args.model_path, variant="baseline")
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=1000)
    goals = env.server.goals

    all_summaries = []

    for variant in variants:
        print(f"\n{'='*60}\n  Variant: {variant}\n{'='*60}")
        agent.variant = variant
        agent.custom_system = ABLATION_PROMPTS.get(variant)

        output_file = f"{args.output_dir}/ablation_{variant}.jsonl"
        summary = run_ablation(agent, env, goals, args.start_idx, args.num_goals, output_file)
        all_summaries.append(summary)

        print(f"\n{variant}: success={summary['success_rate']:.1%} "
              f"avg_r={summary['avg_reward']:.3f} avg_s={summary['avg_steps']:.1f}")

    # Comparison table
    print(f"\n{'='*60}")
    print("ABLATION COMPARISON")
    print(f"{'='*60}")
    print(f"{'Variant':<25} {'Success':>8} {'Avg Reward':>10} {'Avg Steps':>10}")
    print("-" * 53)
    for s in all_summaries:
        print(f"{s['variant']:<25} {s['success_rate']:>7.1%} {s['avg_reward']:>10.3f} {s['avg_steps']:>10.1f}")

    summary_path = f"{args.output_dir}/ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
