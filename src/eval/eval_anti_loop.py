"""
Eval with Anti-Loop Controller: tests baseline + SFT models with and without controller.
"""
import json, sys, os, re, time, torch

sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/src/agent")
WEBSHOP_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop"
sys.path.insert(0, WEBSHOP_PATH)

import gym
import web_agent_site.envs.web_agent_text_env
from react_agent import ReActAgent, SEARCH_PATTERN, CLICK_PATTERN
from anti_loop_controller import AntiLoopController

MODEL_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct"
ADAPTERS = {
    "baseline": None,
    "unified_sft": "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/unified_sft_adapters",
}
OUTDIR = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results/full_env_audit"
NUM_GOALS = 50
START_IDX = 0
MAX_STEPS = 7


def load_agent(name):
    agent = ReActAgent(MODEL_PATH)
    adapter_path = ADAPTERS.get(name)
    if adapter_path:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = agent.tokenizer
        base = agent.model
        agent.model = PeftModel.from_pretrained(base, adapter_path)
        agent.model = agent.model.merge_and_unload()
        agent.model.eval()
    return agent


def run_episode(agent, env, goal_idx, goal, use_controller=False):
    instruction = goal["instruction_text"]
    obs, _ = env.reset(session=goal_idx)
    available = env.get_available_actions()
    clickables = available.get("clickables", [])
    history = []
    total_reward = 0
    done = False
    interventions = []

    controller = None
    if use_controller:
        controller = AntiLoopController(agent, max_steps=MAX_STEPS)
        controller.reset()

    for step in range(MAX_STEPS):
        if controller:
            thought, action, step_interventions = controller.act(instruction, obs, clickables, history)
            interventions.extend(step_interventions)
        else:
            thought, action = agent.act(instruction, obs, clickables, history)

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

    searches = [h["action"] for h in history if h["action"].startswith("search[")]
    first_q = searches[0][7:-1] if searches else ""

    return {
        "goal_idx": goal_idx,
        "instruction": instruction,
        "total_reward": round(total_reward, 3),
        "success": total_reward > 0.5,
        "steps": len(history),
        "first_query": first_q,
        "first_query_words": len(first_q.split()) if first_q.strip() else 0,
        "interventions": interventions,
    }


def run_eval(agent, name, env, goals, use_controller=False):
    suffix = "_with_controller" if use_controller else ""
    results = []
    for i in range(START_IDX, START_IDX + NUM_GOALS):
        goal = goals[i]
        result = run_episode(agent, env, i, goal, use_controller=use_controller)
        results.append(result)
        ctrl_str = f" ctrl={result['interventions']}" if result['interventions'] else ""
        print(f"[{name}{suffix}] {i} r={result['total_reward']:.3f} ok={result['success']}{ctrl_str}")

    n = len(results)
    sn = sum(1 for r in results if r["success"])
    outpath = f"{OUTDIR}/{name}{suffix}_full_synthetic_keyword_50.jsonl"
    with open(outpath, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{name}{suffix}: {sn}/{n} = {sn/n*100:.0f}% | avg_r={sum(r['total_reward'] for r in results)/n:.3f}")
    return results


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
    goals = env.server.goals
    print(f"{len(goals)} goals loaded")

    all_summaries = {}
    for name in ["baseline", "unified_sft"]:
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        agent = load_agent(name)

        # Without controller
        results_no_ctrl = run_eval(agent, name, env, goals, use_controller=False)
        n = len(results_no_ctrl)
        sr = sum(1 for r in results_no_ctrl if r["success"]) / n
        all_summaries[f"{name}"] = {"success_rate": sr, "avg_reward": sum(r["total_reward"] for r in results_no_ctrl) / n}

        # With controller
        results_ctrl = run_eval(agent, name, env, goals, use_controller=True)
        n_ctrl = len(results_ctrl)
        sr_ctrl = sum(1 for r in results_ctrl if r["success"]) / n_ctrl
        intervention_count = sum(len(r["interventions"]) for r in results_ctrl)
        all_summaries[f"{name}+controller"] = {
            "success_rate": sr_ctrl,
            "avg_reward": sum(r["total_reward"] for r in results_ctrl) / n_ctrl,
            "total_interventions": intervention_count,
        }

        del agent
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}\nANTI-LOOP CONTROLLER COMPARISON\n{'='*60}")
    print(f"{'Config':<30} {'SR':>8} {'Avg R':>8}")
    print("-" * 48)
    for k, v in all_summaries.items():
        print(f"{k:<30} {v['success_rate']*100:>7.1f}% {v['avg_reward']:>8.3f}")

    summary_path = f"{OUTDIR}/anti_loop_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
