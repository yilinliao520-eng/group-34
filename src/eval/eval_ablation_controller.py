"""
Controller Ablation: identify which rule causes the gain on unified_sft.
Tests 3 new variants (we already have no_controller=2% and current_controller=48%).
"""
import json, sys, os, re, torch

sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/src/agent")
WEBSHOP_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop"
sys.path.insert(0, WEBSHOP_PATH)

import gym
import web_agent_site.envs.web_agent_text_env
from react_agent import ReActAgent
from anti_loop_controller import AntiLoopController

MODEL_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct"
ADAPTER = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/unified_sft_adapters"
OUTDIR = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results/full_env_audit"
NUM_GOALS = 50
START_IDX = 0
MAX_STEPS = 7

VARIANTS = {
    "repeat_to_click_only": {
        "enable_repeat_to_click": True,
        "enable_search_count_to_click": False,
        "enable_forced_buy": False,
        "conservative_buy": False,
    },
    "no_forced_buy": {
        "enable_repeat_to_click": True,
        "enable_search_count_to_click": True,
        "enable_forced_buy": False,
        "conservative_buy": False,
    },
    "conservative_forced_buy": {
        "enable_repeat_to_click": True,
        "enable_search_count_to_click": True,
        "enable_forced_buy": True,
        "conservative_buy": True,
    },
}


def load_agent():
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model = model.merge_and_unload()
    model.eval()
    return ReActAgent(MODEL_PATH)


def run_variant(variant_name, ctrl_cfg, agent, env, goals):
    controller = AntiLoopController(
        agent, max_steps=MAX_STEPS,
        enable_repeat_to_click=ctrl_cfg["enable_repeat_to_click"],
        enable_search_count_to_click=ctrl_cfg["enable_search_count_to_click"],
        enable_forced_buy=ctrl_cfg["enable_forced_buy"],
        conservative_buy=ctrl_cfg.get("conservative_buy", False),
    )

    results = []
    for i in range(START_IDX, START_IDX + NUM_GOALS):
        goal = goals[i]
        instruction = goal["instruction_text"]
        obs, _ = env.reset(session=i)
        available = env.get_available_actions()
        clickables = available.get("clickables", [])
        history = []
        total_reward = 0
        done = False
        controller.reset()
        interventions = []

        for step in range(MAX_STEPS):
            thought, action, step_interventions = controller.act(instruction, obs, clickables, history)
            interventions.extend(step_interventions)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            history.append({"instruction": instruction, "observation": obs, "thought": thought, "action": action})
            available = env.get_available_actions()
            clickables = available.get("clickables", [])
            if done:
                break

        result = {
            "goal_idx": i, "instruction": instruction,
            "total_reward": round(total_reward, 3),
            "success": total_reward > 0.5,
            "steps": len(history),
            "interventions": interventions,
        }
        results.append(result)
        print(f"[{variant_name}] {i} r={total_reward:.3f} ok={result['success']} ctrl={len(interventions)}")

    return results


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # Load model once
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading model...")
    agent = ReActAgent(MODEL_PATH)

    env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
    goals = env.server.goals
    print(f"{len(goals)} goals loaded")

    # Load the unified SFT adapter
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = agent.tokenizer
    model = agent.model
    print("Loading unified_sft adapter...")
    merged = PeftModel.from_pretrained(model, ADAPTER)
    merged = merged.merge_and_unload()
    merged.eval()
    agent.model = merged

    all_summaries = {}

    for variant_name, cfg in VARIANTS.items():
        print(f"\n{'='*50}\n  {variant_name}\n{'='*50}")
        results = run_variant(variant_name, cfg, agent, env, goals)

        n = len(results)
        sr = sum(1 for r in results if r["success"]) / n
        avg_r = sum(r["total_reward"] for r in results) / n
        interventions = sum(len(r["interventions"]) for r in results)
        outpath = f"{OUTDIR}/controller_ablation_unified_sft_{variant_name}_50.jsonl"
        with open(outpath, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\n{variant_name}: {int(sr*n)}/{n} = {sr*100:.0f}% SR, avg_r={avg_r:.3f}, interventions={interventions}")
        all_summaries[variant_name] = {"success_rate": sr, "avg_reward": avg_r, "total_interventions": interventions}

    # Add pre-existing results
    all_summaries["no_controller"] = {"success_rate": 0.02, "avg_reward": 0.012, "total_interventions": 0}
    all_summaries["current_controller"] = {"success_rate": 0.48, "avg_reward": 0.446, "total_interventions": 114}

    # Summary
    print(f"\n{'='*60}\nCONTROLLER ABLATION COMPARISON\n{'='*60}")
    print(f"{'Config':<35} {'SR':>8} {'Avg R':>8} {'Interventions':>12}")
    print("-" * 65)
    for name in ["no_controller", "repeat_to_click_only", "no_forced_buy", "conservative_forced_buy", "current_controller"]:
        if name in all_summaries:
            v = all_summaries[name]
            print(f"{name:<35} {v['success_rate']*100:>7.1f}% {v['avg_reward']:>8.3f} {v['total_interventions']:>12}")

    outpath = f"{OUTDIR}/controller_ablation_unified_sft_50.json"
    with open(outpath, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
