"""
Smoke test for full WebShop environment.
"""
import os, sys, re, torch

WEBSHOP_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop"
AGENT_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/src/agent"
sys.path.insert(0, AGENT_PATH)
sys.path.insert(0, WEBSHOP_PATH)

import gym
import web_agent_site.envs.web_agent_text_env
from react_agent import ReActAgent

print("Loading environment (full)...")
env = gym.make("WebAgentTextEnv-v0", observation_mode="text")
print(f"Goals: {len(env.server.goals)}")

print("Loading model...")
agent = ReActAgent(
    "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct"
)

for i in range(3):
    goal = env.server.goals[i]
    instruction = goal["instruction_text"]
    print(f"\n=== Goal {i}: {instruction[:80]}... ===")

    obs, _ = env.reset(session=i)
    available = env.get_available_actions()
    clickables = available.get("clickables", [])
    print(f"Step 0 obs ({len(obs)} chars): {obs[:200]}...")

    total_reward = 0
    for step in range(7):
        thought, action = agent.act(instruction, obs, clickables, [])
        print(f"Step {step+1}: Thought={thought[:80]}...")
        print(f"       Action={action}")
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        available = env.get_available_actions()
        clickables = available.get("clickables", [])
        if done:
            break
    print(f"Reward: {total_reward:.3f} | Done: {done}")

print("\nSmoke test PASSED!")
