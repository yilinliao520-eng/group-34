"""
Quick smoke test: load one goal, run agent, check output.
Run on server: python smoke_test.py
"""
import os, sys, json

sys.path.insert(0, "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/src/agent")
WEBSHOP_PATH = os.environ.get("WEBSHOP_PATH", "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop")
sys.path.insert(0, WEBSHOP_PATH)

import gym
import web_agent_site.envs.web_agent_text_env
from react_agent import ReActAgent

MODEL_PATH = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/models/base/qwen/Qwen2.5-7B-Instruct"

print("Loading model...")
agent = ReActAgent(MODEL_PATH)
print("Model loaded.")

print("Creating env...")
env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=1000)
print(f"Env ready. {len(env.server.goals)} goals available.")

goal = env.server.goals[0]
instruction = goal["instruction_text"]
print(f"\nGoal: {instruction}")

obs, _ = env.reset(session=0)
available = env.get_available_actions()
clickables = available.get("clickables", [])

print(f"\nObservation ({len(obs)} chars):")
print(obs[:500])

print(f"\nAvailable clickables: {clickables[:10]}")

print("\n--- Agent thinks ---")
thought, action = agent.act(instruction, obs[:2000], clickables, history=[])
print(f"Thought: {thought}")
print(f"Action: {action}")

print("\n--- Step ---")
obs, reward, done, _ = env.step(action)
print(f"Reward: {reward:.3f} | Done: {done}")
print(f"New obs ({len(obs)} chars): {obs[:300]}")

print("\nSmoke test passed!")
