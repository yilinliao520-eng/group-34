"""
Convert WebShop IL trajectories (HTML) to LLM SFT format (text chat template).
"""
import json, os, re, sys
from bs4 import BeautifulSoup
from bs4.element import Comment

IL_TRAJS = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop/baseline_models/data/il_trajs_finalized_images.jsonl"
OUTPUT = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/data/sft_expert_trajectories.jsonl"


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return element.parent.name not in ignore and not isinstance(element, Comment)


def html_to_text(html):
    """Convert HTML observation to text format (matching WebAgentTextEnv)."""
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    visible = filter(tag_visible, texts)
    return ' [SEP] '.join(t.strip() for t in visible if t != '\n')


def format_trajectory(traj):
    """Convert one IL trajectory to chat format."""
    actions = traj["actions"]
    states = traj["states"]

    # Extract instruction from first state
    first_state = html_to_text(states[0])
    inst_match = re.search(r'Instruction:\s*(.+?)(?:\s*\[SEP\]|$)', first_state)
    instruction = inst_match.group(1).strip() if inst_match else first_state[:200]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a web shopping agent. Use SHORT keyword queries (3-6 words) for search. "
                "Available actions: search[keywords], click[ASIN], click[Buy Now], click[< Prev], "
                "click[Next >], click[Back to Search], click[option_value]. "
                "Format: Thought: <reasoning>\nAction: search[x] or click[x]"
            )
        }
    ]

    for i in range(len(actions)):
        action = actions[i]
        if i == 0:
            # First step: instruction + initial page
            obs = f"Instruction: {instruction}\n\n" + html_to_text(states[i])[:1500]
        else:
            obs = html_to_text(states[i])[:1500]

        # Parse action
        action_name, action_arg = None, None
        sm = re.match(r"search\[(.+)\]", action)
        cm = re.match(r"click\[(.+)\]", action)
        if sm:
            action_name = "search"
            action_arg = sm.group(1).strip()
        elif cm:
            action_name = "click"
            action_arg = cm.group(1).strip()
        else:
            continue

        messages.append({"role": "user", "content": f"Observation: {obs}"})
        messages.append({
            "role": "assistant",
            "content": f"Thought: I will {action_name} for '{action_arg}'.\nAction: {action}"
        })

    return {"messages": messages, "instruction": instruction, "num_steps": len(actions)}


def main():
    print(f"Loading: {IL_TRAJS}")
    trajectories = []
    with open(IL_TRAJS) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    print(f"Loaded {len(trajectories)} trajectories")

    converted = []
    for i, traj in enumerate(trajectories):
        try:
            conv = format_trajectory(traj)
            converted.append(conv)
        except Exception as e:
            print(f"Skipping trajectory {i}: {e}")
            continue

    print(f"Converted {len(converted)} trajectories")

    with open(OUTPUT, "w") as f:
        for c in converted:
            f.write(json.dumps(c) + "\n")

    print(f"Saved to {OUTPUT}")
    print(f"Total steps: {sum(c['num_steps'] for c in converted)}")
    print(f"Avg steps/traj: {sum(c['num_steps'] for c in converted)/len(converted):.1f}")


if __name__ == "__main__":
    main()
