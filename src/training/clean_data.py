"""
Data Cleaning Pipeline: Unify all data sources into the standard ReAct SFT format.
Based on A³T (COLM 2024) / IPR (EMNLP 2024) / ETO (2024) conventions.

Input sources:
  1. il_trajs_finalized_images.jsonl - 1,571 expert trajectories (HTML → convert to text)
  2. goal_query_predict.json - 11,724 (instruction, query) pairs → embed as search step
  3. goal_query_map.json - 1,506 (instruction, query) pairs → same

Output: Clean SFT data in Qwen2.5 chat template format.
  System Prompt → Instruction + Observation → Thought + Action → ... → Buy Now
"""
import json, os, re, random, sys
from bs4 import BeautifulSoup
from bs4.element import Comment

# --- Constants ---
WEBSHOP_DIR = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop"
OUTPUT_DIR = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/data"

SYSTEM_PROMPT = (
    "You are a web shopping agent. Use SHORT keyword queries (3-6 words) for search. "
    "Available actions: search[keywords], click[ASIN], click[Buy Now], click[< Prev], "
    "click[Next >], click[Back to Search], click[option_value]. "
    "Format: Thought: <reasoning>\nAction: search[x] or click[x]"
)

TEMPLATE_THOUGHTS = {
    "search": "I need to search for this product. Let me use focused keywords.",
    "click_asin": "This looks like a good match. Let me check the details.",
    "click_buy": "This product matches all the requirements. I will purchase it.",
    "click_next": "Let me see more results on the next page.",
    "click_prev": "Let me go back to the previous page.",
    "click_back": "Let me return to the search results.",
    "click_option": "I need to select this option to match the requirements.",
    "click_desc": "Let me read the product description.",
    "click_feat": "Let me check the product features.",
    "click_review": "Let me read the reviews.",
}


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return element.parent.name not in ignore and not isinstance(element, Comment)


def html_to_text(html):
    """Convert HTML observation to WebShop text mode format."""
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    visible = filter(tag_visible, texts)
    return ' [SEP] '.join(t.strip() for t in visible if t != '\n')


def get_thought_for_action(action):
    """Generate a template thought for an action."""
    action_lower = action.lower()
    if action_lower.startswith('search'):
        return TEMPLATE_THOUGHTS["search"]
    if 'buy now' in action_lower:
        return TEMPLATE_THOUGHTS["click_buy"]
    if 'next >' in action_lower:
        return TEMPLATE_THOUGHTS["click_next"]
    if '< prev' in action_lower:
        return TEMPLATE_THOUGHTS["click_prev"]
    if 'back to search' in action_lower:
        return TEMPLATE_THOUGHTS["click_back"]
    if action_lower in ['description', 'features', 'reviews']:
        return TEMPLATE_THOUGHTS.get(f"click_{action_lower.lower()}", "Let me check this.")
    # For ASIN clicks and options
    if action_lower.startswith('click['):
        arg = action_lower[6:-1]
        if re.match(r'^[A-Z0-9]{10}$', arg):
            return TEMPLATE_THOUGHTS["click_asin"]
        else:
            return TEMPLATE_THOUGHTS["click_option"]
    return "I will proceed with this action."


def extract_instruction_from_html(first_html):
    """Extract the instruction text from the first page HTML."""
    text = html_to_text(first_html)
    match = re.search(r'Instruction:\s*(.+?)(?:\s*\[SEP\]|$)', text)
    if match:
        return match.group(1).strip()
    return text[:200]


def convert_il_trajectory(traj):
    """Convert one IL trajectory (HTML states) to ReAct chat format."""
    actions = traj["actions"]
    states = traj["states"]

    instruction = extract_instruction_from_html(states[0])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i, action in enumerate(actions):
        # Parse action
        action_clean = action.strip()
        sm = re.match(r"search\[(.+)\]", action_clean)
        cm = re.match(r"click\[(.+)\]", action_clean)

        if not sm and not cm:
            continue

        # Build observation
        if i == 0:
            obs = f"Instruction: {instruction}\n\n" + html_to_text(states[i])[:1500]
        else:
            obs = html_to_text(states[i])[:1500]

        # Build thought + action
        thought = get_thought_for_action(action_clean)
        assistant_msg = f"Thought: {thought}\nAction: {action_clean}"

        messages.append({"role": "user", "content": f"Observation: {obs}"})
        messages.append({"role": "assistant", "content": assistant_msg})

    return {
        "messages": messages,
        "instruction": instruction,
        "num_steps": len(messages) // 2,
        "source": "il_trajectory",
    }


def convert_query_pair(instruction, query):
    """Convert an (instruction, query) pair into the search-step ReAct format."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Instruction: {instruction}\n\n"
            "You are on the search page. What keywords should you search for?"
        )},
        {"role": "assistant", "content": (
            f"Thought: I need to search using focused keywords that capture "
            f"the key product attributes.\nAction: search[{query}]"
        )},
    ]
    return {
        "messages": messages,
        "instruction": instruction,
        "num_steps": 1,
        "source": "query_pair",
    }


def main():
    # === Source 1: IL trajectories ===
    print("=== Processing IL trajectories ===")
    il_path = f"{WEBSHOP_DIR}/baseline_models/data/il_trajs_finalized_images.jsonl"

    il_trajs = []
    with open(il_path) as f:
        for line in f:
            if line.strip():
                il_trajs.append(json.loads(line))
    print(f"Loaded {len(il_trajs)} IL trajectories")

    il_converted = []
    for i, traj in enumerate(il_trajs):
        try:
            conv = convert_il_trajectory(traj)
            il_converted.append(conv)
        except Exception as e:
            if i < 5:
                print(f"  Skip traj {i}: {e}")

    print(f"Converted {len(il_converted)} IL trajectories")

    # === Source 2: Query pairs from goal_query_predict.json ===
    print("\n=== Processing query pairs ===")
    qp_path = f"{WEBSHOP_DIR}/baseline_models/data/goal_query_predict.json"
    with open(qp_path) as f:
        query_data = json.load(f)
    print(f"Loaded {len(query_data)} query predictions")

    # Also load goal_query_map for additional pairs
    qm_path = f"{WEBSHOP_DIR}/baseline_models/data/goal_query_map.json"
    with open(qm_path) as f:
        query_map = json.load(f)
    print(f"Loaded {len(query_map)} query map entries")

    # Merge: use predict.json as primary (has more data), supplement with map.json
    all_instructions = set(query_data.keys()) | set(query_map.keys())
    print(f"Total unique instructions: {len(all_instructions)}")

    query_converted = []
    seen_instructions = set()
    for instruction in all_instructions:
        if instruction in seen_instructions:
            continue
        seen_instructions.add(instruction)

        # Get query (prefer map.json as it has human-annotated queries)
        if instruction in query_map and query_map[instruction]:
            query = query_map[instruction][0]
        elif instruction in query_data and query_data[instruction]:
            query = query_data[instruction][0]
        else:
            continue

        conv = convert_query_pair(instruction, query)
        query_converted.append(conv)

    print(f"Converted {len(query_converted)} query pairs")

    # === Merge and save ===
    all_data = il_converted + query_converted
    random.Random(42).shuffle(all_data)
    print(f"\nTotal unified data: {len(all_data)} examples")

    # Save as JSONL
    output_path = f"{OUTPUT_DIR}/unified_sft_data.jsonl"
    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved to {output_path}")

    # Stats
    il_steps = sum(c["num_steps"] for c in il_converted)
    qp_steps = sum(c["num_steps"] for c in query_converted)
    print(f"\nStats:")
    print(f"  IL trajectories: {len(il_converted)} examples, {il_steps} steps")
    print(f"  Query pairs: {len(query_converted)} examples, {qp_steps} steps")
    print(f"  Total: {len(all_data)} examples, {il_steps + qp_steps} steps")


if __name__ == "__main__":
    main()
