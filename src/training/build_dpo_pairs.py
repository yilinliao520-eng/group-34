"""
Build query-level DPO pairs from diagnostic results.
Chosen: baseline successful short queries
Rejected: SFT failed long/repeated queries

Does NOT train - only constructs the dataset.
"""
import json, os

AUDIT_DIR = "/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/results/full_env_audit"
OUT_PATH = f"{AUDIT_DIR}/query_dpo_pairs_full_synthetic_50.jsonl"


def load_episodes(filepath):
    if not os.path.exists(filepath):
        return []
    episodes = []
    with open(filepath) as f:
        for line in f:
            episodes.append(json.loads(line))
    return episodes


def build_pairs(baseline_eps, sft_eps):
    """
    Pair construction rules (from Review Agent Decision):
    - For each goal where baseline succeeds and SFT fails:
      chosen = baseline's first successful/short query
      rejected = SFT's long/repeated/zero-reward query
    - For each goal where SFT succeeds:
      keep its query as possible chosen example
    """
    pairs = []

    # Index SFT episodes by goal
    sft_by_goal = {e["goal_idx"]: e for e in sft_eps}

    for b in baseline_eps:
        goal_idx = b["goal_idx"]
        s = sft_by_goal.get(goal_idx)
        if not s:
            continue

        # Case 1: baseline succeeds, SFT fails → chosen=baseline, rejected=SFT
        if b.get("success") and not s.get("success"):
            chosen = b.get("first_query", "")
            rejected = s.get("first_query", "")
            if chosen.strip() and rejected.strip() and chosen != rejected:
                pairs.append({
                    "goal_idx": goal_idx,
                    "instruction": b.get("instruction", ""),
                    "chosen_query": chosen,
                    "rejected_query": rejected,
                    "chosen_source": "baseline",
                    "rejected_source": sft_eps[0] if isinstance(sft_eps, list) else "sft",
                    "baseline_reward": b.get("total_reward", 0),
                    "sft_reward": s.get("total_reward", 0),
                    "reason": "baseline_success_sft_fail",
                })

        # Case 2: SFT succeeds → keep as chosen
        elif s.get("success") and not b.get("success"):
            chosen = s.get("first_query", "")
            rejected = b.get("first_query", "")
            if chosen.strip() and rejected.strip() and chosen != rejected:
                pairs.append({
                    "goal_idx": goal_idx,
                    "instruction": s.get("instruction", ""),
                    "chosen_query": chosen,
                    "rejected_query": rejected,
                    "chosen_source": "sft",
                    "rejected_source": "baseline",
                    "baseline_reward": b.get("total_reward", 0),
                    "sft_reward": s.get("total_reward", 0),
                    "reason": "sft_success_baseline_fail",
                })

    return pairs


def main():
    # Load diagnostic files (contain per-step trajectories with first_query)
    baseline_path = f"{AUDIT_DIR}/diagnostic_baseline_synthetic_keyword_50.jsonl"
    sft_path = f"{AUDIT_DIR}/diagnostic_unified_sft_synthetic_keyword_50.jsonl"

    baseline_eps = load_episodes(baseline_path)
    sft_eps = load_episodes(sft_path)

    print(f"Loaded: {len(baseline_eps)} baseline, {len(sft_eps)} SFT episodes")

    # Filter: only goals present in both
    baseline_goals = {e["goal_idx"] for e in baseline_eps}
    sft_goals = {e["goal_idx"] for e in sft_eps}
    common = baseline_goals & sft_goals
    baseline_eps = [e for e in baseline_eps if e["goal_idx"] in common]
    sft_eps = [e for e in sft_eps if e["goal_idx"] in common]
    print(f"Common goals: {len(common)}")

    pairs = build_pairs(baseline_eps, sft_eps)
    print(f"Built {len(pairs)} DPO pairs")

    with open(OUT_PATH, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved to {OUT_PATH}")

    # Print breakdown
    reasons = {}
    for p in pairs:
        r = p["reason"]
        reasons[r] = reasons.get(r, 0) + 1
    for r, c in reasons.items():
        print(f"  {r}: {c}")


if __name__ == "__main__":
    main()
