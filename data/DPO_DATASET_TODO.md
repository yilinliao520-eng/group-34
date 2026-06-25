# DPO Dataset Construction Plan

## Purpose
Construct query-level and action-level preference pairs for DPO training of the WebShop agent.

## Pair Types Needed

### 1. Query-level pairs (search query quality)
- chosen: short successful baseline queries (≤6 words, reward > 0.5)
- rejected: long/repeated zero-result SFT queries (≥10 words, zero reward)
- source: diagnostic_baseline + diagnostic_unified_sft episodes from same goals

### 2. Action-level pairs (transition control)
- chosen: click[ASIN] after 2-3 searches when results exist
- rejected: repeated search[...] from same search result page
- chosen: click[Buy Now] after viewing product details
- rejected: search/back loop from same product page

### Construction Steps
1. Load diagnostic JSONL files from results/full_env_audit/
2. Match episodes by goal_idx
3. For query pairs: extract first_query from each, pair success vs failure
4. For action pairs: identify transition moments (search→click, click→buy) from step sequences
5. Output format: JSONL with chosen/rejected/action fields

### Expected Size
- Query pairs: ~20-30 from 50-goal diagnostic
- Action pairs: ~50-100 from step-level analysis

### Status
- Query pairs built: results/full_env_audit/query_dpo_pairs_full_synthetic_50.jsonl (11 pairs)
- Action pairs: not yet built
- DPO training: not started
