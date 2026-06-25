# AGENT_REVIEW_BOARD

This file is the review/advice board for the WebShop final project.

Role split:
- Decision/Review agent: owns experiment decisions, prioritization, protocol design, result interpretation, and next-step instructions.
- Execution agent: executes the decisions written here, runs data upload, merge, indexing, training, evaluation, and writes logs/results.

The execution agent should treat the latest `Decision` section in this file as the current experimental plan unless the user explicitly overrides it.

The decision/review agent should not run long experiments or modify training code unless the user explicitly asks; it should primarily issue decisions and audit outputs.

## Current Priority

Do not train before the full WebShop environment is verified.

The current top hypothesis is:

> Previous SFT failures were likely caused by full-data/IL-style training being evaluated in the 1K small WebShop index.

Therefore, the next experimental goal is to prove that the environment has truly moved from small to full:

1. Full `items_shuffle_full.json` is valid JSON.
2. `web_agent_site/utils.py` points to the full product file and full attribute file.
3. `search_engine/resources/documents.jsonl` is regenerated from the full product file.
4. `search_engine/indexes/` is rebuilt from full resources.
5. WebShop smoke test works with `num_products=None`.

## Required Audit Artifacts

Please save these under `results/full_env_audit/`:

- `data_integrity.txt`
- `index_stats.txt`
- `smoke_test.jsonl`
- `baseline_full_50.jsonl`
- `baseline_full_50_summary.json`
- `keyword_full_50.jsonl`
- `keyword_full_50_summary.json`
- `unified_sft_full_50.jsonl`
- `unified_sft_full_50_summary.json`
- `model_comparison_full_50.json`

## Minimum Checks Before Any Training

Data integrity:

```bash
cd /inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop/data
python -m json.tool items_shuffle_full.json >/tmp/items_shuffle_check.json
```

Index/resource size:

```bash
cd /inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/env/WebShop
wc -l search_engine/resources/documents.jsonl
du -sh search_engine/resources/documents.jsonl search_engine/indexes
```

Configuration:

```bash
grep -n "DEFAULT_.*PATH\|DEBUG_PROD_SIZE" web_agent_site/utils.py
```

Expected:

- `DEFAULT_FILE_PATH` should point to full data, not `items_shuffle_1000.json`.
- `DEFAULT_ATTR_PATH` should point to `items_ins_v2.json`, not `items_ins_v2_1000.json`.
- full resources/index should be much larger than the old 1K index.

## First Fair Comparison

Use the same split and same max steps for all systems.

Recommended first split:

- `start_idx = 0`
- `num_goals = 50`
- `max_steps = 7`
- `num_products = None`

Systems:

1. `baseline_full_50`: original ReAct + Qwen2.5-7B.
2. `keyword_full_50`: ReAct + keyword-only prompt.
3. `unified_sft_full_50`: existing unified SFT adapter.
4. Optional: `expert_sft_full_50`: existing expert SFT adapter.

Do not compare results across different splits, different `max_steps`, or different product index sizes.

## Metrics To Report

Each summary should include:

- `num_episodes`
- `avg_reward`
- `strong_success_rate`: `reward > 0.5`
- `strict_success_rate`: preferably `reward >= 0.9` or `reward == 1.0`, but define it explicitly.
- `avg_steps`
- `buy_rate`
- `first_query_avg_words`
- `zero_result_rate`
- `invalid_action_rate`
- `parse_failure_rate`

Reason:

- Success rate alone may hide useful changes in partial constraint matching.
- Query length and zero-result rate directly test whether full index fixes the previous SFT failure mode.

## Interpretation Rules

If `unified_sft` improves on full:

- The main previous failure was likely small/full index mismatch.
- Next step: expand to 500 goals before retraining.
- Then consider DPO or Bandit only after stable full-environment gains are confirmed.

If `unified_sft` is still worse on full:

- Do not immediately train longer.
- First inspect:
  - output action format,
  - first query length,
  - zero-result rate,
  - repeated search rate,
  - buy rate,
  - premature buy examples,
  - whether training loss masked user/system tokens.

If `keyword_only` beats baseline on full:

- Keep it as a strong prompt baseline (`System A+`).
- Future SFT/DPO should be compared against both baseline and keyword-only, not only raw baseline.

If full baseline drops sharply:

- Suspect full index/configuration issues before blaming the model.
- Check whether `items_ins_v2.json` and `items_shuffle_full.json` align with the same product universe.

## Notes For Final Report

The final narrative should avoid overstating early SFT failures.

Better framing:

> Naive SFT failed under small-index evaluation because the learned search strategy was distribution-mismatched. After switching to the full WebShop product index, we re-evaluate whether trajectory learning transfers under matched retrieval conditions.

This is a stronger and fairer research story than simply saying "SFT does not work."

## Review Note 2026-06-25 04:57

Current observed status from the shared server:

- `web_agent_site/utils.py` points to full data:
  - `DEFAULT_FILE_PATH = ../data/items_shuffle_full.json`
  - `DEFAULT_ATTR_PATH = ../data/items_ins_v2.json`
  - `DEBUG_PROD_SIZE = None`
- Full resources/index appear built:
  - `search_engine/resources/documents.jsonl`: 1,181,430 lines, 6.4G
  - `search_engine/indexes`: 3.4G
- Damaged chunks `items_shuffle_part_ac` and `items_shuffle_part_ag` now appear to be 500M each.
- A full comparison process is running:
  - command: `python -u src/eval/compare_models.py`
  - log: `/tmp/compare_full.log`

Important audit caveats:

1. The running `compare_models.py` uses `KEYWORD_PROMPT` for all systems. Therefore its `baseline` is not the original raw ReAct baseline; it is closer to a keyword-prompt baseline.
2. The environment reports `Loaded 11674685 goals`, which means the run is using synthetic goals from `items_ins_v2.json` rather than the previous 6,910 human goals. This is acceptable as a full-synthetic evaluation, but it should not be directly compared to earlier small/human-goal results without saying the goal set changed.
3. The script saves `results/model_comparison.json` only after all models complete. If the job is interrupted, partial per-model JSONL results may be lost. Future runs should stream per-episode records into model-specific JSONL files.
4. Current baseline result in the running full-synthetic run:
   - `success = 10/50 = 20%`
   - `avg_reward = 0.208`
   - `avg first-query length = 7.2 words`

Recommended action for the execution agent:

- Let the current run finish, and label it clearly as `full_synthetic_keyword_prompt_50`.
- Do not treat this as the raw baseline.
- After it finishes, save or copy the final output into `results/full_env_audit/model_comparison_full_synthetic_keyword_50.json`.
- Add a separate raw ReAct full run if time permits, using the original `SYSTEM_PROMPT + FEW_SHOT_EXAMPLE` from `react_agent.py`.
- Add a human-goal full run if the project narrative wants continuity with the earlier 50-goal human evaluation. That run should use human goals explicitly rather than the synthetic `items_ins_v2` goal pool.

## Review Note 2026-06-25 05:55

The full synthetic keyword-prompt comparison has finished.

Observed result file:

- `results/model_comparison.json`
- copied/audit file exists: `results/full_env_audit/model_comparison_full_synthetic_keyword_50.json`

Protocol label:

`full_synthetic_keyword_prompt_50`

This is not a raw ReAct baseline because `compare_models.py` uses `KEYWORD_PROMPT` for all models. It also uses the synthetic goal pool from `items_ins_v2.json` (`Loaded 11674685 goals`), not the earlier human-goal pool.

Summary:

| Model | Strong SR (`r > 0.5`) | Strict SR (`r >= 0.9`) | Avg Reward | Avg Steps | Avg First Query Words | Empty First Query | Zero Reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 10/50 = 20% | 0/50 = 0% | 0.2085 | 5.72 | 7.24 | 18/50 | 27/50 |
| expert_sft | 6/50 = 12% | 3/50 = 6% | 0.1100 | 6.76 | 24.28 | 2/50 | 43/50 |
| unified_sft | 4/50 = 8% | 0/50 = 0% | 0.0520 | 6.66 | 25.36 | 0/50 | 46/50 |

Interpretation:

1. Moving to the full index did not by itself rescue the current SFT adapters.
2. Both SFT adapters produce much longer first queries (24-25 words) than the baseline (7.2 words).
3. SFT models almost never produce empty first queries, but their zero-reward rate is worse. This suggests the problem is not merely empty action parsing; it is likely over-specific query generation, retrieval mismatch, or training/inference format mismatch.
4. The expert SFT has some exact/near-exact successes (`strict r >= 0.9`: 3/50), so it may have learned useful behavior in a few cases, but it is not robust.

Recommended next step:

- Do not start another SFT run yet.
- First run a diagnostic evaluation that saves full per-step trajectories for baseline, expert_sft, and unified_sft on the same 50 synthetic goals.
- Add fields per episode:
  - all actions,
  - number of searches,
  - repeated-search count,
  - whether each search returned results,
  - first search result count if available,
  - buy action occurred or not,
  - final reward.
- Then classify failures into:
  - zero-result long query,
  - clicked irrelevant product,
  - no buy / max steps,
  - premature buy,
  - action/click casing mismatch,
  - option selection failure.

Suggested follow-up comparisons:

1. `full_synthetic_raw_react_50`: original `ReActAgent` prompt, no keyword prompt.
2. `full_synthetic_keyword_50`: current keyword-prompt baseline, already done.
3. `full_synthetic_unified_sft_keyword_50`: current unified SFT result, already done.
4. If report continuity matters: `full_human_keyword_50`, using human goals explicitly.

Training advice:

- Before DPO/Bandit, inspect why long SFT queries fail on full. If failures are mostly zero-result long queries, DPO should target query preference pairs rather than whole-trajectory SFT.
- If failures are mostly max-step/no-buy after good retrieval, Bandit/controller may help.
- If failures are action-format/casing issues, fix decoding/action normalization before any model training.

## Decision 2026-06-25 06:05

Do not start new SFT/DPO/Bandit training yet.

Reason:

The first full-index comparison shows that existing SFT adapters still underperform the keyword-prompt baseline:

- baseline: 20% strong SR, avg reward 0.2085, first query 7.24 words
- expert_sft: 12% strong SR, avg reward 0.1100, first query 24.28 words
- unified_sft: 8% strong SR, avg reward 0.0520, first query 25.36 words

This means the current bottleneck is not solved merely by switching to the full index. The next decision-critical question is:

> Why do the SFT adapters fail on full index?

Execution agent must run diagnostics before any new training.

### Required Next Experiment: Failure Diagnosis 50

Run a diagnostic evaluator on the same 50 goals used in `model_comparison_full_synthetic_keyword_50.json`.

Models:

1. baseline with current keyword prompt
2. expert_sft with current keyword prompt
3. unified_sft with current keyword prompt

For every episode, save full per-step trajectory with:

- `goal`
- `instruction`
- `model_name`
- `reward`
- `success`
- `steps`
- every step's:
  - `observation_before` or truncated observation before action
  - `thought`
  - `action`
  - `reward_after_step`
  - `done`
  - `available_clickables`
  - whether action is `search`, `click`, or fallback parse
  - for search actions, approximate result count if extractable from observation after search
- episode-level:
  - `first_query`
  - `first_query_words`
  - `num_searches`
  - `num_clicks`
  - `num_buys`
  - `repeated_search_count`
  - `empty_query_count`
  - `zero_result_search_count`
  - `max_step_exhausted`

Output files:

- `results/full_env_audit/diagnostic_full_synthetic_keyword_50.jsonl`
- `results/full_env_audit/diagnostic_full_synthetic_keyword_50_summary.json`

The summary must include failure categories for each model:

- `zero_result_long_query`
- `clicked_irrelevant_product`
- `max_steps_no_buy`
- `premature_buy`
- `action_or_casing_error`
- `option_selection_failure`
- `other_unclear`

### Decision Rules After Diagnosis

If most SFT failures are `zero_result_long_query`:

- Next action: build query-level DPO/preference dataset.
- Do not do whole-trajectory SFT.
- Chosen queries should be shorter/high-reward baseline or successful SFT queries.
- Rejected queries should be long zero-result SFT queries.

If most failures are `max_steps_no_buy` after nonempty retrieval:

- Next action: Bandit/controller or buy-timing prompt intervention.
- DPO can target buy vs continue decisions.

If most failures are `action_or_casing_error`:

- Next action: fix action normalization/parser first.
- Training is not allowed until parser-normalization is stable.

If failures are mixed:

- Next action: run raw ReAct full baseline and human-goal full baseline before training.

### Secondary Required Experiment

After diagnosis, run:

- `full_synthetic_raw_react_50`

This uses the original `ReActAgent` prompt (`SYSTEM_PROMPT + FEW_SHOT_EXAMPLE`) rather than `KEYWORD_PROMPT`.

Purpose:

- Determine whether keyword prompting helps or hurts on full synthetic goals.
- Separate prompt effect from model/adaptor effect.

Output:

- `results/full_env_audit/baseline_full_synthetic_raw_react_50.jsonl`
- `results/full_env_audit/baseline_full_synthetic_raw_react_50_summary.json`


## Decision 2026-06-25 07:50

Decision owner: Review/Decision agent.

Execution status reviewed:

- `full_synthetic_raw_react_50` completed.
- `diagnostic_*_synthetic_keyword_50.jsonl` completed.
- `diagnostic_summary.json` completed.

### Decision Summary

Do **not** start SFT/DPO/Bandit training yet.

The next action is a controlled inference experiment, not training.

Reason:

The diagnostics show two distinct problems:

1. Prompt effect:
   - raw ReAct: 5/50 = 10%, avg reward 0.077, avg query length 15.8
   - keyword prompt baseline: 10/50 to 13/50 depending script, avg reward around 0.21-0.23, avg query length 7.2-7.5
   - Keyword prompting is practically useful, but with n=50 the raw-vs-keyword success-rate difference is not yet statistically strong.
     - raw 5/50 vs keyword 10/50: Fisher two-sided p ≈ 0.26
     - raw 5/50 vs diagnostic keyword 13/50: Fisher two-sided p ≈ 0.066

2. SFT failure mode:
   - `expert_sft`: long queries plus search/back loops.
   - `unified_sft`: not primarily zero-result retrieval; it repeatedly emits the same long search and rarely transitions to click/buy.
   - `unified_sft` failure distribution: `max_steps_no_buy` 41/50 = 82%, `zero_result_long_query` only 3/50 = 6%.

Interpretation:

- `unified_sft` may have learned instruction-to-query behavior, but lost multi-step policy control.
- Whole-trajectory SFT should not continue until we know whether simple inference-time action control fixes the no-buy loop.
- Bandit should also wait: first test whether a deterministic anti-loop controller is enough to expose latent model ability.

### Required Next Experiment: Anti-Loop Controller 50

Implement an evaluation-only wrapper. Do not train.

Run on the same full synthetic goals 0-49, max_steps=7.

Models:

1. `baseline_keyword`
2. `expert_sft_keyword`
3. `unified_sft_keyword`

Add a deterministic controller after model action generation:

1. Action normalization:
   - Match click actions to available clickables case-insensitively.
   - Normalize `buy now`, `Buy Now`, `buy` to the exact available clickable if present.
   - Normalize `back to search`, `< prev`, `next >`, `description`, `features`, `reviews` similarly.

2. Empty search guard:
   - If the generated action is `search[]` or `search[ ]`, replace it with a short fallback query.
   - Fallback query should use 3-6 informative words from the instruction, dropping filler words such as `find`, `me`, `with`, `and`, `price`, `lower`, `dollars`.

3. Repeated search guard:
   - If the same non-empty search query has already been used in the episode:
     - If the current available clickables include product ASIN-like ids, click the first unvisited ASIN.
     - Else replace with a broadened 3-6 word query.

4. Search-count guard:
   - If `num_searches >= 2` and product ASIN-like clickables are available, click the first unvisited ASIN instead of searching again.

5. Buy-timing guard:
   - If currently on a product page and `Buy Now` is available:
     - allow the model's `click[Buy Now]`;
     - if the model repeats search/back after at least one product has been viewed and no unvisited product clickables remain, force `click[Buy Now]`.
   - Keep this conservative; avoid forcing buy immediately after the first click unless the model is looping.

Save outputs:

- `results/full_env_audit/controller_full_synthetic_keyword_50.json`
- `results/full_env_audit/controller_baseline_keyword_50.jsonl`
- `results/full_env_audit/controller_expert_sft_keyword_50.jsonl`
- `results/full_env_audit/controller_unified_sft_keyword_50.jsonl`

Summary fields:

- strong SR
- strict SR
- avg reward
- avg steps
- avg query words
- zero-result search rate
- repeated-search count
- buy rate
- forced-action count by reason:
  - `empty_search_fallback`
  - `repeat_search_to_click`
  - `repeat_search_to_broaden`
  - `search_count_to_click`
  - `forced_buy_after_loop`
  - `action_normalized`

### Decision Rules After Anti-Loop Controller

If `unified_sft + controller` improves strongly, e.g. reaches or beats keyword baseline:

- Next decision: develop System C as a controller/Bandit problem.
- DPO can be secondary.
- Focus on buy/search/click transition control, not more SFT.

If `unified_sft + controller` still fails:

- Next decision: build query-level DPO dataset.
- DPO target:
  - chosen: short successful baseline queries or successful SFT queries.
  - rejected: long repeated SFT queries and zero-reward queries.
- Do not use whole-trajectory SFT.

If baseline also improves substantially with controller:

- Keep controller as `System A++` and use it as the new non-trained strong baseline.
- Any trained model must beat this stronger baseline, not the weaker raw baseline.

### Data Construction Task, But No Training Yet

In parallel or after controller evaluation, construct but do not train:

- `data/query_dpo_pairs_full_synthetic_50.jsonl`

Pair construction:

- For each goal where baseline succeeds and SFT fails:
  - chosen query = baseline first successful query or shortest query before successful purchase.
  - rejected query = SFT first query if long/repeated/zero-reward.
- For each goal where SFT succeeds:
  - keep its query as possible chosen example.

Fields:

- `goal_idx`
- `instruction`
- `chosen_query`
- `rejected_query`
- `chosen_source`
- `rejected_source`
- `baseline_reward`
- `sft_reward`
- `reason`

This file is for later DPO decision only. Do not start DPO training until the controller result is reviewed.

## Deadline Decision 2026-06-25

The course assignment is due today. Switch to deadline mode.

Decision:

- Do not start any new training today.
- Do not attempt additional large-scale 500-goal evaluations today unless all report materials are already complete.
- The only remaining experiment allowed today is the already-running `eval_anti_loop.py`.
- When `eval_anti_loop.py` finishes, use its result to decide the final narrative:
  - If controller improves SFT: present controller as the final System C direction/result.
  - If controller does not improve SFT: present controller/DPO as follow-up and use the negative SFT result as a key analysis finding.

Minimum final deliverable for today:

1. Full data/index verification:
   - full product file valid and loaded,
   - 1.18M resource docs,
   - 3.4G full index.
2. Results table:
   - small baseline,
   - full raw ReAct 50,
   - full keyword baseline 50,
   - full expert_sft 50,
   - full unified_sft 50,
   - anti-loop controller results if completed in time.
3. Analysis:
   - keyword prompt shortens query and improves practical performance,
   - current SFT adapters underperform,
   - unified_sft failure is mostly repeated/no-buy looping rather than simple zero-result retrieval,
   - naive whole-trajectory SFT is not recommended,
   - next research direction is targeted controller/Bandit or query/action-level DPO.

Execution agent:

- Finish the current anti-loop run.
- Save outputs to `results/full_env_audit/`.
- After that, stop experiments and help package/report if requested.

## Decision Update 2026-06-25

User has overridden deadline mode.

Return to the original experiment-driven workflow:

1. Finish `eval_anti_loop.py`.
2. Review controller results.
3. Decide between:
   - controller/Bandit path if anti-loop restores SFT performance,
   - targeted DPO path if SFT remains weak,
   - parser/action-normalization fixes if failures are mostly format/casing.
4. Do not stop merely because of the course deadline.
5. Continue preserving the full research chain:
   - raw ReAct,
   - keyword prompt,
   - SFT,
   - diagnosis,
   - controller/Bandit,
   - DPO if justified by diagnosis.

Current active decision:

The execution agent should continue with the anti-loop controller evaluation and wait for the next decision after results are available.

## Review And Decision 2026-06-25 09:05

Reviewed completed anti-loop/controller experiment.

### Results Reviewed

Files:

- `results/full_env_audit/anti_loop_comparison.json`
- `results/full_env_audit/baseline_full_synthetic_keyword_50.jsonl`
- `results/full_env_audit/baseline_with_controller_full_synthetic_keyword_50.jsonl`
- `results/full_env_audit/unified_sft_full_synthetic_keyword_50.jsonl`
- `results/full_env_audit/unified_sft_with_controller_full_synthetic_keyword_50.jsonl`

Summary:

| Config | Strong SR | Avg Reward | Interventions |
|---|---:|---:|---:|
| baseline keyword | 6/50 = 12% | 0.097 | - |
| baseline keyword + controller | 14/50 = 28% | 0.282 | 41 |
| unified_sft keyword | 1/50 = 2% | 0.012 | - |
| unified_sft keyword + controller | 24/50 = 48% | 0.446 | 114 |

Statistical note:

- `unified_sft + controller` vs `unified_sft`: Fisher two-sided p ≈ 5.1e-8, clearly meaningful.
- `unified_sft + controller` vs `baseline + controller`: Fisher two-sided p ≈ 0.063, promising but not yet conclusive at 0.05 with n=50.
- `baseline + controller` vs `baseline`: Fisher two-sided p ≈ 0.078, promising but also not conclusive at 0.05.

### Interpretation

This is the first genuinely strong positive result:

> SFT alone fails, but SFT + runtime policy controller becomes the best system so far.

The best current research interpretation is:

- `unified_sft` learned useful retrieval/query behavior but lost transition control.
- The controller supplies missing policy-level decisions: stop repeating search, click products, and buy before max steps.
- This supports the original project architecture: model-level fine-tuning + strategy controller.

However, the current controller is aggressive:

- It forces `click[Buy Now]` at `max_steps - 1` whenever Buy Now is available.
- It intervenes 114 times over 50 unified_sft episodes.

Therefore, do not yet claim "Bandit solved the task" or "SFT alone improved." The correct claim is:

> A deterministic anti-loop controller reveals latent value in the SFT model and improves full synthetic WebShop performance to 48% strong success on 50 goals.

### Important DPO Data Issue

The file `results/full_env_audit/query_dpo_pairs_full_synthetic_50.jsonl` currently appears malformed:

- `rejected_source` sometimes contains an entire diagnostic object rather than a source label.
- Several rejected examples appear to reuse diagnostic goal 0 metadata while paired with other goal indices.

Decision:

- Do not train DPO on this file.
- Rebuild DPO pairs later with strict per-goal alignment.

### Current Decision

System C should now be defined as:

> `unified_sft + deterministic anti-loop controller`

not yet as learned Bandit.

The controller is a valid lightweight strategy controller/prototype. It is adjacent to the proposed Bandit layer, but for honest reporting call it:

- "rule-based anti-loop controller"
- "deterministic strategy controller"
- "controller prototype"

### Required Next Experiment: Controller Ablation 50

Before expanding to 200 goals, run controller ablations on the same 50 goals to identify which rule causes the gain.

Use `unified_sft` only first, because it shows the largest gain.

Run these variants:

1. `unified_sft_no_controller`
   - already available, 2% SR.

2. `unified_sft_repeat_to_click_only`
   - enable only repeated-search-to-click.
   - disable forced buy.
   - disable max search count to click unless needed by implementation.

3. `unified_sft_no_forced_buy`
   - enable repeat-search-to-click and search-count-to-click.
   - disable forced buy.

4. `unified_sft_conservative_forced_buy`
   - force buy only if:
     - current page has Buy Now,
     - at least one product has been clicked/viewed,
     - step >= 6,
     - the previous action was not already `click[Buy Now]`,
     - there are no unvisited product ASINs visible.

5. `unified_sft_current_controller`
   - already available, 48% SR.

Output:

- `results/full_env_audit/controller_ablation_unified_sft_50.json`
- one JSONL per variant.

Metrics:

- strong SR
- strict SR
- avg reward
- avg steps
- buy rate
- intervention counts by rule
- number of successful episodes involving forced buy
- average reward for forced-buy episodes

### Decision Rules After Controller Ablation

If `no_forced_buy` or `conservative_forced_buy` remains close to current 48%:

- Use the conservative controller as final System C.
- Then run 200-goal evaluation for:
  - baseline keyword,
  - baseline keyword + controller,
  - unified_sft,
  - unified_sft + selected controller.

If only aggressive forced buy works:

- Keep current result as an analysis finding, but report it cautiously.
- Do not overstate it as robust final method.
- Then run a smaller human-inspection/case-study analysis of successful forced-buy episodes.

If repeat-to-click alone gives most of the gain:

- The key contribution is loop-breaking transition control.
- Bandit can be simplified to choosing `search vs click`, not full 4-arm policy.

### DPO Decision

DPO remains postponed until after controller ablation.

Reason:

The strongest current failure is policy transition control, and controller already fixes it. DPO should be targeted only after we know whether the transition policy can be represented as preference pairs.

Before any DPO training:

- Rebuild `query_dpo_pairs_full_synthetic_50.jsonl` with strict goal alignment.
- Add action-level DPO pairs, not only query-level pairs:
  - chosen: `click[ASIN]` after repeated search when results exist.
  - rejected: repeated same `search[...]`.
  - chosen: `click[Buy Now]` after verified product page loop.
  - rejected: search/back loop from same state.

## Review And Decision 2026-06-25 10:30

Reviewed controller ablation.

Files:

- `results/full_env_audit/controller_ablation_unified_sft_50.json`
- `results/full_env_audit/controller_ablation_unified_sft_repeat_to_click_only_50.jsonl`
- `results/full_env_audit/controller_ablation_unified_sft_no_forced_buy_50.jsonl`
- `results/full_env_audit/controller_ablation_unified_sft_conservative_forced_buy_50.jsonl`

Results:

| Config | Strong SR | Avg Reward | Interventions |
|---|---:|---:|---:|
| no_controller | 2% | 0.012 | 0 |
| repeat_to_click_only | 22% | 0.182 | 74 |
| no_forced_buy | 22% | 0.192 | 81 |
| conservative_forced_buy | 48% | 0.419 | 106 |
| current_controller | 48% | 0.446 | 114 |

Interpretation:

- The first major gain is from `repeat_to_click`: it raises unified_sft from 2% to 22%.
- The second major gain requires a buy-timing rule: without forced buy, performance stays at 22%; with conservative forced buy, it reaches 48%.
- The conservative controller matches current/aggressive controller success rate while using fewer interventions.

Decision:

Use `unified_sft + conservative anti-loop controller` as the final System C candidate.

This is now the selected controller for larger evaluation.

Do not use the aggressive/current controller as the main reported method, except as an ablation.

### Required Next Experiment: 200-Goal Main Evaluation

Run a 200-goal evaluation on full synthetic goals.

Use the same split for all systems:

- start_idx = 0
- num_goals = 200
- max_steps = 7
- full product index (`num_products=None`)

Systems:

1. `raw_react`
   - original ReAct prompt.
2. `keyword_baseline`
   - base model + keyword prompt.
3. `keyword_baseline_conservative_controller`
   - base model + keyword prompt + conservative controller.
4. `unified_sft`
   - unified SFT adapter + keyword prompt.
5. `unified_sft_conservative_controller`
   - unified SFT adapter + keyword prompt + conservative controller.

Optional only if time/resources allow:

6. `expert_sft`
7. `expert_sft_conservative_controller`

Output:

- `results/full_env_audit/main_eval_full_synthetic_200.json`
- one JSONL per system, e.g.
  - `main_raw_react_full_synthetic_200.jsonl`
  - `main_keyword_baseline_full_synthetic_200.jsonl`
  - `main_keyword_baseline_controller_full_synthetic_200.jsonl`
  - `main_unified_sft_full_synthetic_200.jsonl`
  - `main_unified_sft_controller_full_synthetic_200.jsonl`

Metrics:

- strong SR (`reward > 0.5`)
- strict SR (`reward >= 0.9`)
- avg reward
- avg steps
- buy rate
- avg first query words
- zero reward rate
- intervention counts by rule for controller systems

### DPO Path Decision

DPO is still part of the project chain, but it is no longer the immediate next training step.

Reason:

The controller result is strong enough that the next priority is robust evaluation of System C. DPO should be positioned as the next improvement stage after System C validation.

Execution agent should rebuild DPO pairs only after starting or completing the 200-goal main evaluation.

Required DPO data fix:

- Discard the current malformed `query_dpo_pairs_full_synthetic_50.jsonl`.
- Rebuild:
  - `data/query_dpo_pairs_full_synthetic_aligned.jsonl`
  - `data/action_dpo_pairs_full_synthetic_aligned.jsonl`
- Ensure every chosen/rejected pair uses the same `goal_idx` and same instruction.
- No full diagnostic object should appear inside `chosen_source` or `rejected_source`; source fields must be strings.

Do not train DPO until the 200-goal main evaluation is reviewed.

## Decision Update 2026-06-25 10:45

User reports limited time and wants to move forward instead of running another validation.

Override the 200-goal main evaluation requirement for now.

Current decision:

- Do not run 200-goal evaluation now.
- Treat the 50-goal full synthetic controller ablation as the main experimental result.
- Move forward to final packaging/reporting and minimal DPO-chain completion.

### Final Result To Report For Current Submission

Main system progression:

1. raw ReAct:
   - 10% strong SR on full synthetic 50.
2. keyword prompt baseline:
   - around 20-26% strong SR depending script/protocol.
   - shorter queries than raw ReAct.
3. unified_sft alone:
   - weak alone, 2-10% depending run.
   - diagnosis: repeated search / max-step no-buy loop.
4. unified_sft + conservative anti-loop controller:
   - 48% strong SR on full synthetic 50.
   - best current result.

Main claim:

> SFT alone underperforms because it loses transition control, but SFT combined with a conservative strategy controller substantially improves WebShop performance.

### Immediate Execution Tasks

Execution agent should now stop large evaluations and do only these small tasks:

1. Create a clean final summary file:
   - `results/full_env_audit/final_results_summary.json`
   - include all key 50-goal results and protocol notes.

2. Rebuild DPO pair files only if quick:
   - `data/query_dpo_pairs_full_synthetic_aligned.jsonl`
   - `data/action_dpo_pairs_full_synthetic_aligned.jsonl`
   - no training required.

3. If rebuilding DPO pairs is not quick:
   - write `data/DPO_DATASET_TODO.md` describing how pairs should be built.

4. Create final report support notes:
   - `results/full_env_audit/final_interpretation_notes.md`
   - include:
     - what worked,
     - what failed,
     - why SFT alone failed,
     - why controller helped,
     - limitations,
     - next steps.

### Reporting Language

Use cautious wording:

- "On a 50-goal full synthetic evaluation..."
- "The controller result is promising and should be validated on a larger split."
- "DPO is prepared as the next stage, but not yet trained/evaluated."
- "The controller is deterministic/rule-based, not a learned Bandit yet."

Do not claim:

- "Solved WebShop."
- "Bandit completed."
- "DPO improved performance."
- "SFT alone improved performance."

## Decision Update 2026-06-25 11:05 — Complete Minimal DPO

User wants DPO completed, not only listed as future work.

Decision:

Run a minimal, targeted DPO pipeline. Do not use the existing malformed DPO pair file, and do not use the old `src/training/train_dpo.py` unchanged.

Rationale:

- The existing old DPO script builds random/similarity-proxy pairs from earlier small-environment trajectories. It does not target the current full synthetic failure modes.
- The existing `query_dpo_pairs_full_synthetic_50.jsonl` has alignment/format issues and must not be used for training.
- Current diagnosis says DPO should target two concrete preference types:
  1. repeated long search vs click/product transition;
  2. loop/no-buy vs buy decision when on product page.

### DPO Completion Standard

For this project, DPO is considered completed if all three are done:

1. aligned DPO preference datasets are built;
2. a small DPO LoRA adapter is trained successfully;
3. the DPO adapter is evaluated on the same 50-goal full synthetic split.

It does not need to beat the controller. If it underperforms, report it honestly as a negative/early-stage result.

### Required DPO Data

Build two aligned files:

- `data/query_dpo_pairs_full_synthetic_aligned.jsonl`
- `data/action_dpo_pairs_full_synthetic_aligned.jsonl`

Rules:

- chosen/rejected must use the same `goal_idx` and same `instruction`.
- source fields must be short strings, not full objects.
- Every record must have:
  - `goal_idx`
  - `instruction`
  - `prompt`
  - `chosen`
  - `rejected`
  - `pair_type`
  - `chosen_source`
  - `rejected_source`
  - `reason`

Query-level examples:

- chosen: successful/shorter baseline query or successful SFT query.
- rejected: repeated long SFT query that led to zero/low reward.

Action-level examples:

- chosen: `click[ASIN]` when SFT repeats the same search while product ASINs are visible.
- rejected: repeated same `search[...]`.
- chosen: `click[Buy Now]` when SFT loops on a product page and `Buy Now` is available.
- rejected: repeated search/back/option loop.

### Training Instruction

Create a new script rather than mutating the old one if easier:

- `src/training/train_targeted_dpo.py`

Training target:

- Base model: Qwen2.5-7B-Instruct.
- Optional starting point: use `unified_sft_adapters` as the base policy if implementation is easy; otherwise train DPO LoRA from base model and report that it is DPO-only.
- Output adapter:
  - `models/dpo_targeted_adapters`

Recommended small settings:

- max pairs: use all aligned pairs if <= 300; otherwise cap at 300.
- epochs: 1
- LoRA r: 8 or 16
- learning_rate: 5e-5 to 1e-4
- beta: 0.1
- max_prompt_length: 512
- max_length: 768 or 1024

Do not spend time making it perfect. The goal is a completed DPO pipeline, not a guaranteed improvement.

### DPO Evaluation

Evaluate on the same full synthetic goals 0-49, max_steps=7.

Run at least:

1. `dpo_targeted`
2. `dpo_targeted + conservative_controller` if adapter can be loaded in the same controller wrapper.

Save:

- `results/full_env_audit/dpo_targeted_full_synthetic_50.jsonl`
- `results/full_env_audit/dpo_targeted_controller_full_synthetic_50.jsonl` if run
- `results/full_env_audit/dpo_targeted_summary.json`

Metrics:

- strong SR
- avg reward
- avg steps
- avg first query words
- zero reward rate
- for controller variant: intervention count

### Reporting Rule

If DPO improves:

- Report it as preliminary evidence that targeted preferences can improve the agent.

If DPO does not improve:

- Report it as a completed but inconclusive DPO attempt.
- Main positive result remains `unified_sft + conservative controller`.
- Explain likely reasons:
  - too few preference pairs,
  - DPO pairs built from only 50 goals,
  - action-level DPO requires richer state context than query-only pairs.

## Engineering Speed Optimization Notes 2026-06-25

The current bottleneck is mostly engineering overhead, not only GPU generation.

Observed slow points:

1. Full WebShop environment initialization repeatedly loads:
   - 1.18M products,
   - full attributes,
   - synthetic goal pool with >11M goals.
2. Scripts repeatedly load Qwen2.5-7B checkpoints.
3. Separate scripts re-run the same full environment setup for compare/diagnose/controller/DPO evaluation.

### High-Priority Speed Decisions

Execution agent should prioritize these optimizations for any remaining runs:

1. Use one unified runner when possible.
   - Initialize full env once.
   - Load goals once.
   - Run multiple configs in the same process.
   - Stream each episode result to JSONL immediately.

2. Avoid regenerating the full 11M synthetic goal pool for every script.
   - Create a fixed small evaluation subset file, e.g.:
     - `data/eval_goals_full_synthetic_50.json`
     - `data/eval_goals_full_synthetic_200.json`
   - Future evaluation scripts should load this subset directly if possible.
   - If modifying WebShop env is expensive, at least reuse one process rather than many scripts.

3. Do not reload the same model for every variant.
   - For a given model, run no-controller and controller variants before unloading.
   - Avoid separate scripts for baseline/controller/diagnostic when one script can run all.

4. Reduce generation budget.
   - Use `max_new_tokens=64` for Thought+Action.
   - If output remains valid, try `48`.
   - Keep observation truncation around 1500-2000 chars.
   - Keep history to last 2 turns unless a task specifically needs more.

5. Always support resume.
   - Write JSONL after every episode.
   - On startup, skip completed `goal_idx` values.
   - This avoids losing time after interruption or GPU reset.

6. Keep final validation sizes modest.
   - Use 50 for ablations.
   - Use 100-200 for final confirmation if time allows.
   - Do not run 500+ goals unless the report is already complete.

### Speed Tasks If Time Allows

Optional helper script:

- `src/eval/main_eval_fast.py`

Suggested behavior:

- one env initialization,
- fixed goal subset,
- sequential configs,
- model reuse by grouping configs by model,
- JSONL streaming,
- resume support.

This is optional. Do not delay DPO completion/reporting just to build the perfect runner.

## Decision Update 2026-06-25 10:45 — Let main_eval_fast Finish Before DPO

Execution agent is currently running:

- `src/eval/main_eval_fast.py`

This is the correct next step. Do not interrupt it.

Purpose:

- Run a standardized 50-goal evaluation under one protocol.
- Avoid repeated full environment/model loading.
- Compare the five key configurations fairly:
  1. `baseline_keyword`
  2. `baseline_keyword_ctrl`
  3. `expert_sft`
  4. `unified_sft`
  5. `unified_sft_ctrl`

Decision:

- Do not start DPO until `main_eval_fast.py` completes.
- DPO depends on this standardized result:
  - If `unified_sft_ctrl` remains the strongest, report it as System C and treat DPO as optional follow-up/prototype.
  - If SFT models remain weak without controller, targeted DPO can focus on transition/action preferences.
  - If expert SFT performs unexpectedly well, DPO pair construction should be adjusted to use expert successes as chosen examples.

Allowed parallel work while evaluation runs:

- CPU-only preparation:
  - inspect existing DPO pair files,
  - draft DPO dataset schema,
  - write report text.

Not allowed while evaluation runs:

- starting DPO training,
- starting another large evaluation,
- modifying the running evaluation script.








## Web-Informed Decision Addendum 2026-06-25

Decision after web/literature check: keep the current decision.

Do not start DPO/SFT/Bandit training before the anti-loop/controller evaluation.

External grounding:

- WebShop is explicitly designed around 1.18M real products and ~12k crowd-sourced instructions. The original benchmark emphasizes exactly the bottlenecks we observe here: compositional instruction understanding, query reformulation, noisy webpage text, and strategic exploration. Source: WebShop paper, https://arxiv.org/abs/2207.01206
- ReAct shows that interleaving reasoning and actions helps interactive decision-making tasks including WebShop; this supports keeping a strong prompt/controller baseline rather than jumping directly to weight updates. Source: ReAct paper, https://arxiv.org/abs/2210.03629
- DPO is attractive because it avoids a separate reward model/RL loop and uses pairwise preferences, but it still requires well-formed chosen/rejected pairs. Our current diagnosis says we need to know whether the rejected behavior is bad query generation, repeated search, no-buy looping, or action normalization before training. Source: DPO paper, https://arxiv.org/abs/2305.18290
- Recent environment preference optimization work supports using environment feedback as preference signal for long-horizon agents, but the same logic implies preference construction should be grounded in concrete environment failure modes, not undifferentiated whole trajectories. Source: EPO paper, https://arxiv.org/abs/2408.16090

Implication for this project:

The next experiment remains:

1. Run anti-loop/controller evaluation.
2. Construct query/action preference pairs only after seeing whether controller fixes `unified_sft`.
3. If controller fixes `unified_sft`, prioritize controller/Bandit as System C.
4. If controller does not fix it, use diagnosis to build targeted DPO:
   - query-level DPO for long/repeated bad queries;
   - action-level DPO for search-vs-click-vs-buy transition mistakes;
   - avoid another generic whole-trajectory SFT run.
