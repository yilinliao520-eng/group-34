# Final Interpretation Notes

## What Worked

1. **ReAct prompt format**: Qwen2.5-7B generates valid actions (0 parse failures in 416 steps), proving ReAct is well-internalized.

2. **Keyword-only prompt**: Adding use SHORT keyword queries with 3-6 words doubles success rate (10% → 20%) by fixing the #1 bottleneck: overly verbose search queries.

3. **Anti-loop controller**: A simple rule-based controller (repeat-search-to-click + forced-buy) boosts unified_sft from 2% to 48% SR on 50 goals. The controller fixes a transition-control deficiency that SFT alone cannot solve.

4. **SFT model's search quality**: unified_sft has only 6% zero-result search rate vs 32% for baseline, meaning it learned better search behavior from expert trajectories. But it lost the ability to decide when to buy.

## What Failed

1. **Naive SFT (4 attempts)**: All failed because training data (IL trajectories from 1.18M products) was evaluated in mismatched environments, and later because SFT removes transition-control skills.

2. **DPO not attempted**: Insufficient high-quality paired data (only 11 pairs from 50-goal diagnostic).

3. **Anti-loop controller is deterministic**: Not a learned Bandit yet. The 48% result is promising but the controller cannot adapt or learn from new experience.

## Why SFT Alone Failed

- SFT teaches the model to mimic expert search queries, but it does NOT teach when to stop searching and buy.
- Diagnostics show 82% max_steps_no_buy for unified_sft (only 22% for baseline).
- SFT models generate 25-word queries that rarely return empty results (good search quality) but they never decide to buy (bad transition control).

## Why Controller Helped

- Controller forces the model to break out of search loops (repeat_to_click: +20pp alone).
- Forced buy at max_steps-1 forces a purchase even if the model didn't reach a confident decision.
- Conservative forced buy matches aggressive forced buy (48% vs 48%) with fewer interventions (106 vs 114).

## Limitations

1. Only 50-goal evaluation on synthetic goals, not human goals.
2. Controller is deterministic, not adaptive.
3. No DPO or learned preference optimization yet.
4. Full WebShop benchmark uses 500 test goals; 50 may have variance.
5. Baseline results on full environment (20%) differ from small (28%) - evaluation protocol change, not directly comparable.

## Next Steps

1. Validate controller on 200+ goals for statistical significance.
2. Construct action-level DPO pairs (click vs repeated search, buy vs no-buy).
3. Replace deterministic controller with learned Bandit (Thompson Sampling, 4 arms).
4. Run final comparison: System A (keyword baseline) vs System B (SFT) vs System C (SFT + controller/Bandit).
