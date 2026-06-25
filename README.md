# Group 34 Final Project

**Project:** Fine-Tuned and Controller-Guided LLM Agent for WebShop  
**Course:** CS40008.01 / LLM 2026  
**Group:** 34

## Overview

This project studies LLM-based agents for the WebShop shopping environment. We evaluate a ReAct-style Qwen2.5-7B-Instruct agent, prompt-level query control, supervised fine-tuning (SFT), failure diagnostics, and a lightweight anti-loop strategy controller.

The main finding is that naive SFT alone does not improve the agent: it often repeats search actions and fails to buy. However, combining the SFT model with a conservative anti-loop controller substantially improves performance on a 50-goal full synthetic WebShop evaluation.

## Main Result

Evaluation setting:

- WebShop full product index: 1.18M products
- Test split: synthetic goals 0-49
- Max steps: 7
- Strong success: `reward > 0.5`

| System | Strong SR | Avg Reward | Notes |
|---|---:|---:|---|
| Raw ReAct | 10% | 0.077 | Original ReAct prompt |
| Keyword baseline | 20% | 0.209 | Short keyword query prompt |
| Unified SFT | 2% | 0.012 | Repeated search / no-buy loop |
| Unified SFT + conservative controller | 48% | 0.419 | Best current result |

Controller ablation:

| Controller Variant | Strong SR | Avg Reward |
|---|---:|---:|
| no controller | 2% | 0.012 |
| repeat-to-click only | 22% | 0.182 |
| no forced buy | 22% | 0.192 |
| conservative forced buy | 48% | 0.419 |
| aggressive controller | 48% | 0.446 |

## Repository Layout

```text
src/
  agent/          ReAct agent and anti-loop controller
  eval/           Evaluation, diagnosis, and controller ablation scripts
  training/       SFT, data cleaning, and DPO preparation scripts
results/
  full_env_audit/ Main 50-goal evaluation results and summaries
report/
  final_report_group34_zh.tex  Chinese final report source for Overleaf/ACL
  references.bib              Report bibliography
  figures/                    Report figures in PDF/SVG formats
  README_OVERLEAF.md          Overleaf compilation notes
data/
  DPO_DATASET_TODO.md      Targeted DPO dataset construction plan
docs/
  AGENT_REVIEW_BOARD.md    Experiment decisions and audit notes
```

## Large Files Not Included

The GitHub/course code submission intentionally excludes:

- Qwen2.5-7B base model weights
- LoRA adapter checkpoint files
- WebShop full product JSON files
- Lucene indexes

These files were stored and run on the course server under:

```text
/inspire/hdd/project/fdu-aidake-cfff/public/liaoyilin/final_project/
```

The final course submission directory is expected to contain source code and lightweight result artifacts, not multi-GB data/model files. The report source is included for reproducibility; the final PDF can be compiled from `report/final_report_group34_zh.tex` in Overleaf using the ACL template.

## Reproduction Notes

The original experiments used two server environments:

- `webshop`: WebShop environment, Pyserini, Gym, and evaluation
- LLM/GPU environment for Qwen2.5-7B inference and LoRA adapters

Typical commands:

```bash
# Full-environment smoke test
python src/agent/smoke_test_full.py

# Unified 50-goal evaluation runner
python src/eval/main_eval_fast.py

# Failure diagnosis
python src/eval/diagnose.py

# Anti-loop controller evaluation
python src/eval/eval_anti_loop.py

# Controller ablation
python src/eval/eval_ablation_controller.py
```

Most scripts contain absolute server paths from the experiment environment. To run elsewhere, update:

- `MODEL_PATH`
- `WEBSHOP_PATH`
- output directories

## Method Summary

1. **Raw ReAct baseline:** Qwen2.5-7B-Instruct generates `Thought` and `Action`.
2. **Keyword prompt:** constrains search actions to short keyword queries.
3. **SFT:** trains LoRA adapters from WebShop-style trajectories.
4. **Diagnosis:** categorizes failures into zero-result search, repeated search, max-step no-buy, wrong product, and premature buy.
5. **Anti-loop controller:** deterministic strategy controller that breaks repeated search loops, clicks products, and conservatively buys near the step limit.
6. **DPO preparation:** preference-learning plan for query-level and action-level corrections; DPO training was not used for the main positive result.

## Limitations

- Main evaluation uses 50 full synthetic goals, not the original 500-goal benchmark split.
- The controller is deterministic and rule-based, not a learned Bandit yet.
- DPO is prepared as a next-stage method but not reported as a performance-improving result.
- Results should be validated on a larger split before making benchmark-level claims.

## Key Takeaway

Naive SFT alone can harm a WebShop LLM agent by weakening multi-step transition control. A conservative strategy controller can recover and amplify useful search behavior learned by SFT, producing the strongest result in this project.
