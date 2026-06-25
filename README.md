# Group 34 Final Project

**Project:** Fine-Tuned and Controller-Guided LLM Agent for WebShop  
**Course:** CS40008.01 / LLM 2026  
**Group:** 34

## Overview

This project investigates LLM-based agent strategies for the WebShop e-commerce simulation environment. We implement a ReAct-style Qwen2.5-7B-Instruct agent and evaluate three key hypotheses: (1) prompt-level query constraints improve retrieval effectiveness, (2) supervised fine-tuning (SFT) on expert trajectories enhances agent capabilities, and (3) a lightweight anti-loop strategy controller mitigates repetitive search and inaction failure modes.

Our primary finding is that SFT alone degrades agent performance due to loss of multi-step transition control. However, combining the SFT model with a conservative anti-loop strategy controller yields the strongest result in our evaluation framework.

## Main Results

Evaluation configuration:
- WebShop full product index: 1.18M products
- Evaluation split: synthetic goals 0–49
- Maximum steps per episode: 7
- Primary metric: strong success rate (reward > 0.5)

| System | Strong SR | Avg Reward | Notes |
|---|---:|---:|---|
| Raw ReAct | 10% | 0.077 | Original ReAct prompt baseline |
| Keyword Baseline | 20% | 0.209 | ReAct + short keyword query prompt |
| Unified SFT | 2% | 0.012 | Fine-tuned on 14,544 trajectories |
| Unified SFT + Controller | **48%** | **0.419** | SFT + conservative anti-loop controller |

Controller Ablation Study:

| Controller Variant | Strong SR | Avg Reward |
|---|---:|---:|
| No controller | 2% | 0.012 |
| Repeat-to-click only | 22% | 0.182 |
| No forced buy | 22% | 0.192 |
| Conservative forced buy | **48%** | **0.419** |
| Aggressive controller | 48% | 0.446 |

## Repository Layout

```text
src/
  agent/          ReAct agent implementation and anti-loop controller
  eval/           Evaluation scripts: comparison, diagnosis, and ablation
  training/       SFT, data cleaning, and DPO preparation scripts
results/
  full_env_audit/ Main 50-goal evaluation results and summaries
report/
  group-34.pdf    Final project report (PDF)
  main.tex        LaTeX source for the final report
  acl.sty         ACL-style formatting
  figures/        Report figures in PDF/SVG formats
data/
  eval_goals_full_synthetic_50.json  Fixed 50-goal evaluation subset
requirements.txt  Python dependencies
```

## Large Files Not Included

The following files are excluded from the repository to meet size constraints:
- Qwen2.5-7B-Instruct model weights (~15 GB)
- LoRA adapter checkpoint files (~300 MB)
- WebShop full product JSON files (~5.5 GB)
- Lucene search indexes (~3.5 GB)

These artifacts reside on the course GPU server and can be reproduced by running the scripts in `src/training/`.

## Reproduction Notes

All experiments require:
- WebShop environment (`web_agent_site`, `search_engine/`)
- Qwen2.5-7B-Instruct model weights
- Python 3.8+ with PyTorch, Transformers, PEFT, spaCy, and Pyserini

Key commands:
```bash
python src/agent/smoke_test_full.py     # Smoke test
python src/eval/main_eval_fast.py       # Unified evaluation
python src/eval/diagnose.py             # Failure diagnosis
python src/eval/eval_anti_loop.py       # Controller evaluation
python src/eval/eval_ablation_controller.py  # Controller ablation
```

Scripts contain absolute server paths. To reproduce on a different machine, update `MODEL_PATH`, `WEBSHOP_PATH`, and output directory variables.

## Method Summary

1. Raw ReAct baseline: Qwen2.5-7B-Instruct generates Thought and Action in a ReAct loop.
2. Keyword prompt: constrains search actions to short keyword queries (3–6 words).
3. Supervised fine-tuning: trains LoRA adapters on WebShop expert trajectories and query-pair data.
4. Failure diagnosis: classifies failures as zero-result search, repeated search, max-step no-buy, wrong product, and premature buy.
5. Anti-loop controller: deterministic strategy controller that breaks repeated search loops, triggers product clicks, and forces buy decisions near the step limit.
6. DPO preparation: preference-learning plan for query-level and action-level corrections (not yet executed).

## Limitations

- Primary evaluation uses 50 synthetic goals; the standard 500-goal benchmark split has not been evaluated.
- The anti-loop controller is deterministic and rule-based; future work should incorporate a learned Thompson Sampling Bandit controller.
- DPO data has been constructed but not yet trained or evaluated.
- All results are reported under cautious wording: "On a 50-goal full synthetic evaluation..."

## Conclusion

Naive supervised fine-tuning on WebShop trajectories can degrade agent performance by disrupting multi-step transition control. A conservative strategy controller recovers and amplifies the useful search behaviors learned through SFT, producing the strongest result (48% strong success rate) in our evaluation framework.
