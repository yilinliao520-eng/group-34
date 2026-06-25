# Full Environment Audit Results

All results below are from the full WebShop environment (1.18M products) with synthetic goal split (goals 0-49, max_steps=7, keyword-only prompt unless noted).

## Summary

-  — Consolidated results with protocol metadata
-  — Interpretation, limitations, and next steps

## Model Comparison

-  — Baseline vs unified_sft, 50 goals
-  — Raw ReAct baseline (no keyword prompt)
-  — Keyword prompt baseline
-  — Unified SFT model

## Controller Evaluation

-  — Anti-loop controller vs baseline and SFT
-  — Baseline keyword, 50 goals
-  — Baseline + controller

## Controller Ablation

-  — Full ablation summary
-  — Conservative forced buy variant
-  — No forced buy variant
-  — Repeat-to-click only variant

## Failure Diagnostics

-  — Failure type distribution
-  — Baseline step-by-step diagnostics
-  — Expert SFT diagnostics
-  — Unified SFT diagnostics

## DPO Preparation

-  — Query-level preference pairs (11 pairs, not yet trained)
