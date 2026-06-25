# Fine-Tuned and Controller-Guided LLM Agent for WebShop

## Abstract

We study language-model agents for WebShop, an interactive e-commerce environment requiring natural-language instruction understanding, search query reformulation, product comparison, and purchase decisions. Starting from a ReAct-style Qwen2.5-7B-Instruct baseline, we evaluate prompt-level query control, supervised fine-tuning (SFT), failure diagnosis, and a lightweight anti-loop strategy controller. Raw ReAct suffers from verbose search queries and reaches 10% strong success on a 50-goal full synthetic evaluation. A keyword-only prompt improves the baseline to 20%. Naive SFT alone underperforms, mainly because the model repeatedly searches and fails to buy. After adding a conservative anti-loop controller, the unified SFT model reaches 48% strong success, the best result in our experiments. Our analysis suggests that SFT can learn useful retrieval behavior, but interactive shopping agents still require explicit transition control.

## 1 Introduction

Web shopping is a realistic language-agent task: an agent must interpret user constraints, issue search queries, inspect product pages, select options, and decide when to buy. This project investigates whether a small, reproducible LLM agent can improve over a ReAct baseline in WebShop.

Our original plan followed a three-stage architecture: ReAct baseline, SFT, and reward-aware strategy control. During experimentation we found that naive SFT alone did not improve performance. Instead of treating this as a dead end, we performed failure diagnosis and found that the SFT model often entered repeated-search/no-buy loops. This motivated a deterministic anti-loop controller as a lightweight strategy layer.

## 2 Related Work

WebShop provides a grounded web-shopping benchmark with 1.18M products and natural-language shopping goals. ReAct-style agents interleave reasoning and acting, making them suitable for interactive environments. Preference optimization methods such as DPO are relevant for future work, but they require carefully aligned chosen/rejected action pairs. In our experiments, controller-based transition control became the most effective intervention.

## 3 Task and Dataset

We use WebShop full mode with 1.18M products and a full Lucene search index. The agent receives a shopping instruction and interacts with a text-mode environment using actions such as `search[keywords]`, `click[ASIN]`, and `click[Buy Now]`.

The main evaluation in this report uses synthetic goals 0-49, max 7 steps per episode. We report average reward and strong success rate, where strong success is defined as reward greater than 0.5.

## 4 Methods

### 4.1 Raw ReAct Agent

The raw baseline uses Qwen2.5-7B-Instruct with a ReAct prompt. At each step the model outputs a short thought and a single WebShop action.

### 4.2 Keyword Prompt

Early failures showed that verbose search queries often hurt retrieval. We therefore introduced a keyword-only prompt that asks the model to use short search queries with 3-6 important words.

### 4.3 Supervised Fine-Tuning

We trained LoRA adapters from WebShop-style trajectories. The unified SFT data combines cleaned trajectories and query examples. The goal was to teach the model better search and action patterns.

### 4.4 Anti-Loop Controller

Diagnosis showed that the SFT model often repeated the same search and failed to transition to click or buy. We implemented a deterministic controller with three main rules:

- repeated search -> click a visible product;
- too many searches -> click a visible product;
- conservative forced buy near the step limit when already on a product page.

This controller is not a learned Bandit. It is a rule-based strategy-controller prototype.

### 4.5 DPO Preparation

We identified DPO as a natural next stage, especially for action-level preferences such as repeated search versus click, or no-buy loop versus buy. However, DPO was not used as the main reported performance result.

## 5 Experiments and Results

### 5.1 Main Results

| System | Strong Success | Avg Reward | Notes |
|---|---:|---:|---|
| Raw ReAct | 10% | 0.077 | Original prompt |
| Keyword baseline | 20% | 0.209 | Short search prompt |
| Unified SFT | 2% | 0.012 | Repeated search/no-buy |
| Unified SFT + conservative controller | 48% | 0.419 | Best result |

### 5.2 Controller Ablation

| Variant | Strong Success | Avg Reward |
|---|---:|---:|
| no controller | 2% | 0.012 |
| repeat-to-click only | 22% | 0.182 |
| no forced buy | 22% | 0.192 |
| conservative forced buy | 48% | 0.419 |
| aggressive controller | 48% | 0.446 |

The repeat-to-click rule provides the first major gain, raising success from 2% to 22%. Conservative buy timing is required to reach 48%, matching the aggressive controller while using fewer interventions.

## 6 Analysis

The keyword prompt improves raw ReAct by shortening first queries. However, SFT alone fails: although it can generate more specific queries, it loses transition control and often searches repeatedly. The controller fixes this by forcing progress from search to click and from product inspection to purchase.

This supports our original hypothesis that WebShop agents benefit from a two-layer design: a language model for textual understanding and a strategy controller for interaction policy.

## 7 Limitations

The main evaluation uses only 50 synthetic goals, so the results should be validated on a larger split. The controller is deterministic rather than learned, so it should be interpreted as a prototype, not a completed Bandit algorithm. DPO was prepared as a next-stage method but not shown to improve performance in this report.

## 8 Conclusion

Naive SFT alone does not solve WebShop. In our experiments, it weakens multi-step transition control and causes repeated-search/no-buy loops. However, when paired with a conservative anti-loop controller, the SFT agent reaches 48% strong success on a 50-goal full synthetic evaluation. The key lesson is that fine-tuned language agents need explicit strategy control for interactive web-shopping tasks.

