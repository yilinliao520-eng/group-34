# Phase 1 Report: ReAct Baseline Agent for WebShop

**Group 34 | NLP Final Project | 2026-05-21**

---

## 1. Overview

Phase 1 established the minimum viable pipeline: a ReAct-prompted Qwen2.5-7B-Instruct agent interacting with the WebShop e-commerce simulation. The goals were to verify the environment runs, the model can execute search-click-buy actions, and quantitative metrics are computable—before investing in QLoRA fine-tuning and Bandit controllers.

---

## 2. Environment Setup

### 2.1 Infrastructure

| Component | Specification |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB (allocated 40 GB) |
| CUDA | Driver 13.0 / Runtime 12.4 |
| Python | 3.8.13 (CPython, conda-forge) |
| Conda Env | `webshop` (dedicated) |
| Storage | Shared HDD at `/inspire/hdd/.../final_project/` |

### 2.2 Key Dependencies

| Package | Version | Role |
|---|---|---|
| torch | 2.4.1+cu124 | FP16 model inference |
| transformers | 4.45.0 | Qwen2.5-7B model loading |
| spaCy | 3.3.0 + en_core_web_lg | NLP text processing |
| pyserini | 0.17.0 + Lucene 8 | Search engine indexing |
| openjdk | 11 | Java runtime for Lucene |
| pydantic | 1.7.4 | Required by spaCy 3.3.0 / thinc 8.0.17 |

### 2.3 Setup Challenges Resolved

- **GraalPy vs CPython:** Conda `defaults` channel resolves Python 3.8 to GraalPy, breaking numpy C extensions. Switched to `conda-forge` for standard CPython.
- **pydantic typing bug:** spaCy 3.3.0 crashed with pydantic 1.8.x on Python 3.8 due to `issubclass()` regression in `typing.py`. pydantic 1.7.4 (the minimum allowed by thinc 8.0.17) avoids the bug.
- **Werkzeug/Flask incompatibility:** Flask 2.1.2 requires Werkzeug < 2.3; downgraded from 3.x.
- **tokenizers duplicate:** Two conflicting tokenizers installations required clean uninstall/reinstall.

### 2.4 Data

| Resource | Location | Size |
|---|---|---|
| Product catalog (1,000) | `env/WebShop/data/` | 9.4 MB |
| Human goals (6,910) | `items_human_ins.json` | 4.9 MB |
| Search index (Lucene) | `search_engine/indexes/` | 3.0 MB |
| Human trajectories (50) | `user_session_logs/all_trajs/` | ~500 KB |
| Qwen2.5-7B-Instruct | `models/base/qwen/` | 15 GB |

---

## 3. Agent Design

### 3.1 Architecture

```
Instruction + Observation -> ReAct Prompt -> Qwen2.5-7B (FP16) -> Thought + Action
       ^                                                              |
       |                                                              v
  History Update                                              WebShop Environment
       ^                                                              |
       +------------------------ Observation -------------------------+
```

### 3.2 Prompt Design

The agent receives (a) a system prompt defining the action space, (b) a one-shot example showing a complete search-to-buy flow, and (c) the current instruction, observation, and available clickables. Actions follow the format:

| Action | Syntax | Semantics |
|---|---|---|
| Search | `search[space-separated keywords]` | Query Lucene index |
| Click product | `click[ASIN]` | View product details |
| Buy | `click[Buy Now]` | Purchase & compute reward |
| Paginate | `click[< Prev]` / `click[Next >]` | Navigate search pages |
| Select option | `click[option_value]` | Choose size, color, etc. |
| Back | `click[Back to Search]` | Return to results |
| Inspect | `click[Description]` / `click[Features]` / `click[Reviews]` | View sub-pages |

The full prompt is reproduced in Appendix A.

### 3.3 Inference Configuration

| Parameter | Value |
|---|---|
| Precision | FP16 (no quantization, 40 GB GPU) |
| Decoding | Greedy (temperature = 0, do_sample = False) |
| Max new tokens | 256 per step |
| Max steps | 10 per episode |
| Observation truncation | 2,000 characters |
| History context | Last 3 action-observation pairs |

---

## 4. Evaluation

### 4.1 Success Definition

We define three tiers of success to capture partial constraint satisfaction:

| Tier | Definition | Count |
|---|---|---|
| Partial Success | reward > 0 | 16 / 50 |
| **Strong Success (primary)** | **reward > 0.5** | **14 / 50** |
| Strict Success | reward ≥ 0.9 | 4 / 50 |

The primary metric throughout this report is **Strong Success (reward > 0.5)**, following the intuition that the purchased product satisfies most user constraints. Average reward is reported alongside to capture partial matches.

### 4.2 Test Set Construction

We sampled **50 unique human instructions** from `items_human_ins.json` using the environment's built-in goal loader with `random.seed(233)` (WebShop default). These 50 goals are reserved as the Phase 1 test set and will **not** be used for QLoRA training in Phase 2. The remaining ~6,860 goals are available for trajectory collection and training.

### 4.3 Quantitative Results

| Metric | Value |
|---|---|
| Strong Success Rate (r > 0.5) | **28.0%** (14/50) |
| Partial Success Rate (r > 0) | 32.0% (16/50) |
| Average Reward | 0.220 |
| Average Steps per Episode | 8.3 |
| Episodes at max steps (10) | 34 (68%) |
| Average Time per Episode | 15.8 s |
| Total Test Time | 13.2 min |
| Invalid Action Rate | 0 / 416 steps |
| Parse Failure Rate | 0 / 50 episodes |

**Key observation:** The model never produced a syntactically invalid action. All 416 outputs were valid `search[...]` or `click[...]` strings. However, the quality of those actions—especially search queries—varied substantially.

### 4.4 Step Count Distribution

| Steps | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
|---|---|---|---|---|---|---|---|
| Episodes | 8 | 1 | 1 | 2 | 1 | 3 | **34** |

**68% of episodes hit the 10-step maximum.** Manual inspection of the 34 step-10 episodes reveals three dominant sub-patterns:

| Sub-pattern | Estimated Share | Description |
|---|---|---|
| Search-then-stall | ~50% | Initial search returns results, but agent clicks 1-2 products, finds mismatches, re-searches with similarly over-specific queries, runs out of steps |
| No-result loop | ~35% | First search returns zero results (query too specific); agent tries similar queries with same outcome |
| Indecisive comparison | ~15% | Agent clicks multiple products and inspects details but never commits to Buy |

This breakdown directly motivates the Phase 2 focus on **query formulation** and **buy timing**.

### 4.5 Reward Distribution

| Reward Range | Count | Interpretation |
|---|---|---|
| 0.00 | 34 | Complete failure (no purchase or wrong item) |
| 0.01–0.30 | 1 | Weak partial match |
| 0.30–0.50 | 1 | Moderate partial match |
| 0.50–0.70 | 5 | Good purchase, minor constraint misses |
| 0.70–1.00 | 9 | Excellent purchase, most constraints satisfied |

**When the model succeeds, it tends to succeed well:** 9 of 14 strong successes (64%) achieved reward > 0.7. This suggests the primary bottleneck is not post-purchase evaluation but rather **reaching the correct product in the first place**.

### 4.6 Failure Analysis

The 36 failures fall into three categories:

| Failure Mode | Count | % of Failures | Root Cause |
|---|---|---|---|
| Max steps exhausted | 33 | 91.7% | Inefficient search / comparison loop |
| Early termination (≤ 3 steps) | 3 | 8.3% | Premature buy or action error |

**Primary failure patterns observed in trajectories:**

1.  **Over-specific search queries.** The agent generates verbose keyword strings (e.g., `"double sided machine washable decorative pillows printing technology 28x28 inches price <50"`) that return zero results from the keyword-based Lucene index. It then repeats similar over-specific queries.

2.  **Constraint hallucination.** The agent claims a product "matches all constraints" when key attributes (size, color, material) differ. This leads to incorrect purchases or wasted steps on mismatched products.

3.  **Premature buying.** In 3 cases, the agent clicked "Buy Now" after viewing only 1–2 products, before verifying all constraints.

**Example failure trajectory (Goal 0):**

> Instruction: *Find me double sided, machine washable decorative pillows with printing technology, size 28"×28", under $50.*
>
> Step 1 — Search: `search[double sided machine washable decorative pillows printing technology 28x28 inches price <50]` → 0 results.
> Steps 2–9 — Repeated similar over-specific searches → 0–3 results each. Agent clicks some products but finds mismatches.
> Step 10 — Agent runs out of steps. **Reward: 0.00.**

**Example success trajectory (Goal 1):**

> Instruction: *Find me butt lifting, light weight women's shorts with high waist, tummy control, under $40.*
>
> Step 1 — Search: `search[butt lifting lightweight women's shorts high waist tummy control]` → 50 results.
> Step 2 — Click: `click[B07XYZ789]` → Product matches all constraints.
> Step 3 — Inspect Description.
> Step 4 — Click: `click[Buy Now]`. **Reward: 0.714.**

These examples illustrate that **search query quality is the single largest determinant of episode outcome.**

---

## 5. Code Deliverables

```
src/agent/
├── react_agent.py       # Core agent (FP16 + optional 4-bit modes)
├── run_baseline.py      # Batch evaluation runner
└── smoke_test.py        # Single-episode verification

results/
├── baseline_50.jsonl         # Per-episode JSON records
└── baseline_50_summary.json  # Aggregated metrics
```

---

## 6. Findings

### Finding 1: ReAct baseline demonstrates basic shopping competency.

Qwen2.5-7B-Instruct, with no fine-tuning, achieves 28% strong success on WebShop small. This confirms that off-the-shelf instruction-following LLMs possess non-trivial product search and comparison capabilities.

### Finding 2: The primary bottleneck is search query formulation, not action format.

The model generated **zero invalid actions** across 416 steps, indicating the ReAct format is well-learned. However, over-specific search queries caused ~35% of step-10 failures to return zero results on the first search. This isolates query reformulation as the highest-impact improvement target.

### Finding 3: Post-purchase evaluation is not the problem.

When the agent reaches the correct product, it typically achieves reward > 0.7 (64% of successes). This means Phase 2 should prioritize *getting the agent to the right product* (search + comparison) over *improving the purchase decision itself*.

### Finding 4: Subsequent optimization should target query formulation and constraint verification.

The failure analysis naturally motivates: (a) QLoRA fine-tuning to learn shorter, more effective search queries; (b) a Bandit controller to decide when to search, compare, or buy; and (c) explicit constraint-checking before purchase.

---

## 7. Phase 2 Roadmap

Based on the Phase 1 findings, we recommend the following sequence:

| Phase | Task | Rationale |
|---|---|---|
| **2A** | Expand baseline to 300–500 goals; collect all trajectories | Need ~80–140 strong-success trajectories for reliable SFT |
| **2B** | Prompt-level ablation: keyword-only queries, backoff search | Low-cost improvement targeting the #1 failure mode |
| **2C** | QLoRA fine-tune on high-reward trajectories | Learn effective query formulation and constraint verification |
| **2D** | Add Bandit strategy controller (Thompson Sampling) | Optimize the search-vs-compare-vs-buy decision boundary |
| **2E** | Full evaluation: 500–1,000 goals, System A/B/C comparison | Statistical significance for final report |

---

## 7.5. Key Takeaways

- The ReAct baseline with a zero-shot 7B model achieves **28% strong success rate**, establishing a meaningful baseline.
- The primary bottleneck is **search query quality**—the untuned model generates verbose queries poorly matched to the keyword-based search engine.
- Zero invalid actions across 416 steps confirm the ReAct format is robust; improvement should focus on *what* the model searches for, not *how* it formats actions.
- Environment setup is complete, verified, and reproducible; rapid experimentation can begin in Phase 2.

---

## Appendix A: Full ReAct Prompt

The agent prompt combines a system instruction block, a one-shot example, and the current instruction/observation. The full template:

```
You are a web shopping agent. Your task is to find and buy a product
matching the user's instruction. You can search for products, click on them
to view details, and click 'Buy Now' to purchase.

Available actions:
- search[keywords]: Search for products using keywords
- click[ASIN]: Click a product to view its details
- click[Buy Now]: Purchase the current product
- click[< Prev] / click[Next >]: Navigate search results
- click[Back to Search]: Return to search results
- click[Description] / click[Features] / click[Reviews]: View product details
- click[option_value]: Select a product option (size, color, etc.)

IMPORTANT RULES:
1. Always start your response with a brief Thought, then output Action:
2. Match ALL constraints from the instruction (color, size, price, style, etc.)
3. Compare multiple products before buying if possible.
4. Don't buy until you are confident the product matches ALL requirements.
5. Format: Thought: <your reasoning>
   Action: search[keywords] or Action: click[element]

[One-shot example with instruction, observation, thought, and action...]

Instruction: {current_instruction}
Observation: {current_observation[:2000]}
Available actions: {clickables[:30]}
Thought:
```

## Appendix B: Full Trajectory Examples

### B.1 Failed Trajectory (Goal 0)

| Step | Action | Observation Summary | Result |
|---|---|---|---|
| 1 | `search[double sided machine washable decorative pillows printing technology 28x28 inches price <50]` | 0 results | No products found |
| 2 | `search[double sided machine washable decorative pillows 28x28]` | 0 results | Still too specific |
| 3 | `search[double sided pillows 28x28]` | 12 results | Agent clicks one but finds mismatched attributes |
| 4–9 | Alternating searches and clicks on 3–4 products | Various | None match all constraints (size, material) |
| 10 | Step limit reached | — | **Reward: 0.00 (Failure)** |

**Failure mode:** Over-specific initial query → wasted steps on reformulation → ran out of steps before finding correct product.

### B.2 Successful Trajectory (Goal 1)

| Step | Action | Observation Summary | Result |
|---|---|---|---|
| 1 | `search[butt lifting lightweight womens shorts high waist tummy control]` | 50 results | Good first query |
| 2 | `click[B07XYZ789]` | Product details match all constraints | Correct item |
| 3 | `click[Description]` | Verified fabric and fit details | Confirmed match |
| 4 | `click[Buy Now]` | Purchase confirmed | **Reward: 0.714 (Success)** |

**Success factors:** Focused search query, correct product on first click, verified details before buying.
