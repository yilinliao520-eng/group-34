# Data Directory

This directory contains lightweight data files used for evaluation.

## Files

-  — Fixed 50-goal evaluation subset sampled from WebShop synthetic goal pool (goals 0-49). Used for reproducible model comparison.

## Large Data (Not Included)

Full product data and training trajectories are excluded from this repository:

- WebShop full product catalog (1.18M items, ~5.2 GB)
- IL expert trajectories (1,571 samples)
- Query prediction pairs (11,724 samples)

These reside on the course GPU server and can be regenerated using  scripts.

## Data Split Protocol

- Goals 0–49: held-out evaluation split (used for all reported results)
- Goals 50+: used for trajectory collection only
- No goal overlap between training and evaluation
