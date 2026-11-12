# PRECOG: Temporal Link-Prediction and Group-Level Explainability for Early Warning of Fraud Ring Formation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5+-orange.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PRECOG** (*PREdictive COordinated Group-formation*) is an early-warning research framework for forecasting coordinated financial fraud rings before their transaction topology is fully formed.

Unlike conventional fraud detectors that flag individual accounts or static subgraphs *after* illicit patterns materialize, PRECOG treats temporal link probabilities as intermediate signals—aggregating forecasted edges with observed interactions to detect emerging dense groups, generate group-level structural and temporal explanations, and quantify warning earliness via **Lead Time ($LT$)** and **Normalized Lead Time ($NLT$)**.

---

## Key Capabilities

1. **Temporal Group-Formation Forecasting**
   - Encodes continuous-time transaction streams into time-aware node embeddings $h_v(t)$ and estimates future edge probabilities $p(u, v, t + \Delta)$.
   - Evaluates candidate subgraphs $C$ over the union of observed and predicted edges ($E_{\text{obs}} \cup E_{\text{pred}}$), flagging emerging rings when combined subgraph density $\rho(C, t, t + \Delta)$ exceeds threshold $\tau$.
2. **Group-Level Explainability**
   - Produces structured early-warning packets for flagged candidate rings rather than isolated node scores, reporting:
     - **Shared Structural Evidence:** Common intermediaries, funding sources, and observed-vs-forecasted edge ratios.
     - **Shared Temporal Evidence:** Coordinated transaction cadence and bursty inter-event intervals ($\Delta t$).
     - **Per-Account Confidence ($\text{conf}(v)$):** Probabilistic membership scores acknowledging uncertainty in still-forming rings.
3. **Leakage-Free Lead-Time Evaluation**
   - Enforces strict chronological splits ($t_{\text{train}} < t_{\text{val}} < t_{\text{test}}$) with zero future-information leakage.
   - Evaluates warning earliness using **Lead Time ($LT = T_{\text{formation}} - T_{\text{prediction}}$)** and **Normalized Lead Time ($NLT$)** alongside AUPRC, Precision, Recall, and F1 against tabular (LightGBM), static GNN (GraphSAGE), and standard temporal link-prediction baselines.

---

## Repository Structure

```text
precog-fraud-ring/
├── configs/                 # Experiment hyperparameters (delta, theta, tau) & dataset splits
├── data/                    # Chronological dataset loaders & semi-synthetic ring injector
├── src/precog/
│   ├── data/                # Leakage-free temporal batching & formation-time labeling
│   ├── models/              # Temporal encoder, link predictor, and subgraph density detector
│   ├── explainability/      # Group-level structural/temporal attribution & conf(v)
│   └── evaluation/          # LT, NLT, classification metrics, and baseline wrappers
├── experiments/             # Reproducible training, baseline comparison, and ablation scripts
└── tests/                   # Unit tests for temporal leakage, density math, and lead-time metrics
