<!-- BlackRoad SEO Enhanced -->

# ulackroad au testing lab

> Part of **[BlackRoad OS](https://blackroad.io)** — Sovereign Computing for Everyone

[![BlackRoad OS](https://img.shields.io/badge/BlackRoad-OS-ff1d6c?style=for-the-badge)](https://blackroad.io)
[![BlackRoad Labs](https://img.shields.io/badge/Org-BlackRoad-Labs-2979ff?style=for-the-badge)](https://github.com/BlackRoad-Labs)
[![License](https://img.shields.io/badge/License-Proprietary-f5a623?style=for-the-badge)](LICENSE)

**ulackroad au testing lab** is part of the **BlackRoad OS** ecosystem — a sovereign, distributed operating system built on edge computing, local AI, and mesh networking by **BlackRoad OS, Inc.**

## About BlackRoad OS

BlackRoad OS is a sovereign computing platform that runs AI locally on your own hardware. No cloud dependencies. No API keys. No surveillance. Built by [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc), a Delaware C-Corp founded in 2025.

### Key Features
- **Local AI** — Run LLMs on Raspberry Pi, Hailo-8, and commodity hardware
- **Mesh Networking** — WireGuard VPN, NATS pub/sub, peer-to-peer communication
- **Edge Computing** — 52 TOPS of AI acceleration across a Pi fleet
- **Self-Hosted Everything** — Git, DNS, storage, CI/CD, chat — all sovereign
- **Zero Cloud Dependencies** — Your data stays on your hardware

### The BlackRoad Ecosystem
| Organization | Focus |
|---|---|
| [BlackRoad OS](https://github.com/BlackRoad-OS) | Core platform and applications |
| [BlackRoad OS, Inc.](https://github.com/BlackRoad-OS-Inc) | Corporate and enterprise |
| [BlackRoad AI](https://github.com/BlackRoad-AI) | Artificial intelligence and ML |
| [BlackRoad Hardware](https://github.com/BlackRoad-Hardware) | Edge hardware and IoT |
| [BlackRoad Security](https://github.com/BlackRoad-Security) | Cybersecurity and auditing |
| [BlackRoad Quantum](https://github.com/BlackRoad-Quantum) | Quantum computing research |
| [BlackRoad Agents](https://github.com/BlackRoad-Agents) | Autonomous AI agents |
| [BlackRoad Network](https://github.com/BlackRoad-Network) | Mesh and distributed networking |
| [BlackRoad Education](https://github.com/BlackRoad-Education) | Learning and tutoring platforms |
| [BlackRoad Labs](https://github.com/BlackRoad-Labs) | Research and experiments |
| [BlackRoad Cloud](https://github.com/BlackRoad-Cloud) | Self-hosted cloud infrastructure |
| [BlackRoad Forge](https://github.com/BlackRoad-Forge) | Developer tools and utilities |

### Links
- **Website**: [blackroad.io](https://blackroad.io)
- **Documentation**: [docs.blackroad.io](https://docs.blackroad.io)
- **Chat**: [chat.blackroad.io](https://chat.blackroad.io)
- **Search**: [search.blackroad.io](https://search.blackroad.io)

---


> A/B testing framework for ML and product experiments

Part of the [BlackRoad OS](https://blackroad.io) ecosystem — [BlackRoad-Labs](https://github.com/BlackRoad-Labs)

---

# blackroad-ab-testing-lab

> A/B testing framework for ML and product experiments using pure Python stdlib

Run statistically rigorous A/B tests with Welch's t-test, confidence intervals, and winner determination. Uses only Python's `statistics` and `math` stdlib — no scipy or numpy required.

## Features

- 🧪 **Experiment management** — Create, start, stop experiments
- 🎯 **Variant assignment** — Multiple variants with configurable traffic splits
- 📊 **Welch's t-test** — Two-sample t-test with unequal variances
- 📏 **Confidence intervals** — 95% CI using t-distribution
- 🏆 **Winner detection** — Statistical significance check (α=0.05)
- 🔢 **Pure Python** — No external statistical libraries required

## Statistical Methodology

This library implements **Welch's t-test** (two-tailed) for comparing variant means:

1. **t-statistic**: `t = (μ₁ - μ₂) / √(s₁²/n₁ + s₂²/n₂)`
2. **Degrees of freedom**: Welch–Satterthwaite approximation
3. **p-value**: Computed via regularized incomplete beta function (Lentz continued fraction)
4. **Confidence interval**: `μ ± t* × (s/√n)`

A result is considered significant when `p < α` (default α = 0.05).

## Installation

```bash
git clone https://github.com/BlackRoad-Labs/blackroad-ab-testing-lab
cd blackroad-ab-testing-lab
```

## Usage

### Create an experiment

```bash
python ab_testing.py create "checkout_button" \
  "New green CTA increases checkout conversion" \
  "conversion_rate" \
  --min-sample 200
```

### Add variants

```bash
EXP_ID=<experiment-id>
python ab_testing.py add-variant $EXP_ID control --pct 50 --desc "Blue button"
python ab_testing.py add-variant $EXP_ID treatment --pct 50 \
  --config '{"color": "green", "text": "Buy Now"}' --desc "Green button"
```

### Record results

```bash
python ab_testing.py record $EXP_ID user-001 control 0
python ab_testing.py record $EXP_ID user-002 treatment 1
python ab_testing.py record $EXP_ID user-003 treatment 1
```

### Analyze

```bash
python ab_testing.py analyze $EXP_ID
```

Output:
```
=== Analysis: checkout_button ===
Metric: conversion_rate
Variants:
  control:   n=100 mean=0.2340 CI=[0.1980, 0.2700]
  treatment: n=100 mean=0.3120 CI=[0.2750, 0.3490]

Comparisons:
  control vs treatment: p=0.0023 ✓ SIGNIFICANT
```

### Check significance

```bash
python ab_testing.py check-significance $EXP_ID --alpha 0.05
```

### Declare winner

```bash
python ab_testing.py winner $EXP_ID
```

## Tests

```bash
pytest tests/ -v
```
