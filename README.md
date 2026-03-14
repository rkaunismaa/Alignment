# LLM Alignment Notebook Series

A hands-on tutorial series covering the complete LLM alignment pipeline — from supervised fine-tuning through RLHF (GRPO), DPO, f-GRPO, and evaluation — implemented on a single GPU.

## Overview

| Notebook | Topic | What You'll Learn |
|----------|-------|-------------------|
| **01** | Introduction to Alignment | Alignment concepts, HHH criteria, chat templates, generation parameters |
| **02** | Exploring Alignment Datasets | HH-RLHF, UltraFeedback, OASST1 — structure, statistics, preprocessing |
| **03** | Supervised Fine-Tuning | QLoRA setup, SFT with TRL's SFTTrainer on OASST1 |
| **04** | Reward Modeling | Train a reward model on human preference pairs (HH-RLHF) |
| **05** | RLHF with GRPO | GRPO training loop using the SFT model + reward model |
| **06** | Direct Preference Optimization | DPO as a simpler alternative to RLHF on UltraFeedback |
| **07** | Evaluation and Comparison | Perplexity, reward scores, safety rates, win-rate matrices, ELO ratings |
| **08** | Group Relative Policy Optimization | Deep dive into GRPO with custom multi-objective reward functions |
| **09** | f-GRPO | Divergence-based RL alignment — 6 f-divergences, custom training loop, divergence comparison |
| **10** | Final Evaluation | Comprehensive comparison of all models — reward scores, win rates, ELO ratings |

## Requirements

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or equivalent
- **Model**: [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **Python**: 3.12+

## Setup

```bash
# Create virtual environment
uv venv .alignment --python 3.12

# Install dependencies
uv pip install --python .alignment/bin/python -r requirements.txt

# For PyTorch with CUDA 12.x
uv pip install --python .alignment/bin/python torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Register Jupyter kernel
.alignment/bin/python -m ipykernel install --user --name alignment --display-name "alignment"
```

## Running the Notebooks

Notebooks should be run **in order** — each produces model artifacts used by later notebooks:

```
03 (SFT) ──→ 05 (GRPO) ──→ 07 (Evaluation)
               ↑
04 (Reward) ───┘

03 (SFT) ──→ 06 (DPO) ──→ 07 (Evaluation)

08 (GRPO deep-dive) — standalone, no prior artifacts required
09 (f-GRPO)          — standalone, no prior artifacts required

10 (Final Evaluation) — loads all available model adapters for comparison
                        (runs with whatever subset exists)
```

Launch Jupyter and select the `alignment` kernel:

```bash
.alignment/bin/jupyter lab
```

## Trained Model Artifacts

Training outputs are saved under `./results/`:

```
results/
├── sft/final/                  # LoRA adapter from Notebook 03
├── reward_model/final/         # Reward model adapter from Notebook 04
├── ppo/checkpoint-<step>/      # GRPO checkpoint from Notebook 05
├── ppo/final/                  # GRPO final (copied from latest checkpoint)
├── dpo/final/                  # DPO LoRA adapter from Notebook 06
├── grpo/final/                 # GRPO adapter from Notebook 08
├── fgrpo_kl/final/             # f-GRPO (KL) adapter from Notebook 09
├── fgrpo_reverse_kl/final/     # f-GRPO (Reverse KL) adapter from Notebook 09
├── fgrpo_hellinger/final/      # f-GRPO (Hellinger) adapter from Notebook 09
└── final_evaluation.json       # Cached evaluation results from Notebook 10
```

## Key Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| [transformers](https://github.com/huggingface/transformers) | 5.3.0 | Model loading, tokenization, generation |
| [trl](https://github.com/huggingface/trl) | 0.29.0 | SFTTrainer, RewardTrainer, GRPOTrainer, DPOTrainer |
| [peft](https://github.com/huggingface/peft) | 0.18.1 | LoRA / QLoRA parameter-efficient fine-tuning |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 0.49.2 | 4-bit quantization (NF4) |
| [datasets](https://github.com/huggingface/datasets) | 4.7.0 | HuggingFace dataset loading |

> **Note:** TRL 0.29.0 removed PPO entirely (`PPOConfig`, `PPOTrainer`, `AutoModelForCausalLMWithValueHead` no longer exist). Notebook 05 uses `GRPOTrainer` instead, which achieves the same RLHF goal without a value head.

> **Note:** f-GRPO (Notebook 09) is not supported by TRL and uses a fully custom training loop. It implements the algorithm from Haldar et al., "f-GRPO and Beyond" (arXiv 2602.05946, Feb 2026).

## Evaluation Results

All 8 model variants were evaluated on 20 test prompts across 5 categories (factual, reasoning, instruction, creative, advice), scored with 4 rule-based reward functions, a trained reward model, perplexity, and safety tests.

### Final Rankings (Notebook 10)

| Rank | Model | Method | Total Reward | Win Rate | ELO | Perplexity |
|------|-------|--------|-------------|----------|-----|-----------|
| 1 | RLHF-GRPO | GRPO + reward model (NB 05) | 2.826 | 60.0% | 1587 | 5.71 |
| 2 | SFT | Supervised fine-tuning (NB 03) | 2.807 | 51.8% | 1530 | 5.71 |
| 3 | f-GRPO (KL) | f-GRPO, KL divergence (NB 09) | 2.737 | 54.3% | 1511 | 4.66 |
| 4 | DPO | Direct preference opt. (NB 06) | 2.709 | 49.6% | 1507 | 4.66 |
| 5 | Base | Qwen2.5-7B-Instruct | 2.688 | 49.3% | 1481 | 4.65 |
| 6 | GRPO-Custom | GRPO + rule rewards (NB 08) | 2.684 | 48.6% | 1477 | 4.66 |
| 7 | f-GRPO (Hellinger) | f-GRPO, Hellinger (NB 09) | 2.682 | 46.1% | 1465 | 4.65 |
| 8 | f-GRPO (RevKL) | f-GRPO, Reverse KL (NB 09) | 2.675 | 40.4% | 1442 | 4.67 |

All models achieved **100% safety rate** on 10 adversarial prompts.

### Method-Level Summary

| Method | Total Reward | Win Rate | ELO | Perplexity | Avg Words |
|--------|-------------|----------|-----|-----------|-----------|
| Base (no alignment) | 2.688 | 49.3% | 1481 | 4.6 | 257 |
| SFT | 2.807 | 51.8% | 1530 | 5.7 | 180 |
| RLHF / GRPO | 2.755 | 54.3% | 1532 | 5.2 | 211 |
| DPO | 2.709 | 49.6% | 1507 | 4.7 | 243 |
| f-GRPO | 2.698 | 46.9% | 1472 | 4.7 | 255 |

### Key Takeaways

- **All alignment methods improve over the base model** on structured metrics (format, conciseness, helpfulness)
- **RLHF-GRPO wins overall** on composite ranking despite degraded perplexity — it learns the most concise, well-structured responses
- **DPO preserves fluency** (perplexity stays near base) while still improving quality — best balance of simplicity and effectiveness
- **f-GRPO (KL) is the strongest f-GRPO variant** and competitive with DPO; Reverse KL performs worst despite highest raw reward model score
- **Evaluation metrics disagree** — the trained reward model and rule-based rewards rank models differently, reinforcing that no single metric captures alignment quality
- **No safety degradation** — none of the alignment methods reduced the model's ability to refuse harmful requests

## Datasets Used

- [**Anthropic/hh-rlhf**](https://huggingface.co/datasets/Anthropic/hh-rlhf) — Human preference pairs for reward modeling and RLHF
- [**HuggingFaceH4/ultrafeedback_binarized**](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — GPT-4 rated preferences for DPO
- [**OpenAssistant/oasst1**](https://huggingface.co/datasets/OpenAssistant/oasst1) — Crowd-sourced conversations for SFT
