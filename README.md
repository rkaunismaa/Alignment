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
└── fgrpo_hellinger/final/      # f-GRPO (Hellinger) adapter from Notebook 09
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

## Datasets Used

- [**Anthropic/hh-rlhf**](https://huggingface.co/datasets/Anthropic/hh-rlhf) — Human preference pairs for reward modeling and RLHF
- [**HuggingFaceH4/ultrafeedback_binarized**](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — GPT-4 rated preferences for DPO
- [**OpenAssistant/oasst1**](https://huggingface.co/datasets/OpenAssistant/oasst1) — Crowd-sourced conversations for SFT
