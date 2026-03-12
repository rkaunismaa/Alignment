# LLM Alignment Notebook Series

A hands-on tutorial series covering the complete LLM alignment pipeline — from supervised fine-tuning through RLHF and DPO — implemented on a single GPU.

## Overview

| Notebook | Topic | What You'll Learn |
|----------|-------|-------------------|
| **01** | Introduction to Alignment | Alignment concepts, HHH criteria, chat templates, generation parameters |
| **02** | Exploring Alignment Datasets | HH-RLHF, UltraFeedback, OASST1 — structure, statistics, preprocessing |
| **03** | Supervised Fine-Tuning | QLoRA setup, SFT with TRL's SFTTrainer on OASST1 |
| **04** | Reward Modeling | Train a reward model on human preference pairs (HH-RLHF) |
| **05** | RLHF with PPO | PPO training loop using the SFT model + reward model |
| **06** | Direct Preference Optimization | DPO as a simpler alternative to RLHF on UltraFeedback |
| **07** | Evaluation and Comparison | Perplexity, reward scores, safety rates, win-rate matrices, ELO ratings |

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
03 (SFT) ──→ 05 (PPO) ──→ 07 (Evaluation)
              ↑
04 (Reward) ──┘

03 (SFT) ──→ 06 (DPO) ──→ 07 (Evaluation)
```

Launch Jupyter and select the `alignment` kernel:

```bash
.alignment/bin/jupyter lab
```

## Trained Model Artifacts

Training outputs are saved under `./results/`:

```
results/
├── sft/final/           # LoRA adapter from Notebook 03
├── reward_model/final/  # Reward model adapter from Notebook 04
├── ppo/final/           # PPO-trained model from Notebook 05
└── dpo/final/           # DPO LoRA adapter from Notebook 06
```

## Key Libraries

| Library | Purpose |
|---------|---------|
| [transformers](https://github.com/huggingface/transformers) | Model loading, tokenization, generation |
| [trl](https://github.com/huggingface/trl) | SFTTrainer, RewardTrainer, PPOTrainer, DPOTrainer |
| [peft](https://github.com/huggingface/peft) | LoRA / QLoRA parameter-efficient fine-tuning |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization (NF4) |
| [datasets](https://github.com/huggingface/datasets) | HuggingFace dataset loading |

## Datasets Used

- [**Anthropic/hh-rlhf**](https://huggingface.co/datasets/Anthropic/hh-rlhf) — Human preference pairs for reward modeling and PPO
- [**HuggingFaceH4/ultrafeedback_binarized**](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) — GPT-4 rated preferences for DPO
- [**OpenAssistant/oasst1**](https://huggingface.co/datasets/OpenAssistant/oasst1) — Crowd-sourced conversations for SFT
