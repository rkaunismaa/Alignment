# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 10-notebook Jupyter series teaching LLM alignment techniques hands-on, from foundational concepts through SFT, reward modeling, RLHF (GRPO), DPO, f-GRPO, and comprehensive evaluation. All training targets **Qwen2.5-7B-Instruct** on a local **RTX 4090 (24GB VRAM)**.

## Environment

- **Python venv**: `/home/rob/PythonEnvironments/Alignment/.alignment/`
- **Package manager**: `uv`
- **Jupyter kernel**: `alignment` (registered via `ipykernel`)
- **Python**: 3.12

### Library Versions (as of March 2026)
| Library | Version |
|---------|---------|
| torch | 2.6.0+cu124 |
| transformers | 5.3.0 |
| trl | 0.29.0 |
| peft | 0.18.1 |
| bitsandbytes | 0.49.2 |
| datasets | 4.7.0 |

```bash
# Install packages
uv pip install --python /home/rob/PythonEnvironments/Alignment/.alignment/bin/python <package>

# Validate notebook JSON
python -c "import json; json.load(open('<notebook>.ipynb'))"
```

## Notebook Series (must be run in order)

Each notebook produces artifacts consumed by later notebooks:

1. **01_Introduction_to_Alignment** — Load model, explore chat templates, generation parameters, observe alignment behaviors
2. **02_Exploring_Alignment_Datasets** — HH-RLHF, UltraFeedback, OASST1 dataset exploration and preprocessing
3. **03_Supervised_Fine_Tuning** — QLoRA SFT on OASST1 → saves adapter to `./results/sft/final/`
4. **04_Reward_Modeling** — Train reward model on HH-RLHF preferences → saves to `./results/reward_model/final/`
5. **05_RLHF_with_PPO** — GRPO training (PPO was removed in TRL 0.9+; rewritten to use GRPOTrainer) using SFT model + reward model → saves to `./results/ppo/checkpoint-*/`
6. **06_Direct_Preference_Optimization** — DPO on UltraFeedback → saves to `./results/dpo/final/`
7. **07_Evaluation_and_Comparison** — Loads all models sequentially for evaluation
8. **08_Group_Relative_Policy_Optimization** — Deep dive into GRPO with custom reward functions
9. **09_f_GRPO** — f-GRPO: divergence-based RL alignment using f-divergence variational bounds (custom implementation, not TRL)
10. **10_Final_Evaluation** — Comprehensive comparison of all models: reward scores, win-rate matrix, ELO ratings, response analysis

## Key Technical Constraints

### VRAM Management (24GB limit)
- All training uses **4-bit QLoRA** (`BitsAndBytesConfig` with NF4, `bnb_4bit_compute_dtype=torch.bfloat16`)
- **Never load two 7B models simultaneously** — always `del model; torch.cuda.empty_cache()` before loading the next
- In comparison/evaluation cells, load models one at a time: load → generate → delete → next
- The reward model (~12–13 GB in memory) must be loaded **last**, after all generation is complete, to avoid crowding out other models
- Notebook 05 cell 11 is a **restart checkpoint** — it reloads all state from saved checkpoints and can run after a kernel restart without re-training

### Mixed Precision — Always Use bf16
- **Always use `bf16=True`**, never `fp16=True`
- The RTX 4090 has native bfloat16 support; `fp16=True` fails with `"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'` because LoRA adapter weights are in bfloat16
- Always set `bnb_4bit_compute_dtype=torch.bfloat16` in `BitsAndBytesConfig`

### TRL 0.29.0 API Changes
- **PPO is fully removed**: `PPOConfig`, `PPOTrainer`, and `AutoModelForCausalLMWithValueHead` no longer exist. Notebook 05 uses `GRPOTrainer`/`GRPOConfig` instead.
- **`tokenizer=` is removed**: Use `processing_class=tokenizer` in `SFTTrainer`, `RewardTrainer`, and `GRPOTrainer`
- **`warmup_ratio` is deprecated**: Use `warmup_steps` instead
- **GRPO reward functions** must have signature `func(prompts: list[str], completions: list[str], **kwargs) -> list[float]`

### f-GRPO (Notebook 09)
- **Not supported by TRL** — uses a fully custom training loop
- Requires a reference model; uses `model.disable_adapter_layers()` / `model.enable_adapter_layers()` to share weights between policy and reference (no extra VRAM copy)
- Implements 6 f-divergences: KL, Reverse KL, Pearson χ², Hellinger, Jensen-Shannon, Total Variation
- Each divergence has a generator `f(t)`, Fenchel conjugate `f*(s)`, and link function `g(r)`
- Completions are split into D+ (positive advantage) and D- (negative advantage); different transformations applied to each set
- LoRA weights are reset between divergence comparison runs via `kaiming_uniform_` (A) and `zeros_` (B)
- Paper: Haldar et al., "f-GRPO and Beyond", arXiv 2602.05946, Feb 2026

### GRPO Completions Format (Notebooks 05 & 08)
When calling reward functions manually (e.g. in comparison cells), pass completions as `[[{"content": "..."}]]` — a list containing one message-dict list. Do **not** use `[c[0] for c in comps]` — that unwraps to a bare dict and breaks `re.search`.

```python
# Correct
comp = [[{"content": response}]]
format_reward_func(comp)[0]
helpfulness_reward_func([[{"content": prompt}]], comp)[0]
```

### Transformers 5.x Notes
- `tokenizer.apply_chat_template(tokenize=True)` returns a **dict**, not a list. Use `tokenize=False` to get a string, then tokenize separately.
- `warmup_ratio` is deprecated in `SFTConfig`/`RewardConfig`/`GRPOConfig` — use `warmup_steps`

### Saved Artifacts
- Notebook 05 (GRPO) saves checkpoints to `./results/ppo/checkpoint-<step>/` during training. The final save cell copies the latest checkpoint to `./results/ppo/final/`.
- To find the latest checkpoint programmatically:
```python
import glob
checkpoints = sorted(glob.glob("./results/ppo/checkpoint-*"), key=lambda p: int(p.split("-")[-1]))
latest = checkpoints[-1]
```

### LoRA Config (consistent across notebooks)
```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
           task_type=TaskType.CAUSAL_LM)
```

## Evaluation Results Summary

### Notebook 07: Core Pipeline Evaluation (Base, SFT, PPO/GRPO, DPO)

| Model | Perplexity | Reward Model | Safety | Win Rate | ELO | Avg Words |
|-------|-----------|-------------|--------|----------|-----|-----------|
| Base  | 4.65      | -8.08       | 100%   | 58.3%    | 1551 | 166      |
| DPO   | 4.66      | -8.54       | 100%   | 56.5%    | 1571 | 165      |
| SFT   | 5.71      | -8.53       | 100%   | 42.6%    | 1444 | 121      |
| PPO   | 5.71      | -8.38       | 100%   | 42.6%    | 1434 | 118      |

- DPO and Base are essentially tied; DPO wins on ELO, Base on reward model score
- SFT/PPO produce shorter responses but degrade perplexity
- All Wilcoxon signed-rank tests non-significant (p > 0.05) — differences are small

### Notebook 10: Full Comparison (all 8 models)

| Rank | Model | Total Reward | Reward Model | Perplexity | Win Rate | ELO |
|------|-------|-------------|-------------|-----------|----------|-----|
| 1 | RLHF-GRPO | 2.826 | -10.75 | 5.71 | 60.0% | 1587 |
| 2 | SFT | 2.807 | -9.77 | 5.71 | 51.8% | 1530 |
| 3 | f-GRPO (KL) | 2.737 | -7.44 | 4.66 | 54.3% | 1511 |
| 4 | DPO | 2.709 | -7.03 | 4.66 | 49.6% | 1507 |
| 5 | Base | 2.688 | -7.72 | 4.65 | 49.3% | 1481 |
| 6 | GRPO-Custom | 2.684 | -9.09 | 4.66 | 48.6% | 1477 |
| 7 | f-GRPO (Hellinger) | 2.682 | -8.31 | 4.65 | 46.1% | 1465 |
| 8 | f-GRPO (RevKL) | 2.675 | -6.74 | 4.67 | 40.4% | 1442 |

### Key Findings

1. **RLHF-GRPO is the overall winner** by composite score (ELO + win rate + reward) despite worst perplexity and reward model scores — it excels at conciseness (0.700) and instruction-following
2. **Metrics disagree**: the trained reward model ranks f-GRPO (RevKL) best (-6.74) while rule-based rewards + ELO rank RLHF-GRPO first — different metrics capture different quality aspects
3. **SFT/RLHF degrade perplexity** (5.71 vs 4.65 base) but improve conciseness and structure; DPO/f-GRPO preserve perplexity while still improving
4. **All models maintain 100% safety** — no alignment method degraded safety refusal behaviour
5. **Response length trade-off**: RLHF-GRPO and SFT are most concise (171-180 words avg), f-GRPO variants and Base are most verbose (255-258 words)
6. **f-GRPO divergence choice matters**: KL is the best f-GRPO variant (rank 3 overall); Reverse KL is worst (rank 8) despite highest reward model score

## Datasets
- **Anthropic/hh-rlhf** — preference pairs (chosen/rejected conversation strings with `\n\nHuman:`/`\n\nAssistant:` format)
- **HuggingFaceH4/ultrafeedback_binarized** — binarized preference data for DPO
- **OpenAssistant/oasst1** — conversational tree structure, filter for English + rank==0
