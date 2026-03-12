# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 7-notebook Jupyter series teaching LLM alignment techniques hands-on, from foundational concepts through RLHF and DPO training. All training targets **Qwen2.5-7B-Instruct** on a local **RTX 4090 (24GB VRAM)**.

## Environment

- **Python venv**: `/home/rob/PythonEnvironments/Alignment/.alignment/`
- **Package manager**: `uv`
- **Jupyter kernel**: `alignment` (registered via `ipykernel`)
- **Python**: 3.12

```bash
# Install packages
uv pip install --python /home/rob/PythonEnvironments/Alignment/.alignment/bin/python <package>

# Run a notebook
.alignment/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=alignment <notebook>.ipynb

# Validate notebook JSON
python -c "import json; json.load(open('<notebook>.ipynb'))"
```

## Notebook Series (must be run in order)

Each notebook produces artifacts consumed by later notebooks:

1. **01_Introduction_to_Alignment** — Load model, explore chat templates, generation parameters, observe alignment behaviors
2. **02_Exploring_Alignment_Datasets** — HH-RLHF, UltraFeedback, OASST1 dataset exploration and preprocessing
3. **03_Supervised_Fine_Tuning** — QLoRA SFT on OASST1 → saves adapter to `./results/sft/final/`
4. **04_Reward_Modeling** — Train reward model on HH-RLHF preferences → saves to `./results/reward_model/final/`
5. **05_RLHF_with_PPO** — PPO training using SFT model (from 03) + reward model (from 04) → saves to `./results/ppo/final/`
6. **06_Direct_Preference_Optimization** — DPO on UltraFeedback → saves to `./results/dpo/final/`
7. **07_Evaluation_and_Comparison** — Loads all 4 models (base, SFT, PPO, DPO) sequentially for evaluation

## Key Technical Constraints

### VRAM Management (24GB limit)
- Training notebooks use **4-bit QLoRA** (BitsAndBytesConfig with NF4)
- Notebook 05 (PPO) is the tightest: policy + value head + reward model all in VRAM simultaneously. Uses batch_size=8, mini_batch_size=2.
- Notebook 07 loads models **one at a time** (load → generate → unload with `gc.collect()` + `torch.cuda.empty_cache()`)
- DPO uses batch_size=1 with gradient_accumulation_steps=16

### Transformers 5.x API
- `tokenizer.apply_chat_template(tokenize=True)` returns a **dict**, not a list. Always use `tokenize=False` to get a string, then `tokenizer(text, return_tensors="pt")` for tensors.
- Use `dtype=` instead of deprecated `torch_dtype=` in `from_pretrained()`
- Use `torch.cuda.get_device_properties(0).total_memory` (not `.total_mem`)

### LoRA Config (consistent across notebooks)
```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
           target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"])
```

## Datasets
- **Anthropic/hh-rlhf** — preference pairs (chosen/rejected conversation strings with `\n\nHuman:`/`\n\nAssistant:` format)
- **HuggingFaceH4/ultrafeedback_binarized** — binarized preference data for DPO
- **OpenAssistant/oasst1** — conversational tree structure, filter for English + rank==0
