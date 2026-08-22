# Kaggle LoRA Training Plan

## Goal
Train two LoRA adapters on **Qwen3.5-0.8B-Base** for inference-time proxy guidance of a frozen **Qwen3.5-2B** target model.

## Dataset
Source: `top2000_four_negatives.jsonl`

- **2,000** unique specification → correct RTL pairs
- **4 verified incorrect RTL variants per specification**
- **8,000** total negative RTL variants
- Negatives must remain compile/synthesis valid but functionally incorrect

Split **by design/source ID** so all variants of the same design stay in the same split.

Suggested split:
- 1,800 designs: train
- 100 designs: validation
- 100 designs: test

## Training

### Positive LoRA (`M+`)
Base: `Qwen3.5-0.8B-Base`

Input:
```text
Specification:
<spec>
```

Target:
```text
<correct_rtl>
```

Train on the 2,000 unique correct pairs.

### Negative LoRA (`M-`)
Use the same base model and input format, but target an incorrect RTL variant.

Do **not** train on all four negatives per design in every epoch. Rotate/select negatives so `M+` and `M-` receive roughly comparable optimizer steps/token exposure.

## Recommended Settings
- LoRA rank: **16**
- LoRA alpha: **32**
- LoRA dropout: **0.05**
- Learning rate: **1e-4**
- Epochs: **3**
- Precision: **BF16**
- Max sequence length: **2048**
- Base-model weights: **frozen**

Save only:
```text
positive_lora/
negative_lora/
```

## Inference
No further training is performed.

Load:
- Frozen `Qwen3.5-2B` target
- Frozen `Qwen3.5-0.8B` proxy
- `positive_lora`
- `negative_lora`

Use the proxy logits to modify the frozen target model's decoding at inference time.
