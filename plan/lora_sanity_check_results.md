# LoRA Sanity Check Results

## Evaluation setup

- Base model: `Qwen/Qwen3.5-0.8B-Base`
- Positive expert: base model with `positive_lora`
- Negative expert: base model with `negative_lora`
- Split: held-out test set only
- Test designs: 100
- Verified negative variants: 400 (four per design)
- Maximum sequence length: 2,048 tokens
- Hardware: one NVIDIA H100 PCIe
- Runtime: approximately 413 seconds

Likelihoods were normalized by the number of output tokens. Evaluation used the
same specification prompt and truncation policy as LoRA training.

## Results

| Metric | Result | Passed / total |
|---|---:|---:|
| Positive LoRA improves correct RTL likelihood | 99.00% | 99 / 100 |
| Negative LoRA improves negative RTL likelihood | 100.00% | 400 / 400 |
| Positive LoRA prefers correct over negative RTL | 99.25% | 397 / 400 |
| `C+ > 0` | 62.00% | 62 / 100 |
| `C- < 0` | 97.50% | 390 / 400 |

## Likelihood margins

| Margin | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| `LL(M+, R+) - LL(M0, R+)` | +0.225983 | +0.218533 | -0.013929 | +0.591714 |
| `LL(M-, R-) - LL(M0, R-)` | +0.281657 | +0.262405 | +0.018507 | +0.788666 |
| `LL(M+, R+) - LL(M+, R-)` | +0.112296 | +0.090544 | -0.011185 | +0.425664 |
| `C+` | +0.006261 | +0.005165 | -0.071958 | +0.059200 |
| `C-` | -0.099395 | -0.076484 | -0.418068 | +0.083410 |

## Interpretation

The primary sanity checks pass strongly:

- The positive LoRA improves correct RTL likelihood on 99% of unseen designs.
- The negative LoRA improves likelihood for every verified negative sample.
- The most important check passes on 397 of 400 comparisons: the positive LoRA
  prefers correct RTL over the corresponding incorrect RTL.
- Negative-direction expert separation is strong, with `C- < 0` for 97.5% of
  negative samples.

Positive-direction expert separation is weaker. Although the mean and median
`C+` margins are positive, only 62% of correct designs have `C+ > 0`. Both
experts appear to learn substantial shared RTL structure, while the negative
expert specializes more clearly on localized incorrect variants.

## Verdict

Proceed with a limited ProxyRTL decoding pilot. The learned likelihood signals
are strong enough for initial integration, but the positive expert coefficient
should be monitored carefully because `M+` versus `M-` separation on correct RTL
is modest.

## Raw artifacts

- Detailed summary: `output/lora_sanity/local_20260822T164726Z/summary.json`
- Correct RTL likelihoods: `output/lora_sanity/local_20260822T164726Z/correct_likelihoods.csv`
- Negative RTL likelihoods: `output/lora_sanity/local_20260822T164726Z/negative_likelihoods.csv`
