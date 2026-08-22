# LoRA Sanity Check

## Goal
Verify that the trained **positive LoRA (`M+`)** and **negative LoRA (`M-`)** learned the intended RTL behavior before implementing ProxyRTL inference.

## Models
- `M0`: Qwen3.5-0.8B base model
- `M+`: `M0` + positive LoRA
- `M-`: `M0` + negative LoRA

Use only held-out validation/test designs. Do not use training samples.

## Required Checks

For each validation sample with specification `S`, correct RTL `R+`, and negative RTL `R-`, compute normalized sequence log-likelihood:

```text
LL(M, R | S) = sum(token log-probabilities) / number_of_output_tokens
```

### Check 1 — Positive LoRA learned correct RTL
Expect:

```text
LL(M+, R+ | S) > LL(M0, R+ | S)
```

Report the percentage of designs where this is true.

### Check 2 — Negative LoRA learned incorrect RTL
Expect:

```text
LL(M-, R- | S) > LL(M0, R- | S)
```

Report the percentage of negative samples where this is true.

### Check 3 — Positive LoRA prefers correct over incorrect RTL
Expect:

```text
LL(M+, R+ | S) > LL(M+, R- | S)
```

This is the most important sanity check.

### Check 4 — Positive and negative experts diverge correctly
Compute:

```text
C+ = LL(M+, R+ | S) - LL(M-, R+ | S)
C- = LL(M+, R- | S) - LL(M-, R- | S)
```

Desired behavior:

```text
C+ > 0
C- < 0
```

## Evaluation Output

Produce:

| Metric | Result |
|---|---:|
| Positive LoRA improves correct RTL likelihood | % |
| Negative LoRA improves negative RTL likelihood | % |
| Positive LoRA prefers correct over negative RTL | % |
| `C+ > 0` | % |
| `C- < 0` | % |

Also report mean and median likelihood margins.

## Pass Criterion
Proceed to ProxyRTL decoding if:

- the positive LoRA consistently increases likelihood of correct RTL,
- the negative LoRA increases likelihood of verified incorrect RTL,
- `M+` prefers correct RTL over its corresponding negatives,
- the positive/negative directions show clear separation on unseen designs.

If these signals are weak or reversed, inspect LoRA training and data formatting before continuing.
