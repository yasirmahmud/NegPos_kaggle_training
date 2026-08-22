# NegPos Qwen3.5 LoRA training

This repository trains the positive and negative adapters described in
`plan/kaggle_lora_training_plan.md` on a private Kaggle T4 job. The positive
and negative runs use the same 1,800 training source IDs and optimizer exposure.
The negative run rotates one of four variants per source ID across its three
epochs. Validation and test IDs are disjoint at the source/design level.

Run the full job from a persistent tmux session:

```bash
tmux new -s negpos-lora
bash scripts/run_overnight_training.sh
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t negpos-lora`. The launcher
publishes a private payload, submits one private GPU kernel, polls it, and
downloads both adapters beneath `output/kaggle_lora/<UTC timestamp>/trained_loras/`.
Logs are written under `logs/`.

If local monitoring is interrupted after Kaggle accepted the kernel, resume it
without submitting a new kernel version:

```bash
bash scripts/run_overnight_training.sh --resume
```

The requested precision is BF16 on compatible GPUs. Kaggle T4 workers
automatically use FP16 because Turing GPUs do not provide native BF16 training.
