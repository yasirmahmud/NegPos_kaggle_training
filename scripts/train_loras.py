#!/usr/bin/env python3
"""Train positive and negative Qwen3.5 LoRA adapters from the RTL dataset."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def stable_order(source_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()


def load_and_split(path: Path, seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            required = {
                "source_id",
                "negative_index",
                "specification",
                "correct_rtl",
                "negative_rtl",
            }
            missing = required.difference(row)
            if missing:
                raise ValueError(f"Line {line_number} is missing {sorted(missing)}")
            grouped.setdefault(str(row["source_id"]), []).append(row)

    if len(grouped) != 2000:
        raise ValueError(f"Expected 2,000 source IDs, found {len(grouped):,}")
    for source_id, variants in grouped.items():
        indices = sorted(int(row["negative_index"]) for row in variants)
        if indices != [1, 2, 3, 4]:
            raise ValueError(f"{source_id} has negative indices {indices}, expected 1..4")
        specs = {row["specification"] for row in variants}
        positives = {row["correct_rtl"] for row in variants}
        if len(specs) != 1 or len(positives) != 1:
            raise ValueError(f"{source_id} has inconsistent positive examples")
        variants.sort(key=lambda row: int(row["negative_index"]))

    source_ids = sorted(grouped, key=lambda value: stable_order(value, seed))
    partitions = {
        "train": source_ids[:1800],
        "validation": source_ids[1800:1900],
        "test": source_ids[1900:],
    }
    return {
        name: [
            {"source_id": source_id, "variants": grouped[source_id]}
            for source_id in ids
        ]
        for name, ids in partitions.items()
    }


def split_summary(splits: dict[str, list[dict[str, Any]]], seed: int) -> dict[str, Any]:
    return {
        "method": "sha256(seed:source_id), sorted; exact 1800/100/100",
        "seed": seed,
        "counts": {name: len(items) for name, items in splits.items()},
        "source_ids": {
            name: [item["source_id"] for item in items]
            for name, items in splits.items()
        },
    }


@dataclass
class RTLExampleDataset:
    items: list[dict[str, Any]]
    tokenizer: Any
    max_length: int
    kind: str
    seed: int
    epoch: int = 0

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def negative_index(self, source_id: str) -> int:
        # Each design sees one negative per epoch. The starting variant is
        # source-specific and balanced; three epochs expose three variants.
        digest = hashlib.sha256(f"{self.seed}:{source_id}".encode()).digest()
        return (int.from_bytes(digest[:4], "big") + self.epoch) % 4

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        item = self.items[index]
        variants = item["variants"]
        row = variants[0]
        target = (
            row["correct_rtl"]
            if self.kind == "positive"
            else variants[self.negative_index(item["source_id"])]["negative_rtl"]
        )
        prompt = f"Specification:\n{row['specification'].strip()}\n\n"
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target.strip(), add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer has no EOS token")
        # A pathological long RTL target must not consume the entire window;
        # reserve at least one quarter (up to 512 tokens) for its specification.
        minimum_prompt = min(512, self.max_length // 4)
        target_ids = target_ids[: self.max_length - minimum_prompt - 1] + [eos]
        prompt_budget = self.max_length - len(target_ids)
        # Keep the beginning of a specification, where module/interface details live.
        prompt_ids = prompt_ids[: max(0, prompt_budget)]
        input_ids = prompt_ids + target_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + target_ids,
        }


class Collator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in examples)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for item in examples:
            padding = width - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(item["attention_mask"] + [0] * padding)
            batch["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace, dtype: Any) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3_5ForCausalLM

    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def evaluate(model: Any, loader: Any, device: Any, autocast_dtype: Any) -> float:
    import torch

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                loss = model(**batch).loss
            total += float(loss.detach())
            count += 1
    model.train()
    return total / max(1, count)


def train_one(
    kind: str,
    splits: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    args: argparse.Namespace,
    dtype: Any,
    precision_name: str,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    seed_everything(args.seed + (0 if kind == "positive" else 10_000))
    model = build_model(args, dtype).to("cuda")
    train_data = RTLExampleDataset(
        splits["train"], tokenizer, args.max_length, kind, args.seed
    )
    validation_data = RTLExampleDataset(
        splits["validation"], tokenizer, args.max_length, kind, args.seed
    )
    collator = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collator)
    validation_loader = DataLoader(
        validation_data, batch_size=1, shuffle=False, collate_fn=collator
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    use_fp16_scaler = precision_name == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    history: list[dict[str, Any]] = []
    global_update = 0
    started = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to("cuda") for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype):
                loss = model(**batch).loss
                scaled_loss = loss / args.gradient_accumulation
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach())
            should_update = step % args.gradient_accumulation == 0 or step == len(train_loader)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if global_update % 25 == 0:
                    elapsed = time.time() - started
                    print(
                        f"[{kind}] epoch={epoch + 1}/{args.epochs} "
                        f"update={global_update}/{total_updates} "
                        f"loss={running_loss / step:.4f} elapsed_s={elapsed:.0f}",
                        flush=True,
                    )
        # Keep validation fixed so epoch-to-epoch losses are comparable.
        validation_data.set_epoch(0)
        validation_loss = evaluate(model, validation_loader, "cuda", dtype)
        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "validation_loss": validation_loss,
            "updates": global_update,
        }
        history.append(record)
        print(f"[{kind}] {json.dumps(record, sort_keys=True)}", flush=True)

    destination = args.output_root / f"{kind}_lora"
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    manifest = {
        "adapter": kind,
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "precision": precision_name,
        "hyperparameters": {
            "epochs": args.epochs,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.gradient_accumulation,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": LORA_TARGETS,
        },
        "negative_rotation": (
            "one source-specific variant per epoch; index advances modulo four"
            if kind == "negative"
            else None
        ),
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    (destination / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return manifest


def main() -> int:
    args = parse_args()
    splits = load_and_split(args.data, args.seed)
    summary = split_summary(splits, args.seed)
    print(json.dumps({"split_counts": summary["counts"]}, sort_keys=True), flush=True)
    if args.validate_only:
        return 0
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full training")
    # torch.cuda.is_bf16_supported() can return true on a T4 even though
    # Turing has no native BF16 tensor-core path. Require Ampere (SM80+) so
    # Kaggle T4 jobs use stable/native FP16 as intended.
    compute_capability = torch.cuda.get_device_capability(0)
    use_bf16 = compute_capability[0] >= 8 and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    precision_name = "bf16" if use_bf16 else "fp16"
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": ".".join(map(str, compute_capability)),
                "precision": precision_name,
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    manifests = []
    for kind in ("positive", "negative"):
        manifests.append(train_one(kind, splits, tokenizer, args, dtype, precision_name))
    (args.output_root / "split_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "adapters": manifests}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
