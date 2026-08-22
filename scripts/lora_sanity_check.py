#!/usr/bin/env python3
"""Evaluate held-out RTL likelihoods for the base, positive, and negative models."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--positive-lora", type=Path, required=True)
    parser.add_argument("--negative-lora", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    return parser.parse_args()


def load_test_rows(data_path: Path, split_path: Path) -> list[dict[str, Any]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = list(split["source_ids"]["test"])
    if len(test_ids) != 100 or len(set(test_ids)) != 100:
        raise ValueError("Expected exactly 100 unique test source IDs")
    test_set = set(test_ids)
    grouped: dict[str, list[dict[str, Any]]] = {}
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            source_id = str(row["source_id"])
            if source_id in test_set:
                grouped.setdefault(source_id, []).append(row)
    if set(grouped) != test_set:
        raise ValueError(f"Missing test IDs: {sorted(test_set.difference(grouped))}")
    result = []
    for source_id in test_ids:
        variants = sorted(grouped[source_id], key=lambda row: int(row["negative_index"]))
        if [int(row["negative_index"]) for row in variants] != [1, 2, 3, 4]:
            raise ValueError(f"{source_id} does not have negative indices 1..4")
        if len({row["specification"] for row in variants}) != 1:
            raise ValueError(f"Inconsistent specification for {source_id}")
        if len({row["correct_rtl"] for row in variants}) != 1:
            raise ValueError(f"Inconsistent correct RTL for {source_id}")
        result.append({"source_id": source_id, "variants": variants})
    return result


def encode_candidate(
    tokenizer: Any, specification: str, target: str, max_length: int
) -> tuple[list[int], list[int]]:
    prompt = f"Specification:\n{specification.strip()}\n\n"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target.strip(), add_special_tokens=False)
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer has no EOS token")
    minimum_prompt = min(512, max_length // 4)
    target_ids = target_ids[: max_length - minimum_prompt - 1] + [eos]
    prompt_ids = prompt_ids[: max_length - len(target_ids)]
    return prompt_ids + target_ids, target_ids


def normalized_log_likelihood(
    model: Any, input_ids: list[int], target_ids: list[int], device: Any
) -> float:
    import torch

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    # Keep target_count + 1 positions: the first predicts the first target,
    # and the final unused position is discarded after causal shifting.
    with torch.inference_mode():
        output = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=len(target_ids) + 1,
        )
    logits = output.logits[0, :-1]
    if logits.shape[0] != len(target_ids):
        raise RuntimeError(
            f"Expected {len(target_ids)} prediction positions, got {logits.shape[0]}"
        )
    targets = torch.tensor(target_ids, dtype=torch.long, device=device)
    total = torch.zeros((), dtype=torch.float64, device="cpu")
    # Chunk across positions to avoid materializing a target_len x 248k FP32 tensor.
    for start in range(0, len(target_ids), 64):
        chunk = logits[start : start + 64].float()
        chosen = chunk.gather(1, targets[start : start + 64, None]).squeeze(1)
        log_prob = chosen - torch.logsumexp(chunk, dim=-1)
        total += log_prob.double().sum().cpu()
    return float(total / len(target_ids))


def score_all_models(
    model: Any,
    tokenizer: Any,
    specification: str,
    target: str,
    max_length: int,
    device: Any,
) -> tuple[dict[str, float], int]:
    input_ids, target_ids = encode_candidate(tokenizer, specification, target, max_length)
    with model.disable_adapter():
        base = normalized_log_likelihood(model, input_ids, target_ids, device)
    model.set_adapter("positive")
    positive = normalized_log_likelihood(model, input_ids, target_ids, device)
    model.set_adapter("negative")
    negative = normalized_log_likelihood(model, input_ids, target_ids, device)
    return {"base": base, "positive": positive, "negative": negative}, len(target_ids)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def percentage(flags: list[bool]) -> float:
    return 100.0 * sum(flags) / len(flags)


def summarize(correct_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_improvement = [
        row["ll_positive"] - row["ll_base"] for row in correct_rows
    ]
    negative_improvement = [
        row["ll_negative"] - row["ll_base"] for row in negative_rows
    ]
    positive_preference = []
    correct_by_id = {row["source_id"]: row for row in correct_rows}
    for row in negative_rows:
        correct = correct_by_id[row["source_id"]]
        positive_preference.append(correct["ll_positive"] - row["ll_positive"])
    c_plus = [
        row["ll_positive"] - row["ll_negative"] for row in correct_rows
    ]
    c_minus = [
        row["ll_positive"] - row["ll_negative"] for row in negative_rows
    ]
    return {
        "sample_counts": {
            "test_designs": len(correct_rows),
            "verified_negative_samples": len(negative_rows),
        },
        "metrics_percent": {
            "positive_lora_improves_correct_rtl": percentage(
                [value > 0 for value in positive_improvement]
            ),
            "negative_lora_improves_negative_rtl": percentage(
                [value > 0 for value in negative_improvement]
            ),
            "positive_lora_prefers_correct_over_negative_rtl": percentage(
                [value > 0 for value in positive_preference]
            ),
            "c_plus_gt_zero": percentage([value > 0 for value in c_plus]),
            "c_minus_lt_zero": percentage([value < 0 for value in c_minus]),
        },
        "margins": {
            "ll_positive_correct_minus_base_correct": distribution(positive_improvement),
            "ll_negative_negative_minus_base_negative": distribution(negative_improvement),
            "ll_positive_correct_minus_positive_negative": distribution(positive_preference),
            "c_plus": distribution(c_plus),
            "c_minus": distribution(c_minus),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    test_items = load_test_rows(args.data, args.split_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full sanity check")
    capability = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.positive_lora)
    base = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    model = PeftModel.from_pretrained(
        base, args.positive_lora, adapter_name="positive", is_trainable=False
    )
    model.load_adapter(args.negative_lora, adapter_name="negative", is_trainable=False)
    model.to(device).eval()
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "compute_capability": ".".join(map(str, capability)),
                "dtype": str(dtype),
                "test_designs": len(test_items),
                "negative_samples": len(test_items) * 4,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    correct_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    started = time.time()
    for design_number, item in enumerate(test_items, 1):
        first = item["variants"][0]
        correct_scores, correct_tokens = score_all_models(
            model,
            tokenizer,
            first["specification"],
            first["correct_rtl"],
            args.max_length,
            device,
        )
        correct_rows.append(
            {
                "source_id": item["source_id"],
                "output_tokens": correct_tokens,
                "ll_base": correct_scores["base"],
                "ll_positive": correct_scores["positive"],
                "ll_negative": correct_scores["negative"],
            }
        )
        for variant in item["variants"]:
            scores, target_tokens = score_all_models(
                model,
                tokenizer,
                variant["specification"],
                variant["negative_rtl"],
                args.max_length,
                device,
            )
            negative_rows.append(
                {
                    "source_id": item["source_id"],
                    "negative_index": int(variant["negative_index"]),
                    "output_tokens": target_tokens,
                    "ll_base": scores["base"],
                    "ll_positive": scores["positive"],
                    "ll_negative": scores["negative"],
                }
            )
        if design_number % 5 == 0:
            print(
                f"scored_designs={design_number}/100 elapsed_s={time.time() - started:.0f}",
                flush=True,
            )

    summary = summarize(correct_rows, negative_rows)
    summary["evaluation"] = {
        "split": "test",
        "base_model": MODEL_ID,
        "base_model_revision": MODEL_REVISION,
        "max_length": args.max_length,
        "all_four_negatives_per_design": True,
        "elapsed_seconds": time.time() - started,
    }
    write_csv(args.output_dir / "correct_likelihoods.csv", correct_rows)
    write_csv(args.output_dir / "negative_likelihoods.csv", negative_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
