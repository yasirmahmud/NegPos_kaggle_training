#!/usr/bin/env python3
"""Run the held-out LoRA sanity check as a private Kaggle GPU job."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kaggle_lora_offload as common


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / ".tmp" / "kaggle_sanity"
TRAINING_ROOT = (
    REPO_ROOT
    / "output"
    / "kaggle_lora"
    / "20260822T100854Z"
    / "trained_loras"
)
PAYLOAD_SLUG = "negpos-lora-sanity-payload"
PAYLOAD_ARCHIVE = "negpos_sanity_payload.bin"
KERNEL_SLUG = "negpos-lora-sanity-check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "status", "retrieve"), default="run")
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--username", default=None)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    return parser.parse_args()


def add_file(bundle: tarfile.TarFile, source: Path, destination: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    info = bundle.gettarinfo(str(source), arcname=destination)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    with source.open("rb") as handle:
        bundle.addfile(info, handle)


def build_payload(account: str) -> tuple[Path, str]:
    stage = WORK_ROOT / "payload"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    archive = stage / PAYLOAD_ARCHIVE
    raw = stage / "payload.tar"
    inputs = [
        (REPO_ROOT / "scripts" / "lora_sanity_check.py", "sanity/scripts/lora_sanity_check.py"),
        (
            REPO_ROOT / "data" / "top2000_four_negatives.jsonl",
            "sanity/data/top2000_four_negatives.jsonl",
        ),
        (TRAINING_ROOT / "split_manifest.json", "sanity/adapters/split_manifest.json"),
    ]
    positive_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    )
    negative_files = ("adapter_config.json", "adapter_model.safetensors")
    inputs.extend(
        (TRAINING_ROOT / "positive_lora" / name, f"sanity/adapters/positive_lora/{name}")
        for name in positive_files
    )
    inputs.extend(
        (TRAINING_ROOT / "negative_lora" / name, f"sanity/adapters/negative_lora/{name}")
        for name in negative_files
    )
    with tarfile.open(raw, "w") as bundle:
        for source, destination in inputs:
            add_file(bundle, source, destination)
    with (
        raw.open("rb") as source,
        archive.open("wb") as destination,
        gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0, compresslevel=1) as compressed,
    ):
        shutil.copyfileobj(source, compressed)
    raw.unlink()
    digest = common.sha256(archive)
    (stage / "payload-manifest.json").write_text(
        json.dumps({"archive": archive.name, "sha256": digest}, indent=2) + "\n",
        encoding="utf-8",
    )
    (stage / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "NegPos Private LoRA Sanity Payload",
                "id": f"{account}/{PAYLOAD_SLUG}",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return stage, digest


def publish(account: str, environment: dict[str, str]) -> str:
    stage, digest = build_payload(account)
    reference = f"{account}/{PAYLOAD_SLUG}"
    exists = common.run_kaggle(
        ["datasets", "files", reference], environment, capture=True, check=False
    ).returncode == 0
    if exists and common.remote_hash(reference, environment) == digest:
        print(f"Sanity payload already current: {digest[:12]}", flush=True)
    elif exists:
        common.run_kaggle(
            ["datasets", "version", "-p", str(stage), "-m", f"Sanity payload {digest[:12]}", "-r", "zip"],
            environment,
        )
    else:
        common.run_kaggle(["datasets", "create", "-p", str(stage), "-r", "zip"], environment)
    common.wait_for_dataset(reference, digest, environment)
    return reference


def kernel_source() -> str:
    return '''"""Generated private Kaggle LoRA sanity-check entry point."""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

work = Path("/kaggle/working")
archives = list(Path("/kaggle/input").rglob("negpos_sanity_payload.bin"))
if len(archives) != 1:
    raise RuntimeError(f"Expected one sanity payload, found {archives}")
with tarfile.open(archives[0], "r:gz") as bundle:
    bundle.extractall(work, filter="data")
project = work / "sanity"
subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchao"], check=False)
subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
    "transformers==5.12.0", "peft==0.19.1", "accelerate==1.14.0",
    "safetensors>=0.6", "einops>=0.8",
], check=True)
fla = subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
    "fla-core", "flash-linear-attention",
], check=False)
print(f"FLA_INSTALL_RETURN_CODE={fla.returncode}", flush=True)
causal = subprocess.run([
    sys.executable, "-m", "pip", "install", "--quiet", "--no-build-isolation",
    "causal-conv1d>=1.5.0",
], check=False)
print(f"CAUSAL_CONV1D_INSTALL_RETURN_CODE={causal.returncode}", flush=True)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(work / "hf_cache")
output = work / "sanity_results"
command = [
    sys.executable, str(project / "scripts" / "lora_sanity_check.py"),
    "--data", str(project / "data" / "top2000_four_negatives.jsonl"),
    "--split-manifest", str(project / "adapters" / "split_manifest.json"),
    "--positive-lora", str(project / "adapters" / "positive_lora"),
    "--negative-lora", str(project / "adapters" / "negative_lora"),
    "--output-dir", str(output),
]
telemetry = None
if shutil.which("nvidia-smi"):
    telemetry = subprocess.Popen([
        "nvidia-smi",
        "--query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits", "--loop=30",
    ])
failure = None
try:
    subprocess.run(command, cwd=project, check=True)
except Exception as exc:
    failure = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()
finally:
    if telemetry is not None:
        telemetry.terminate()
        try:
            telemetry.wait(timeout=5)
        except subprocess.TimeoutExpired:
            telemetry.kill()
    (work / "job-result.json").write_text(json.dumps({
        "status": "failed" if failure else "complete", "failure": failure,
    }, indent=2) + "\\n")
    if output.exists():
        with tarfile.open(work / "lora_sanity_results.tar.gz", "w:gz") as bundle:
            bundle.add(output, arcname="sanity_results")
if failure:
    raise RuntimeError(failure)
'''


def build_kernel(account: str, accelerator: str) -> tuple[Path, str]:
    stage = WORK_ROOT / "kernel"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    reference = f"{account}/{KERNEL_SLUG}"
    (stage / "run_sanity.py").write_text(kernel_source(), encoding="utf-8")
    metadata: dict[str, Any] = {
        "id": reference,
        "title": "NegPos LoRA Sanity Check",
        "code_file": "run_sanity.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": accelerator,
        "dataset_sources": [f"{account}/{PAYLOAD_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (stage / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return stage, reference


def submit(account: str, accelerator: str, environment: dict[str, str]) -> str:
    stage, reference = build_kernel(account, accelerator)
    common.run_kaggle(
        ["kernels", "push", "-p", str(stage), "--accelerator", accelerator], environment
    )
    return reference


def retrieve(reference: str, environment: dict[str, str]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = REPO_ROOT / "output" / "lora_sanity" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    common.run_kaggle(
        ["kernels", "output", reference, "-p", str(destination), "--force"], environment
    )
    archive = destination / "lora_sanity_results.tar.gz"
    if archive.is_file():
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(destination, filter="data")
    summary = destination / "sanity_results" / "summary.json"
    if not summary.is_file():
        raise FileNotFoundError(f"Sanity summary missing; inspect {destination}")
    return destination


def main() -> int:
    args = parse_args()
    environment = common.load_environment(args.env_file)
    account = common.username(args, environment)
    reference = f"{account}/{KERNEL_SLUG}"
    if args.command == "status":
        common.status(reference, environment)
    elif args.command == "retrieve":
        print(retrieve(reference, environment))
    else:
        publish(account, environment)
        reference = submit(account, args.accelerator, environment)
        print(f"Submitted {reference}", flush=True)
        common.wait_for_kernel(reference, environment)
        print(f"Sanity outputs: {retrieve(reference, environment)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
