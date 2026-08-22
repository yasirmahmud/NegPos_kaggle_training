#!/usr/bin/env python3
"""Publish, run, monitor, and retrieve the private Kaggle LoRA job."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT / ".tmp" / "kaggle_lora"
PAYLOAD_SLUG = "negpos-qwen35-lora-training-payload"
PAYLOAD_ARCHIVE = "negpos_training_payload.bin"
KERNEL_SLUG = "negpos-qwen3-5-positive-negative-loras"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("publish", "submit", "status", "retrieve", "resume", "run"),
    )
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--username", default=None)
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--skip-publish", action="store_true")
    return parser.parse_args()


def load_environment(path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    if path.is_file():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                continue
            try:
                values = shlex.split(raw_value, comments=True, posix=True)
                value = values[0] if values else ""
            except ValueError:
                value = raw_value.strip().strip("'\"")
            environment.setdefault(key, value)
    token_auth = bool(environment.get("KAGGLE_API_TOKEN"))
    legacy_auth = bool(environment.get("KAGGLE_USERNAME") and environment.get("KAGGLE_KEY"))
    if not token_auth and not legacy_auth:
        raise RuntimeError(
            "Set KAGGLE_API_TOKEN, or KAGGLE_USERNAME plus KAGGLE_KEY, in .env"
        )
    return environment


def username(args: argparse.Namespace, environment: dict[str, str]) -> str:
    value = args.username or environment.get("KAGGLE_USERNAME")
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise RuntimeError("A valid KAGGLE_USERNAME is required")
    return value.lower()


def kaggle_binary() -> Path:
    candidates = [
        REPO_ROOT / ".tools" / "kaggle-cli" / "bin" / "kaggle",
        Path("/home/touhid/code/COIL_RTL/.tools/kaggle-cli-current/bin/kaggle"),
        Path("/home/touhid/code/COIL_RTL/.tools/kaggle-cli-venv/bin/kaggle"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    discovered = shutil.which("kaggle")
    if discovered:
        return Path(discovered)
    raise RuntimeError("Kaggle CLI is unavailable; run scripts/run_overnight_training.sh")


def run_kaggle(
    arguments: list[str],
    environment: dict[str, str],
    *,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(kaggle_binary()), *arguments],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=capture,
        check=check,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(account: str) -> tuple[Path, str]:
    stage = WORK_ROOT / "payload"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    uncompressed = stage / "payload.tar"
    archive = stage / PAYLOAD_ARCHIVE
    inputs = [
        (REPO_ROOT / "scripts" / "train_loras.py", "negpos/scripts/train_loras.py"),
        (
            REPO_ROOT / "data" / "top2000_four_negatives.jsonl",
            "negpos/data/top2000_four_negatives.jsonl",
        ),
    ]
    with tarfile.open(uncompressed, "w") as bundle:
        for source, destination in inputs:
            if not source.is_file():
                raise FileNotFoundError(source)
            info = bundle.gettarinfo(str(source), arcname=destination)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as handle:
                bundle.addfile(info, handle)
    with (
        uncompressed.open("rb") as source,
        archive.open("wb") as destination,
        gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0, compresslevel=1) as compressed,
    ):
        shutil.copyfileobj(source, compressed)
    uncompressed.unlink()
    payload_hash = sha256(archive)
    (stage / "payload-manifest.json").write_text(
        json.dumps(
            {"schema_version": "1.0.0", "archive": archive.name, "sha256": payload_hash},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (stage / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "NegPos Qwen3.5 Private LoRA Training Payload",
                "id": f"{account}/{PAYLOAD_SLUG}",
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return stage, payload_hash


def remote_hash(reference: str, environment: dict[str, str]) -> str | None:
    destination = WORK_ROOT / "remote_manifest"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    result = run_kaggle(
        [
            "datasets",
            "download",
            reference,
            "-f",
            "payload-manifest.json",
            "-p",
            str(destination),
            "--unzip",
            "--force",
            "--quiet",
        ],
        environment,
        capture=True,
        check=False,
    )
    manifests = list(destination.rglob("payload-manifest.json"))
    if result.returncode or len(manifests) != 1:
        return None
    return str(json.loads(manifests[0].read_text(encoding="utf-8")).get("sha256") or "")


def wait_for_dataset(
    reference: str, expected_hash: str, environment: dict[str, str], timeout: int = 3600
) -> None:
    deadline = time.monotonic() + timeout
    next_notice = time.monotonic()
    while time.monotonic() < deadline:
        # The token-auth CLI can temporarily return 403 from the private
        # dataset-status endpoint immediately after creation. The payload is
        # ready only when its manifest is downloadable and has the expected
        # hash, so use that as the authoritative readiness check.
        if remote_hash(reference, environment) == expected_hash:
            return
        result = run_kaggle(
            ["datasets", "status", reference], environment, capture=True, check=False
        )
        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode == 0 and (
            "error" in output.lower() or "failed" in output.lower()
        ):
            raise RuntimeError(f"Kaggle dataset processing failed: {output}")
        if result.returncode != 0 and time.monotonic() >= next_notice:
            print(
                "Waiting for the private Kaggle payload to become readable "
                f"(status endpoint returned code {result.returncode})",
                flush=True,
            )
            next_notice = time.monotonic() + 60
        time.sleep(10)
    raise TimeoutError(f"Timed out waiting for dataset {reference}")


def publish(account: str, environment: dict[str, str]) -> str:
    stage, payload_hash = build_payload(account)
    reference = f"{account}/{PAYLOAD_SLUG}"
    exists = (
        run_kaggle(
            ["datasets", "files", reference], environment, capture=True, check=False
        ).returncode
        == 0
    )
    if exists and remote_hash(reference, environment) == payload_hash:
        print(f"Payload already current: {payload_hash[:12]}", flush=True)
    elif exists:
        run_kaggle(
            [
                "datasets",
                "version",
                "-p",
                str(stage),
                "-m",
                f"Training payload {payload_hash[:12]}",
                "-r",
                "zip",
            ],
            environment,
        )
    else:
        run_kaggle(["datasets", "create", "-p", str(stage), "-r", "zip"], environment)
    wait_for_dataset(reference, payload_hash, environment)
    return reference


def kernel_source() -> str:
    return '''"""Generated private Kaggle entry point for both LoRA training runs."""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
from pathlib import Path

work = Path("/kaggle/working")
archives = list(Path("/kaggle/input").rglob("negpos_training_payload.bin"))
if len(archives) != 1:
    raise RuntimeError(f"Expected one training payload, found {archives}")
with tarfile.open(archives[0], "r:gz") as bundle:
    bundle.extractall(work, filter="data")
project = work / "negpos"

# Kaggle's base image supplies CUDA PyTorch. Install only the model/fine-tuning
# stack, then try the optimized Qwen Gated DeltaNet backend without replacing torch.
# Its image currently contains torchao 0.10, which PEFT 0.19 explicitly rejects
# merely by detecting it. TorchAO is unused here, so remove it before importing PEFT.
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "--yes", "torchao"],
    check=False,
)
subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
        "transformers==5.12.0", "peft==0.19.1", "accelerate==1.14.0",
        "safetensors>=0.6", "einops>=0.8",
    ],
    check=True,
)
fla = subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet", "--no-deps",
        "fla-core", "flash-linear-attention",
    ],
    check=False,
)
print(f"FLA_INSTALL_RETURN_CODE={fla.returncode}", flush=True)
causal_conv = subprocess.run(
    [
        sys.executable, "-m", "pip", "install", "--quiet",
        "--no-build-isolation", "causal-conv1d>=1.5.0",
    ],
    check=False,
)
print(f"CAUSAL_CONV1D_INSTALL_RETURN_CODE={causal_conv.returncode}", flush=True)

os.environ["PYTHONPATH"] = str(project)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = str(work / "hf_cache")
output_root = work / "trained_loras"
command = [
    sys.executable,
    str(project / "scripts" / "train_loras.py"),
    "--data", str(project / "data" / "top2000_four_negatives.jsonl"),
    "--output-root", str(output_root),
]

print("GPU_TELEMETRY index,timestamp,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_w", flush=True)
telemetry = None
if shutil.which("nvidia-smi"):
    telemetry = subprocess.Popen([
        "nvidia-smi",
        "--query-gpu=index,timestamp,utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
        "--loop=30",
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
    result = {"schema_version": "1.0.0", "status": "failed" if failure else "complete", "failure": failure}
    (work / "job-result.json").write_text(json.dumps(result, indent=2) + "\\n")
    if output_root.exists():
        with tarfile.open(work / "positive_negative_loras.tar.gz", "w:gz") as bundle:
            bundle.add(output_root, arcname="trained_loras")
    shutil.rmtree(project, ignore_errors=True)
    shutil.rmtree(work / "hf_cache", ignore_errors=True)
    shutil.rmtree(output_root, ignore_errors=True)
if failure:
    raise RuntimeError(failure)
'''


def build_kernel(account: str, accelerator: str) -> tuple[Path, str]:
    stage = WORK_ROOT / "kernel"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    reference = f"{account}/{KERNEL_SLUG}"
    (stage / "run_training.py").write_text(kernel_source(), encoding="utf-8")
    metadata: dict[str, Any] = {
        "id": reference,
        "title": "NegPos Qwen3.5 Positive Negative LoRAs",
        "code_file": "run_training.py",
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
    run_kaggle(
        ["kernels", "push", "-p", str(stage), "--accelerator", accelerator],
        environment,
    )
    return reference


def status(reference: str, environment: dict[str, str]) -> str:
    result = run_kaggle(
        ["kernels", "status", reference], environment, capture=True, check=False
    )
    output = f"{result.stdout}\n{result.stderr}".strip()
    print(f"[{datetime.now(timezone.utc).isoformat()}] {output}", flush=True)
    if result.returncode != 0:
        return "unavailable"
    lowered = output.lower()
    for state in ("complete", "error", "failed", "cancel", "running", "queued"):
        if state in lowered:
            return state
    return "unknown"


def wait_for_kernel(reference: str, environment: dict[str, str], timeout: int = 46800) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = status(reference, environment)
        if state == "complete":
            return
        if state in {"error", "failed", "cancel"}:
            raise RuntimeError(f"Kaggle job ended with status {state}: {reference}")
        time.sleep(60)
    raise TimeoutError(f"Timed out waiting for {reference}")


def retrieve(reference: str, environment: dict[str, str]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = REPO_ROOT / "output" / "kaggle_lora" / stamp
    destination.mkdir(parents=True, exist_ok=False)
    run_kaggle(
        ["kernels", "output", reference, "-p", str(destination), "--force"],
        environment,
    )
    archive = destination / "positive_negative_loras.tar.gz"
    if archive.is_file():
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(destination, filter="data")
    result_path = destination / "job-result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            raise RuntimeError(f"Downloaded Kaggle job reports failure: {result}")
    positive = destination / "trained_loras" / "positive_lora" / "adapter_config.json"
    negative = destination / "trained_loras" / "negative_lora" / "adapter_config.json"
    if not positive.is_file() or not negative.is_file():
        raise FileNotFoundError("Downloaded output does not contain both LoRA adapters")
    return destination


def main() -> int:
    args = parse_args()
    environment = load_environment(args.env_file)
    account = username(args, environment)
    reference = f"{account}/{KERNEL_SLUG}"
    if args.command == "publish":
        print(publish(account, environment))
    elif args.command == "submit":
        print(submit(account, args.accelerator, environment))
    elif args.command == "status":
        status(reference, environment)
    elif args.command == "retrieve":
        print(retrieve(reference, environment))
    elif args.command == "resume":
        print(f"Monitoring existing job {reference}", flush=True)
        wait_for_kernel(reference, environment)
        print(f"Training outputs: {retrieve(reference, environment)}", flush=True)
    elif args.command == "run":
        if not args.skip_publish:
            publish(account, environment)
        reference = submit(account, args.accelerator, environment)
        print(f"Submitted {reference}", flush=True)
        wait_for_kernel(reference, environment)
        print(f"Training outputs: {retrieve(reference, environment)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
