"""Generated private Kaggle entry point for both LoRA training runs."""
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
    (work / "job-result.json").write_text(json.dumps(result, indent=2) + "\n")
    if output_root.exists():
        with tarfile.open(work / "positive_negative_loras.tar.gz", "w:gz") as bundle:
            bundle.add(output_root, arcname="trained_loras")
    shutil.rmtree(project, ignore_errors=True)
    shutil.rmtree(work / "hf_cache", ignore_errors=True)
    shutil.rmtree(output_root, ignore_errors=True)
if failure:
    raise RuntimeError(failure)
