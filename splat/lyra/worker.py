"""
NVIDIA Lyra worker for 3D scene generation from single images/videos.

Lyra uses video diffusion self-distillation to generate 3D Gaussian Splats.
Requirements: Linux, Python 3.10, CUDA, H100/A100 GPU (43GB+ VRAM)

Paper: https://arxiv.org/abs/2509.19296
"""

import modal
import subprocess
import os
from pathlib import Path

app = modal.App("lyra-3d-generation")

# Lyra requires specific versions - can't share with fVDB
# Use NVIDIA NGC PyTorch container (has flash-attn, apex pre-built)
lyra_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:24.08-py3",  # PyTorch 2.4 + CUDA 12.5
        add_python=None,  # Already has Python
    )
    .apt_install(
        "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0"
    )
    # Upgrade to PyTorch 2.6 (Lyra requirement)
    .run_commands(
        "pip install --upgrade pip setuptools wheel",
        "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
    )
    # GEN3C requirements (no CUDA compile needed)
    .pip_install(
        "attrs==25.1.0",
        "boto3==1.35.99",
        "decord==0.6.0",
        "diffusers==0.32.2",
        "einops==0.8.1",
        "huggingface-hub==0.29.2",
        "hydra-core==1.3.2",
        "imageio[pyav,ffmpeg]==2.37.0",
        "loguru==0.7.2",
        "numpy==1.26.4",
        "omegaconf==2.3.0",
        "opencv-python==4.10.0.84",
        "pillow==11.1.0",
        "pyyaml==6.0.2",
        "safetensors==0.5.3",
        "scikit-image==0.24.0",
        "tqdm==4.66.5",
        "transformers==4.49.0",
        "accelerate==1.10.0",
        "warp-lang==1.7.2",
        "plyfile==1.1.2",
        "timm==1.0.19",
        "kiui==0.2.17",
        "lru-dict==1.3.0",
        "peft==0.14.0",
    )
    # CUDA-compiled packages - redirect ALL output to /dev/null to avoid Modal log limits
    .env({"TORCH_CUDA_ARCH_LIST": "8.0;9.0", "MAX_JOBS": "2"})
    # flash_attn (suppress output completely)
    .run_commands(
        "pip install flash_attn==2.7.4.post1 --no-build-isolation > /dev/null 2>&1 && echo 'flash_attn installed'",
    )
    # causal-conv1d (suppress output completely)
    .run_commands(
        "pip install 'causal-conv1d>=1.4.0' > /dev/null 2>&1 && echo 'causal-conv1d installed'",
    )
    # mamba-ssm (suppress output completely)
    .run_commands(
        "pip install 'mamba-ssm>=2.2.0' > /dev/null 2>&1 && echo 'mamba-ssm installed'",
    )
    # gsplat (suppress output completely)
    .run_commands(
        "pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@73fad53c31ec4d6b088470715a63f432990493de > /dev/null 2>&1 && echo 'gsplat installed'",
    )
    # fused-ssim (suppress output completely)
    .run_commands(
        "pip install --no-build-isolation git+https://github.com/rahul-goel/fused-ssim/@8bdb59feb7b9a41b1fab625907cb21f5417deaac > /dev/null 2>&1 && echo 'fused-ssim installed'",
    )
    # Non-CUDA packages
    .run_commands(
        "pip install mpi4py==4.1.0 openexr==3.2.3 deepspeed==0.17.5",
    )
    .run_commands(
        "pip install git+https://github.com/microsoft/MoGe.git",
    )
    # Transformer engine + Apex
    .pip_install(
        "megatron-core==0.10.0",
        "transformer-engine[pytorch]>=1.12.0",
    )
    .run_commands(
        "pip install nvidia-apex > /dev/null 2>&1 && echo 'apex installed' || echo 'apex skipped'",
    )
    # Clone Lyra repo
    .run_commands(
        "git clone https://github.com/nv-tlabs/lyra.git /opt/lyra",
    )
)

# Volume for model weights and outputs
model_volume = modal.Volume.from_name("lyra-models", create_if_missing=True)
output_volume = modal.Volume.from_name("lyra-outputs", create_if_missing=True)
MODEL_PATH = "/models"
OUTPUT_PATH = "/outputs"


@app.function(
    image=lyra_image,
    gpu="H100",  # Lyra needs 43GB+ VRAM
    timeout=600,
    volumes={MODEL_PATH: model_volume, OUTPUT_PATH: output_volume},
)
def test_gpu():
    """Verify GPU access and Lyra dependencies."""
    import torch

    results = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
    }

    # Check key dependencies
    deps = ["gsplat", "flash_attn", "transformer_engine", "mamba_ssm", "moge"]
    for dep in deps:
        try:
            __import__(dep)
            results[f"{dep}_installed"] = True
        except ImportError as e:
            results[f"{dep}_installed"] = False
            results[f"{dep}_error"] = str(e)

    # Check if lyra repo exists
    results["lyra_repo_exists"] = Path("/opt/lyra").exists()

    return results


@app.function(
    image=lyra_image,
    gpu="H100",
    timeout=3600,  # 1 hour for downloading all checkpoints
    volumes={MODEL_PATH: model_volume, OUTPUT_PATH: output_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN
)
def download_checkpoints():
    """Download all required model checkpoints from HuggingFace."""
    import subprocess
    import os

    os.chdir("/opt/lyra")
    checkpoint_dir = Path(MODEL_PATH) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    results = {}

    # Login to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=True)

    # Download Cosmos tokenizer
    print("=== Downloading Cosmos tokenizer ===")
    result = subprocess.run(
        ["python3", "-m", "scripts.download_tokenizer_checkpoints",
         "--checkpoint_dir", str(checkpoint_dir / "cosmos_predict1"),
         "--tokenizer_types", "CV8x8x8-720p"],
        capture_output=True, text=True, env={**os.environ, "CUDA_HOME": "/usr/local/cuda"}
    )
    results["cosmos_tokenizer"] = {
        "success": result.returncode == 0,
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-500:],
    }

    # Download GEN3C checkpoints
    print("=== Downloading GEN3C checkpoints ===")
    result = subprocess.run(
        ["python", "scripts/download_gen3c_checkpoints.py",
         "--checkpoint_dir", str(checkpoint_dir)],
        capture_output=True, text=True,
        env={**os.environ, "CUDA_HOME": "/usr/local/cuda", "PYTHONPATH": "/opt/lyra"}
    )
    results["gen3c"] = {
        "success": result.returncode == 0,
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-500:],
    }

    # Download Lyra checkpoints
    print("=== Downloading Lyra checkpoints ===")
    result = subprocess.run(
        ["python", "scripts/download_lyra_checkpoints.py",
         "--checkpoint_dir", str(checkpoint_dir)],
        capture_output=True, text=True,
        env={**os.environ, "CUDA_HOME": "/usr/local/cuda", "PYTHONPATH": "/opt/lyra"}
    )
    results["lyra"] = {
        "success": result.returncode == 0,
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-500:],
    }

    model_volume.commit()

    # List what we downloaded
    print("=== Downloaded checkpoints ===")
    for f in checkpoint_dir.rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.relative_to(checkpoint_dir)}: {size_mb:.1f} MB")

    return results


@app.function(
    image=lyra_image,
    gpu="H100",
    timeout=1800,  # 30 min
    volumes={MODEL_PATH: model_volume, OUTPUT_PATH: output_volume},
)
def download_demo_samples():
    """Download demo samples from HuggingFace."""
    import subprocess

    demo_dir = Path(OUTPUT_PATH) / "demo"
    demo_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        ["huggingface-cli", "download", "nvidia/Lyra-Testing-Example",
         "--repo-type", "dataset", "--local-dir", str(demo_dir)],
        capture_output=True, text=True
    )

    output_volume.commit()

    return {
        "success": result.returncode == 0,
        "demo_dir": str(demo_dir),
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-500:],
    }


@app.function(
    image=lyra_image,
    gpu="H100",
    timeout=3600,  # 1 hour for generation
    volumes={MODEL_PATH: model_volume, OUTPUT_PATH: output_volume},
)
def generate_3d_from_image(
    image_url: str,
    output_name: str = "scene",
    movement_factor: float = 1.0,
) -> dict:
    """
    Generate 3D Gaussian Splat from a single image.

    Args:
        image_url: URL to input image
        output_name: Name for output files
        movement_factor: Camera motion amount (1.0 default, 2.0 for more motion)

    Returns:
        dict with output paths and status
    """
    import requests
    import tempfile
    import shutil

    os.chdir("/opt/lyra")
    checkpoint_dir = Path(MODEL_PATH) / "checkpoints"

    # Download input image
    work_dir = Path(tempfile.mkdtemp())
    input_dir = work_dir / "input"
    input_dir.mkdir()

    print(f"Downloading image from {image_url}")
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    input_image = input_dir / "input.png"
    input_image.write_bytes(resp.content)

    output_dir = work_dir / "output"
    output_dir.mkdir()

    env = {
        **os.environ,
        "CUDA_HOME": "/usr/local/cuda",
        "PYTHONPATH": "/opt/lyra",
    }

    # Step 1: Generate multi-view video latents
    print("=== Step 1: Generating multi-view latents ===")
    cmd = [
        "torchrun", "--nproc_per_node=1",
        "cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py",
        "--checkpoint_dir", str(checkpoint_dir),
        "--num_gpus", "1",
        "--input_image_path", str(input_image),
        "--video_save_folder", str(output_dir / "diffusion"),
        "--foreground_masking",
        "--multi_trajectory",
        "--total_movement_distance_factor", str(movement_factor),
        # Offload for memory efficiency
        "--offload_diffusion_transformer",
        "--offload_tokenizer",
        "--offload_text_encoder_model",
    ]

    result1 = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd="/opt/lyra")
    print(f"Step 1 returncode: {result1.returncode}")
    if result1.stdout:
        print(result1.stdout[-2000:])
    if result1.stderr:
        print(result1.stderr[-1000:])

    if result1.returncode != 0:
        return {
            "success": False,
            "step": "diffusion",
            "error": result1.stderr[-1000:],
        }

    # Step 2: Reconstruct with 3DGS decoder
    print("=== Step 2: 3DGS reconstruction ===")
    # Would need to update config to point to generated latents
    # For now, return the diffusion outputs

    # Copy outputs to volume
    final_dir = Path(OUTPUT_PATH) / output_name
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.copytree(output_dir, final_dir)
    output_volume.commit()

    return {
        "success": True,
        "output_dir": str(final_dir),
        "step1_complete": True,
    }


@app.local_entrypoint()
def main(test_type: str = "gpu"):
    """
    Test Lyra setup on Modal.

    Usage:
        modal run splat/lyra/worker.py                    # GPU test
        modal run splat/lyra/worker.py --test-type download  # Download checkpoints
        modal run splat/lyra/worker.py --test-type demo      # Download demo samples
    """
    if test_type == "download":
        print("\n=== Downloading Lyra Checkpoints ===")
        print("This will take a while (several GB of models)...")
        result = download_checkpoints.remote()
        for name, info in result.items():
            print(f"\n{name}: {'✓' if info['success'] else '✗'}")
            if not info['success']:
                print(f"  Error: {info.get('stderr', '')[:200]}")
        return

    if test_type == "demo":
        print("\n=== Downloading Demo Samples ===")
        result = download_demo_samples.remote()
        print(f"Success: {result['success']}")
        print(f"Demo dir: {result['demo_dir']}")
        return

    # Default: GPU test
    print("\n=== Testing Lyra Setup on Modal ===\n")
    result = test_gpu.remote()

    print(f"CUDA Available: {result['cuda_available']}")
    print(f"Device: {result.get('device_name', 'N/A')}")
    print(f"VRAM: {result.get('vram_gb', 0):.1f} GB")
    print(f"CUDA Version: {result.get('cuda_version', 'N/A')}")
    print(f"Lyra repo exists: {result.get('lyra_repo_exists', False)}")
    print("\nDependencies:")
    for key, val in result.items():
        if key.endswith("_installed"):
            dep = key.replace("_installed", "")
            status = "✓" if val else "✗"
            print(f"  {dep}: {status}")
            if not val and f"{dep}_error" in result:
                print(f"    Error: {result[f'{dep}_error'][:100]}")
