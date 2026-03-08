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

# Shared env dict for all subprocess calls inside the container
LYRA_ENV_KEYS = {
    "CUDA_HOME": "/usr/local/cuda",
    "PYTHONPATH": "/opt/lyra",
    "NVTE_FRAMEWORK": "pytorch",
}


def _lyra_env():
    return {**os.environ, **LYRA_ENV_KEYS}


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

    # Check if checkpoints exist
    ckpt_dir = Path(MODEL_PATH) / "checkpoints"
    results["checkpoints_exist"] = ckpt_dir.exists()
    if ckpt_dir.exists():
        ckpt_files = list(ckpt_dir.rglob("*.pt")) + list(ckpt_dir.rglob("*.jit"))
        results["checkpoint_count"] = len(ckpt_files)
        results["checkpoint_files"] = [str(f.relative_to(ckpt_dir)) for f in ckpt_files[:20]]

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
    os.chdir("/opt/lyra")
    checkpoint_dir = Path(MODEL_PATH) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    env = _lyra_env()

    results = {}

    # HF_TOKEN env var is picked up automatically by huggingface_hub.
    # No CLI login needed — just make sure the env is passed to subprocesses.
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Download Cosmos tokenizer directly via huggingface_hub (the script may not work in NGC)
    print("=== Downloading Cosmos tokenizer ===")
    try:
        from huggingface_hub import snapshot_download
        tokenizer_dir = checkpoint_dir / "Cosmos-Tokenize1-CV8x8x8-720p"
        snapshot_download(
            "nvidia/Cosmos-Tokenize1-CV8x8x8-720p",
            local_dir=str(tokenizer_dir),
            token=hf_token,
        )
        results["cosmos_tokenizer"] = {"success": True, "stdout": str(tokenizer_dir), "stderr": ""}
    except Exception as e:
        results["cosmos_tokenizer"] = {"success": False, "stdout": "", "stderr": str(e)}
    print(f"  Cosmos tokenizer: {'OK' if results['cosmos_tokenizer']['success'] else 'FAILED'}")

    # Download GEN3C checkpoints
    print("=== Downloading GEN3C checkpoints ===")
    result = subprocess.run(
        ["python", "scripts/download_gen3c_checkpoints.py",
         "--checkpoint_dir", str(checkpoint_dir)],
        capture_output=True, text=True, env=env,
    )
    results["gen3c"] = {
        "success": result.returncode == 0,
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-500:],
    }
    print(f"  GEN3C: {'OK' if result.returncode == 0 else 'FAILED'}")

    # Download Lyra checkpoints
    print("=== Downloading Lyra checkpoints ===")
    result = subprocess.run(
        ["python", "scripts/download_lyra_checkpoints.py",
         "--checkpoint_dir", str(checkpoint_dir)],
        capture_output=True, text=True, env=env,
    )
    results["lyra"] = {
        "success": result.returncode == 0,
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-500:],
    }
    print(f"  Lyra: {'OK' if result.returncode == 0 else 'FAILED'}")

    model_volume.commit()

    # List what we downloaded
    print("\n=== Downloaded checkpoints ===")
    total_size = 0
    for f in sorted(checkpoint_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            total_size += size_mb
            print(f"  {f.relative_to(checkpoint_dir)}: {size_mb:.1f} MB")
    print(f"\nTotal: {total_size / 1024:.1f} GB")

    return results


@app.function(
    image=lyra_image,
    gpu="H100",
    timeout=3600,  # 1 hour for full pipeline
    volumes={MODEL_PATH: model_volume, OUTPUT_PATH: output_volume},
)
def generate_3d_from_image(
    image_bytes: bytes,
    image_name: str = "input.png",
    output_name: str = "scene",
    movement_factor: float = 1.0,
) -> dict:
    """
    Full pipeline: single image -> multi-view latents -> 3D Gaussian PLY.

    Args:
        image_bytes: Raw bytes of the input image (PNG/JPG).
        image_name: Filename for the input.
        output_name: Name prefix for output files on the volume.
        movement_factor: Camera motion magnitude (1.0 default).

    Returns:
        dict with success, PLY paths, timing, and logs.
    """
    import time
    import shutil
    import tempfile

    os.chdir("/opt/lyra")
    checkpoint_dir = Path(MODEL_PATH) / "checkpoints"
    env = _lyra_env()

    # Install missing runtime deps and fix version conflicts
    subprocess.run(["pip", "install", "-q", "--no-deps", "iopath"], capture_output=True)
    subprocess.run(["pip", "install", "-q", "portalocker", "termcolor"], capture_output=True)
    subprocess.run(["pip", "install", "-q", "huggingface-hub>=0.26.0,<1.0"], capture_output=True)

    # --- Fix 0: Disable broken .so files from NGC container (apex) ---
    # The NGC container (PyTorch 2.5) has C extensions that are ABI-incompatible
    # with our PyTorch 2.6 upgrade. Only disable known NGC packages (apex).
    # Do NOT disable packages we installed ourselves (mamba-ssm, causal-conv1d, etc.)
    _site = Path("/usr/local/lib/python3.10/dist-packages")
    _apex_modules = [
        "amp_C", "bnp", "cudnn_gbn_lib", "distributed_adam_cuda",
        "distributed_lamb_cuda", "fast_bottleneck", "fast_layer_norm",
        "fast_multihead_attn", "fmhalib", "focal_loss_cuda", "fused_adam_cuda",
        "fused_conv_bias_relu", "fused_dense_cuda", "fused_lamb_cuda",
        "fused_layer_norm_cuda", "fused_rotary_positional_embedding",
        "fused_weight_gradient_mlp_cuda", "generic_scaled_masked_softmax_cuda",
        "group_norm_cuda", "instance_norm_nvfuser_cuda", "mlp_cuda",
        "nccl_p2p_cuda", "peer_memory_cuda", "scaled_masked_softmax_cuda",
        "scaled_softmax_cuda", "scaled_upper_triang_masked_softmax_cuda",
        "syncbn", "transducer_joint_cuda", "transducer_loss_cuda",
        "xentropy_cuda", "_apex_gpu_direct_storage", "_apex_nccl_allocator",
    ]
    _disabled = []
    for mod_name in _apex_modules:
        so_files = list(_site.glob(f"{mod_name}.cpython-310-x86_64-linux-gnu.so"))
        for so_path in so_files:
            so_path.rename(so_path.with_suffix(".so.disabled"))
            so_path.with_name(mod_name + ".py").write_text(
                "# Auto-stub: original .so had ABI mismatch (NGC PyTorch 2.5 vs 2.6)\n"
                "def __getattr__(name): return None\n"
            )
            _disabled.append(mod_name)
    if _disabled:
        print(f"Disabled {len(_disabled)} NGC apex .so files")

    # --- Fix 1: OpenCV cv2.dnn.DictValue AttributeError in NGC container ---
    cv2_typing = Path("/usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py")
    if cv2_typing.exists():
        content = cv2_typing.read_text()
        if "cv2.dnn.DictValue" in content and "try:" not in content.split("cv2.dnn.DictValue")[0][-20:]:
            content = content.replace(
                "LayerId = cv2.dnn.DictValue",
                "try:\n    LayerId = cv2.dnn.DictValue\nexcept AttributeError:\n    LayerId = int"
            )
            cv2_typing.write_text(content)
            print("Patched cv2.typing")

    # --- Fix 2: Replace transformer_engine with stub package ---
    # Root cause: the TE C++ extension (.so) was compiled against PyTorch 2.5
    # (from the NGC container) but we upgraded to PyTorch 2.6. The .so has
    # undefined symbols from PyTorch's C10 CUDA library, causing ImportError.
    # TE is only needed for training (FP8, fused kernels); inference is fine
    # with standard PyTorch ops. So we replace TE with a lightweight stub.
    import shutil as _shutil
    _site = Path("/usr/local/lib/python3.10/dist-packages")
    te_dir = _site / "transformer_engine"
    te_bak = _site / "_transformer_engine_real"
    if te_dir.exists() and not te_bak.exists():
        _shutil.move(str(te_dir), str(te_bak))
        te_dir.mkdir()
        # Main __init__.py: register a meta path finder that creates stub
        # modules for ANY transformer_engine.* submodule import on the fly.
        # This handles deep imports like transformer_engine.pytorch.attention
        # without needing to create every possible submodule file.
        (te_dir / "__init__.py").write_text(
            'import sys, types, importlib.abc, importlib.machinery\n'
            '\n'
            '__version__ = "2.12.0"\n'
            '\n'
            'class _TEStubFinder(importlib.abc.MetaPathFinder):\n'
            '    def find_spec(self, fullname, path, target=None):\n'
            '        if fullname.startswith("transformer_engine"):\n'
            '            return importlib.machinery.ModuleSpec(\n'
            '                fullname, _TEStubLoader(), is_package=True,\n'
            '            )\n'
            '        return None\n'
            '\n'
            'class _TEStubLoader(importlib.abc.Loader):\n'
            '    def create_module(self, spec): return None\n'
            '    def exec_module(self, module):\n'
            '        module.__path__ = []\n'
            '        module.__getattr__ = lambda name: None\n'
            '\n'
            '# Append (not insert) so real files are tried first\n'
            'sys.meta_path.append(_TEStubFinder())\n'
            '\n'
            'def __getattr__(name): return None\n'
        )
        print("Replaced transformer_engine with stub package + meta finder")

    # --- Fix 3: Stub megatron's TE extension module ---
    # megatron.core.extensions.transformer_engine defines classes like
    # TELinear(te.pytorch.Linear) and exports fused_apply_rotary_pos_emb_thd.
    # Replace with a module that returns None for any attribute access, so
    # `from megatron.core.extensions.transformer_engine import X` always
    # returns None (callers check `if X is not None` before using).
    megatron_te = _site / "megatron" / "core" / "extensions" / "transformer_engine.py"
    if megatron_te.exists():
        content = megatron_te.read_text()
        if "PATCHED_BY_LYRA" not in content:
            megatron_te.write_text(
                '# PATCHED_BY_LYRA: stub - TE not available (ABI mismatch)\n'
                'def __getattr__(name):\n'
                '    return None\n'
            )
            print("Stubbed megatron TE extension")

    # Clear __pycache__ so torchrun subprocesses load patched .py files
    for pkg in ["transformer_engine", "_transformer_engine_real", "megatron", "cv2"]:
        pkg_dir = _site / pkg
        if pkg_dir.exists():
            for pycache in pkg_dir.rglob("__pycache__"):
                _shutil.rmtree(pycache, ignore_errors=True)
    print("Cleared __pycache__")

    work_dir = Path(tempfile.mkdtemp())
    input_dir = work_dir / "input"
    latent_dir = work_dir / "latents"
    recon_dir = work_dir / "recon"
    for d in [input_dir, latent_dir, recon_dir]:
        d.mkdir(parents=True)

    # Save input image
    input_image = input_dir / image_name
    input_image.write_bytes(image_bytes)
    print(f"Saved input image: {input_image} ({len(image_bytes) / 1024:.0f} KB)")

    # ------------------------------------------------------------------
    # Step 1: Generate multi-view video latents from single image
    # Uses Gen3C diffusion model to create 6 camera trajectories
    # ------------------------------------------------------------------
    print("\n========== Step 1: Multi-view latent generation ==========")
    t0 = time.time()

    step1_cmd = [
        "torchrun", "--nproc_per_node=1",
        "cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py",
        "--checkpoint_dir", str(checkpoint_dir),
        "--num_gpus", "1",
        "--input_image_path", str(input_image),
        "--video_save_folder", str(latent_dir),
        "--foreground_masking",
        "--multi_trajectory",
        "--total_movement_distance_factor", str(movement_factor),
        # Offload to system RAM to stay under VRAM budget
        "--offload_diffusion_transformer",
        "--offload_tokenizer",
        "--offload_text_encoder_model",
    ]

    print(f"CMD: {' '.join(step1_cmd)}")
    proc1 = subprocess.run(
        step1_cmd, capture_output=True, text=True,
        env=env, cwd="/opt/lyra", timeout=2400,
    )
    step1_time = time.time() - t0
    print(f"Step 1 done in {step1_time:.0f}s (rc={proc1.returncode})")
    if proc1.stdout:
        print(f"STDOUT (last 2000):\n{proc1.stdout[-2000:]}")
    if proc1.returncode != 0:
        print(f"STDERR:\n{proc1.stderr[-2000:]}")
        return {
            "success": False,
            "step": 1,
            "error": proc1.stderr[-2000:] or proc1.stdout[-2000:],
            "step1_time_s": round(step1_time),
        }

    # List what Step 1 produced
    latent_files = sorted(latent_dir.rglob("*"))
    print(f"\nStep 1 produced {len(latent_files)} files:")
    for f in latent_files[:30]:
        if f.is_file():
            print(f"  {f.relative_to(latent_dir)} ({f.stat().st_size / 1024:.0f} KB)")

    # ------------------------------------------------------------------
    # Step 2: Reconstruct 3D Gaussians from latents via Lyra decoder
    #
    # Lyra's sample.py reads a YAML config that points to:
    #   - The generated latent directory (dataset_name → registry.py path)
    #   - The Lyra checkpoint (ckpt_path)
    #   - Training configs for model architecture
    #
    # The dataset registry entry "lyra_static_demo_generated" expects
    # latents at: assets/demo/static/diffusion_output_generated
    # So we symlink our output there.
    # ------------------------------------------------------------------
    print("\n========== Step 2: 3D Gaussian reconstruction ==========")
    t1 = time.time()

    # Symlink generated latents to where Lyra's registry expects them
    expected_latent_path = Path("/opt/lyra/assets/demo/static/diffusion_output_generated")
    expected_latent_path.parent.mkdir(parents=True, exist_ok=True)
    if expected_latent_path.exists() or expected_latent_path.is_symlink():
        expected_latent_path.unlink()
    expected_latent_path.symlink_to(latent_dir)
    print(f"Symlinked {latent_dir} -> {expected_latent_path}")

    # Write custom inference config
    lyra_config = recon_dir / "lyra_inference.yaml"
    lyra_config.write_text(f"""\
# Auto-generated config for single-image reconstruction
out_dir_inference: {recon_dir / 'output'}

# Use the generated latents (matches registry entry)
dataset_name: lyra_static_demo_generated

# 6 camera viewpoints in optimal order
static_view_indices_fixed: ['5', '0', '1', '2', '3', '4']

# Render every 4th frame (faster, still good quality)
target_index_subsample: 4

# Static scene mode
set_manual_time_idx: true

# Model architecture configs (from Lyra repo)
config_path:
  - configs/training/default.yaml
  - configs/training/3dgs_res_704_1280_views_121_multi_6_prune.yaml

# Lyra static checkpoint
ckpt_path: {checkpoint_dir}/Lyra/lyra_static.pt

# Export Gaussian PLY files
save_gaussians: true
save_gaussians_orig: true
""")

    step2_cmd = [
        "accelerate", "launch",
        "--num_processes", "1",
        "sample.py",
        "--config", str(lyra_config),
    ]

    print(f"CMD: {' '.join(step2_cmd)}")
    proc2 = subprocess.run(
        step2_cmd, capture_output=True, text=True,
        env=env, cwd="/opt/lyra", timeout=2400,
    )
    step2_time = time.time() - t1
    print(f"Step 2 done in {step2_time:.0f}s (rc={proc2.returncode})")
    if proc2.stdout:
        print(f"STDOUT (last 2000):\n{proc2.stdout[-2000:]}")
    if proc2.returncode != 0:
        print(f"STDERR:\n{proc2.stderr[-2000:]}")
        return {
            "success": False,
            "step": 2,
            "error": proc2.stderr[-2000:] or proc2.stdout[-2000:],
            "step1_time_s": round(step1_time),
            "step2_time_s": round(step2_time),
        }

    # ------------------------------------------------------------------
    # Collect outputs: PLY files + rendered videos
    # ------------------------------------------------------------------
    output_root = recon_dir / "output"
    ply_files = sorted(output_root.rglob("*.ply"))
    video_files = sorted(output_root.rglob("*.mp4"))

    print(f"\nFound {len(ply_files)} PLY, {len(video_files)} MP4 files")

    # Copy everything to the persistent volume
    final_dir = Path(OUTPUT_PATH) / output_name
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True)

    saved_plys = []
    saved_videos = []

    for ply in ply_files:
        dest = final_dir / "gaussians" / ply.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ply, dest)
        size_mb = ply.stat().st_size / 1024 / 1024
        saved_plys.append(str(dest.relative_to(OUTPUT_PATH)))
        print(f"  PLY: {dest.relative_to(OUTPUT_PATH)} ({size_mb:.1f} MB)")

    for vid in video_files:
        dest = final_dir / "videos" / vid.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vid, dest)
        saved_videos.append(str(dest.relative_to(OUTPUT_PATH)))

    output_volume.commit()

    total_time = step1_time + step2_time
    print(f"\nTotal pipeline time: {total_time:.0f}s")

    return {
        "success": True,
        "step1_time_s": round(step1_time),
        "step2_time_s": round(step2_time),
        "total_time_s": round(total_time),
        "ply_files": saved_plys,
        "video_files": saved_videos,
        "output_dir": str(final_dir.relative_to(OUTPUT_PATH)),
    }


@app.function(
    image=lyra_image,
    volumes={OUTPUT_PATH: output_volume},
    timeout=120,
)
def list_outputs() -> list[dict]:
    """List all files on the output volume."""
    results = []
    for p in sorted(Path(OUTPUT_PATH).rglob("*")):
        if p.is_file():
            results.append({
                "path": str(p.relative_to(OUTPUT_PATH)),
                "size_mb": round(p.stat().st_size / 1024 / 1024, 2),
            })
    return results


@app.function(
    image=lyra_image,
    volumes={OUTPUT_PATH: output_volume},
    timeout=120,
)
def get_ply(name: str) -> bytes:
    """Download a PLY file from the output volume by relative path."""
    path = Path(OUTPUT_PATH) / name
    if not path.exists():
        available = [
            str(p.relative_to(OUTPUT_PATH))
            for p in Path(OUTPUT_PATH).rglob("*.ply")
        ]
        raise FileNotFoundError(f"PLY '{name}' not found. Available: {available}")
    return path.read_bytes()


@app.local_entrypoint()
def main(
    action: str = "test",
    image_path: str = "",
    output_name: str = "scene",
    movement_factor: float = 1.0,
):
    """
    Lyra 3D reconstruction on Modal.

    Usage:
        modal run splat/lyra/worker.py                                         # Test setup
        modal run splat/lyra/worker.py --action download                       # Download checkpoints
        modal run splat/lyra/worker.py --action reconstruct --image-path p.png # Full pipeline
        modal run splat/lyra/worker.py --action list                           # List outputs
        modal run splat/lyra/worker.py --action get-ply --image-path scene/gaussians/gaussians_0.ply
    """
    if action == "test":
        print("\n=== Testing Lyra Setup on Modal ===\n")
        result = test_gpu.remote()
        print(f"CUDA Available: {result['cuda_available']}")
        print(f"Device: {result.get('device_name', 'N/A')}")
        print(f"VRAM: {result.get('vram_gb', 0):.1f} GB")
        print(f"CUDA Version: {result.get('cuda_version', 'N/A')}")
        print(f"Lyra repo exists: {result.get('lyra_repo_exists', False)}")
        print(f"Checkpoints exist: {result.get('checkpoints_exist', False)}")
        if result.get("checkpoint_files"):
            print(f"Checkpoint files ({result['checkpoint_count']}):")
            for f in result["checkpoint_files"]:
                print(f"  {f}")
        print("\nDependencies:")
        for key, val in result.items():
            if key.endswith("_installed"):
                dep = key.replace("_installed", "")
                status = "OK" if val else "MISSING"
                print(f"  {dep}: {status}")
                if not val and f"{dep}_error" in result:
                    print(f"    Error: {result[f'{dep}_error'][:100]}")

    elif action == "download":
        print("\n=== Downloading Lyra Checkpoints ===")
        print("This will take 15-30 min (several GB of models)...\n")
        result = download_checkpoints.remote()
        for name, info in result.items():
            ok = info["success"]
            print(f"  {name}: {'OK' if ok else 'FAILED'}")
            if not ok:
                print(f"    stderr: {info.get('stderr', '')[:300]}")

    elif action == "reconstruct":
        if not image_path:
            print("ERROR: --image-path is required")
            return

        img = Path(image_path)
        if not img.exists():
            print(f"ERROR: File not found: {image_path}")
            return

        print(f"\n=== Lyra 3D Reconstruction ===")
        print(f"  Input: {img.name} ({img.stat().st_size / 1024:.0f} KB)")
        print(f"  Output name: {output_name}")
        print(f"  Movement factor: {movement_factor}\n")

        result = generate_3d_from_image.remote(
            image_bytes=img.read_bytes(),
            image_name=img.name,
            output_name=output_name,
            movement_factor=movement_factor,
        )

        if result["success"]:
            print(f"\nSUCCESS")
            print(f"  Step 1 (diffusion): {result['step1_time_s']}s")
            print(f"  Step 2 (3DGS):      {result['step2_time_s']}s")
            print(f"  Total:              {result['total_time_s']}s")
            print(f"\n  PLY files:")
            for p in result["ply_files"]:
                print(f"    {p}")
            print(f"  Videos:")
            for v in result["video_files"]:
                print(f"    {v}")
            print(f"\n  To download: modal run splat/lyra/worker.py --action get-ply --image-path {result['ply_files'][0] if result['ply_files'] else '...'}")
        else:
            print(f"\nFAILED at step {result.get('step')}")
            print(f"  Error: {result.get('error', 'unknown')[:500]}")

    elif action == "list":
        print("\n=== Lyra Output Volume ===\n")
        files = list_outputs.remote()
        if not files:
            print("  (empty — run reconstruct first)")
        for f in files:
            print(f"  {f['path']}  ({f['size_mb']} MB)")

    elif action == "get-ply":
        if not image_path:
            print("ERROR: --image-path should be the PLY path on the volume")
            return
        print(f"Downloading: {image_path}")
        data = get_ply.remote(image_path)
        local_out = Path(image_path).name
        Path(local_out).write_bytes(data)
        print(f"Saved to ./{local_out} ({len(data) / 1024 / 1024:.1f} MB)")

    else:
        print(f"Unknown action: {action}")
        print("Actions: test, download, reconstruct, list, get-ply")
