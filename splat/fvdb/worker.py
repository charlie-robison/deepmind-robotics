"""
fVDB Reality Capture worker for Gaussian Splatting 3D reconstruction.

Uses NVIDIA's open-source fVDB Reality Capture toolbox (Apache 2.0 license).
Requirements: Linux, Python 3.10-3.13, CUDA 12.8, Ampere+ GPU (compute 8.0+)
"""

import modal
import subprocess
import os
import tempfile
from pathlib import Path

# Create a separate app for testing (can merge into main app later)
app = modal.App("fvdb-reality-capture")

# GPU image with fVDB Reality Capture
# Using NVIDIA CUDA 12.8 base + pre-built wheels from NVIDIA's index
fvdb_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg", "git", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands(
        # Install PyTorch 2.8 + torchvision 0.23.0 with CUDA 12.8 (must stay on 2.8 for fVDB)
        "pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128",
        # Install fVDB with NVIDIA's wheel index (built for PyTorch 2.8)
        "pip install fvdb-reality-capture fvdb-core==0.3.0+pt28.cu128 "
        "--extra-index-url https://d36m13axqqhiit.cloudfront.net/simple",
    )
    .pip_install("boto3", "requests", "pillow")
)

# Volume for caching models and outputs
output_volume = modal.Volume.from_name("fvdb-outputs", create_if_missing=True)
OUTPUT_PATH = "/outputs"


@app.function(
    image=fvdb_image,
    gpu="A10G",  # 24GB VRAM - good for Gaussian splatting
    timeout=1800,  # 30 min for reconstruction
    volumes={OUTPUT_PATH: output_volume},
)
def test_gpu():
    """Simple test to verify GPU access and fVDB installation."""
    import torch

    results = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
    }

    # Check if fvdb is importable
    try:
        import fvdb
        results["fvdb_installed"] = True
        results["fvdb_version"] = getattr(fvdb, "__version__", "unknown")
    except ImportError as e:
        results["fvdb_installed"] = False
        results["fvdb_error"] = str(e)

    # Check for frgs CLI
    try:
        result = subprocess.run(["frgs", "--help"], capture_output=True, text=True, timeout=10)
        results["frgs_cli_available"] = result.returncode == 0
        results["frgs_help"] = result.stdout[:500] if result.stdout else result.stderr[:500]
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        results["frgs_cli_available"] = False
        results["frgs_error"] = str(e)

    # List installed packages related to fvdb/gaussian
    result = subprocess.run(
        ["pip", "list"], capture_output=True, text=True
    )
    packages = [line for line in result.stdout.split("\n")
                if any(x in line.lower() for x in ["fvdb", "gaussian", "splat", "nerf", "torch"])]
    results["relevant_packages"] = packages

    return results


@app.function(
    image=fvdb_image,
    gpu="A10G",
    timeout=3600,  # 1 hour for full reconstruction
    volumes={OUTPUT_PATH: output_volume},
)
def reconstruct_from_images(
    image_urls: list[str],
    output_name: str = "scene",
    output_format: str = "ply",  # ply, usdz, glb
) -> dict:
    """
    Run 3D Gaussian Splatting reconstruction from a set of images.

    Args:
        image_urls: List of URLs to input images
        output_name: Name for the output file
        output_format: Output format (ply, usdz, glb)

    Returns:
        dict with status and output URL
    """
    import requests
    from PIL import Image
    from io import BytesIO

    work_dir = Path(tempfile.mkdtemp())
    input_dir = work_dir / "images"
    output_dir = work_dir / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Download images
    print(f"Downloading {len(image_urls)} images...")
    for i, url in enumerate(image_urls):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()

            # Save image
            img = Image.open(BytesIO(resp.content))
            img_path = input_dir / f"image_{i:04d}.jpg"
            img.convert("RGB").save(img_path, "JPEG", quality=95)
            print(f"  Downloaded: {img_path.name}")
        except Exception as e:
            print(f"  Failed to download {url}: {e}")

    # Count downloaded images
    downloaded = list(input_dir.glob("*.jpg"))
    if len(downloaded) < 3:
        return {
            "success": False,
            "error": f"Need at least 3 images, only got {len(downloaded)}"
        }

    print(f"\nStarting reconstruction with {len(downloaded)} images...")

    # Try frgs CLI first
    try:
        output_file = output_dir / f"{output_name}.{output_format}"

        cmd = [
            "frgs", "reconstruct",
            "--input", str(input_dir),
            "--output", str(output_file),
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3000,  # 50 min timeout
            cwd=str(work_dir),
        )

        if result.returncode == 0 and output_file.exists():
            # Copy to volume for persistence
            final_path = Path(OUTPUT_PATH) / f"{output_name}.{output_format}"
            import shutil
            shutil.copy(output_file, final_path)
            output_volume.commit()

            return {
                "success": True,
                "output_path": str(final_path),
                "stdout": result.stdout[-2000:],  # Last 2000 chars
            }
        else:
            return {
                "success": False,
                "error": "frgs failed",
                "stdout": result.stdout[-1000:],
                "stderr": result.stderr[-1000:],
            }

    except FileNotFoundError:
        return {
            "success": False,
            "error": "frgs CLI not found - may need different package name"
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Reconstruction timed out after 50 minutes"
        }


@app.function(
    image=fvdb_image,
    gpu="A10G",
    timeout=300,  # 5 min is enough to see sizes
    volumes={OUTPUT_PATH: output_volume},
)
def check_dataset_sizes() -> dict:
    """Start downloading each dataset just long enough to see its size."""
    import re

    datasets = ["miris_factory", "safety_park", "gettysburg", "mipnerf360"]
    sizes = {}

    for dataset in datasets:
        print(f"\n=== Checking size of {dataset} ===")
        work_dir = Path(tempfile.mkdtemp())

        process = subprocess.Popen(
            ["frgs", "download", dataset, "--download-path", str(work_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Read output until we see the size (format: "X.XG/Y.YG" or "X.XM/Y.YM")
        size_found = None
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                # Look for pattern like "50.0M/2.3G" or "1.2G/13.6G"
                match = re.search(r'[\d.]+[MG]/(\d+\.?\d*[MG])', line)
                if match:
                    size_found = match.group(1)
                    print(f"\n>>> Found size: {size_found}")
                    break
                # Also check for completed small downloads
                if "100%" in line and size_found is None:
                    match = re.search(r'(\d+\.?\d*[MG])/\1', line)
                    if match:
                        size_found = match.group(1)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            process.kill()
            process.wait()

        sizes[dataset] = size_found or "unknown"
        print(f">>> {dataset}: {sizes[dataset]}")

    # Sort by size
    print("\n=== Dataset Sizes (smallest first) ===")
    for ds, size in sorted(sizes.items(), key=lambda x: (x[1] or "zzz")):
        print(f"  {ds}: {size}")

    return sizes


@app.function(
    image=fvdb_image,
    gpu="A10G",
    timeout=600,
    volumes={OUTPUT_PATH: output_volume},
)
def test_frgs_help() -> dict:
    """Check what frgs commands are available and their usage."""
    results = {}

    # Get main help
    result = subprocess.run(["frgs", "--help"], capture_output=True, text=True)
    results["main_help"] = result.stdout

    # Get reconstruct help
    result = subprocess.run(["frgs", "reconstruct", "--help"], capture_output=True, text=True)
    results["reconstruct_help"] = result.stdout

    # Get download help (for sample data)
    result = subprocess.run(["frgs", "download", "--help"], capture_output=True, text=True)
    results["download_help"] = result.stdout

    return results


@app.function(
    image=fvdb_image,
    gpu="A10G",
    timeout=5400,  # 1.5 hours
    volumes={OUTPUT_PATH: output_volume},
)
def test_with_sample_data(dataset: str = "gettysburg") -> dict:
    """
    Test reconstruction with fVDB's built-in sample data.
    Downloads a sample dataset and runs reconstruction.

    Available datasets: mipnerf360, gettysburg, safety_park, miris_factory
    """
    import shutil

    work_dir = Path(tempfile.mkdtemp())
    data_dir = work_dir / "data"
    data_dir.mkdir(exist_ok=True)

    # Download dataset (longer timeout for large datasets)
    print(f"\n=== Downloading {dataset} dataset ===")
    result = subprocess.run(
        ["frgs", "download", dataset, "--download-path", str(data_dir)],
        capture_output=True, text=True, cwd=str(work_dir), timeout=1200  # 20 min
    )
    print(f"Download returncode: {result.returncode}")
    if result.stdout:
        print(f"stdout: {result.stdout[-2000:]}")
    if result.stderr:
        print(f"stderr: {result.stderr[-1000:]}")

    # List what we downloaded
    import os
    print("\n=== Downloaded files ===")
    file_count = 0
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            file_count += 1
            if file_count <= 30:
                rel_path = os.path.relpath(os.path.join(root, f), data_dir)
                print(f"  {rel_path}")
    print(f"  ... total {file_count} files")

    # Find the actual dataset directory (mipnerf360 extracts to 360_v2/bicycle, etc)
    dataset_path = data_dir / dataset
    if not dataset_path.exists():
        # Check for 360_v2 subdirectory (mipnerf360 specific)
        alt_path = data_dir / "360_v2"
        if alt_path.exists():
            # Use the bicycle scene (first scene in the dataset)
            bicycle_path = alt_path / "bicycle"
            if bicycle_path.exists():
                dataset_path = bicycle_path
            else:
                # Pick first subdirectory that has an images folder
                for subdir in alt_path.iterdir():
                    if subdir.is_dir() and (subdir / "images").exists():
                        dataset_path = subdir
                        break
                else:
                    dataset_path = alt_path
        else:
            subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
            if subdirs:
                dataset_path = subdirs[0]
            else:
                dataset_path = data_dir

    print(f"\n=== Dataset path: {dataset_path} ===")
    print(f"Contents: {list(dataset_path.iterdir())[:10]}")

    # Run reconstruction
    output_file = work_dir / "output.ply"

    print(f"\n=== Running reconstruction ===")
    print(f"Input: {dataset_path}")
    print(f"Output: {output_file}")

    # Use the correct CLI format: frgs reconstruct <path> -o <output>
    cmd = [
        "frgs", "reconstruct",
        str(dataset_path),
        "-o", str(output_file),
    ]
    print(f"Command: {' '.join(cmd)}")
    print("\n=== frgs output (streaming) ===")

    # Stream output in real-time instead of capturing (so we see it if it crashes)
    import sys
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Combine stderr into stdout
        text=True,
        cwd=str(work_dir),
        bufsize=1,  # Line buffered
    )

    stdout_lines = []
    try:
        for line in process.stdout:
            print(line, end='', flush=True)  # Print in real-time
            stdout_lines.append(line)
            if len(stdout_lines) > 500:  # Keep last 500 lines
                stdout_lines.pop(0)

        process.wait(timeout=5000)  # ~83 min timeout (within 1.5hr container limit)
    except subprocess.TimeoutExpired:
        process.kill()
        print("\n!!! TIMEOUT - process killed after 83 minutes !!!")

    print(f"\n=== frgs finished with returncode: {process.returncode} ===")

    # Check output
    output_exists = output_file.exists()
    output_size = output_file.stat().st_size if output_exists else 0

    # Copy to volume if successful
    if output_exists and output_size > 0:
        final_path = Path(OUTPUT_PATH) / f"{dataset}_output.ply"
        shutil.copy(output_file, final_path)
        output_volume.commit()
        print(f"\n✓ Output saved to volume: {final_path}")
        print(f"  Size: {output_size / 1024 / 1024:.2f} MB")

    stdout_text = ''.join(stdout_lines)
    return {
        "success": process.returncode == 0 and output_exists,
        "output_exists": output_exists,
        "output_size_mb": output_size / 1024 / 1024 if output_exists else 0,
        "stdout": stdout_text[-5000:] if stdout_text else "",
        "stderr": "",  # Already combined into stdout
        "returncode": process.returncode,
    }


@app.local_entrypoint()
def main(test_type: str = "gpu"):
    """
    Test the fVDB Reality Capture setup.

    Usage:
        modal run gsplat/worker.py                          # Quick GPU test
        modal run gsplat/worker.py --test-type help         # Show frgs CLI help
        modal run gsplat/worker.py --test-type sample       # Test with safety_park dataset
        modal run gsplat/worker.py --test-type sample:miris_factory  # Specify dataset

    Available datasets: mipnerf360, gettysburg, safety_park, miris_factory
    """
    if test_type == "sizes":
        print("\n=== Checking Dataset Sizes ===\n")
        sizes = check_dataset_sizes.remote()
        print("\n=== Results ===")
        for ds, size in sizes.items():
            print(f"  {ds}: {size}")
        return

    if test_type == "help":
        print("\n=== frgs CLI Help ===\n")
        result = test_frgs_help.remote()
        print("=== Main Help ===")
        print(result["main_help"])
        print("\n=== Reconstruct Help ===")
        print(result["reconstruct_help"])
        print("\n=== Download Help ===")
        print(result["download_help"])
        return

    if test_type.startswith("sample"):
        # Allow specifying dataset: sample, sample:safety_park, sample:miris_factory, etc.
        if ":" in test_type:
            dataset = test_type.split(":")[1]
        else:
            dataset = "safety_park"  # Default to safety_park (likely smaller than gettysburg)

        print(f"\n=== Testing with Sample Data: {dataset} ===\n")
        print("Available datasets: mipnerf360, gettysburg, safety_park, miris_factory")
        print()
        result = test_with_sample_data.remote(dataset=dataset)
        print(f"\nSuccess: {result.get('success', False)}")
        print(f"Output exists: {result.get('output_exists', False)}")
        print(f"Output size: {result.get('output_size_mb', 0):.2f} MB")
        print(f"Return code: {result.get('returncode', 'N/A')}")
        print(f"\n=== stdout ===\n{result.get('stdout', '')}")
        if result.get('stderr'):
            print(f"\n=== stderr ===\n{result['stderr']}")
        return

    # Default: GPU test
    print("\n=== Testing fVDB Reality Capture on Modal ===\n")

    result = test_gpu.remote()

    print("GPU Test Results:")
    print(f"  CUDA Available: {result['cuda_available']}")
    print(f"  Device: {result.get('device_name', 'N/A')}")
    print(f"  CUDA Version: {result.get('cuda_version', 'N/A')}")
    print()
    print(f"  fVDB Installed: {result.get('fvdb_installed', False)}")
    if result.get('fvdb_installed'):
        print(f"  fVDB Version: {result.get('fvdb_version', 'unknown')}")
    else:
        print(f"  fVDB Error: {result.get('fvdb_error', 'N/A')}")
    print()
    print(f"  frgs CLI Available: {result.get('frgs_cli_available', False)}")
    if result.get('frgs_cli_available'):
        print(f"  frgs Help: {result.get('frgs_help', '')[:200]}...")
    else:
        print(f"  frgs Error: {result.get('frgs_error', 'N/A')}")
    print()
    print("Relevant Packages:")
    for pkg in result.get('relevant_packages', []):
        print(f"  {pkg}")
