"""
Server-side integration for the Lyra Modal worker.

Calls the remote Modal functions to run single-image -> 3D Gaussian reconstruction.
Lyra repo: https://github.com/nv-tlabs/lyra.git
"""

import base64
import modal


def _get_fn(name: str):
    """Lazy-lookup a function from the deployed lyra-3d-generation Modal app."""
    return modal.Function.from_name("lyra-3d-generation", name)


async def reconstruct_3d_from_image(
    image_base64: str,
    image_name: str = "input.png",
    output_name: str = "scene",
    movement_factor: float = 1.0,
) -> dict:
    """
    Send a base64-encoded image to Lyra on Modal for 3D Gaussian reconstruction.

    Returns dict with success status, PLY paths, timing info.
    """
    # Decode base64 -> raw bytes
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    image_base64 += "=" * (-len(image_base64) % 4)
    image_bytes = base64.b64decode(image_base64)

    fn = _get_fn("generate_3d_from_image")
    return await fn.remote.aio(
        image_bytes=image_bytes,
        image_name=image_name,
        output_name=output_name,
        movement_factor=movement_factor,
    )


async def list_lyra_outputs() -> list[dict]:
    """List all files on the Lyra output volume."""
    fn = _get_fn("list_outputs")
    return await fn.remote.aio()


async def get_lyra_ply(name: str) -> bytes:
    """Download a PLY file from the Lyra output volume."""
    fn = _get_fn("get_ply")
    return await fn.remote.aio(name)
