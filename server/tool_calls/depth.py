import base64
import io

import numpy as np
import matplotlib.cm as cm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pydantic import BaseModel
from transformers import pipeline


pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")


class DepthResult(BaseModel):
    original_image: str
    depth_map_image: str


def generate_depth_map(base64_image: str) -> DepthResult:
    """
    Returns a depth map for the given image and the original image.
    Saves to Logs in DB.
    :param base64_image: The image to generate the depth map for.
    :return: The DepthResult class.
    """
    # Strip data URI prefix if present (e.g. "data:image/png;base64,...")
    if "," in base64_image:
        base64_image = base64_image.split(",", 1)[1]
    # Fix padding
    base64_image += "=" * (-len(base64_image) % 4)
    image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")

    depth = pipe(image)["depth"]

    # Apply Spectral colormap like the HF Space does
    depth_array = np.array(depth).astype(np.float32)
    depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min() + 1e-8)
    colored = cm.Spectral(depth_normalized)[:, :, :3]  # drop alpha channel
    depth_colored = Image.fromarray((colored * 255).astype(np.uint8))

    buffer = io.BytesIO()
    depth_colored.save(buffer, format="PNG")
    depth_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return DepthResult(original_image=base64_image, depth_map_image=depth_b64)
