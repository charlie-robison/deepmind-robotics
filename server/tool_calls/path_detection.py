"""
Detect waypoints and obstacles from a top-down image using Gemini Vision.
Takes a single base64 image, returns structured waypoint + obstacle data.
"""

import base64
import json
import os
from pydantic import BaseModel
from google import genai


DETECTION_PROMPT = """You are a precision path planner analyzing a TOP-DOWN bird's-eye view of a 3D room.

COORDINATE SYSTEM:
- The image shows an 8x8 unit area rendered by an orthographic camera.
- Image left edge = world X = -4, image right edge = world X = +4
- Image top edge = world Z = -4, image bottom edge = world Z = +4
- To convert pixel position to world coords:
  worldX = (pixelX / imageWidth) * 8 - 4
  worldZ = (pixelY / imageHeight) * 8 - 4

WHAT YOU SEE:
- A room viewed from directly above with furniture (couch, coffee table, chairs, lamp, rug, etc.)
- A BLUE dot marking the robot START position (bottom-left corner)
- A RED dot marking the DESTINATION (top-left area, near the lamp)
- GREEN lines and GREEN DOTS showing a pre-planned path the robot should follow

OBSTACLE DETECTION RULES (CRITICAL):
- Identify EACH piece of furniture as a SEPARATE obstacle with its own tight bounding box.
- Bounding boxes must FIT TIGHTLY around the visible furniture — do NOT inflate or pad them.
- Do NOT merge multiple pieces of furniture into one large box.
- Typical obstacles: couch (~2.5 x 1.5 units), coffee table (~1.5 x 1.0), chairs (~0.8 x 0.8), lamp (~0.5 x 0.5), rug edges.
- Each obstacle width and depth should match what you actually see — most furniture is 0.5 to 3.0 units, never 4+ units.

WAYPOINT DETECTION RULES:
- Trace the GREEN path from the BLUE start dot to the RED destination dot.
- Record each GREEN DOT waypoint along the path IN ORDER.
- The path should navigate AROUND the furniture cluster, staying on open floor.
- Typical path: starts bottom-left, goes right along the bottom, then up the right side past furniture, then left across the top to the destination.
- There should be approximately 5-8 waypoints forming an L-shape or zigzag around obstacles.

Respond with ONLY valid JSON (no markdown, no code fences) in this exact format:
{
  "waypoints": [
    {"x": -3.0, "z": 4.0, "description": "start position"},
    {"x": 1.0, "z": 2.0, "description": "turn point"}
  ],
  "obstacles": [
    {"label": "couch", "center_x": -2.0, "center_z": -1.5, "width": 2.5, "depth": 1.5}
  ],
  "reasoning": "Brief explanation"
}"""


class Waypoint(BaseModel):
    x: float
    z: float
    description: str = ""

class Obstacle(BaseModel):
    label: str
    center_x: float
    center_z: float
    width: float
    depth: float

class PathDetectionResult(BaseModel):
    waypoints: list[Waypoint]
    obstacles: list[Obstacle] = []
    reasoning: str = ""


async def detect_path_from_image(base64_image: str) -> PathDetectionResult:
    """
    Send a single top-down image to Gemini and get back detected waypoints + obstacles.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    # Strip data URI prefix if present
    if "," in base64_image:
        base64_image = base64_image.split(",", 1)[1]

    # Decode to bytes for Gemini
    image_bytes = base64.b64decode(base64_image)

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai.types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            DETECTION_PROMPT,
        ],
    )

    # Parse the JSON response
    raw = response.text.strip()
    # Strip markdown code fences if the model wraps it
    if raw.startswith("```"):
        raw = raw.replace("```json", "").replace("```", "").strip()

    data = json.loads(raw)

    return PathDetectionResult(
        waypoints=[Waypoint(**wp) for wp in data.get("waypoints", [])],
        obstacles=[Obstacle(**obs) for obs in data.get("obstacles", [])],
        reasoning=data.get("reasoning", ""),
    )
