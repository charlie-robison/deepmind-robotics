import os
import base64
import json
import heapq
import struct
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from dotenv import load_dotenv
from fastapi import HTTPException

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

DETECT_PROMPT = """You are analyzing a top-down view of a room for robotic path planning.

Identify the following and return their bounding boxes in pixel coordinates:

1. "lamp" — the glowing octagonal/circular light fixture in the top-left area (above the couch). Return its CENTER point.
2. "obstacles" — list ALL furniture/objects that block the floor: the couch, rug/carpet, coffee table, chairs, any other items on the floor. For each, return a bounding box with some padding.

The image may have a dark border — ignore it. Only analyze the room interior.

Return ONLY a JSON object, no markdown fences:
{
  "image_width": <width in pixels>,
  "image_height": <height in pixels>,
  "lamp": {"x": <center_x>, "y": <center_y>},
  "obstacles": [
    {"label": "<name>", "x_min": <left>, "y_min": <top>, "x_max": <right>, "y_max": <bottom>},
    ...
  ]
}

Be generous with obstacle bounding boxes — add 10-15px padding around each object to ensure clearance. The rug/carpet counts as an obstacle."""

LAMP_ONLY_PROMPT = """You are analyzing a top-down view of a room for robotic path planning.

Find the glowing octagonal/circular light fixture (lamp) in the top-left area of the room (above the couch). Return its CENTER point in pixel coordinates.

The image may have a dark border — ignore it. Only analyze the room interior.

Return ONLY a JSON object, no markdown fences:
{
  "image_width": <width in pixels>,
  "image_height": <height in pixels>,
  "lamp": {"x": <center_x>, "y": <center_y>}
}"""


def _png_dimensions(data: bytes) -> tuple[int, int]:
    """Read width and height from PNG IHDR chunk."""
    w = struct.unpack('>I', data[16:20])[0]
    h = struct.unpack('>I', data[20:24])[0]
    return w, h


def _astar(start, end, obstacles, width, height, step=10):
    """Simple A* grid pathfinder avoiding obstacle rectangles."""

    def blocked(x, y):
        if x < 5 or y < 5 or x >= width - 5 or y >= height - 5:
            return True
        for obs in obstacles:
            if obs["x_min"] <= x <= obs["x_max"] and obs["y_min"] <= y <= obs["y_max"]:
                return True
        return False

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # Snap start/end to grid
    sx, sy = round(start["x"] / step) * step, round(start["y"] / step) * step
    ex, ey = round(end["x"] / step) * step, round(end["y"] / step) * step

    # If end is blocked (lamp is on furniture), find nearest unblocked
    if blocked(ex, ey):
        best = None
        for dx in range(-100, 101, step):
            for dy in range(-100, 101, step):
                nx, ny = ex + dx, ey + dy
                if not blocked(nx, ny):
                    d = abs(dx) + abs(dy)
                    if best is None or d < best[0]:
                        best = (d, nx, ny)
        if best:
            ex, ey = best[1], best[2]

    open_set = [(0, sx, sy)]
    came_from = {}
    g_score = {(sx, sy): 0}

    directions = [(step, 0), (-step, 0), (0, step), (0, -step),
                  (step, step), (step, -step), (-step, step), (-step, -step)]

    while open_set:
        _, cx, cy = heapq.heappop(open_set)

        if abs(cx - ex) <= step and abs(cy - ey) <= step:
            # Reconstruct path
            path = []
            node = (cx, cy)
            while node in came_from:
                path.append({"x": node[0], "y": node[1]})
                node = came_from[node]
            path.reverse()
            return path

        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if blocked(nx, ny):
                continue
            cost = step if dx == 0 or dy == 0 else int(step * 1.414)
            new_g = g_score[(cx, cy)] + cost
            if new_g < g_score.get((nx, ny), float("inf")):
                g_score[(nx, ny)] = new_g
                f = new_g + heuristic((nx, ny), (ex, ey))
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(open_set, (f, nx, ny))

    # Fallback: straight line
    return [{"x": end["x"], "y": end["y"]}]


def _simplify_path(waypoints, tolerance=15):
    """Remove waypoints that are roughly collinear (Ramer-Douglas-Peucker)."""
    if len(waypoints) < 3:
        return waypoints

    def point_line_dist(p, a, b):
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        if dx == 0 and dy == 0:
            return ((p["x"] - a["x"])**2 + (p["y"] - a["y"])**2) ** 0.5
        t = max(0, min(1, ((p["x"] - a["x"]) * dx + (p["y"] - a["y"]) * dy) / (dx*dx + dy*dy)))
        proj_x = a["x"] + t * dx
        proj_y = a["y"] + t * dy
        return ((p["x"] - proj_x)**2 + (p["y"] - proj_y)**2) ** 0.5

    max_dist = 0
    max_idx = 0
    for i in range(1, len(waypoints) - 1):
        d = point_line_dist(waypoints[i], waypoints[0], waypoints[-1])
        if d > max_dist:
            max_dist = d
            max_idx = i

    if max_dist > tolerance:
        left = _simplify_path(waypoints[:max_idx + 1], tolerance)
        right = _simplify_path(waypoints[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [waypoints[0], waypoints[-1]]


async def _call_gemini(contents):
    """Call Gemini with fallback across models."""
    last_error = None
    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            return response
        except ClientError as e:
            last_error = e
            if "RESOURCE_EXHAUSTED" in str(e) or "NOT_FOUND" in str(e):
                continue
            raise HTTPException(status_code=502, detail=f"Gemini API error: {e}")
    raise HTTPException(
        status_code=429,
        detail=f"Gemini rate limit exceeded on all models. Please wait and try again. ({last_error})"
    )


def _parse_gemini_json(response) -> dict:
    """Extract JSON from Gemini response text."""
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


async def trace_path(image_bytes: bytes, mime_type: str = "image/png", start_hint: dict | None = None, precomputed_obstacles: list | None = None) -> dict:
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Read actual image dimensions from PNG header
    actual_w, actual_h = _png_dimensions(image_bytes)

    # Choose prompt based on whether obstacles are precomputed
    prompt = LAMP_ONLY_PROMPT if precomputed_obstacles else DETECT_PROMPT

    contents = [
        types.Content(
            parts=[
                types.Part(
                    inline_data=types.Blob(mime_type=mime_type, data=b64)
                ),
                types.Part(text=prompt),
            ]
        )
    ]

    response = await _call_gemini(contents)
    scene = _parse_gemini_json(response)

    print(f"[Gemini] Actual PNG: {actual_w}x{actual_h}, Gemini reported: {scene.get('image_width')}x{scene.get('image_height')}")
    print(f"[Gemini] Lamp: {scene['lamp']}")

    # Use actual image dimensions, not Gemini's guess
    width = actual_w
    height = actual_h

    # Scale Gemini's lamp coordinates if it reported different dimensions
    gw = scene.get("image_width", actual_w)
    gh = scene.get("image_height", actual_h)
    if gw != actual_w or gh != actual_h:
        sx, sy = actual_w / gw, actual_h / gh
        scene["lamp"]["x"] = round(scene["lamp"]["x"] * sx)
        scene["lamp"]["y"] = round(scene["lamp"]["y"] * sy)

    if precomputed_obstacles:
        obstacles = precomputed_obstacles
        print(f"[Gemini] Using {len(obstacles)} precomputed obstacles from scene geometry")
    else:
        obstacles = scene.get("obstacles", [])
        # Scale Gemini obstacle coordinates if dimensions differ
        if gw != actual_w or gh != actual_h:
            sx, sy = actual_w / gw, actual_h / gh
            for obs in obstacles:
                obs["x_min"] = round(obs["x_min"] * sx)
                obs["y_min"] = round(obs["y_min"] * sy)
                obs["x_max"] = round(obs["x_max"] * sx)
                obs["y_max"] = round(obs["y_max"] * sy)
        print(f"[Gemini] Using {len(obstacles)} Gemini-detected obstacles")

    start = start_hint or {"x": 50, "y": height - 50}
    end = scene["lamp"]

    raw_waypoints = _astar(start, end, obstacles, width, height, step=10)
    waypoints = _simplify_path(raw_waypoints)

    return {
        "start": start,
        "end": end,
        "waypoints": waypoints,
        "obstacles": obstacles,
        "image_width": width,
        "image_height": height,
        "description": f"A* path with {len(waypoints)} waypoints, avoiding {len(obstacles)} obstacles.",
    }
