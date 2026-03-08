from fastapi import FastAPI, Form, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import json
import os
from dotenv import load_dotenv

load_dotenv()

from tool_calls.depth import DepthResult
from tool_calls.three_js import camera_update, update_position as _update_position, update_rotation as _update_rotation
from tool_calls.mcp import start_mcp_session
from tool_calls.gemini_path import trace_path
from database import init_db
from models import Environment, Pose, Trajectory, ActivityLog
from ws_manager import ws_manager

app = FastAPI(title="Robotics Vision API")

# Serve the project root so the GLB file is accessible at /static/cozy_living_room_baked.glb
_project_root = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(_project_root)), name="static")


@app.on_event("startup")
async def on_startup():
    await init_db()


class ImageRequest(BaseModel):
    image: str
    target_object: str = ""

class PositionRequest(BaseModel):
    x: float
    y: float
    z: float

class RotationRequest(BaseModel):
    x: float
    y: float
    z: float
    w: float ## By how much to rotate.

class CameraRequest(BaseModel):
    zoom_percentage: float

class EnvironmentRequest(BaseModel):
    x: float
    y: float
    z: float
    scaleX: float
    scaleY: float
    scaleZ: float
    rotX: float
    rotY: float
    rotZ: float
    rotW: float
    r: float
    g: float
    b: float
    opacity: float

class ImageResponse(BaseModel):
    image: str

class StatusResponse(BaseModel):
    status: str
    id: str | None = None

class PoseResponse(BaseModel):
    id: str
    iteration_num: int
    trajectory_fk: str
    xPos: float
    yPos: float
    zPos: float
    rotX: float | None
    rotY: float | None
    rotZ: float | None
    rotW: float | None
    deltaX: float | None
    deltaY: float | None
    deltaZ: float | None
    deltaRotX: float | None
    deltaRotY: float | None
    deltaRotZ: float | None
    timestamp: datetime | None

class TrajectoryResponse(BaseModel):
    id: str
    environment_fk: str
    poses: list[PoseResponse]

class LogResponse(BaseModel):
    id: str
    toolCall: str
    pose_fk: str | None
    base64_image: str | None

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "example_client_1.html")

@app.get("/client-1")
async def client_page():
    return FileResponse(Path(__file__).parent / "example_client_1.html")

@app.get("/button")
async def button_page():
    return FileResponse(Path(__file__).parent / "button_page.html")

@app.get("/overshoot-key")
async def overshoot_key():
    return {"key": os.environ.get("OVERSHOOT_API", "")}

@app.get("/health")
async def health():
    return {"status": "ok"}

## GENERAL API
@app.post("/saveSplat")
async def save_splat(request: EnvironmentRequest) -> StatusResponse:
    env = Environment(**request.model_dump())
    await env.insert()
    return StatusResponse(status="saved", id=str(env.id))

class StartMCPRequest(BaseModel):
    environment_id: str

@app.post("/startMCP")
async def start_mcp(request: StartMCPRequest) -> StatusResponse:
    trajectory = await start_mcp_session(request.environment_id)
    ## TODO: CALL GEMINI TO START MCP!!
    return StatusResponse(status="started", id=str(trajectory.id))

@app.get("/currPosRot")
async def current_pos_rot() -> PoseResponse:
    pose = await Pose.find_all().sort("-timestamp").first_or_none()
    if not pose:
        raise HTTPException(status_code=404, detail="No poses found")
    return PoseResponse(
        id=str(pose.id),
        iteration_num=pose.iteration_num,
        trajectory_fk=str(pose.trajectory_fk.ref.id),
        xPos=pose.xPos, yPos=pose.yPos, zPos=pose.zPos,
        rotX=pose.rotX, rotY=pose.rotY, rotZ=pose.rotZ, rotW=pose.rotW,
        deltaX=pose.deltaX, deltaY=pose.deltaY, deltaZ=pose.deltaZ,
        deltaRotX=pose.deltaRotX, deltaRotY=pose.deltaRotY, deltaRotZ=pose.deltaRotZ,
        timestamp=pose.timestamp,
    )

@app.get("/latestLog")
async def get_latest_log() -> LogResponse:
    log = await ActivityLog.find_all().sort("-_id").first_or_none()
    if not log:
        raise HTTPException(status_code=404, detail="No logs found")
    return LogResponse(
        id=str(log.id),
        toolCall=log.toolCall,
        pose_fk=str(log.pose_fk.ref.id) if log.pose_fk else None,
        base64_image=log.base64_image,
    )

@app.get("/trajectory/{trajectory_id}")
async def get_trajectory(trajectory_id: str) -> TrajectoryResponse:
    trajectory = await Trajectory.get(trajectory_id)
    if not trajectory:
        raise HTTPException(status_code=404, detail="Trajectory not found")
    poses = await Pose.find(Pose.trajectory_fk.id == trajectory.id).sort("iteration_num").to_list()
    return TrajectoryResponse(
        id=str(trajectory.id),
        environment_fk=str(trajectory.environment_fk.ref.id),
        poses=[
            PoseResponse(
                id=str(p.id),
                iteration_num=p.iteration_num,
                trajectory_fk=str(p.trajectory_fk.ref.id),
                xPos=p.xPos, yPos=p.yPos, zPos=p.zPos,
                rotX=p.rotX, rotY=p.rotY, rotZ=p.rotZ, rotW=p.rotW,
                deltaX=p.deltaX, deltaY=p.deltaY, deltaZ=p.deltaZ,
                deltaRotX=p.deltaRotX, deltaRotY=p.deltaRotY, deltaRotZ=p.deltaRotZ,
                timestamp=p.timestamp,
            )
            for p in poses
        ],
    )

## WEBSOCKET
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            ws_manager.receive_response(json.loads(data))
    except WebSocketDisconnect:
        ws_manager.disconnect()


## TOOL CALLS FOR MCP
@app.post("/updatePosition")
async def update_position(request: PositionRequest) -> DepthResult:
    return await _update_position(request.x, request.y, request.z)


@app.post("/updateRotation")
async def update_rotation(request: RotationRequest) -> DepthResult:
    return await _update_rotation(request.x, request.y, request.z, request.w)


@app.post("/updateCamera")
async def update_camera(request: CameraRequest) -> DepthResult:
    return await camera_update(request.zoom_percentage)


@app.post("/tracePath")
async def trace_path_endpoint(
    file: UploadFile = File(...),
    start_x: int | None = Form(None),
    start_y: int | None = Form(None),
    obstacles: str | None = Form(None),
):
    image_bytes = await file.read()
    mime = file.content_type or "image/png"
    start_hint = None
    if start_x is not None and start_y is not None:
        start_hint = {"x": start_x, "y": start_y}
    precomputed = json.loads(obstacles) if obstacles else None
    result = await trace_path(image_bytes, mime, start_hint=start_hint, precomputed_obstacles=precomputed)
    return result
