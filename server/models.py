from beanie import Document, Link
from pydantic import Field
from typing import Optional
from datetime import datetime, timezone


class Environment(Document):
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

    class Settings:
        name = "environment"


class Trajectory(Document):
    environment_fk: Link[Environment]
    poses: list["Pose"] = []

    class Settings:
        name = "trajectory"


class Pose(Document):
    iteration_num: int
    trajectory_fk: Link[Trajectory]
    xPos: float
    yPos: float
    zPos: float
    rotX: Optional[float] = None
    rotY: Optional[float] = None
    rotZ: Optional[float] = None
    rotW: Optional[float] = None
    deltaX: Optional[float] = None
    deltaY: Optional[float] = None
    deltaZ: Optional[float] = None
    deltaRotX: Optional[float] = None
    deltaRotY: Optional[float] = None
    deltaRotZ: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "pose"


class ActivityLog(Document):
    toolCall: str
    pose_fk: Optional[Link[Pose]] = None
    base64_image: Optional[str] = None

    class Settings:
        name = "activity_log"
