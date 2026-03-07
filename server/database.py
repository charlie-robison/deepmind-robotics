import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
DB_NAME = os.getenv("DB_NAME", "wapp")

client = AsyncIOMotorClient(MONGODB_URL)


async def init_db():
    from models import Environment, Trajectory, Pose, ActivityLog
    await init_beanie(
        database=client[DB_NAME],
        document_models=[Environment, Trajectory, Pose, ActivityLog],
    )
