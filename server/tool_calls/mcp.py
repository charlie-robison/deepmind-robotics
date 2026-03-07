from models import Trajectory, Environment


async def start_mcp_session(environment_id: str) -> Trajectory:
    environment = await Environment.get(environment_id)
    if not environment:
        raise ValueError(f"Environment {environment_id} not found")

    trajectory = Trajectory(environment_fk=environment, poses=[])
    await trajectory.insert()

    ## TODO: Call Gemini Agent with trajectory context.

    return trajectory
