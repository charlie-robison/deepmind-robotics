"""Convert trajectory data from Three.js format or MongoDB to MuJoCo training format."""

import argparse
import asyncio
import json
import pathlib


def convert_threejs_trajectory(
    poses: list[dict],
    target_name: str = "path_1",
) -> dict:
    """Convert [{x, y, z}, ...] to {"target": name, "waypoints": [[x,-z,y], ...]}

    Applies coordinate swap: Three.js Y-up -> MuJoCo Z-up
    Three.js (X, Y, Z) -> MuJoCo (X, -Z, Y)
    """
    waypoints = [[p["x"], -p["z"], p["y"]] for p in poses]
    return {"target": target_name, "waypoints": waypoints}


def convert_threejs_file(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> None:
    """Read JSON of [{x,y,z},...] or list-of-lists, write mujoco_example format."""
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)

    with open(input_path) as f:
        data = json.load(f)

    # Handle both single trajectory [{x,y,z},...] and multiple [[{x,y,z},...],...]
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], dict) and "x" in data[0]:
            # Single trajectory: [{x, y, z}, ...]
            trajectories = [convert_threejs_trajectory(data, "path_1")]
        elif isinstance(data[0], list):
            # Multiple trajectories: [[{x,y,z},...], ...]
            trajectories = [
                convert_threejs_trajectory(traj, f"path_{i+1}")
                for i, traj in enumerate(data)
            ]
        else:
            # Already in mujoco_example format
            trajectories = data
    else:
        trajectories = data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"Converted {len(trajectories)} trajectories -> {output_path}")


async def fetch_from_mongodb(
    mongodb_url: str,
    db_name: str = "wapp",
    output_path: str | pathlib.Path = "trajectories/trajectories.json",
) -> None:
    """Query Trajectory + Pose documents from MongoDB, produce output JSON.

    Uses motor.motor_asyncio.AsyncIOMotorClient (same as server/database.py).
    Queries 'trajectory' collection, then for each trajectory queries 'pose'
    collection sorted by iteration_num, extracts [xPos, yPos, zPos].
    """
    from motor.motor_asyncio import AsyncIOMotorClient

    client = AsyncIOMotorClient(mongodb_url)
    db = client[db_name]

    trajectory_col = db["trajectory"]
    pose_col = db["pose"]

    trajectories = []
    async for traj_doc in trajectory_col.find():
        traj_id = traj_doc["_id"]
        poses_cursor = pose_col.find(
            {"trajectory_fk.$id": traj_id}
        ).sort("iteration_num", 1)

        threejs_poses = []
        async for pose_doc in poses_cursor:
            threejs_poses.append({
                "x": pose_doc["xPos"],
                "y": pose_doc["yPos"],
                "z": pose_doc["zPos"],
            })

        if threejs_poses:
            converted = convert_threejs_trajectory(
                threejs_poses, f"path_{len(trajectories)+1}"
            )
            trajectories.append(converted)

    client.close()

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trajectories, f, indent=2)

    print(f"Fetched {len(trajectories)} trajectories from MongoDB -> {output_path}")


def main():
    """CLI: --input PATH | --mongodb-url URL, --output PATH, --db-name NAME"""
    parser = argparse.ArgumentParser(
        description="Convert trajectories from Three.js or MongoDB to MuJoCo format"
    )
    parser.add_argument("--input", type=str, help="Path to Three.js JSON file")
    parser.add_argument("--mongodb-url", type=str, help="MongoDB connection URL")
    parser.add_argument(
        "--output", type=str, default="trajectories/converted.json",
        help="Output path for converted JSON"
    )
    parser.add_argument("--db-name", type=str, default="wapp", help="MongoDB database name")
    args = parser.parse_args()

    if args.input:
        convert_threejs_file(args.input, args.output)
    elif args.mongodb_url:
        asyncio.run(fetch_from_mongodb(args.mongodb_url, args.db_name, args.output))
    else:
        parser.error("Either --input or --mongodb-url is required")


if __name__ == "__main__":
    main()
