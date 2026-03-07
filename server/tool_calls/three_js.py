from tool_calls.depth import generate_depth_map, DepthResult
from models import Pose, ActivityLog
from ws_manager import ws_manager


async def update_position(dx: float, dy: float, dz: float) -> DepthResult:
    """
    Moves the object by the given delta units relative to its current position.

    :param dx: Units to move along the x-axis.
    :param dy: Units to move along the y-axis.
    :param dz: Units to move along the z-axis.
    :return: DepthResult containing the generated depth map and the original image.
    """
    image: str = await ws_manager.send_command("updatePosition", {"dx": dx, "dy": dy, "dz": dz})

    # Generates depth map for the image and returns both the depth map and the original image.
    depth_map_object: DepthResult = generate_depth_map(image)

    # Save position change in DB.
    last_pose = await Pose.find_all().sort("-timestamp").first_or_none()
    prev_x = last_pose.xPos if last_pose else 0
    prev_y = last_pose.yPos if last_pose else 0
    prev_z = last_pose.zPos if last_pose else 0
    pose = Pose(
        iteration_num=(last_pose.iteration_num + 1) if last_pose else 0,
        trajectory_fk=last_pose.trajectory_fk if last_pose else None,
        xPos=prev_x + dx, yPos=prev_y + dy, zPos=prev_z + dz,
        deltaX=dx, deltaY=dy, deltaZ=dz,
    )
    await pose.insert()

    # Logs activity
    log = ActivityLog(toolCall="updatePosition", pose_fk=pose, base64_image=image)
    await log.insert()

    return depth_map_object

async def update_rotation(dx: float, dy: float, dz: float) -> DepthResult:
    """
    Updates the rotation of the object by Euler angle deltas (radians).

    :param dx: Delta rotation around X axis (pitch).
    :param dy: Delta rotation around Y axis (yaw).
    :param dz: Delta rotation around Z axis (roll).
    :return: DepthResult containing the generated depth map and the original image.
    """
    image: str = await ws_manager.send_command("updateRotation", {"dx": dx, "dy": dy, "dz": dz})

    # Generates depth map for the image and returns both the depth map and the original image.
    depth_map_object: DepthResult = generate_depth_map(image)

    # Save rotation change in DB.
    last_pose = await Pose.find_all().sort("-timestamp").first_or_none()
    prev_rx = last_pose.rotX if last_pose and last_pose.rotX else 0
    prev_ry = last_pose.rotY if last_pose and last_pose.rotY else 0
    prev_rz = last_pose.rotZ if last_pose and last_pose.rotZ else 0
    pose = Pose(
        iteration_num=(last_pose.iteration_num + 1) if last_pose else 0,
        trajectory_fk=last_pose.trajectory_fk if last_pose else None,
        xPos=last_pose.xPos if last_pose else 0,
        yPos=last_pose.yPos if last_pose else 0,
        zPos=last_pose.zPos if last_pose else 0,
        rotX=prev_rx + dx, rotY=prev_ry + dy, rotZ=prev_rz + dz,
        deltaRotX=dx, deltaRotY=dy, deltaRotZ=dz,
    )
    await pose.insert()

    # Logs activity
    log = ActivityLog(toolCall="updateRotation", pose_fk=pose, base64_image=image)
    await log.insert()

    return depth_map_object

async def camera_update(zoom_percentage: float) -> DepthResult:
    """
    Updates the zoom level of the camera in the 3D scene and logs the activity.

    :param zoom_percentage: The new zoom level as a percentage.
    :return: DepthResult containing the generated depth map and the original image.
    """
    image: str = await ws_manager.send_command("cameraUpdate", {"zoomPercentage": zoom_percentage})

    # Generates depth map for the image and returns both the depth map and the original image.
    depth_map_object: DepthResult = generate_depth_map(image)

    ## Logs activity.
    log = ActivityLog(toolCall="cameraUpdate", base64_image=image)
    await log.insert()

    return depth_map_object
