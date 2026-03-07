from tool_calls.depth import generate_depth_map, DepthResult
from models import Pose, ActivityLog
from ws_manager import ws_manager


async def update_position(x: float, y: float, z: float) -> DepthResult:
    """
    Updates the position of the camera in the 3D scene and logs the activity.

    :param x: The new x-coordinate of the camera position.
    :param y: The new y-coordinate of the camera position.
    :param z: The new z-coordinate of the camera position.
    :return: DepthResult containing the generated depth map and the original image.
    """
    image: str = await ws_manager.send_command("updatePosition", {"x": x, "y": y, "z": z})

    # Generates depth map for the image and returns both the depth map and the original image.
    depth_map_object: DepthResult = generate_depth_map(image)

    # Save position change in DB.
    last_pose = await Pose.find_all().sort("-timestamp").first_or_none()
    pose = Pose(
        iteration_num=(last_pose.iteration_num + 1) if last_pose else 0,
        trajectory_fk=last_pose.trajectory_fk if last_pose else None,
        xPos=x, yPos=y, zPos=z,
        deltaX=(x - last_pose.xPos) if last_pose else 0,
        deltaY=(y - last_pose.yPos) if last_pose else 0,
        deltaZ=(z - last_pose.zPos) if last_pose else 0,
    )
    await pose.insert()

    # Logs activity
    log = ActivityLog(toolCall="updatePosition", pose_fk=pose, base64_image=image)
    await log.insert()

    return depth_map_object

async def update_rotation(x: float, y: float, z: float, w: float) -> DepthResult:
    """
    Updates the rotation of the camera in the 3D scene and logs the activity.

    :param x: The new x-coordinate of the camera rotation.
    :param y: The new y-coordinate of the camera rotation.
    :param z: The new z-coordinate of the camera rotation.
    :param w: The new w-coordinate of the camera rotation.
    :return: DepthResult containing the generated depth map and the original image.
    """
    image: str = await ws_manager.send_command("updateRotation", {"x": x, "y": y, "z": z, "w": w})

    # Generates depth map for the image and returns both the depth map and the original image.
    depth_map_object: DepthResult = generate_depth_map(image)

    # Save rotation change in DB.
    last_pose = await Pose.find_all().sort("-timestamp").first_or_none()
    pose = Pose(
        iteration_num=(last_pose.iteration_num + 1) if last_pose else 0,
        trajectory_fk=last_pose.trajectory_fk if last_pose else None,
        xPos=last_pose.xPos if last_pose else 0,
        yPos=last_pose.yPos if last_pose else 0,
        zPos=last_pose.zPos if last_pose else 0,
        rotX=x, rotY=y, rotZ=z, rotW=w,
        deltaRotX=(x - last_pose.rotX) if last_pose and last_pose.rotX else 0,
        deltaRotY=(y - last_pose.rotY) if last_pose and last_pose.rotY else 0,
        deltaRotZ=(z - last_pose.rotZ) if last_pose and last_pose.rotZ else 0,
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
