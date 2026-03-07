# API Endpoints

Base URL: `http://localhost:8000`

---

## Health

### `GET /`
Returns a message confirming the API is running.

**Response:** `{ "message": "Robotics Vision API is running" }`

### `GET /health`
Health check.

**Response:** `{ "status": "ok" }`

---

## General API

### `POST /saveSplat`
Save a Gaussian Splat environment to the database.

**Request Body:**
```json
{
  "x": 0.0, "y": 0.0, "z": 0.0,
  "scaleX": 1.0, "scaleY": 1.0, "scaleZ": 1.0,
  "rotX": 0.0, "rotY": 0.0, "rotZ": 0.0, "rotW": 1.0,
  "r": 255.0, "g": 255.0, "b": 255.0,
  "opacity": 1.0
}
```

**Response:** `{ "status": "saved" }`

### `POST /startMCP`
Start the MCP (Gemini Agent). *(Not yet implemented)*

**Response:** `{ "status": "string" }`

### `GET /currPosRot`
Get the most recent pose (position + rotation).

**Response:**
```json
{
  "id": 1,
  "iteration_num": 0,
  "trajectory_fk": 1,
  "xPos": 0.0, "yPos": 0.0, "zPos": 0.0,
  "rotX": 0.0, "rotY": 0.0, "rotZ": 0.0, "rotW": 1.0,
  "deltaX": 0.0, "deltaY": 0.0, "deltaZ": 0.0,
  "deltaRotX": 0.0, "deltaRotY": 0.0, "deltaRotZ": 0.0,
  "timestamp": "2026-03-06T15:00:00Z"
}
```

**Error:** `404` if no poses exist.

### `GET /latestLog`
Get the most recent activity log entry.

**Response:**
```json
{
  "id": 1,
  "toolCall": "updatePosition",
  "pose_fk": 1,
  "base64_image": "data:image/png;base64,..."
}
```

**Error:** `404` if no logs exist.

### `GET /trajectory/{trajectory_id}`
Get a trajectory and its ordered list of poses.

**Path Params:** `trajectory_id` (int)

**Response:**
```json
{
  "id": 1,
  "environment_fk": 1,
  "poses": [
    {
      "id": 1,
      "iteration_num": 0,
      "trajectory_fk": 1,
      "xPos": 0.0, "yPos": 0.0, "zPos": 0.0,
      "rotX": null, "rotY": null, "rotZ": null, "rotW": null,
      "deltaX": null, "deltaY": null, "deltaZ": null,
      "deltaRotX": null, "deltaRotY": null, "deltaRotZ": null,
      "timestamp": "2026-03-06T15:00:00Z"
    }
  ]
}
```

**Error:** `404` if trajectory not found.

---

## Tool Calls (MCP)

### `POST /updatePosition`
Update the robot's position. Returns a depth map result.

**Request Body:**
```json
{ "x": 1.0, "y": 2.0, "z": 3.0 }
```

**Response:** `DepthResult`

### `POST /updateRotation`
Update the robot's rotation. Returns a depth map result.

**Request Body:**
```json
{ "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 }
```

**Response:** `DepthResult`

### `POST /updateCamera`
Update the camera zoom. Returns a depth map result.

**Request Body:**
```json
{ "zoom_percentage": 50.0 }
```

**Response:** `DepthResult`
