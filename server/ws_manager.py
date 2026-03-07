import asyncio
import json
from fastapi import WebSocket


class WebSocketManager:
    """Manages a single WebSocket connection to the Three.js frontend."""

    def __init__(self):
        self._ws: WebSocket | None = None
        self._response_event: asyncio.Event = asyncio.Event()
        self._response_data: dict | None = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._ws = ws

    def disconnect(self):
        self._ws = None
        self._response_data = None
        self._response_event.clear()

    def receive_response(self, data: dict):
        """Called when the frontend sends a response back."""
        self._response_data = data
        self._response_event.set()

    async def send_command(self, action: str, params: dict, timeout: float = 30.0) -> str:
        """Send a command to the Three.js client and await a base64 image response."""
        if not self._ws:
            raise RuntimeError("No Three.js client connected")

        self._response_event.clear()
        self._response_data = None

        message = {"action": action, **params}
        await self._ws.send_json(message)

        try:
            await asyncio.wait_for(self._response_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"Timeout waiting for Three.js response to '{action}'")

        image = self._response_data.get("image", "")
        self._response_data = None
        return image


ws_manager = WebSocketManager()
