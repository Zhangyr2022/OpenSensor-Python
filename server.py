from kalman_filter import KalmanFilter
from message import Message, SensorMessage
import asyncio
import json
import logging
from typing import Optional, Dict, Callable
from uuid import UUID
import websockets
from utilities import convert_camel_to_snake


class Server:
    """WebSocket server class."""

    def __init__(
        self,
        port: int,
        after_message_receive_event: Optional[Callable[..., None]] = None,
    ):
        self.after_message_receive_event = after_message_receive_event
        self._logger = logging.getLogger("Server")
        self._sockets: Dict[UUID, websockets.WebSocketServerProtocol] = {}
        self._is_running = False
        self._ws_server: Optional[websockets.WebSocketServer] = None
        self._port = port

    async def start(self):
        """Starts the WebSocket server."""
        if self._is_running:
            raise RuntimeError("Server is already running")

        if self._ws_server is not None:
            raise RuntimeError("WebSocketServer instance already exists")

        self._logger.info("Starting...")

        try:
            self._ws_server = await websockets.serve(
                self._on_connect, host="183.172.226.9", port=self._port
            )
            self._is_running = True
            self._logger.info("Started.")
            await self._ws_server.wait_closed()
        except Exception as ex:
            self._logger.error(f"Failed to start server: {ex}")

    async def stop(self):
        """Stops the WebSocket server."""
        if not self._is_running:
            raise RuntimeError("Server is not running")

        self._logger.info("Stopping...")

        if self._ws_server is None:
            raise RuntimeError("WebSocketServer instance is not available")

        await self._ws_server.close()
        self._is_running = False

        self._logger.info("Stopped.")

    async def publish(self, message: dict):
        """Publishes a message to all connected sockets."""
        json_string = json.dumps(message)

        for socket in self._sockets.values():
            try:
                await socket.send(json_string)
            except Exception as ex:
                self._logger.error(
                    f"Failed to send message to {socket.remote_address}: {ex}"
                )

    async def _on_connect(
        self, websocket: websockets.WebSocketServerProtocol, path: str
    ):
        """Handles the WebSocket on connect event."""
        connection_id = len(self._sockets) + 1  # 或者使用其他唯一标识符生成方式
        self._logger.debug("Connection %d opened.", connection_id)
        self._sockets[connection_id] = websocket

        try:
            async for message in websocket:
                await self._handle_message(message)
        finally:
            await self._on_disconnect(connection_id)

    async def _on_disconnect(self, connection_id: int):
        """Handles the WebSocket on disconnect event."""
        self._logger.debug("Connection %d closed.", connection_id)
        self._sockets.pop(connection_id, None)

    async def _handle_message(self, text: str):
        """Handles a received WebSocket message."""
        general_message = json.loads(text)

        if general_message["message_type"] == "SENSOR":
            sensor_message = SensorMessage(**general_message)
            if self.after_message_receive_event is not None:
                self.after_message_receive_event(sender=self, e=sensor_message)
            else:
                self._logger.warning("No event handler is set.")
        else:
            raise ValueError(f"Invalid message type: {general_message['message_type']}")
