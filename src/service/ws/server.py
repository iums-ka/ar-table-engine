import asyncio
import threading
import json
import websockets
from typing import Set


class WebSocketServer:
    def __init__(self, host="0.0.0.0", port=5001):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.loop = None

    async def _handler(self, websocket):
        print("WebSocket client connected")
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print("WebSocket client disconnected")

    async def _run(self):
        async with websockets.serve(self._handler, self.host, self.port):
            print(f"WebSocket server listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

    def start(self):
        def runner():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._run())

        threading.Thread(target=runner, daemon=True).start()

    def broadcast(self, payload: dict):
        if not self.loop or not self.clients:
            return

        message = json.dumps(payload)

        async def _send():
            dead = []
            for ws in self.clients:
                try:
                    await ws.send(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.clients.discard(ws)

        asyncio.run_coroutine_threadsafe(_send(), self.loop)