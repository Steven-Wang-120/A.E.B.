from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
import unittest
import uuid
from pathlib import Path

import zmq
import zmq.asyncio

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))

from zmq_transport import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    ZmqReply,
    ZmqRouterChannel,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def envelope(
    channel: str,
    kind: str,
    method: str,
    payload: dict,
    *,
    reply_to: str | None = None,
) -> dict:
    data = {
        "protocol": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "channel": channel,
        "kind": kind,
        "id": uuid.uuid4().hex,
        "method": method,
        "timestamp": time.time(),
        "payload": payload,
    }
    if reply_to is not None:
        data["reply_to"] = reply_to
    return data


class ZmqRouterChannelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.endpoint = f"tcp://127.0.0.1:{free_port()}"
        self.channel = ZmqRouterChannel(
            "text",
            self.endpoint,
            request_timeout_sec=1.0,
        )
        self.client = zmq.asyncio.Context.instance().socket(zmq.DEALER)
        self.client.setsockopt(zmq.LINGER, 0)
        self.client.setsockopt(zmq.IDENTITY, b"test-ex")

    async def asyncTearDown(self) -> None:
        self.client.close(linger=0)
        await self.channel.close()

    async def _client_request(
        self,
        method: str,
        payload: dict,
        binary: bytes | None = None,
    ) -> tuple[dict, bytes | None]:
        request = envelope("text", "request", method, payload)
        frames = [json.dumps(request).encode("utf-8")]
        if binary is not None:
            frames.append(binary)
        await self.client.send_multipart(frames)
        response_frames = await asyncio.wait_for(self.client.recv_multipart(), 1.0)
        response = json.loads(response_frames[0].decode("utf-8"))
        response_binary = response_frames[1] if len(response_frames) == 2 else None
        self.assertEqual(response["reply_to"], request["id"])
        return response, response_binary

    async def test_handshake_and_binary_handler(self) -> None:
        async def echo(_peer, request, binary):
            return ZmqReply({"ok": True, "value": request["payload"]["value"]}, binary)

        self.channel.register_handler("test.echo", echo)
        await self.channel.start()
        self.client.connect(self.endpoint)

        hello, _ = await self._client_request("system.hello", {"client": "EX"})
        self.assertTrue(hello["payload"]["ok"])
        self.assertEqual(self.channel.default_peer, b"test-ex")

        response, response_binary = await self._client_request(
            "test.echo",
            {"value": 7},
            b"binary-payload",
        )
        self.assertEqual(response["payload"]["value"], 7)
        self.assertEqual(response_binary, b"binary-payload")

    async def test_astrbot_can_request_the_connected_ex_peer(self) -> None:
        await self.channel.start()
        self.client.connect(self.endpoint)
        await self._client_request("system.hello", {"client": "EX"})

        async def ex_responder() -> None:
            request_frames = await self.client.recv_multipart()
            request = json.loads(request_frames[0].decode("utf-8"))
            response = envelope(
                "text",
                "response",
                request["method"],
                {"ok": True, "context_id": "ctx-1"},
                reply_to=request["id"],
            )
            await self.client.send_json(response)

        responder = asyncio.create_task(ex_responder())
        reply = await self.channel.request("bridge.context.get", {})
        await responder
        self.assertEqual(reply.payload["context_id"], "ctx-1")


if __name__ == "__main__":
    unittest.main()
