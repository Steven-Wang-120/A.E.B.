from __future__ import annotations

import asyncio
import importlib
import json
import socket
import sys
import time
import unittest
import uuid
from pathlib import Path
from typing import ClassVar

import zmq
import zmq.asyncio

WORK_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK_ROOT))
plugin_module = importlib.import_module("astrbot_plugin_astrbotex_interaction.main")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def envelope(channel: str, method: str, payload: dict) -> dict:
    return {
        "protocol": "astrbotex-zmq",
        "version": 1,
        "channel": channel,
        "kind": "request",
        "id": uuid.uuid4().hex,
        "method": method,
        "timestamp": time.time(),
        "payload": payload,
    }


class FakePlatformManager:
    platform_insts: ClassVar[list] = []


class FakeContext:
    def __init__(self) -> None:
        self.platform_manager = FakePlatformManager()
        self.tools = []

    def add_llm_tools(self, *tools) -> None:
        self.tools.extend(tools)

    def get_using_stt_provider(self):
        return None

    def get_config(self) -> dict:
        return {"provider_tts_settings": {"provider_id": ""}}

    def get_provider_by_id(self, _provider_id):
        return None

    def get_all_tts_providers(self) -> list:
        return []


class PluginChannelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.plugin = plugin_module.AstrBotEXInteractionPlugin(FakeContext())
        self.plugin.zmq_bind_host = "127.0.0.1"
        self.plugin.text_port = free_port()
        self.plugin.audio_port = free_port()
        self.plugin.vision_port = free_port()
        await self.plugin.initialize()
        self.clients: list[zmq.asyncio.Socket] = []

    async def asyncTearDown(self) -> None:
        for client in self.clients:
            client.close(linger=0)
        await self.plugin.terminate()

    async def _connect(self, channel: str, port: int) -> zmq.asyncio.Socket:
        client = zmq.asyncio.Context.instance().socket(zmq.DEALER)
        client.setsockopt(zmq.LINGER, 0)
        client.setsockopt(zmq.IDENTITY, f"test-{channel}".encode())
        client.connect(f"tcp://127.0.0.1:{port}")
        self.clients.append(client)
        hello, _ = await self._request(client, channel, "system.hello", {})
        self.assertTrue(hello["payload"]["ok"])
        return client

    async def _request(
        self,
        client: zmq.asyncio.Socket,
        channel: str,
        method: str,
        payload: dict,
        binary: bytes | None = None,
    ) -> tuple[dict, bytes | None]:
        request = envelope(channel, method, payload)
        frames = [json.dumps(request).encode()]
        if binary is not None:
            frames.append(binary)
        await client.send_multipart(frames)
        response_frames = await asyncio.wait_for(client.recv_multipart(), 1.0)
        response = json.loads(response_frames[0])
        self.assertEqual(response["reply_to"], request["id"])
        return response, response_frames[1] if len(response_frames) == 2 else None

    async def test_three_channels_are_independent_and_respond(self) -> None:
        text = await self._connect("text", self.plugin.text_port)
        audio = await self._connect("audio", self.plugin.audio_port)
        vision = await self._connect("vision", self.plugin.vision_port)

        status, _ = await self._request(text, "text", "transport.status", {})
        self.assertEqual(status["payload"]["channels"]["text"]["peers"], 1)
        self.assertEqual(status["payload"]["channels"]["audio"]["peers"], 1)
        self.assertEqual(status["payload"]["channels"]["vision"]["peers"], 1)

        providers, _ = await self._request(audio, "audio", "providers.status", {})
        self.assertIsNone(providers["payload"]["stt"])
        self.assertIsNone(providers["payload"]["tts"])

        published, _ = await self._request(
            vision,
            "vision",
            "vision.publish",
            {
                "stream_id": "front-camera-vector",
                "dtype": "float32",
                "shape": [4],
            },
            b"\x00\x01\x02\x03",
        )
        self.assertEqual(published["payload"]["bytes"], 4)

        vision_status, _ = await self._request(
            vision,
            "vision",
            "vision.status",
            {},
        )
        self.assertEqual(
            vision_status["payload"]["streams"][0]["stream_id"],
            "front-camera-vector",
        )


if __name__ == "__main__":
    unittest.main()
