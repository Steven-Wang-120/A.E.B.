from __future__ import annotations

import asyncio
import base64
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

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
        self.context = FakeContext()
        self.plugin = plugin_module.AstrBotEXInteractionPlugin(self.context)
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

        json_payload = {
            "stream_id": "front-camera-yolo",
            "frame_id": 1,
            "objects": [{"class_name": "person", "bbox_xyxy": [1, 2, 3, 4]}],
        }
        published_json, _ = await self._request(
            text,
            "text",
            "vision.json.publish",
            json_payload,
        )
        self.assertEqual(published_json["payload"]["items"], 1)

        frame = b"\xff\xd8jpeg\xff\xd9"
        image_payload = {
            "stream_id": "front-camera-yolo",
            "frame_id": 1,
            "frame_content_type": "image/jpeg",
        }
        published_jpeg, _ = await self._request(
            vision,
            "vision",
            "vision.jpeg.publish",
            image_payload,
            frame,
        )
        self.assertEqual(published_jpeg["payload"]["bytes"], len(frame))

        json_status, _ = await self._request(text, "text", "vision.json.status", {})
        self.assertEqual(json_status["payload"]["json"]["count"], 1)
        self.assertEqual(
            json_status["payload"]["json"]["items"][0]["payload"], json_payload
        )

        vision_status, _ = await self._request(
            vision,
            "vision",
            "vision.status",
            {},
        )
        self.assertEqual(vision_status["payload"]["json"]["count"], 1)
        self.assertEqual(vision_status["payload"]["jpeg"]["count"], 1)
        self.assertEqual(
            vision_status["payload"]["jpeg"]["items"][0]["stream_id"],
            "front-camera-yolo",
        )

    async def test_split_vision_tools_return_exact_json_and_jpeg(self) -> None:
        frame = b"\xff\xd8" + bytes(index % 251 for index in range(630)) + b"\xff\xd9"
        self.assertEqual(len(frame), 634)
        json_payload = {
            "stream_id": "step5-flow",
            "frame_id": 5,
            "source": "step5-verification",
            "objects": [
                {
                    "bbox_xyxy": [1, 2, 3, 4],
                    "class_name": "test-object",
                    "confidence": 0.99,
                }
            ],
        }
        image_payload = {
            "stream_id": "step5-flow",
            "frame_id": 5,
            "source": "step5-verification",
            "frame_content_type": "image/jpeg",
        }
        json_response = await self.plugin._handle_vision_json_publish(
            b"test-peer",
            {"payload": json_payload},
            None,
        )
        jpeg_response = await self.plugin._handle_vision_jpeg_publish(
            b"test-peer",
            {"payload": image_payload},
            frame,
        )
        self.assertEqual(json_response["items"], 1)
        self.assertEqual(jpeg_response["bytes"], 634)

        json_tool = next(
            item
            for item in self.context.tools
            if item.name == "get_astrbotex_vision_json_buffer"
        )
        json_result = await json_tool.call(None)
        json_text = next(
            item
            for item in json_result.content
            if isinstance(item, plugin_module.mcp.types.TextContent)
        )
        json_metadata = json.loads(json_text.text)
        self.assertEqual(json_metadata["count"], 1)
        self.assertEqual(json_metadata["items"][0]["payload"], json_payload)

        jpeg_tool = next(
            item
            for item in self.context.tools
            if item.name == "get_astrbotex_vision_jpeg_buffer"
        )
        jpeg_result = await jpeg_tool.call(None)
        jpeg_text = next(
            item
            for item in jpeg_result.content
            if isinstance(item, plugin_module.mcp.types.TextContent)
        )
        image_content = next(
            item
            for item in jpeg_result.content
            if isinstance(item, plugin_module.mcp.types.ImageContent)
        )
        jpeg_metadata = json.loads(jpeg_text.text)
        self.assertEqual(jpeg_metadata["stream_id"], "step5-flow")
        self.assertEqual(jpeg_metadata["bytes"], 634)
        self.assertEqual(jpeg_metadata["payload"], image_payload)
        self.assertEqual(image_content.mimeType, "image/jpeg")
        self.assertEqual(base64.b64decode(image_content.data), frame)

    async def test_vision_caches_retain_eight_json_items_and_eight_jpeg_items(
        self,
    ) -> None:
        for frame_id in range(10):
            await self.plugin._handle_vision_json_publish(
                b"test-peer",
                {"payload": {"stream_id": "front", "frame_id": frame_id}},
                None,
            )
            await self.plugin._handle_vision_jpeg_publish(
                b"test-peer",
                {
                    "payload": {
                        "stream_id": "front",
                        "frame_id": frame_id,
                        "frame_content_type": "image/jpeg",
                    }
                },
                b"\xff\xd8" + bytes([frame_id]) + b"\xff\xd9",
            )

        self.assertEqual(len(self.plugin._vision_json_cache), 8)
        self.assertEqual(len(self.plugin._vision_jpeg_cache), 8)
        self.assertEqual(self.plugin._vision_json_cache[0]["frame_id"], 2)
        self.assertEqual(self.plugin._vision_jpeg_cache[0]["frame_id"], 2)

        json_tool = next(
            item
            for item in self.context.tools
            if item.name == "get_astrbotex_vision_json_buffer"
        )
        json_result = await json_tool.call(None, limit=8)
        json_text = next(
            item
            for item in json_result.content
            if isinstance(item, plugin_module.mcp.types.TextContent)
        )
        json_metadata = json.loads(json_text.text)
        self.assertEqual(
            [item["frame_id"] for item in json_metadata["items"]],
            [9, 8, 7, 6, 5, 4, 3, 2],
        )

        jpeg_tool = next(
            item
            for item in self.context.tools
            if item.name == "get_astrbotex_vision_jpeg_buffer"
        )
        jpeg_result = await jpeg_tool.call(None, index=7)
        jpeg_text = next(
            item
            for item in jpeg_result.content
            if isinstance(item, plugin_module.mcp.types.TextContent)
        )
        jpeg_metadata = json.loads(jpeg_text.text)
        self.assertEqual(jpeg_metadata["frame_id"], 2)

    async def test_vision_tools_do_not_mutate_automatic_llm_context(self) -> None:
        marker = "vision-buffer-must-stay-on-demand"
        await self.plugin._handle_vision_json_publish(
            b"test-peer",
            {
                "payload": {
                    "stream_id": "on-demand-only",
                    "objects": [{"class_name": marker}],
                }
            },
            None,
        )
        await self.plugin._handle_vision_jpeg_publish(
            b"test-peer",
            {
                "payload": {
                    "stream_id": "on-demand-only",
                    "frame_content_type": "image/jpeg",
                }
            },
            b"\xff\xd8" + b"x" * 630 + b"\xff\xd9",
        )
        for tool_name in (
            "get_astrbotex_vision_json_buffer",
            "get_astrbotex_vision_jpeg_buffer",
        ):
            tool = next(item for item in self.context.tools if item.name == tool_name)
            await tool.call(None)

        async def fake_request_text(method: str, payload: dict) -> dict:
            self.assertEqual(method, "bridge.context.get")
            self.assertEqual(payload, {})
            return {
                "context_id": "ctx-1",
                "blocks": [],
                "affordances": [],
                "proposal_schema": {},
                "rules": [],
                "vision_marker": marker,
            }

        class FakeProviderRequest:
            def __init__(self) -> None:
                self.extra_user_content_parts = []

        self.plugin.request_text = fake_request_text
        request = FakeProviderRequest()
        await self.plugin.inject_ex_context(None, request)
        injected_text = "\n".join(
            part.text for part in request.extra_user_content_parts
        )
        self.assertNotIn(marker, injected_text)
        self.assertNotIn("vision_marker", injected_text)


if __name__ == "__main__":
    unittest.main()
