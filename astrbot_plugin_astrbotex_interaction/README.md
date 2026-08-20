# AstrBotEX ZeroMQ Integration Plugin

This plugin merges the former AstrBotEX interaction and bridge plugins. It
registers the `astrbotex` platform adapter, injects EX context into LLM
requests, exposes proposal and on-demand vision-buffer tools, proxies STT/TTS
providers, and maintains three independent ZeroMQ channels.

## Channel layout

AstrBot binds three ROUTER sockets. AstrBotEX must connect one DEALER socket to
each endpoint and send `system.hello` on every connection.

| Channel | Default endpoint | Purpose |
|---------|------------------|---------|
| `text` | `tcp://0.0.0.0:8766` | Text interaction, replies, context, proposals, runtime commands, vision JSON fields |
| `audio` | `tcp://0.0.0.0:8767` | Provider status, STT input, TTS output |
| `vision` | `tcp://0.0.0.0:8768` | JPEG image frames and vision binary payloads |

The channels are intentionally separate. Slow STT/TTS or high-volume vision
traffic cannot consume the text channel's queue.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASTRBOTEX_ZMQ_BIND_HOST` | `0.0.0.0` | Bind address for all three ROUTER sockets |
| `ASTRBOTEX_ZMQ_TEXT_PORT` | `8766` | Text/JSON channel port |
| `ASTRBOTEX_ZMQ_AUDIO_PORT` | `8767` | STT/TTS channel port |
| `ASTRBOTEX_ZMQ_VISION_PORT` | `8768` | JPEG/binary vision channel port |
| `ASTRBOTEX_ZMQ_TIMEOUT_SEC` | `10` | Outbound request timeout |
| `ASTRBOTEX_ZMQ_VISION_CACHE_ITEMS` | `8` | Number of latest JSON records and JPEG frames retained in each separate vision cache |
| `ASTRBOTEX_ZMQ_VISION_CACHE_STREAMS` | `8` | Deprecated compatibility alias for `ASTRBOTEX_ZMQ_VISION_CACHE_ITEMS` |
| `ASTRBOTEX_SESSION_ID` | `astrbotex_default` | Default platform session |

When both containers are attached to the same Docker bridge network, no host
port publication is required. The EX side connects to `tcp://astrbot:8766`,
`tcp://astrbot:8767`, and `tcp://astrbot:8768`.

## Protocol envelope

The first multipart frame is UTF-8 JSON:

```json
{
  "protocol": "astrbotex-zmq",
  "version": 1,
  "channel": "text",
  "kind": "request",
  "id": "request-uuid",
  "method": "interaction.message",
  "timestamp": 1786320000.0,
  "payload": {}
}
```

Responses use `kind=response` and include `reply_to` with the request ID. Audio
and vision messages may append one binary multipart frame after the JSON frame.

## Methods

Text channel:

- EX -> AstrBot: `interaction.message`, `transport.status`, `vision.json.publish`,
  `vision.json.status`
- AstrBot -> EX: `interaction.reply`, `bridge.context.get`,
  `bridge.proposal.submit`, `runtime.status`
- `vision.json.publish` accepts YOLO/object-detection JSON fields only. It does
  not accept a binary multipart frame.

Audio channel:

- EX -> AstrBot: `providers.status`, `stt.transcribe`, `tts.synthesize`
- `stt.transcribe` accepts a binary audio frame plus `filename` metadata.
- `tts.synthesize` returns metadata plus a binary audio frame.

Vision channel:

- EX -> AstrBot: `vision.jpeg.publish`, `vision.status`
- `vision.jpeg.publish` accepts JPEG metadata plus one binary multipart frame.
- `vision.publish` remains as a backward-compatible combined JSON + optional
  binary method, but new clients should use the split JSON/JPEG methods above.
- A.E.B keeps two independent bounded caches: latest 8 JSON records and latest
  8 JPEG frames by default.
- The `get_astrbotex_vision_json_buffer` LLM tool returns buffered JSON fields.
- The `get_astrbotex_vision_jpeg_buffer` LLM tool returns buffered JPEG images
  as MCP image content without changing their bytes.
- Vision payloads are not added by the automatic LLM request hook. JSON and
  images become visible to the model only after it explicitly calls the vision
  buffer tools.

## Installation

1. Remove or disable the old `astrbot_plugin_astrbotex_bridge` plugin.
2. Replace the old interaction plugin directory with this directory.
3. Install `requirements.txt`; the required package is `pyzmq`.
4. Keep the `astrbotex` platform enabled in `cmd_config.json`.
5. Restart AstrBot and connect the three EX DEALER sockets.

The old port `8766` is retained for the text channel, but it is no longer an
HTTP server. HTTP requests sent to that port will not work.

## Tests

Build and run the isolated Docker test image from this directory:

```sh
docker build -f Dockerfile.test -t local/astrbotex-plugin-zmq-test .
docker run --rm local/astrbotex-plugin-zmq-test
```
