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

## Private task planning (B05/B06, default OFF)

`ASTRBOTEX_TASK_PLANNING=1` requires `ASTRBOTEX_TASK_DB_PATH` (an existing writable
parent directory) and `ASTRBOTEX_TASK_PROVIDER_ID`. It uses the public Host
`Context.llm_generate(chat_provider_id=..., tools=ToolSet, contexts=..., stream=False)`
with a private, bounded tool loop. It does not register these tools in global
chat, create ordinary user events, parse prose commands, or require a final reply.
`save_plan`, `submit_current_goal`, `emit_user_message`, `finish_planning_turn` are
the only allowed tools. `finish` has no public_text. Skill content is explicitly loaded.

### Administrator task commands

Default OFF remains `ASTRBOTEX_TASK_PLANNING=0`. When enabled, also configure
`ASTRBOTEX_TASK_ROBOT_ID` from trusted deployment configuration, plus DB/provider
above. Optional `ASTRBOTEX_TASK_PEER_ID` is the exact text DEALER identity; it is
not a model argument. Commands use the real Host ADMIN filter and additionally
check event.is_admin() before constructing authority:

- `ex_task <all remaining task text>`
- `ex_task_update <task_id> <all remaining replacement text>`
- `ex_task_review <task_id>`
- `ex_task_cancel <task_id>`

Successful commands consume the Host event, disable its automatic LLM path and
produce no JSON/final/start announcement; only private emit may speak. Entry
errors expose fixed bounded diagnostic codes. Obtain task_id from the local
TaskStore/admin diagnostic interface (no new public success receipt).

Authority uses configured robot_id, real event sender, unified_msg_origin (as
session_id, including platform/session) and a framework route hash; durable
routing also stores source session and exact peer. Model/tool arguments and
ordinary interaction.message payloads do not construct TaskAuthority. Routing
chooses the framework-owned EX event peer, explicit configured peer, or exactly
one hello-completed peer seen in the last 30 seconds; multiple unbound peers are
rejected instead of using default_peer. A persisted route's complete identity
AND peer are immutable. Peer rotation needs a separately reviewed migration,
not a second user's rebind. Recovered tasks require same authorized identity
and explicit ex_task_review, never old start/renew replay.

Underlying `admit_task_route` / `begin_robot_task` hooks remain available to
trusted framework callers. Plan changes preserve completed steps, CAS revisions
and future intent-only steps. Goal+params stay atomic in the durable outbox;
submit timeouts replay only the same ID within the first lease. Only the private
framework decision.goal.submit seam accepts strictly validated business
GoalSubmitResult {ok:false,phase:rejected,...}; arbitrary remote errors, malformed
IDs/shapes and non-submit methods still raise ZmqRemoteError. A valid rejection
is persisted and clears current_goal/lease; it does not count as uncertain IO.

### Interaction turn extension (separate from frozen B00 decision methods)

`interaction.task.turn` uses unchanged outer `astrbotex-zmq`, version 1, text
request/response. Business payload has exactly:

```json
{"task_schema_version":1,"operation":"bind","ex_session":"ex-boot","task_id":"task-id","robot_id":"robot-config-id","session_id":"platform:FriendMessage:session","user_id":"host-sender","route_ref":"framework-hash","turn_id":"framework-turn","generation":1,"expected_revision":0}
```

operation is bind|invalidate; invalidate permits turn_id:null. generation is
integer >=1; expected_revision is integer >=0. All fields come from saved
framework state, with no model recipients. EX must respond {ok:true}; failed
bind prevents LLM/tool execution. `turn_sync=None` keeps existing offline
coordinator tests compatible; the real plugin always injects the extension.
Begin-turn binds immediately when session is known, then refreshes after context
binding and after goal admission/revision assignment. Updates/cancel/review,
planning failure, lease loss and invalidated terminal review revoke immediately,
not after a late LLM response. Finish alone does NOT invalidate already legal
queued emit; next generation/target replacement or EX voice-generation change
must discard old queued text/audio.

EX composition must associate the configured trusted text connection, full
source identity, single-robot task ownership, revision and generation watermarks;
capture its own voice generation on bind and recheck at public send/audio commit.
This extension is NOT added to task_contracts or the seven decision methods.
Its real EX composition counterpart is an outstanding integration test.


Feedback is durably received before ACK. `acked_event_seq` is this receipt's
sequence, NOT a contiguous applied cursor or completion assertion. Replay cursor
is separate. Gaps/terminal facts reconcile events.get and state.get before step
advance. Independent heartbeat renews only live local leases; unknown/offline
execution stays owned and cannot silently replay. A bounded reconciler recovers
missed feedback even after ACK/connection loss without per-frame LLM calls.

B04 integration hooks still require agreement (not frozen wire field changes):
- context.get result: `ex_session`, strict integer `revision`, `actions` list of
  `{action_id, schema, ...}` and relevant descriptions/observations.
- state.get's existing free-form `execution.feedback`: current goal-level B00
  Feedback; completion uses `details.completion_evidence` with `verified=true`,
  matching goal_id/goal_revision and succeeded_actions covering required actions.
  Missing proof never advances; timed_out/unknown require user review.
- route identity must be authenticated outside LLM/network claims.
- EX transport `set_decision_handler(handler, public_validator=..., public_handler=...)`
  gets trusted connection_id. Task replies NEVER fall through to legacy
  InteractionCore; the dedicated handler must enforce single text-or-TTS delivery
  and recheck generation at asynchronous audio playback commit.

TaskOutputRouter emits trusted task/turn/generation/route/message IDs and claims
public messages before IO (at-most-once attempt; ambiguous failed notice is not
action retry). Explicit public text is a model-controlled channel; the framework
prevents automatic raw/tool/error/chunk forwarding, not arbitrary exfiltration
inside malicious explicit emit text. No real LLM, account, TTS or hardware checks
are claimed by deterministic tests.

## Tests

Functional units and their fixtures remain under this plugin's `tests/`.
System/composition tests and the finite campaign now live in the independent
[AstrBotVLA-tests](https://github.com/Steven-Wang-120/AstrBotVLA-tests) repository,
under `validation_tests.host`; drivers use `validation_drivers`.

From that repository's root, with disjoint EX/A.E.B checkouts and the already
prepared local Host 4.26.7 image (no pull/build/network):

```sh
python -B run_validation.py host-unit --ex-checkout ../ex --aeb-checkout ../aeb --host-image local/hzf-aeb-baseline:20260927
python -B run_validation.py host-integration --ex-checkout ../ex --aeb-checkout ../aeb --host-image local/hzf-aeb-baseline:20260927
python -B run_validation.py campaign --ex-checkout ../ex --aeb-checkout ../aeb --host-image local/hzf-aeb-baseline:20260927
```

See its README for pinned commits, explicit development-only `--allow-dirty`,
strict no-skips/thread-leaks/source-changes evidence and isolated ROS checks.
The campaign is deterministic fake-provider/EX repetition, not real B04
integration. Historical limitations above are retained; real NapCat, TTS and
mechanical stop are not established by these offline checks.
