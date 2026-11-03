# AstrBotEX ZeroMQ Integration Plugin

This plugin merges the former AstrBotEX interaction and bridge plugins. It
registers the `astrbotex` platform adapter, injects EX context into LLM
requests, exposes proposal and on-demand vision-buffer tools, proxies STT/TTS
providers, and maintains three independent ZeroMQ channels.

An optional fourth channel carries upper-control execution steps. It is off by
default and never moves the three legacy endpoints.

## Channel layout

AstrBot binds three ROUTER sockets. AstrBotEX must connect one DEALER socket to
each endpoint and send `system.hello` on every connection.

| Channel | Default endpoint | Purpose |
|---------|------------------|---------|
| `text` | `tcp://0.0.0.0:8766` | Text interaction, replies, context, proposals, runtime commands, vision JSON fields |
| `audio` | `tcp://0.0.0.0:8767` | Provider status, STT input, TTS output |
| `vision` | `tcp://0.0.0.0:8768` | JPEG image frames and vision binary payloads |
| `control` | `tcp://127.0.0.1:8769` | Upper-control step injection only (`upper.*`), off by default |

The channels are intentionally separate. Slow STT/TTS or high-volume vision
traffic cannot consume the text channel's queue. The `control` socket is bound
only when `control_enabled` is set; execution steps never travel on `text`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASTRBOTEX_ZMQ_BIND_HOST` | `0.0.0.0` | Bind address for the three legacy ROUTER sockets |
| `ASTRBOTEX_ZMQ_TEXT_PORT` | `8766` | Text/JSON channel port |
| `ASTRBOTEX_ZMQ_AUDIO_PORT` | `8767` | STT/TTS channel port |
| `ASTRBOTEX_ZMQ_VISION_PORT` | `8768` | JPEG/binary vision channel port |
| `ASTRBOTEX_ZMQ_TIMEOUT_SEC` | `10` | Outbound request timeout |
| `ASTRBOTEX_ZMQ_VISION_CACHE_ITEMS` | `8` | Number of latest JSON records and JPEG frames retained in each separate vision cache |
| `ASTRBOTEX_ZMQ_VISION_CACHE_STREAMS` | `8` | Deprecated compatibility alias for `ASTRBOTEX_ZMQ_VISION_CACHE_ITEMS` |
| `ASTRBOTEX_SESSION_ID` | `astrbotex_default` | Default platform session |
| `ASTRBOTEX_UPPER_ENABLED` | `0` | Fallback for `upper_enabled` when no Host config is injected |
| `ASTRBOTEX_UPPER_CONTROL_ENABLED` | `0` | Fallback for `control_enabled` |
| `ASTRBOTEX_UPPER_CONTROL_HOST` | `127.0.0.1` | Fallback for `control_bind_host` |
| `ASTRBOTEX_UPPER_CONTROL_PORT` | `8769` | Fallback for `control_port` |
| `ASTRBOTEX_UPPER_CONTROL_PEER_ID` | empty | Fallback for the trusted `control_peer_id` |
| `ASTRBOTEX_UPPER_ROBOT_ID` | empty | Fallback for `robot_id` |
| `ASTRBOTEX_UPPER_MODEL_TIMEOUT_S` | `30` | Fallback for `model_timeout_s` |
| `ASTRBOTEX_UPPER_DATA_ROOT` | plugin data dir | Overrides the upper-control data root (tests/isolated runs only) |

Plugin configuration injected by the Host wins over these fallbacks.

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

## Upper control (B10 wiring, default OFF)

When `upper_enabled` is set, one trusted text input is classified by a private
model into `speech_only` / `action_only` / `speech_and_action`, then a private
generation model produces a public reply plus ordered execution segments. The
public reply leaves through the original platform (or, for EX voice input, the
original `interaction.reply` peer and its TTS chain). Execution segments leave
only through the `control` channel. Both model calls are non-streaming and use
the Host provider IDs below; no key is stored by this plugin.

Assembly order is fixed: the three legacy channels and their one `TaskStore` →
the upper components layered on that store → management API → optional `control`
channel → input takeover. A missing module, a rejected setting, or a failed
control bind degrades to "upper control unavailable" with a recorded reason; the
legacy channels keep running.

`upper_enabled` alone is not enough: a turn always ends at the control channel,
so without `control_enabled` the plugin records `control_disabled` and leaves
input takeover off rather than classifying an input it cannot deliver.

| Config key | Default | Meaning |
|------------|---------|---------|
| `upper_enabled` | `false` | Master switch for classification and takeover |
| `routing_provider_id` | empty | Host provider ID used for classification |
| `generation_provider_id` | empty | Host provider ID used for content generation |
| `control_enabled` | `false` | Bind the fourth channel and allow injection |
| `control_bind_host` | `127.0.0.1` | Control ROUTER bind address |
| `control_port` | `8769` | Control ROUTER port |
| `control_peer_id` | empty | Exact trusted EX DEALER identity; never inferred from a model |
| `robot_id` | empty | Robot identity for the injected batch |
| `control_route_ref` | `upper-control` | Route id EX checks against its registered set |
| `model_timeout_s` | `30` | Per-call budget for classification and generation |

**Management page.** The integrated B07/B08 page appears as "上位控制" in the
Host's native plugin pages. Its ten management APIs are available while both
feature switches are off, before robot/peer configuration, and while EX is offline.
Settings, prompts and descriptions persist locally; saving never calls a model
or sends a control task. The Host repository remains a read-only dependency.

Provider IDs, model timeout, prompt edits and `upper_enabled` apply to the next
input. Each input freezes its own model settings. Control connection changes
require reloading the plugin. Text/voice replies use the original text peer;
executable segments use the independently configured control peer.

Complete B11 verification requires a real disposable Dashboard via `--host-url`
and authenticated Playwright storage. The real mode opens the Dashboard route,
iframe and bridge. A missing or failed mandatory page stage yields `partial`
and exits non-zero. See `docs/upper-control-deploy.md` for commands and limits.

**Session binding.** EX compares the injected `ex_session` against the session it
advertises and refuses a mismatch, so `ASTRBOTEX_SESSION_ID` must be the same on
both sides. An EX voice event carries its own session; any other input uses the
plugin's configured one.

**Takeover rules.** Only already-woken text messages are considered. Registered
commands (`/ex_task`, `/ex_status`, …) keep their priority and are never
classified. Non-text input, and the original STT path, keep their existing
handling — the decision runs on the transcribed text. A session with an active
legacy control task is refused with `input_busy` instead of preempting that
lease. One Host event produces at most one classification: the default LLM run
and the legacy `manage_astrbotex_task` / `submit_astrbotex_proposal` tools are
suppressed for that input, and a failure is suppressed too rather than falling
back to a second reply.

**Where the input came from.** `TrustedUpperInput` is issued by the framework
from the Host event and the deployment config only: robot, user, session, route
and the idempotency key never come from message content, model output or a Page
body. A repeated Host event, or a new event carrying the same Host message id,
is claimed once and never re-generated.

**Delivery states.** `pending` means the segment is stored locally and not sent.
`received` means EX recorded the batch. `injected` means the batch crossed the
configured backend input boundary. `rejected` is a determinate refusal. Neither
`received` nor `injected` is execution feedback: `delivery_unknown` means the
batch may or may not have arrived, and nothing is retried automatically.

**Explicitly not implemented in this round:** execution feedback uplink,
automatic suffix sending, error replanning, parameter completion, ROS/low-level
control, physical action, task success judgement, and A/C comparison. Handshake
and delivery receipts prove delivery only.

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
