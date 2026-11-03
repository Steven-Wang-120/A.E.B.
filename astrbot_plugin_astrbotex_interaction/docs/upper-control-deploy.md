# Upper control deployment (B11)

Step-by-step deployment for the AstrBot/AEB upper-control chain. Every step is
reversible and none of them opens a public port or touches a ROS device
configuration.

Placeholders you must fill in yourself:

| Placeholder | Meaning |
|---|---|
| `<AEB_ROOT>` | A.E.B checkout that contains `astrbot_plugin_astrbotex_interaction` |
| `<EX_ROOT>` | AstrBotEX checkout |
| `<HOST_ROOT>` | AstrBot Host checkout (read-only dependency) |
| `<AEB_HOST>` / `<EX_HOST>` | Hostnames or IPs of the two machines |
| `<EX_DEALER_ID>` | The DEALER identity EX will use on the `control` channel |
| `<ROBOT_ID>` | Robot identity bound to that connection |
| `<PROVIDER_ROUTING>` / `<PROVIDER_GENERATION>` | Existing Host provider IDs |

The default layout keeps both processes on one machine with
`control_bind_host=127.0.0.1`. Cross-machine deployment is an explicit change of
that field plus a trusted network boundary you already own — this document does
not open anything to the internet.

## 0. Back up and record versions

```bash
cd <AEB_ROOT> && git rev-parse HEAD && git status --porcelain > /tmp/aeb-dirty.txt
cd <EX_ROOT>  && git rev-parse HEAD && git status --porcelain > /tmp/ex-dirty.txt
```

Copy the existing plugin data directory (`data/plugin_data/astrbot_plugin_astrbotex_interaction`)
and the EX data directory somewhere outside both checkouts. The rollback path
below depends on that copy. Record the Host version (`python -c "import astrbot;print(astrbot.__version__)"`).

Stop the Host before replacing files; do not hot-swap a running plugin directory.

## 1. Install this task's commit

```bash
cd <AEB_ROOT> && git checkout <upper-control-commit>
cd <EX_ROOT>  && git checkout <upper-control-commit>
```

Only the files listed in `handoff.json` change. `main.py` is owned by B10, the EX
connection and backend entry points by B09. Do not merge a partial set: a new
`main.py` without the B01-B09 modules degrades to "upper control unavailable",
which is safe but useless.

## 2. Prepare directories and providers

1. Confirm the Host provider IDs you intend to use exist in the Host UI. This
   plugin never stores a key.
2. Create the plugin data directory if it does not exist; the plugin resolves it
   through `StarTools.get_data_dir(...)`.
3. Do **not** pre-create the SQLite file. It is created on first use with WAL.

## 3. Start with the feature off

Leave `upper_enabled=false` and `control_enabled=false` and start the Host.
Expected observations:

- Three channels bind on 8766/8767/8768 as before; no 8769 listener.
- `plugin.upper_status` reports `enabled=false`, `reason="upper_disabled"`.
- Ordinary chat and the legacy `/ex_task` commands behave exactly as before.
- The native page can read/save settings, prompts and action descriptions before enabling.
- No robot or control peer is required to maintain those local descriptions.

If any of that fails, stop here: the regression is in the legacy path, not in the
new feature.

## 4. Check the management page and API

The integrated candidate ships B07/B08. Management repositories and all ten
API routes are initialized even when `upper_enabled=false`, `control_enabled=false`,
the robot/peer is not configured, or EX is offline. Saving metadata never starts
a model or sends a control request. The Host must expose `register_web_api` and
its native plugin-page bridge.

1. Open the plugin detail page. The native page appears as "上位控制".
2. If the Host does not expose the plugin-page API, the log records
   `host_page_api_unavailable`. That is a Host capability gap, not a plugin
   error; the Host repository must not be patched for it.
3. Call the endpoints while logged out and confirm 401/403.
4. Save a small edit, refresh, and confirm it persisted. Saving must not call a
   model or the control channel.

## 5. Configure both ends of the control channel

AEB side (`_conf_schema.json` / plugin config, or the env fallbacks):

```json
{
  "control_enabled": true,
  "control_bind_host": "127.0.0.1",
  "control_port": 8769,
  "control_peer_id": "<EX_DEALER_ID>",
  "control_route_ref": "upper-control",
  "robot_id": "<ROBOT_ID>"
}
```

`ASTRBOTEX_SESSION_ID` must match the session EX advertises on the control
connection. EX refuses a batch whose `ex_session` differs (`session_mismatch`),
and the refusal is recorded, never retried. `control_route_ref` must be one of
the route ids EX registers for that connection.

EX side: create a connection with `protocol_profile=astrbotex`, `channel=control`,
endpoint `tcp://<AEB_HOST>:8769`, and `upper_robot_id=<ROBOT_ID>`. The control DEALER identity may differ from the text DEALER identity. Public
voice replies go back to the event's text peer; executable segments use only
the configured control peer. Keep the old
`text` connection on 8766; `decision_connection_id` must still point at `text`.

Cross-machine: set `control_bind_host` to the AEB interface that faces EX. Do not
use `0.0.0.0` unless the network is already trusted and firewalled.

## 6. Local test before enabling

```bash
cd <AEB_ROOT>/astrbot_plugin_astrbotex_interaction
python -B scripts/verify_upper_control.py \
  --aeb-root <AEB_ROOT> --ex-root <EX_ROOT> --host-root <HOST_ROOT> \
  --evidence-dir ./port_artifacts/upper/B11/run-001 --scope all \
  --host-image <EXISTING_LOCAL_HOST_IMAGE> \
  --ui-mode real_host --host-url <ISOLATED_DASHBOARD_URL> \
  --baseline ./port_artifacts/upper/B11/baseline-legacy.json
```

Use a new `--evidence-dir` every time; the runner refuses to overwrite one. Exit
code 0 means every stage passed. A non-zero exit with `NOT_RUN` means a
dependency is missing — read `report.md` before enabling anything.

For real page checks, set `PLAYWRIGHT_MODULE` if Playwright is not on the module
search path. Set `UPPER_HOST_STORAGE` to a Playwright storage state obtained from
an authenticated **disposable** Dashboard. Alternatively set `UPPER_HOST_COOKIE`
to its dashboard JWT (`UPPER_HOST_COOKIE_NAME` defaults to `astrbot_dashboard_jwt`).
The test creates local descriptions and changes a timeout/prompt; never point it
at a production Dashboard. Credentials stay outside committed evidence.

`--scope all` and `--scope host` require the real Dashboard route, iframe and bridge.
`--scope page --ui-mode fake_bridge` runs browser UI tests against substitutes and
cannot satisfy real Host acceptance. Missing real Host URL, browser or credentials
must produce NOT_RUN/FAIL and a non-zero exit. Every recorded stage participates
in the verdict. The runner uses `--pull never`, read-only source mounts and
container-local temporary data for Python suites.

Stages are split per repository and per surface, so a missing piece names its
owner instead of hiding the rest:

| Stage | Runs | Missing owner reported |
|---|---|---|
| `unit_aeb` | B00-B06 module and contract tests, plus B10 composition | B00-B06 |
| `unit_ex` | EX ingress contract tests | B00, B09 |
| `loopback_aeb` | real pipeline + real ROUTER/DEALER peer, scripted models | B00-B06, B11 |
| `loopback_ex` | EX ingress with a local backend request capture | B00, B09 |
| `host` | Python composition/API tests using Host libraries; no Dashboard is started here | B07, B10 |
| `page_spec` | actual Dashboard iframe → bridge → API → repositories; mandatory | B07, B08 |
| `legacy_regression` | feature-off suites compared to `--baseline` | - |

`--baseline` pins today's green set; a suite that used to pass and no longer does
turns the stage into `FAIL` with `new_failures` naming it.

## 7. Enable upper control

After control connection changes have been applied by reloading the plugin,
set `upper_enabled=true`. The enable flag, Provider IDs, prompts and model timeout
take effect on the next input. Each input freezes its Provider IDs and timeout,
so changing settings cannot redirect an in-flight generation. Disabling immediately
suppresses late output; the next input refreshes admission. Changing control
host/port/peer/robot or `control_enabled` still requires a plugin reload. Expected log lines:

- `AstrBotEX upper control active: control=on api=on` (when native API registration is available)
- `AstrBotEX ZeroMQ channels started: text=..., audio=..., vision=...`

Then verify:

1. One ordinary text message produces exactly one classification and one public
   reply. It must not also produce a legacy task.
2. A `/ex_task ...` command is still handled by the legacy path.
3. `runtime.status` shows a `control` channel entry with the configured port.
4. With EX offline, a message must fail closed (no duplicate reply) and record a
   diagnostic; it must not fall back to the default LLM answer.

## 8. Observe delivery state

Read the plan table for `pending` rows. `received` and `injected` are delivery
evidence only. `delivery_unknown` means the batch may or may not have arrived —
never resend it manually; the framework will not create a second batch id.

## 9. Rollback

1. Set `upper_enabled=false`, then `control_enabled=false`, and reload.
2. Confirm 8769 is no longer listening and that three legacy channels still work.
3. Restore the data directory copy only if you also intend to discard new plans.
   Keeping the new tables and `unknown` rows is the safe default: they are
   evidence, not garbage.
4. `git checkout <previous-commit>` in both checkouts if you need the code gone.

Rollback never requires deleting the database, resetting batch ids or replaying a
delivery.

## What this deployment does not claim

- No execution feedback uplink, no automatic suffix sending, no replanning.
- No ROS or low-level motion is exercised; nothing here validates a physical task.
- `injected` is not success. Task success needs the external acceptance path.

## 2026-11-03 repair verification

The candidate additionally covers feature-off management, real SettingsSnapshot
reads, real ActionCatalog create/read/edit/delete, frozen Provider/timeout values,
distinct voice text/control peers, shared EX ingress parsing and truthful page gating.
The real browser spec tests offline catalog persistence, settings/prompt saves and
refreshes, anonymous read/write refusal, revision conflicts, and draft retention
under injected HTTP/network failures. Scripted models and loopback receipts do
not prove a live model, execution success, or the complete crash/restart matrix.

## Offline system acceptance and forced process recovery

The mandatory Host stage also runs `test_upper_acceptance.py` and
`test_upper_crash_recovery.py`. In image mode the runner mounts both candidate
repositories read-only and writes captures/projections under `system/` in the
new evidence directory. Native Host mode uses the explicit AEB/EX/Host roots.
No package or image is downloaded.

The system matrix uses real `Context.llm_generate`, upper model result objects,
the OneBot event serializer/send adapter, EX text ConnectionManager and
InteractionCore, the control ROUTER/DEALER and real JevBackend HTTP transport.
Only provider responses, the OneBot API peer, and audio providers/devices are
local substitutes. R1-R5 run for both platform and voice, with replay and
sentinel/counter assertions. S1-S4 capture the local HTTP model request and
database snapshots; S4 retains its third segment across two timeout windows
and restart. A wrong-session batch is refused without another model request.

Crash tests start independent children, wait for durable gates, force kill
them, then recover the same unchanged database/log in fresh interpreters.
Gates cover reserve-before-send, send-before-receipt, EX consumption started,
and public-send-before-confirmation. Recovery retains unknown states and does
not resend/reconsume/re-speak; this is conservative delivery, not exactly-once.

Real Dashboard tests additionally cover primitive/skill CRUD, parameters,
both prompt overrides and builtin restoration, real plugin-manager reload,
and light/dark 1280/390px layouts. Use only a disposable Host. Temporary
dashboard credentials and Host databases must stay outside committed evidence.

2026-11-03 acceptance also fixed the B03/B05 result boundary: classification
probabilities and generated content are read from their production typed
wrappers before validation and persistence. Model evidence remains private.
The complete matrix is a wiring check, not a paid-model or hardware task test.

Snapshot conversion is confined to `upper_store.py`; legacy `task_models.py`
retains its source-baseline behavior. An explicit public-sender `False` records
`delivery_unknown` and never authorizes an automatic retry.

Records translate the actual UpperStore batch phase, selected prefix and
snapshot hashes to the management API. Before reservation the record is
`unsent` with zero sent segments; database failures remain dependency errors.
