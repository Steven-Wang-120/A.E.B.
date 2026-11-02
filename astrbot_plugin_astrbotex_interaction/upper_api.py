"""Dashboard management API for the AEB upper-control feature (task B07).

This module is deliberately a *wiring* layer.  It validates HTTP input, calls
the injected repositories and shapes the response body.  It never keeps a
second copy of the settings values or of the action catalog, and it never
calls the upper pipeline, a model provider or the ZeroMQ control channel.

Ownership (C00): this file is created by B07.  ``main.py`` is owned by B10 and
is not touched here; B10 calls :func:`register_upper_api` from the plugin
lifecycle.  The Host Vue/backend code is a read-only dependency.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

logger = logging.getLogger(__name__)

try:  # Host public helpers; unavailable only in Host-less unit-test processes.
    from astrbot.api.web import error_response as _host_error_response
    from astrbot.api.web import json_response as _host_json_response
    from astrbot.api.web import request as _host_request
except Exception:  # pragma: no cover - exercised by the Host-less test process
    _host_error_response = None
    _host_json_response = None
    _host_request = None


PLUGIN_NAME = "astrbot_plugin_astrbotex_interaction"

#: Stable route table.  The order and the method tuples are part of the
#: contract: ``Context.register_web_api`` replaces an existing registration only
#: when both the route *and* the methods list match exactly, so re-registering
#: after a plugin reload must produce byte-identical values or handlers stack up.
ROUTE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("upper/settings", ("GET",)),
    ("upper/settings/save", ("POST",)),
    ("upper/providers", ("GET",)),
    ("upper/actions", ("GET",)),
    ("upper/actions/save", ("POST",)),
    ("upper/actions/delete", ("POST",)),
    ("upper/prompts", ("GET",)),
    ("upper/prompts/save", ("POST",)),
    ("upper/plans", ("GET",)),
    ("upper/plans/get", ("GET",)),
)

_HANDLER_NAMES = {
    "upper/settings": "get_settings",
    "upper/settings/save": "save_settings",
    "upper/providers": "get_providers",
    "upper/actions": "get_actions",
    "upper/actions/save": "save_action",
    "upper/actions/delete": "delete_action",
    "upper/prompts": "get_prompts",
    "upper/prompts/save": "save_prompt",
    "upper/plans": "get_plans",
    "upper/plans/get": "get_plan",
}

#: error_code -> HTTP status.  C01 fixes the transport codes; the Dashboard
#: codes below are the ones C03 §3 lists for this API.
ERROR_STATUS = {
    "revision_conflict": 409,
    "invalid_request": 400,
    "field_not_allowed": 400,
    "path_not_allowed": 400,
    "cursor_invalid": 400,
    "not_found": 404,
    "dependency_unavailable": 503,
    "internal_error": 500,
}

#: Where a saved value starts to matter.  ``reload_required`` mirrors B07 §2:
#: anything that changes a listening socket or the trusted identity needs a
#: plugin reload before it is true.
APPLY_SCOPE: dict[str, str] = {
    "upper_enabled": "next_input",
    "routing_provider_id": "next_input",
    "generation_provider_id": "next_input",
    "routing_prompt": "next_input",
    "planning_prompt": "next_input",
    "model_timeout_s": "next_input",
    "control_enabled": "reload_required",
    "control_bind_host": "reload_required",
    "control_port": "reload_required",
    "control_peer_id": "reload_required",
    "robot_id": "reload_required",
}

SETTING_FIELDS: tuple[str, ...] = tuple(APPLY_SCOPE)
PROMPT_KINDS: tuple[str, ...] = ("routing", "planning")
_PROMPT_SETTING = {"routing": "routing_prompt", "planning": "planning_prompt"}

ACTION_LEVELS = ("primitive", "basic", "skill")

#: Record keys a Page is never allowed to send.  Machine contracts come from
#: EX and are read-only; execution rights are never granted by a form.
FORBIDDEN_RECORD_KEYS = frozenset(
    {
        "machine_schema",
        "machine",
        "exec_permission",
        "exec_permissions",
        "executable",
        "permissions",
        "permission",
        "params",
        "command",
        "commands",
        "shell",
        "script",
        "ros_action",
        "ros_action_type",
        "driver_settings",
    }
)

_PATH_KEYS = frozenset(
    {"path", "file", "file_path", "skill_path", "prompt_path", "source_path", "content_path"}
)

MAX_PROMPT_BYTES = 64 * 1024
MAX_SKILL_BYTES = 64 * 1024
MAX_DESCRIPTION_CHARS = 8192
MAX_NAME_CHARS = 128
MAX_RECORD_BYTES = 256 * 1024
MAX_ID_CHARS = 128
MAX_PARAMETERS = 32
MAX_USAGE_STAGES = 16
PLAN_LIMIT_DEFAULT = 20
PLAN_LIMIT_MAX = 100

#: Delivery phases the Dashboard may render.  ``received`` and ``injected``
#: only prove that the text reached EX; none of them means "executed".
DELIVERY_PHASES = ("unsent", "sending", "received", "injected", "rejected", "unknown")
_PHASE_ALIASES = {
    None: "unsent",
    "": "unsent",
    "pending": "unsent",
    "queued": "unsent",
    "unsent": "unsent",
    "reserved": "unsent",
    "sending": "sending",
    "sending.": "sending",
    "in_flight": "sending",
    "delivering": "sending",
    "received": "received",
    "recorded": "received",
    "input_received": "received",
    "injected": "injected",
    "accepted": "injected",
    "rejected": "rejected",
    "refused": "rejected",
    "error": "rejected",
    "failed": "rejected",
    "unknown": "unknown",
    "delivery_unknown": "unknown",
}

#: Provider rows come from the Host; only these fields are ever returned.
_PROVIDER_PUBLIC_FIELDS = ("id", "name", "model", "type")

_SCHEMA_DEFAULTS: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# Errors and results
# --------------------------------------------------------------------------


class UpperApiError(Exception):
    """A rejected request carrying a stable machine code."""

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_STATUS:
            raise ValueError(f"unknown upper api error code: {code}")
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")

    @property
    def status_code(self) -> int:
        return ERROR_STATUS[self.code]


@dataclass(frozen=True)
class ApiResult:
    """Transport-neutral handler result, used by tests and by the Host adapter."""

    status_code: int
    body: Any


class HostResponder:
    """Adapter over ``astrbot.api.web`` helpers."""

    @staticmethod
    def available() -> bool:
        return _host_json_response is not None and _host_error_response is not None

    def ok(self, payload: Any, status_code: int = 200) -> Any:
        if _host_json_response is None:  # pragma: no cover - Host always present in production
            raise RuntimeError("AstrBot web helpers are unavailable")
        return _host_json_response(payload, status_code=status_code)

    def error(self, message: str, status_code: int, data: Any = None) -> Any:
        if _host_error_response is None:  # pragma: no cover
            raise RuntimeError("AstrBot web helpers are unavailable")
        return _host_error_response(message, status_code=status_code, data=data)


# --------------------------------------------------------------------------
# Repository contracts (dependency requests towards B01 / B02 / B04 / B06+B10)
# --------------------------------------------------------------------------


@runtime_checkable
class SettingsRepository(Protocol):
    """B01 ``SettingsRepository``.

    ``load()`` returns either ``{"values": {...}, "revision": int, "sources": {...}}``
    or a plain mapping of setting values (then ``revision()`` is consulted).
    """

    def load(self) -> Any: ...

    def save(self, data: Mapping[str, Any], expected_revision: int) -> Any: ...


@runtime_checkable
class ActionCatalogRepository(Protocol):
    """B01 ``ActionCatalog``."""

    def snapshot(self) -> Any: ...

    def save(self, record: Mapping[str, Any], expected_revision: int) -> Any: ...

    def delete(self, action_ref: str, expected_revision: int) -> Any: ...


@runtime_checkable
class PlanRepository(Protocol):
    """B04 ``UpperStore`` read projection.

    ``list_plans(limit, offset)`` returns ``(items, has_more)`` or a mapping with
    ``plans`` / ``has_more``.  ``get_plan(plan_id)`` returns ``None`` when the
    plan is unknown to this installation.
    """

    def list_plans(self, limit: int, offset: int) -> Any: ...

    def get_plan(self, plan_id: str) -> Any: ...


@dataclass
class UpperApiDependencies:
    """Everything the API may call.  Missing members disable their endpoints.

    B10 builds this once during plugin start-up and hands it to
    :func:`register_upper_api`; nothing here is constructed from a Page request.
    """

    settings: SettingsRepository | None = None
    actions: ActionCatalogRepository | None = None
    plans: PlanRepository | None = None
    #: ``() -> {"state": "online"|"offline"|"unknown", "catalog_version": str|None,
    #:          "action_ids": [str], "checked_at": str|None}``
    ex_availability: Any = None
    #: ``() -> {"control_state": str, "control_enabled": bool, "control_peer": str|None,
    #:          "ex_session": str|None, "input_ready": bool, "last_delivery": dict|None}``
    runtime_status: Any = None
    #: ``() -> {"routing": str, "planning": str}`` built-in prompt text from B02.
    prompt_defaults: Any = None
    #: ``() -> [{"id": str, "name": str|None, "model": str|None, "type": str|None}]``
    providers: Any = None
    #: Extra read-only attributes exposed to the Page without further parsing.
    notes: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Small coercion helpers
# --------------------------------------------------------------------------


def _get(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _as_int(value: Any) -> int | None:
    """Reject bool even though ``isinstance(True, int)`` is True."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _require_int(value: Any, field_name: str) -> int:
    number = _as_int(value)
    if number is None:
        raise UpperApiError("invalid_request", f"{field_name} 必须是整数")
    return number


def _require_revision(value: Any) -> int:
    number = _require_int(value, "expected_revision")
    if number < 0:
        raise UpperApiError("invalid_request", "expected_revision 不能为负数")
    return number


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UpperApiError("invalid_request", f"{field_name} 必须是对象")
    return value


def _require_str(
    value: Any,
    field_name: str,
    *,
    max_chars: int,
    allow_empty: bool = True,
) -> str:
    if not isinstance(value, str):
        raise UpperApiError("invalid_request", f"{field_name} 必须是字符串")
    if not allow_empty and not value.strip():
        raise UpperApiError("invalid_request", f"{field_name} 不能为空")
    if len(value) > max_chars:
        raise UpperApiError("invalid_request", f"{field_name} 超过 {max_chars} 字符上限")
    return value


def _utf8_len(value: str) -> int:
    return len(value.encode("utf-8"))


def _looks_like_path(value: str) -> bool:
    """True for drive-qualified, UNC or rooted paths and parent escapes."""

    text = value.strip()
    if not text:
        return False
    if text.startswith(("/", "\\", "~")):
        return True
    if ".." in text.replace("\\", "/").split("/"):
        return True
    if len(text) >= 2 and text[1] == ":" and text[0].isalpha():
        return True
    return False


def _restricted_id(value: Any, field_name: str, *, allow_empty: bool = False) -> str:
    text = _require_str(value, field_name, max_chars=MAX_ID_CHARS, allow_empty=allow_empty)
    if not text:
        return ""
    if _looks_like_path(text):
        raise UpperApiError("path_not_allowed", f"{field_name} 不接受文件路径")
    if any(ch.isspace() for ch in text):
        raise UpperApiError("invalid_request", f"{field_name} 不能包含空白字符")
    return text


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _schema() -> dict[str, Any]:
    global _SCHEMA_DEFAULTS
    if _SCHEMA_DEFAULTS is None:
        path = Path(__file__).with_name("_conf_schema.json")
        try:
            _SCHEMA_DEFAULTS = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("upper api: _conf_schema.json unreadable; defaults unavailable")
            _SCHEMA_DEFAULTS = {}
    return _SCHEMA_DEFAULTS


def schema_defaults() -> dict[str, Any]:
    """Declared defaults, straight from ``_conf_schema.json`` (single source)."""

    defaults: dict[str, Any] = {}
    for name in SETTING_FIELDS:
        entry = _schema().get(name) or {}
        if "default" in entry:
            defaults[name] = entry["default"]
    return defaults


def _schema_type(name: str) -> str:
    entry = _schema().get(name) or {}
    return str(entry.get("type") or "string")


# --------------------------------------------------------------------------
# Settings snapshot normalisation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _SettingsView:
    values: dict[str, Any]
    revision: int
    sources: dict[str, str]
    raw: Any


def _settings_view(repo: SettingsRepository) -> _SettingsView:
    try:
        raw = repo.load()
    except UpperApiError:
        raise
    except Exception as exc:  # repository failure must not leak a traceback
        logger.warning("upper api: settings load failed: %s", exc)
        raise UpperApiError("dependency_unavailable", "设置仓储暂时不可用") from None

    values: dict[str, Any] = {}
    revision: Any = None
    sources: dict[str, str] = {}

    if isinstance(raw, Mapping) and "values" in raw:
        inner = raw.get("values")
        values = dict(inner) if isinstance(inner, Mapping) else {}
        revision = raw.get("revision")
        maybe_sources = raw.get("sources")
        if isinstance(maybe_sources, Mapping):
            sources = {str(k): str(v) for k, v in maybe_sources.items()}
    elif isinstance(raw, Mapping):
        values = dict(raw)
    else:
        inner = getattr(raw, "values", None)
        if isinstance(inner, Mapping):
            values = dict(inner)
        revision = getattr(raw, "revision", None)
        maybe_sources = getattr(raw, "sources", None)
        if isinstance(maybe_sources, Mapping):
            sources = {str(k): str(v) for k, v in maybe_sources.items()}

    if revision is None:
        revision_fn = getattr(repo, "revision", None)
        revision = revision_fn() if callable(revision_fn) else 0

    number = _as_int(revision)
    if number is None or number < 0:
        raise UpperApiError("dependency_unavailable", "设置仓储未提供可用的 revision")

    merged = schema_defaults()
    merged.update({k: v for k, v in values.items() if k in APPLY_SCOPE})
    return _SettingsView(values=merged, revision=number, sources=sources, raw=raw)


def _validate_setting_value(name: str, value: Any) -> Any:
    kind = _schema_type(name)
    if kind == "bool":
        if not isinstance(value, bool):
            raise UpperApiError("invalid_request", f"{name} 必须是布尔值")
        return value
    if kind == "int":
        number = _as_int(value)
        if number is None:
            raise UpperApiError("invalid_request", f"{name} 必须是整数")
        if name == "control_port" and not (1 <= number <= 65535):
            raise UpperApiError("invalid_request", "control_port 必须在 1..65535 之间")
        return number
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UpperApiError("invalid_request", f"{name} 必须是数值")
        number = float(value)
        if name == "model_timeout_s" and not (1.0 <= number <= 600.0):
            raise UpperApiError("invalid_request", "model_timeout_s 必须在 1..600 秒之间")
        return number
    # string / text
    text = _require_str(value, name, max_chars=MAX_PROMPT_BYTES)
    if kind == "text":
        if _utf8_len(text) > MAX_PROMPT_BYTES:
            raise UpperApiError(
                "invalid_request", f"{name} 超过 {MAX_PROMPT_BYTES} 字节上限"
            )
        return text
    if name in ("routing_provider_id", "generation_provider_id"):
        if not text:
            return ""
        if any(ch.isspace() for ch in text):
            raise UpperApiError("invalid_request", f"{name} 不能包含空白字符")
        return text
    if name == "control_bind_host":
        if not text:
            raise UpperApiError("invalid_request", "control_bind_host 不能为空")
        if "://" in text or "/" in text or "\\" in text:
            raise UpperApiError("invalid_request", "control_bind_host 只接受主机名或地址")
        return text
    if name == "control_peer_id":
        return _restricted_id(text, name, allow_empty=True)
    if name == "robot_id":
        return _restricted_id(text, name, allow_empty=True)
    return text


# --------------------------------------------------------------------------
# Action record validation
# --------------------------------------------------------------------------


def _validate_parameters(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise UpperApiError("invalid_request", "parameters 必须是列表")
    if len(value) > MAX_PARAMETERS:
        raise UpperApiError("invalid_request", f"parameters 最多 {MAX_PARAMETERS} 项")
    cleaned: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        entry = _require_mapping(item, f"parameters[{index}]")
        lowered = {str(key).lower() for key in entry}
        if lowered & FORBIDDEN_RECORD_KEYS:
            raise UpperApiError(
                "field_not_allowed", f"parameters[{index}] 不接受执行权限或机器契约字段"
            )
        name = _require_str(
            entry.get("name"), f"parameters[{index}].name", max_chars=64, allow_empty=False
        )
        description = _require_str(
            entry.get("description", ""), f"parameters[{index}].description", max_chars=512
        )
        unit = _require_str(entry.get("unit", ""), f"parameters[{index}].unit", max_chars=32)
        required = entry.get("required", False)
        if not isinstance(required, bool):
            raise UpperApiError(
                "invalid_request", f"parameters[{index}].required 必须是布尔值"
            )
        cleaned.append(
            {
                "name": name,
                "description": description,
                "unit": unit,
                "required": required,
            }
        )
    return cleaned


def _validate_action_record(raw: Any) -> dict[str, Any]:
    record = _require_mapping(raw, "record")
    for key in record:
        lowered = str(key).lower()
        if lowered in FORBIDDEN_RECORD_KEYS:
            raise UpperApiError(
                "field_not_allowed", f"动作记录不接受字段 {key}（机器契约与执行权限来自 EX）"
            )
        if lowered in _PATH_KEYS:
            raise UpperApiError(
                "path_not_allowed", f"动作记录不接受文件路径字段 {key}"
            )

    level = record.get("level")
    if level not in ACTION_LEVELS:
        raise UpperApiError(
            "invalid_request", "level 必须是 primitive、basic 或 skill"
        )

    name = _require_str(
        record.get("name"), "name", max_chars=MAX_NAME_CHARS, allow_empty=False
    )
    description = _require_str(
        record.get("description", ""), "description", max_chars=MAX_DESCRIPTION_CHARS
    )
    skill_markdown = _require_str(
        record.get("skill_markdown", ""), "skill_markdown", max_chars=MAX_SKILL_BYTES
    )
    if _utf8_len(skill_markdown) > MAX_SKILL_BYTES:
        raise UpperApiError("invalid_request", f"skill_markdown 超过 {MAX_SKILL_BYTES} 字节上限")

    action_id: str | None
    raw_action_id = record.get("action_id")
    if raw_action_id in (None, ""):
        action_id = None
    else:
        action_id = _restricted_id(raw_action_id, "action_id")

    if level in ("primitive", "basic") and not action_id:
        raise UpperApiError("invalid_request", "primitive/basic 动作必须关联 EX action_id")
    if level == "skill" and action_id:
        raise UpperApiError("invalid_request", "skill 层不关联可执行 action_id")

    enabled = record.get("enabled", True)
    if not isinstance(enabled, bool):
        raise UpperApiError("invalid_request", "enabled 必须是布尔值")

    usage_stages = record.get("usage_stages", [])
    if usage_stages is None:
        usage_stages = []
    if not isinstance(usage_stages, list):
        raise UpperApiError("invalid_request", "usage_stages 必须是列表")
    if len(usage_stages) > MAX_USAGE_STAGES:
        raise UpperApiError("invalid_request", f"usage_stages 最多 {MAX_USAGE_STAGES} 项")
    stages = [
        _require_str(item, "usage_stages[]", max_chars=64, allow_empty=False)
        for item in usage_stages
    ]

    action_ref = None
    raw_action_ref = record.get("action_ref")
    if raw_action_ref not in (None, ""):
        action_ref = _restricted_id(raw_action_ref, "action_ref")

    cleaned = {
        "action_ref": action_ref,
        "action_id": action_id,
        "level": level,
        "name": name,
        "description": description,
        "parameters": _validate_parameters(record.get("parameters")),
        "usage_stages": stages,
        "skill_markdown": skill_markdown,
        "enabled": enabled,
    }
    if len(json.dumps(cleaned, ensure_ascii=False).encode("utf-8")) > MAX_RECORD_BYTES:
        raise UpperApiError("invalid_request", "动作记录超过 256 KiB 上限")
    return cleaned


# --------------------------------------------------------------------------
# Plan projection
# --------------------------------------------------------------------------


def _normalise_phase(value: Any) -> str:
    if isinstance(value, str):
        return _PHASE_ALIASES.get(value.strip().lower(), "unknown")
    return _PHASE_ALIASES.get(value, "unknown")


def _normalise_route(value: Any) -> str | None:
    if value in ("speech_only", "action_only", "speech_and_action"):
        return str(value)
    return None


def _segments_projection(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    segments: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if isinstance(item, Mapping):
            segments.append(
                {
                    "segment_id": item.get("segment_id"),
                    "ordinal": _as_int(item.get("ordinal")) if _as_int(item.get("ordinal")) is not None else index,
                    "text": item.get("text") if isinstance(item.get("text"), str) else "",
                }
            )
        elif isinstance(item, str):
            segments.append({"segment_id": None, "ordinal": index, "text": item})
    return segments


def _deliveries_projection(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    deliveries: list[dict[str, Any]] = []
    for item in raw:
        entry = _require_mapping(item, "delivery") if isinstance(item, Mapping) else {}
        deliveries.append(
            {
                "batch_id": entry.get("batch_id"),
                "phase": _normalise_phase(entry.get("phase")),
                "payload_hash": entry.get("payload_hash"),
                "segment_ids": list(entry.get("segment_ids") or []),
                "attempted_at": entry.get("attempted_at") or entry.get("created_at"),
            }
        )
    return deliveries


def _plan_summary(raw: Any) -> dict[str, Any]:
    segments = _segments_projection(_get(raw, "segments"))
    sent_prefix = _as_int(_get(raw, "sent_prefix"))
    if sent_prefix is None:
        sent_prefix = _as_int(_get(raw, "send_count")) or 0
    sent_prefix = max(0, min(sent_prefix, len(segments)))
    pending = segments[sent_prefix:]
    return {
        "plan_id": _get(raw, "plan_id"),
        "route": _normalise_route(_get(raw, "route")),
        "input_key": _get(raw, "input_key"),
        "plan_revision": _as_int(_get(raw, "plan_revision")),
        "segment_total": len(segments),
        "sent_prefix": sent_prefix,
        "pending_suffix": max(len(segments) - sent_prefix, 0),
        "delivery_phase": _normalise_phase(_get(raw, "delivery_phase")),
        "prompt_version": _get(raw, "prompt_version"),
        "catalog_version": _get(raw, "catalog_version"),
        "created_at": _get(raw, "created_at"),
    }, segments, pending


# --------------------------------------------------------------------------
# Query helpers
# --------------------------------------------------------------------------


def _query_int(source: Any, key: str, default: int) -> int:
    """Read an int query parameter, tolerating the Host ``MultiDict`` API."""

    raw: Any
    try:
        raw = source.get(key, default, type=int)  # type: ignore[call-arg]
    except TypeError:
        raw = source.get(key, default)
    if raw is None or raw == "":
        return default
    number = _as_int(raw)
    if number is None:
        try:
            number = int(str(raw))
        except (TypeError, ValueError):
            raise UpperApiError("invalid_request", f"{key} 必须是整数") from None
    return number


def _query_str(source: Any, key: str, default: str | None = None) -> str | None:
    try:
        raw = source.get(key, default)
    except Exception:  # pragma: no cover - defensive
        return default
    if raw is None:
        return default
    return raw if isinstance(raw, str) else str(raw)


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(str(offset).encode("ascii")).decode("ascii").rstrip("=")


def _decode_cursor(raw: str) -> int:
    text = raw.strip()
    if not text or len(text) > 64:
        raise UpperApiError("cursor_invalid", "游标格式不正确")
    padded = text + "=" * (-len(text) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("ascii")
        offset = int(decoded)
    except Exception:
        raise UpperApiError("cursor_invalid", "游标格式不正确") from None
    if offset < 0:
        raise UpperApiError("cursor_invalid", "游标格式不正确")
    return offset


# --------------------------------------------------------------------------
# The API
# --------------------------------------------------------------------------


class UpperApi:
    """Ten handlers over injected repositories.  No model, no control channel."""

    def __init__(
        self,
        dependencies: UpperApiDependencies | None = None,
        *,
        host: Any = None,
        responder: Any = None,
        request_proxy: Any = None,
    ) -> None:
        self.dependencies = dependencies or UpperApiDependencies()
        self.host = host
        self.responder = responder or HostResponder()
        self._request_proxy = request_proxy

    # -- plumbing ---------------------------------------------------------

    @property
    def request(self) -> Any:
        if self._request_proxy is not None:
            return self._request_proxy
        if _host_request is None:  # pragma: no cover
            raise UpperApiError("dependency_unavailable", "Host 请求上下文不可用")
        return _host_request

    def _ok(self, payload: dict[str, Any], status_code: int = 200) -> Any:
        # The Page bridge unwraps ``response.data.data`` and treats a top-level
        # ``status: "error"`` as failure, so business payloads never use either key.
        if "data" in payload or payload.get("status") == "error":
            raise UpperApiError("internal_error", "响应体包含保留字段")
        return self.responder.ok(payload, status_code=status_code)

    def _fail(self, error: UpperApiError) -> Any:
        message = f"[UPPER:{error.code}] {error.message}"
        return self.responder.error(
            message, status_code=error.status_code, data={"error_code": error.code}
        )

    async def _dispatch(self, handler: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await handler(*args, **kwargs)
        except UpperApiError as error:
            return self._fail(error)
        except Exception as exc:  # never leak a stack trace or a private payload
            logger.exception("upper api: unhandled failure in %s", getattr(handler, "__name__", "handler"))
            return self._fail(UpperApiError("internal_error", "请求处理失败"))

    def _require(self, name: str) -> Any:
        repository = getattr(self.dependencies, name, None)
        if repository is None:
            raise UpperApiError("dependency_unavailable", f"{name} 尚未装配")
        return repository

    def _repository_call(self, label: str, call: Any) -> Any:
        try:
            return call()
        except UpperApiError:
            raise
        except Exception as exc:
            text = str(exc)
            logger.warning("upper api: %s failed: %s", label, text)
            code = getattr(exc, "code", None)
            if code == "revision_conflict":
                raise UpperApiError("revision_conflict", "目录或设置已被其他页面更新") from None
            if code in ("not_found", "unknown_action"):
                raise UpperApiError("not_found", "未找到对应条目") from None
            if code in ("invalid_record", "invalid_settings", "invalid_field"):
                raise UpperApiError("invalid_request", "提交内容不符合目录规则") from None
            raise UpperApiError("dependency_unavailable", f"{label} 暂时不可用") from None

    # -- settings ---------------------------------------------------------

    async def get_settings(self) -> Any:
        return await self._dispatch(self._get_settings)

    async def _get_settings(self) -> Any:
        repository = self._require("settings")
        view = _settings_view(repository)
        return self._ok(
            {
                "settings": {name: view.values.get(name) for name in SETTING_FIELDS},
                "revision": view.revision,
                "apply_scope": dict(APPLY_SCOPE),
                "sources": {name: view.sources.get(name, "default") for name in SETTING_FIELDS},
                "runtime": self._runtime_status(),
                "fields": [
                    {
                        "name": name,
                        "type": _schema_type(name),
                        "apply_scope": APPLY_SCOPE[name],
                        "default": schema_defaults().get(name),
                    }
                    for name in SETTING_FIELDS
                ],
            }
        )

    def _runtime_status(self) -> dict[str, Any]:
        default = {
            "control_state": "unavailable",
            "control_enabled": False,
            "control_peer": None,
            "ex_session": None,
            "input_ready": False,
            "last_delivery": None,
        }
        provider = self.dependencies.runtime_status
        if not callable(provider):
            return default
        try:
            raw = provider()
        except Exception as exc:
            logger.warning("upper api: runtime status unavailable: %s", exc)
            return default
        if not isinstance(raw, Mapping):
            return default
        merged = dict(default)
        merged.update({key: raw.get(key, merged[key]) for key in default})
        state = merged.get("control_state")
        if state not in ("running", "stopped", "unavailable", "error"):
            merged["control_state"] = "unavailable"
        merged["control_enabled"] = bool(merged.get("control_enabled"))
        merged["input_ready"] = bool(merged.get("input_ready"))
        return merged

    async def save_settings(self) -> Any:
        return await self._dispatch(self._save_settings)

    async def _save_settings(self) -> Any:
        repository = self._require("settings")
        payload = _require_mapping(await self.request.json(default={}), "请求体")
        expected_revision = _require_revision(payload.get("expected_revision"))

        raw_settings = payload.get("settings")
        if raw_settings is None:
            # Tolerate a flat body so the Page may post either shape.
            raw_settings = {
                key: value for key, value in payload.items() if key not in ("expected_revision",)
            }
        settings = _require_mapping(raw_settings, "settings")
        if not settings:
            raise UpperApiError("invalid_request", "settings 不能为空")

        unknown = [key for key in settings if key not in APPLY_SCOPE]
        if unknown:
            raise UpperApiError(
                "field_not_allowed", "不接受的设置字段: " + ", ".join(sorted(unknown))
            )

        cleaned = {key: _validate_setting_value(key, value) for key, value in settings.items()}

        current = _settings_view(repository)
        if expected_revision != current.revision:
            raise UpperApiError("revision_conflict", "设置已被其他页面更新，请重新载入")

        result = self._repository_call(
            "settings save", lambda: repository.save(cleaned, expected_revision)
        )
        new_revision = _as_int(_get(result, "revision"))
        if new_revision is None:
            new_revision = expected_revision + 1

        changed = [key for key, value in cleaned.items() if current.values.get(key) != value]
        reload_required = [key for key in changed if APPLY_SCOPE[key] == "reload_required"]
        return self._ok(
            {
                "revision": new_revision,
                "changed": changed,
                "reload_required": reload_required,
                "next_input": [key for key in changed if APPLY_SCOPE[key] == "next_input"],
            }
        )

    # -- providers --------------------------------------------------------

    async def get_providers(self) -> Any:
        return await self._dispatch(self._get_providers)

    def _provider_rows(self) -> tuple[list[dict[str, Any]], str]:
        """Return public provider rows plus a reason string when unavailable."""

        rows: list[dict[str, Any]] = []
        reason = ""
        provider_fn = self.dependencies.providers
        if callable(provider_fn):
            try:
                raw_rows = provider_fn() or []
            except Exception as exc:
                logger.warning("upper api: provider listing failed: %s", exc)
                return [], "provider_listing_failed"
            for item in raw_rows:
                if not isinstance(item, Mapping):
                    continue
                provider_id = item.get("id")
                if not isinstance(provider_id, str) or not provider_id:
                    continue
                rows.append(
                    {
                        "id": provider_id,
                        "name": item.get("name") or provider_id,
                        "model": item.get("model"),
                        "type": item.get("type"),
                        "available": True,
                        "reason": None,
                    }
                )
            return rows, reason

        getter = getattr(self.host, "get_all_providers", None)
        if not callable(getter):
            return [], "host_provider_api_unavailable"
        try:
            instances = getter() or []
        except Exception as exc:
            logger.warning("upper api: Host provider listing failed: %s", exc)
            return [], "provider_listing_failed"
        for instance in instances:
            try:
                meta = instance.meta()
            except Exception:
                continue
            provider_id = getattr(meta, "id", None)
            if not isinstance(provider_id, str) or not provider_id:
                continue
            config = getattr(instance, "provider_config", {}) or {}
            display = config.get("name") if isinstance(config, Mapping) else None
            rows.append(
                {
                    "id": provider_id,
                    "name": display or provider_id,
                    "model": getattr(meta, "model", None),
                    "type": getattr(meta, "type", None),
                    "available": True,
                    "reason": None,
                }
            )
        return rows, reason

    async def _get_providers(self) -> Any:
        rows, reason = self._provider_rows()
        known = {row["id"] for row in rows}

        selected: dict[str, str] = {"routing_provider_id": "", "generation_provider_id": ""}
        stale: list[str] = []
        settings_error = None
        if self.dependencies.settings is not None:
            try:
                view = _settings_view(self.dependencies.settings)
                for key in selected:
                    value = view.values.get(key)
                    selected[key] = value if isinstance(value, str) else ""
                    if selected[key] and selected[key] not in known:
                        stale.append(selected[key])
            except UpperApiError as error:
                settings_error = error.code

        payload: dict[str, Any] = {
            "providers": rows,
            "selected": selected,
            "stale": sorted(set(stale)),
            "installed": bool(rows),
        }
        if reason:
            payload["unavailable_reason"] = reason
        if settings_error:
            payload["settings_error"] = settings_error
        # Selecting a provider here never installs or switches a model: the Page
        # only ever sends an existing Host provider id back to settings/save.
        return self._ok(payload)

    # -- action catalog ---------------------------------------------------

    def _catalog_view(self) -> tuple[list[dict[str, Any]], int, str]:
        catalog = self._require("actions")
        snapshot = self._repository_call("action catalog", catalog.snapshot)
        records = _get(snapshot, "records")
        if records is None:
            records = _get(snapshot, "actions")
        if records is None and isinstance(snapshot, (list, tuple)):
            records = snapshot
        revision = _as_int(_get(snapshot, "revision"))
        if revision is None:
            revision = 0
        version = _get(snapshot, "catalog_version") or _get(snapshot, "version") or ""
        clean: list[dict[str, Any]] = []
        for item in records or []:
            if not isinstance(item, Mapping):
                continue
            clean.append(
                {
                    "action_ref": item.get("action_ref"),
                    "action_id": item.get("action_id"),
                    "level": item.get("level"),
                    "name": item.get("name"),
                    "description": item.get("description", ""),
                    "parameters": list(item.get("parameters") or []),
                    "usage_stages": list(item.get("usage_stages") or []),
                    "skill_markdown": item.get("skill_markdown", ""),
                    "enabled": bool(item.get("enabled", True)),
                    "revision": _as_int(item.get("revision")),
                }
            )
        return clean, revision, str(version)

    def _ex_availability(self) -> dict[str, Any]:
        default = {
            "state": "unknown",
            "catalog_version": None,
            "action_ids": [],
            "checked_at": None,
        }
        provider = self.dependencies.ex_availability
        if not callable(provider):
            return default
        try:
            raw = provider()
        except Exception as exc:
            logger.warning("upper api: EX availability unavailable: %s", exc)
            return default
        if not isinstance(raw, Mapping):
            return default
        state = raw.get("state")
        return {
            "state": state if state in ("online", "offline", "unknown") else "unknown",
            "catalog_version": raw.get("catalog_version"),
            "action_ids": [str(item) for item in (raw.get("action_ids") or [])],
            "checked_at": raw.get("checked_at"),
        }

    async def get_actions(self) -> Any:
        return await self._dispatch(self._get_actions)

    async def _get_actions(self) -> Any:
        records, revision, version = self._catalog_view()
        return self._ok(
            {
                "actions": records,
                "revision": revision,
                "catalog_version": version,
                "ex_availability": self._ex_availability(),
            }
        )

    async def save_action(self) -> Any:
        return await self._dispatch(self._save_action)

    async def _save_action(self) -> Any:
        catalog = self._require("actions")
        payload = _require_mapping(await self.request.json(default={}), "请求体")
        expected_revision = _require_revision(payload.get("expected_revision"))
        record = _validate_action_record(payload.get("record"))

        _, current_revision, _ = self._catalog_view()
        if expected_revision != current_revision:
            raise UpperApiError("revision_conflict", "动作目录已被其他页面更新，请重新载入")

        result = self._repository_call(
            "action save", lambda: catalog.save(record, expected_revision)
        )
        action_ref = _get(result, "action_ref") or record.get("action_ref")
        if not isinstance(action_ref, str) or not action_ref:
            raise UpperApiError("dependency_unavailable", "目录未返回稳定的 action_ref")
        new_revision = _as_int(_get(result, "revision"))
        if new_revision is None:
            new_revision = expected_revision + 1
        return self._ok(
            {
                "action_ref": action_ref,
                "revision": new_revision,
                "catalog_version": _get(result, "catalog_version")
                or _get(result, "version")
                or "",
                "action": _get(result, "record") or dict(record, action_ref=action_ref),
            }
        )

    async def delete_action(self) -> Any:
        return await self._dispatch(self._delete_action)

    async def _delete_action(self) -> Any:
        catalog = self._require("actions")
        payload = _require_mapping(await self.request.json(default={}), "请求体")
        expected_revision = _require_revision(payload.get("expected_revision"))
        action_ref = _restricted_id(payload.get("action_ref"), "action_ref")

        records, current_revision, _ = self._catalog_view()
        if expected_revision != current_revision:
            raise UpperApiError("revision_conflict", "动作目录已被其他页面更新，请重新载入")
        if action_ref not in {row.get("action_ref") for row in records}:
            raise UpperApiError("not_found", "未找到该动作条目")

        result = self._repository_call(
            "action delete", lambda: catalog.delete(action_ref, expected_revision)
        )
        new_revision = _as_int(_get(result, "revision"))
        if new_revision is None:
            new_revision = expected_revision + 1
        return self._ok(
            {
                "action_ref": action_ref,
                "deleted": True,
                "revision": new_revision,
                "catalog_version": _get(result, "catalog_version")
                or _get(result, "version")
                or "",
                # Deleting the AEB entry never uninstalls the EX action itself.
                "ex_action_untouched": True,
            }
        )

    # -- prompts ----------------------------------------------------------

    def _prompt_defaults(self) -> dict[str, str]:
        provider = self.dependencies.prompt_defaults
        if not callable(provider):
            return {}
        try:
            raw = provider()
        except Exception as exc:
            logger.warning("upper api: prompt defaults unavailable: %s", exc)
            return {}
        if not isinstance(raw, Mapping):
            return {}
        return {
            kind: raw.get(kind) if isinstance(raw.get(kind), str) else ""
            for kind in PROMPT_KINDS
        }

    async def get_prompts(self) -> Any:
        return await self._dispatch(self._get_prompts)

    async def _get_prompts(self) -> Any:
        repository = self._require("settings")
        view = _settings_view(repository)
        defaults = self._prompt_defaults()
        payload: dict[str, Any] = {"revision": view.revision, "prompts": {}}
        for kind in PROMPT_KINDS:
            override = view.values.get(_PROMPT_SETTING[kind]) or ""
            override = override if isinstance(override, str) else ""
            builtin = defaults.get(kind, "")
            effective = override or builtin
            payload["prompts"][kind] = {
                "kind": kind,
                "content": override,
                "effective_content": effective,
                "source": "override" if override else "builtin",
                "hash": _canonical_hash({"kind": kind, "content": effective}),
                "default_hash": _canonical_hash({"kind": kind, "content": builtin})
                if builtin
                else None,
                "bytes": _utf8_len(effective),
            }
        return self._ok(payload)

    async def save_prompt(self) -> Any:
        return await self._dispatch(self._save_prompt)

    async def _save_prompt(self) -> Any:
        repository = self._require("settings")
        payload = _require_mapping(await self.request.json(default={}), "请求体")
        expected_revision = _require_revision(payload.get("expected_revision"))
        kind = payload.get("kind")
        if kind not in PROMPT_KINDS:
            raise UpperApiError("invalid_request", "kind 只接受 routing 或 planning")
        content = _require_str(payload.get("content", ""), "content", max_chars=MAX_PROMPT_BYTES)
        if _utf8_len(content) > MAX_PROMPT_BYTES:
            raise UpperApiError("invalid_request", f"content 超过 {MAX_PROMPT_BYTES} 字节上限")

        current = _settings_view(repository)
        if expected_revision != current.revision:
            raise UpperApiError("revision_conflict", "设置已被其他页面更新，请重新载入")

        field_name = _PROMPT_SETTING[kind]
        # Only the edited field is written, so a concurrent settings save from
        # another tab cannot lose the other fields.
        result = self._repository_call(
            "prompt save",
            lambda: repository.save({field_name: content}, expected_revision),
        )
        new_revision = _as_int(_get(result, "revision"))
        if new_revision is None:
            new_revision = expected_revision + 1
        builtin = self._prompt_defaults().get(kind, "")
        effective = content or builtin
        return self._ok(
            {
                "kind": kind,
                "revision": new_revision,
                "source": "override" if content else "builtin",
                "hash": _canonical_hash({"kind": kind, "content": effective}),
                "bytes": _utf8_len(effective),
                "reload_required": [],
            }
        )

    # -- plans ------------------------------------------------------------

    async def get_plans(self) -> Any:
        return await self._dispatch(self._get_plans)

    async def _get_plans(self) -> Any:
        repository = self._require("plans")
        limit = _query_int(self.request.query, "limit", PLAN_LIMIT_DEFAULT)
        if limit < 1 or limit > PLAN_LIMIT_MAX:
            raise UpperApiError("invalid_request", f"limit 必须在 1..{PLAN_LIMIT_MAX} 之间")
        cursor_raw = _query_str(self.request.query, "cursor")
        offset = _decode_cursor(cursor_raw) if cursor_raw else 0

        result = self._repository_call(
            "plan listing", lambda: repository.list_plans(limit, offset)
        )
        if isinstance(result, Mapping):
            items = result.get("plans") or result.get("items") or []
            has_more = bool(result.get("has_more"))
        elif isinstance(result, (list, tuple)) and len(result) == 2:
            items, has_more = result[0] or [], bool(result[1])
        else:
            items, has_more = result or [], False

        summaries = []
        for item in items:
            summary, _segments, _pending = _plan_summary(item)
            if not summary.get("plan_id"):
                continue
            summaries.append(summary)
        next_cursor = _encode_cursor(offset + len(summaries)) if has_more else None
        return self._ok(
            {
                "plans": summaries,
                "limit": limit,
                "next_cursor": next_cursor,
                "count": len(summaries),
            }
        )

    async def get_plan(self) -> Any:
        return await self._dispatch(self._get_plan)

    async def _get_plan(self) -> Any:
        repository = self._require("plans")
        plan_id = _query_str(self.request.query, "plan_id")
        if not plan_id:
            raise UpperApiError("invalid_request", "缺少 plan_id")
        plan_id = _restricted_id(plan_id, "plan_id")
        raw = self._repository_call("plan lookup", lambda: repository.get_plan(plan_id))
        if raw is None:
            raise UpperApiError("not_found", "未找到该计划")

        summary, segments, pending = _plan_summary(raw)
        return self._ok(
            {
                "plan": {
                    **summary,
                    "segments": segments,
                    "pending_segments": pending,
                    "deliveries": _deliveries_projection(_get(raw, "deliveries")),
                    # Visibility only.  There is no execution-success field, no
                    # resume action and no motion retry anywhere in this payload.
                    "visibility": "decision_only",
                }
            }
        )


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UpperApiRegistration:
    """What B10 receives: the concrete routes and the handlers behind them."""

    plugin_name: str
    api: UpperApi
    routes: tuple[tuple[str, tuple[str, ...], str], ...]

    def full_routes(self) -> tuple[tuple[str, list[str], str], ...]:
        return tuple(
            (f"/{self.plugin_name}/{route}", list(methods), desc)
            for route, methods, desc in self.routes
        )

    def handlers(self) -> dict[str, Any]:
        return {route: getattr(self.api, _HANDLER_NAMES[route]) for route, _, _ in self.routes}

    def describe(self) -> dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "routes": [
                {"route": f"/{self.plugin_name}/{route}", "methods": list(methods), "desc": desc}
                for route, methods, desc in self.routes
            ],
            "lifecycle": (
                "register during plugin start-up; Host replaces a registration only when "
                "route and methods match exactly, so a reload re-binds instead of stacking"
            ),
        }


_DESCRIPTIONS = {
    "upper/settings": "Read upper-control settings, revision and runtime state",
    "upper/settings/save": "Save upper-control settings with revision CAS",
    "upper/providers": "List configured Host chat providers (ids only, no secrets)",
    "upper/actions": "Read the AEB action catalog and EX availability",
    "upper/actions/save": "Create or update one AEB action record",
    "upper/actions/delete": "Delete one AEB action record",
    "upper/prompts": "Read routing and planning prompts with source and hash",
    "upper/prompts/save": "Save one prompt override",
    "upper/plans": "List local upper plans (paged)",
    "upper/plans/get": "Read one upper plan with segments and deliveries",
}


def register_upper_api(
    context: Any,
    dependencies: UpperApiDependencies | None = None,
    *,
    responder: Any = None,
    request_proxy: Any = None,
) -> UpperApiRegistration:
    """Register the ten upper endpoints on the Host ``Context``.

    Called by B10 from the plugin lifecycle.  Registering twice with the same
    routes replaces the previous handlers, so a plugin reload does not stack
    duplicate registrations and does not leave a handler bound to a dead
    plugin instance.
    """

    if not hasattr(context, "register_web_api"):
        raise TypeError("context must expose register_web_api (AstrBot Star Context)")

    api = UpperApi(
        dependencies,
        host=context,
        responder=responder,
        request_proxy=request_proxy,
    )
    routes: list[tuple[str, tuple[str, ...], str]] = []
    for route, methods in ROUTE_SPECS:
        desc = _DESCRIPTIONS[route]
        context.register_web_api(
            f"/{PLUGIN_NAME}/{route}",
            getattr(api, _HANDLER_NAMES[route]),
            list(methods),
            desc,
        )
        routes.append((route, methods, desc))
    return UpperApiRegistration(
        plugin_name=PLUGIN_NAME, api=api, routes=tuple(routes)
    )


__all__ = [
    "ACTION_LEVELS",
    "APPLY_SCOPE",
    "ApiResult",
    "DELIVERY_PHASES",
    "ERROR_STATUS",
    "HostResponder",
    "PLUGIN_NAME",
    "PROMPT_KINDS",
    "ROUTE_SPECS",
    "SETTING_FIELDS",
    "UpperApi",
    "UpperApiDependencies",
    "UpperApiError",
    "UpperApiRegistration",
    "register_upper_api",
    "schema_defaults",
]
