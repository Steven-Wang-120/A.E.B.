"""B01 settings truth, backed by the Host's own config object.

There is deliberately no second ``settings.json``. This repository wraps the
``AstrBotConfig`` instance B10 injects and persists through the Host's
``save_config()``, so a write failure stays visible instead of being hidden
behind a private file that the Host's own configuration page knows nothing
about.

Concurrency and staleness
-------------------------
A single serial write lock guards every mutation. The business revision is
derived from the settings content hash and stored as a metadata record inside
the same config object, which gives two properties:

* an edit made through the Host's generic configuration page changes the
  content hash, so the next :meth:`load` bumps the revision and an older Page
  save is rejected with ``revision_conflict`` instead of overwriting the newer
  value;
* the revision survives a restart, because it lives in the config, not in
  process memory.

Prompts share the same lock and the same revision as every other setting, so a
settings save can never overwrite a prompt that changed meanwhile.

No model call, socket or chat send happens here.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Mapping

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorCode:
    OK = "ok"
    INVALID_SETTINGS = "invalid_settings"
    SETTINGS_WRITE_FAILED = "settings_write_failed"
    REVISION_CONFLICT = "revision_conflict"
    PROMPT_TOO_LARGE = "prompt_too_large"


class UpperSettingsError(ValueError):
    """A settings operation that must be reported, never swallowed."""

    def __init__(self, code: str, message: str = "", *, detail: Any = None) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


def _fail(code: str, message: str, detail: Any = None) -> None:
    raise UpperSettingsError(code, message, detail=detail)


def canonical_hash(value: Any) -> str:
    """Same normalisation as B00 ``upper_contracts.canonical_hash``.

    B02 needs this value for ``PromptSnapshotMeta.settings_version``, so the
    two must agree byte for byte.
    """
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

#: C01 §3. The feature ships off, and the control channel binds loopback only.
#: The old three text/audio/vision channel addresses are not touched here.
DEFAULT_SETTINGS: dict[str, Any] = {
    "upper_enabled": False,
    "routing_provider_id": "",
    "generation_provider_id": "",
    "routing_prompt": "",
    "planning_prompt": "",
    "control_enabled": False,
    "control_bind_host": "127.0.0.1",
    "control_port": 8769,
    "control_peer_id": "",
    "robot_id": "",
    "model_timeout_s": 30.0,
}

#: Changing any of these needs the control channel rebuilt, not just a reload
#: of the prompt.
RELOAD_FIELDS = frozenset(
    {"control_enabled", "control_bind_host", "control_port", "control_peer_id", "robot_id"}
)
#: Changing any of these takes effect on the next input. A plan already running
#: keeps the snapshot it was built with.
NEXT_INPUT_FIELDS = frozenset(
    {
        "upper_enabled",
        "routing_provider_id",
        "generation_provider_id",
        "routing_prompt",
        "planning_prompt",
        "model_timeout_s",
    }
)

APPLY_NEXT_INPUT = "next_input"
APPLY_RELOAD_REQUIRED = "reload_required"

MAX_PROVIDER_ID_LEN = 128
MAX_HOST_LEN = 253
MAX_PROMPT_CHARS = 200_000
MIN_TIMEOUT_S = 0.1
MAX_TIMEOUT_S = 600.0

#: Metadata record inside the same config object. Excluded from the fingerprint
#: so that recording a revision never changes the content it describes.
META_KEY = "upper_settings_meta"

_HOST_RE = re.compile(r"^[A-Za-z0-9._:\[\]-]+$")
_PROVIDER_RE = re.compile(r"^[A-Za-z0-9._:-]*$")
_WIRE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


# --------------------------------------------------------------------------
# Field validation
# --------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be a boolean")
    return value


def _validate_provider(value: Any, name: str) -> str:
    if not isinstance(value, str):
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be a string")
    if len(value) > MAX_PROVIDER_ID_LEN:
        _fail(
            ErrorCode.INVALID_SETTINGS,
            f"{name} must not exceed {MAX_PROVIDER_ID_LEN} characters",
        )
    if not _PROVIDER_RE.match(value):
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} contains unsupported characters")
    return value


def _validate_host(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be a non-empty string")
    if len(value) > MAX_HOST_LEN:
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} is too long")
    if not _HOST_RE.match(value):
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be a host name or address")
    return value


def _validate_port(value: Any, name: str) -> int:
    if type(value) is not int:
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be an integer")
    if not 1 <= value <= 65535:
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be within 1..65535")
    return value


def _validate_timeout(value: Any, name: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be a number")
    number = float(value)
    if not MIN_TIMEOUT_S <= number <= MAX_TIMEOUT_S:
        _fail(
            ErrorCode.INVALID_SETTINGS,
            f"{name} must be within {MIN_TIMEOUT_S}..{MAX_TIMEOUT_S} seconds",
        )
    return number


def _validate_prompt(value: Any, name: str) -> str:
    if not isinstance(value, str):
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be a string")
    if len(value) > MAX_PROMPT_CHARS:
        _fail(
            ErrorCode.PROMPT_TOO_LARGE,
            f"{name} exceeds {MAX_PROMPT_CHARS} characters",
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be valid UTF-8")
    # An empty override is meaningful: it means "use the built-in B02 file".
    return value


def _validate_wire_id(value: Any, name: str) -> str:
    if not isinstance(value, str):
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} must be a string")
    if len(value) > MAX_PROVIDER_ID_LEN:
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} is too long")
    if value and not _WIRE_ID_RE.match(value):
        _fail(ErrorCode.INVALID_SETTINGS, f"{name} contains unsupported characters")
    return value


VALIDATORS = {
    "upper_enabled": _validate_bool,
    "control_enabled": _validate_bool,
    "routing_provider_id": _validate_provider,
    "generation_provider_id": _validate_provider,
    "routing_prompt": _validate_prompt,
    "planning_prompt": _validate_prompt,
    "control_bind_host": _validate_host,
    "control_port": _validate_port,
    "control_peer_id": _validate_wire_id,
    "robot_id": _validate_wire_id,
    "model_timeout_s": _validate_timeout,
}


@dataclass(frozen=True, slots=True)
class SettingsSnapshot:
    """One immutable view of the settings plus its revision identity."""

    settings: dict[str, Any]
    revision: int
    fingerprint: str
    apply_scope: str = APPLY_NEXT_INPUT

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": dict(self.settings),
            "revision": self.revision,
            "fingerprint": self.fingerprint,
            "apply_scope": self.apply_scope,
        }

    def get(self, name: str) -> Any:
        return self.settings.get(name, DEFAULT_SETTINGS.get(name))


class SettingsRepository:
    """Settings truth backed by the Host's own config object.

    ``config`` is duck-typed: any mutable mapping exposing ``save_config()``
    works, which is what lets the tests run without a Host.
    """

    def __init__(self, config: Any) -> None:
        if not hasattr(config, "save_config"):
            _fail(ErrorCode.INVALID_SETTINGS, "config object must expose save_config()")
        self._config = config
        self._lock = threading.RLock()
        self._revision = 0
        self._fingerprint = ""
        self._settings: dict[str, Any] = dict(DEFAULT_SETTINGS)

    # -- reading ----------------------------------------------------------

    def _raw(self, name: str) -> Any:
        value = self._config.get(name, DEFAULT_SETTINGS[name])
        return DEFAULT_SETTINGS[name] if value is None else value

    def _read_raw(self) -> dict[str, Any]:
        return {name: self._raw(name) for name in DEFAULT_SETTINGS}

    def _read_meta(self) -> dict[str, Any]:
        meta = self._config.get(META_KEY)
        return meta if isinstance(meta, dict) else {}

    def _normalise(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Validate every field; a bad external value falls back to its default.

        A broken hand edit must not crash the plugin. The fallback is visible in
        the returned settings, so the operator can see which key is wrong.
        """
        settings: dict[str, Any] = {}
        for name, default in DEFAULT_SETTINGS.items():
            try:
                settings[name] = VALIDATORS[name](raw[name], name)
            except UpperSettingsError:
                settings[name] = default
        return settings

    def load(self) -> SettingsSnapshot:
        """Recompute from the config object, detecting an external edit."""
        with self._lock:
            settings = self._normalise(self._read_raw())
            fingerprint = canonical_hash(settings)
            meta = self._read_meta()
            stored = meta.get("revision")
            if type(stored) is not int or stored < 1:
                stored = 1
            meta_fingerprint = meta.get("fingerprint")
            drifted = (meta_fingerprint and meta_fingerprint != fingerprint) or (
                self._fingerprint and self._fingerprint != fingerprint
            )
            self._revision = max(self._revision, stored) + (1 if drifted else 0)
            self._fingerprint = fingerprint
            self._settings = settings
            return SettingsSnapshot(
                settings=dict(settings),
                revision=self._revision,
                fingerprint=fingerprint,
                apply_scope=APPLY_NEXT_INPUT,
            )

    def load_prompts(self) -> tuple[str, str, int]:
        snapshot = self.load()
        return (
            snapshot.settings["routing_prompt"],
            snapshot.settings["planning_prompt"],
            snapshot.revision,
        )

    @property
    def revision(self) -> int:
        return self.load().revision

    # -- writing ----------------------------------------------------------

    def _commit(self, patch: Mapping[str, Any], *, revision: int, fingerprint: str) -> None:
        """Write the settings and the metadata in one Host save.

        If the Host write fails, every touched key is restored so the previously
        valid configuration stays readable.
        """
        touched = {name: self._config.get(name) for name in patch}
        touched[META_KEY] = self._config.get(META_KEY)
        try:
            for name, value in patch.items():
                self._config[name] = value
            self._config[META_KEY] = {"revision": revision, "fingerprint": fingerprint}
            self._config.save_config()
        except Exception as exc:  # noqa: BLE001 - a failed save must stay visible
            for name, value in touched.items():
                if value is None:
                    try:
                        del self._config[name]
                    except (KeyError, TypeError):
                        pass
                else:
                    self._config[name] = value
            raise UpperSettingsError(
                ErrorCode.SETTINGS_WRITE_FAILED,
                f"host config save failed: {type(exc).__name__}",
            ) from exc

    def _apply_scope_for(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
        if any(before.get(name) != after.get(name) for name in RELOAD_FIELDS):
            return APPLY_RELOAD_REQUIRED
        return APPLY_NEXT_INPUT

    def save(self, data: Mapping[str, Any], expected_revision: int) -> SettingsSnapshot:
        """Merge a *partial* patch into the live config under one write lock.

        Only a partial patch is accepted, which makes "keep the fields the
        caller did not edit" structural rather than a convention a Page has to
        remember.
        """
        if not isinstance(data, Mapping):
            _fail(ErrorCode.INVALID_SETTINGS, "settings payload must be an object")
        unknown = sorted(set(data) - set(DEFAULT_SETTINGS))
        if unknown:
            _fail(ErrorCode.INVALID_SETTINGS, f"unknown setting {unknown[0]!r}")

        with self._lock:
            current = self.load()
            if type(expected_revision) is not int or expected_revision != current.revision:
                _fail(
                    ErrorCode.REVISION_CONFLICT,
                    f"expected revision {expected_revision} but current is {current.revision}",
                    detail={"current_revision": current.revision},
                )
            merged = dict(current.settings)
            for name, value in data.items():
                merged[name] = VALIDATORS[name](value, name)
            fingerprint = canonical_hash(merged)
            scope = self._apply_scope_for(current.settings, merged)
            # The revision only moves when the content moves.
            revision = (
                current.revision
                if fingerprint == current.fingerprint
                else current.revision + 1
            )
            self._commit(
                {name: merged[name] for name in data},
                revision=revision,
                fingerprint=fingerprint,
            )
            self._settings = merged
            self._fingerprint = fingerprint
            self._revision = revision
            return SettingsSnapshot(
                settings=dict(merged),
                revision=revision,
                fingerprint=fingerprint,
                apply_scope=scope,
            )

    def save_prompts(
        self,
        *,
        routing_prompt: str | None = None,
        planning_prompt: str | None = None,
        expected_revision: int,
    ) -> SettingsSnapshot:
        """Prompt overrides share the settings lock and revision.

        ``None`` leaves that prompt untouched, so the settings page and the
        prompts page cannot clobber each other with a stale full dict.
        """
        patch: dict[str, Any] = {}
        if routing_prompt is not None:
            patch["routing_prompt"] = routing_prompt
        if planning_prompt is not None:
            patch["planning_prompt"] = planning_prompt
        return self.save(patch, expected_revision)

    async def save_async(
        self, data: Mapping[str, Any], expected_revision: int
    ) -> SettingsSnapshot:
        """Async facade over the same single write path.

        The Host write itself is synchronous; running it in a worker thread
        keeps the event loop free while still taking the one write lock.
        """
        import asyncio

        return await asyncio.to_thread(self.save, data, expected_revision)
