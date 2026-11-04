"""B01 action description catalog: versioned, atomic, offline-tolerant.

The catalog owns *descriptions*. It never owns execution: an EX action's real
range, unit mapping and preconditions come from the EX capability snapshot and
are read-only here. Deleting a catalog entry removes a description; it does not
and cannot delete an EX action.

Publication is copy-on-write. A save builds a complete new version in a scratch
directory, verifies the bytes it just wrote, and only then atomically repoints
``current.json``. An interruption therefore leaves an unreferenced temporary
version, never a half-published catalog. Old versions are never deleted, because
a task that already ran keeps its own text snapshot and may still reference them.

This module is self-contained (no package-relative imports) so the existing
standalone test loader can exercise it, matching the convention already used by
``task_contracts.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorCode:
    OK = "ok"
    INVALID_ACTION = "invalid_action"
    ACTION_NOT_FOUND = "action_not_found"
    DUPLICATE_PARAMETER = "duplicate_parameter"
    CATALOG_WRITE_FAILED = "catalog_write_failed"
    CATALOG_CORRUPT = "catalog_corrupt"
    REVISION_CONFLICT = "revision_conflict"


class CatalogError(ValueError):
    """A catalog operation that must be reported to the caller."""

    def __init__(self, code: str, message: str = "", *, detail: Any = None) -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


def _fail(code: str, message: str, detail: Any = None) -> None:
    raise CatalogError(code, message, detail=detail)


def canonical_hash(value: Any) -> str:
    """Same normalisation as B00 ``upper_contracts.canonical_hash``."""
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
# Layout constants
# --------------------------------------------------------------------------

CATALOG_DIR = "upper"
CATALOG_ROOT = "catalog"
VERSIONS_DIR = "versions"
CURRENT_FILE = "current.json"
MANIFEST_FILE = "manifest.json"
SKILL_FILE = "SKILL.md"

LEVELS = frozenset({"primitive", "basic", "skill"})
AVAILABILITY = frozenset({"unknown", "described_only", "unavailable", "available"})

#: Stable ids only: lowercase, and no separator that could escape the data root.
_ACTION_REF_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ACTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_VERSION_RE = re.compile(r"^[0-9a-f]{64}$")

MAX_NAME_CHARS = 200
MAX_DESCRIPTION_CHARS = 4000
MAX_SKILL_CHARS = 200_000
MAX_PARAMETERS = 64
MAX_STAGES = 16
MAX_STAGE_CHARS = 200


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActionParameter:
    """Explanation only. The machine contract stays with EX."""

    name: str
    description: str = ""
    unit: str = ""
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "unit": self.unit,
            "required": self.required,
        }

    @classmethod
    def parse(cls, payload: Any) -> "ActionParameter":
        if not isinstance(payload, dict):
            _fail(ErrorCode.INVALID_ACTION, "parameter must be an object")
        unknown = sorted(set(payload) - {"name", "description", "unit", "required"})
        if unknown:
            _fail(ErrorCode.INVALID_ACTION, f"unknown parameter field {unknown[0]!r}")
        name = payload.get("name")
        if not isinstance(name, str) or not name.strip():
            _fail(ErrorCode.INVALID_ACTION, "parameter name is required")
        description = payload.get("description", "")
        unit = payload.get("unit", "")
        required = payload.get("required", False)
        if not isinstance(description, str) or not isinstance(unit, str):
            _fail(
                ErrorCode.INVALID_ACTION,
                "parameter description and unit must be strings",
            )
        if not isinstance(required, bool):
            _fail(ErrorCode.INVALID_ACTION, "parameter required must be a boolean")
        return cls(
            name=name.strip(),
            description=description,
            unit=unit,
            required=required,
        )


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """One catalog entry (C01 §3).

    ``action_id`` links to an EX capability. It is *not* an execution permission:
    a caller may not treat its presence as authority to act.
    """

    action_ref: str
    level: str
    name: str
    description: str = ""
    action_id: str | None = None
    parameters: tuple[ActionParameter, ...] = ()
    usage_stages: tuple[str, ...] = ()
    skill_markdown: str = ""
    enabled: bool = True
    revision: int = 1

    def validate(self) -> "ActionRecord":
        if not isinstance(self.action_ref, str) or not _ACTION_REF_RE.match(self.action_ref):
            _fail(
                ErrorCode.INVALID_ACTION,
                "action_ref must match [a-z0-9][a-z0-9._-]{0,63}",
            )
        if self.level not in LEVELS:
            _fail(ErrorCode.INVALID_ACTION, "level must be primitive, basic or skill")
        if self.action_id is not None:
            if self.level == "skill":
                # A skill is a composition note; it has no independent EX action.
                _fail(
                    ErrorCode.INVALID_ACTION,
                    "a skill entry must not carry an action_id",
                )
            if not isinstance(self.action_id, str) or not _ACTION_ID_RE.match(self.action_id):
                _fail(ErrorCode.INVALID_ACTION, "action_id has unsupported characters")
        if not isinstance(self.name, str) or not self.name.strip():
            _fail(ErrorCode.INVALID_ACTION, "name is required")
        if len(self.name) > MAX_NAME_CHARS:
            _fail(ErrorCode.INVALID_ACTION, "name is too long")
        if not isinstance(self.description, str) or len(self.description) > MAX_DESCRIPTION_CHARS:
            _fail(ErrorCode.INVALID_ACTION, "description is missing or too long")
        if len(self.parameters) > MAX_PARAMETERS:
            _fail(ErrorCode.INVALID_ACTION, "too many parameters")
        seen: set[str] = set()
        for parameter in self.parameters:
            if not isinstance(parameter, ActionParameter):
                _fail(ErrorCode.INVALID_ACTION, "parameters must be ActionParameter objects")
            if parameter.name in seen:
                _fail(
                    ErrorCode.DUPLICATE_PARAMETER,
                    f"duplicate parameter {parameter.name!r}",
                )
            seen.add(parameter.name)
        if len(self.usage_stages) > MAX_STAGES:
            _fail(ErrorCode.INVALID_ACTION, "too many usage stages")
        for stage in self.usage_stages:
            if not isinstance(stage, str) or len(stage) > MAX_STAGE_CHARS:
                _fail(ErrorCode.INVALID_ACTION, "invalid usage stage")
        if not isinstance(self.skill_markdown, str):
            _fail(ErrorCode.INVALID_ACTION, "skill_markdown must be a string")
        if len(self.skill_markdown) > MAX_SKILL_CHARS:
            _fail(ErrorCode.INVALID_ACTION, "skill markdown is too large")
        try:
            self.skill_markdown.encode("utf-8")
        except UnicodeEncodeError:
            _fail(ErrorCode.INVALID_ACTION, "skill markdown must be valid UTF-8")
        if not isinstance(self.enabled, bool):
            _fail(ErrorCode.INVALID_ACTION, "enabled must be a boolean")
        if type(self.revision) is not int or self.revision < 1:
            _fail(ErrorCode.INVALID_ACTION, "revision must be an integer >= 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_ref": self.action_ref,
            "action_id": self.action_id,
            "level": self.level,
            "name": self.name,
            "description": self.description,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "usage_stages": list(self.usage_stages),
            "skill_markdown": self.skill_markdown,
            "enabled": self.enabled,
            "revision": self.revision,
        }

    @classmethod
    def parse(cls, payload: Any) -> "ActionRecord":
        if not isinstance(payload, dict):
            _fail(ErrorCode.INVALID_ACTION, "action record must be an object")
        allowed = {
            "action_ref",
            "action_id",
            "level",
            "name",
            "description",
            "parameters",
            "usage_stages",
            "skill_markdown",
            "enabled",
            "revision",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            _fail(ErrorCode.INVALID_ACTION, f"unknown action field {unknown[0]!r}")
        parameters = payload.get("parameters", [])
        if not isinstance(parameters, list):
            _fail(ErrorCode.INVALID_ACTION, "parameters must be an array")
        stages = payload.get("usage_stages", [])
        if not isinstance(stages, list):
            _fail(ErrorCode.INVALID_ACTION, "usage_stages must be an array")
        return cls(
            action_ref=payload.get("action_ref", ""),
            level=payload.get("level", ""),
            name=payload.get("name", ""),
            description=payload.get("description", ""),
            action_id=payload.get("action_id"),
            parameters=tuple(ActionParameter.parse(item) for item in parameters),
            usage_stages=tuple(stages),
            skill_markdown=payload.get("skill_markdown", ""),
            enabled=payload.get("enabled", True),
            revision=payload.get("revision", 1),
        ).validate()

    def skill_hash(self) -> str:
        return hashlib.sha256(self.skill_markdown.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Capability view and snapshots
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CapabilityView:
    """Read-only EX capability cache.

    Used to decide what the prompt may *show* as executable. Never used as an
    authorisation, and never written back to EX.
    """

    ex_session: str
    catalog_version: str
    fetched_at: float
    connected: bool = True
    #: action_id -> {"available": bool, "reason": str}
    actions: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def availability_of(self, record: ActionRecord) -> tuple[str, str]:
        if not record.enabled:
            return "unavailable", "disabled"
        if record.action_id is None:
            return "described_only", "no EX action bound"
        if not self.connected:
            # Offline is unknown. It must never be promoted to available.
            return "unknown", "capability cache is stale"
        entry = self.actions.get(record.action_id)
        if entry is None:
            return "described_only", "not reported by EX"
        if entry.get("available") is True:
            return "available", ""
        return "unavailable", str(entry.get("reason", ""))


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    record: ActionRecord
    availability: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = self.record.to_dict()
        payload["availability"] = self.availability
        payload["availability_reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class CatalogSnapshot:
    """Immutable catalog view pinned for one planning turn.

    Pinning is the point: a model generation never reads half of an updated
    catalog. ``settings_hash`` and ``capability_version`` travel with it so a
    saved plan can explain exactly which world it was built against.
    """

    version: str
    revision: int
    entries: tuple[CatalogEntry, ...]
    settings_hash: str = ""
    capability_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "revision": self.revision,
            "settings_hash": self.settings_hash,
            "capability_version": self.capability_version,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def enabled(self) -> tuple[CatalogEntry, ...]:
        return tuple(entry for entry in self.entries if entry.record.enabled)

    def available(self) -> tuple[CatalogEntry, ...]:
        return tuple(entry for entry in self.entries if entry.availability == "available")

    def get(self, action_ref: str) -> CatalogEntry | None:
        for entry in self.entries:
            if entry.record.action_ref == action_ref:
                return entry
        return None

    def content_hash(self) -> str:
        return canonical_hash(self.to_dict())


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------


class ActionCatalog:
    """Versioned action description directory under the plugin data root.

    Layout::

        upper/catalog/
            current.json                       {"revision": n, "version": <hash>}
            versions/<version>/manifest.json   {"records": {...}}
            versions/<version>/actions/<action_ref>/SKILL.md
    """

    def __init__(self, data_root: str | os.PathLike[str], *, clock=None) -> None:
        self._root = Path(data_root) / CATALOG_DIR / CATALOG_ROOT
        self._versions = self._root / VERSIONS_DIR
        self._current = self._root / CURRENT_FILE
        self._lock = threading.RLock()
        self._clock = clock or (lambda: 0.0)

    @property
    def root(self) -> Path:
        return self._root

    # -- safe paths -------------------------------------------------------

    def _record_dir(self, version: str, action_ref: str) -> Path:
        # Both components are pattern-checked before the join, so no user input
        # can escape the data root.
        if not isinstance(version, str) or not _VERSION_RE.match(version):
            _fail(ErrorCode.CATALOG_CORRUPT, "invalid catalog version")
        if not isinstance(action_ref, str) or not _ACTION_REF_RE.match(action_ref):
            _fail(ErrorCode.INVALID_ACTION, "invalid action_ref")
        return self._versions / version / "actions" / action_ref

    # -- pointer and manifest --------------------------------------------

    def _read_current(self) -> dict[str, Any]:
        try:
            raw = self._current.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {"revision": 0, "version": ""}
        except OSError as exc:
            raise CatalogError(ErrorCode.CATALOG_CORRUPT, "current.json unreadable") from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise CatalogError(ErrorCode.CATALOG_CORRUPT, "current.json is not JSON") from exc
        if not isinstance(data, dict):
            raise CatalogError(ErrorCode.CATALOG_CORRUPT, "current.json is not an object")
        return data

    def _read_manifest(self, version: str) -> dict[str, Any]:
        if not isinstance(version, str) or not _VERSION_RE.match(version):
            raise CatalogError(ErrorCode.CATALOG_CORRUPT, "invalid version pointer")
        path = self._versions / version / MANIFEST_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise CatalogError(
                ErrorCode.CATALOG_CORRUPT, f"manifest missing for version {version}"
            ) from exc
        except (OSError, ValueError) as exc:
            raise CatalogError(
                ErrorCode.CATALOG_CORRUPT, f"manifest unreadable for version {version}"
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("records"), dict):
            raise CatalogError(ErrorCode.CATALOG_CORRUPT, "manifest has no records")
        return data

    def revision(self) -> int:
        with self._lock:
            value = self._read_current().get("revision", 0)
            return value if type(value) is int else 0

    # -- reading ----------------------------------------------------------

    def _read_skill(self, version: str, ref: str) -> str:
        path = self._record_dir(version, ref) / SKILL_FILE
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            # A missing file is a configuration error, not an invitation to fall
            # back to an older document and call it current.
            raise CatalogError(
                ErrorCode.CATALOG_CORRUPT,
                f"skill file missing for {ref} in version {version}",
            ) from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise CatalogError(
                ErrorCode.CATALOG_CORRUPT, f"skill file unreadable for {ref}"
            ) from exc

    def _records_of(self, version: str, manifest: dict[str, Any]) -> tuple[dict[str, ActionRecord], bool]:
        records: dict[str, ActionRecord] = {}
        drifted = False
        for ref in sorted(manifest["records"]):
            payload = dict(manifest["records"][ref])
            expected = payload.pop("skill_hash", "")
            text = self._read_skill(version, ref)
            if expected and expected != hashlib.sha256(text.encode("utf-8")).hexdigest():
                # The file was edited on disk after publication. Valid new text
                # is republished as a new version below; nothing is lost.
                drifted = True
            payload["skill_markdown"] = text
            records[ref] = ActionRecord.parse(payload)
        return records, drifted

    def records(self) -> tuple[ActionRecord, ...]:
        """Current records with their published skill text.

        If a skill file was edited externally, valid content is published as a
        new version here; invalid or missing content raises ``catalog_corrupt``.
        """
        with self._lock:
            current = self._read_current()
            version = current.get("version") or ""
            if not version:
                return ()
            manifest = self._read_manifest(version)
            records, drifted = self._records_of(version, manifest)
            if drifted:
                revision = current.get("revision", 0)
                revision = revision if type(revision) is int else 0
                self._publish(records, revision=revision + 1)
            return tuple(records[ref] for ref in sorted(records))

    def list(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records()]

    # -- writing ----------------------------------------------------------

    def _manifest_of(self, records: Mapping[str, ActionRecord]) -> dict[str, Any]:
        body: dict[str, Any] = {"records": {}}
        for ref in sorted(records):
            payload = records[ref].to_dict()
            payload["skill_hash"] = records[ref].skill_hash()
            payload.pop("skill_markdown")
            body["records"][ref] = payload
        return body

    def _publish(self, records: Mapping[str, ActionRecord], *, revision: int) -> str:
        """Write a complete version, verify it, then repoint the pointer."""
        manifest = self._manifest_of(records)
        version = canonical_hash(manifest)
        target = self._versions / version
        if not target.exists():
            scratch = self._versions / f".tmp-{uuid.uuid4().hex}"
            try:
                # An empty catalog (the result of deleting the last entry) still
                # needs a real directory to hold its manifest.
                scratch.mkdir(parents=True, exist_ok=True)
                for ref in sorted(records):
                    directory = scratch / "actions" / ref
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / SKILL_FILE).write_text(
                        records[ref].skill_markdown, encoding="utf-8"
                    )
                (scratch / MANIFEST_FILE).write_text(
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
                # Verify the bytes that actually landed before publishing them.
                landed = json.loads((scratch / MANIFEST_FILE).read_text(encoding="utf-8"))
                if canonical_hash(landed) != version:
                    _fail(ErrorCode.CATALOG_WRITE_FAILED, "manifest changed while writing")
                for ref, payload in manifest["records"].items():
                    text = (scratch / "actions" / ref / SKILL_FILE).read_text(encoding="utf-8")
                    if hashlib.sha256(text.encode("utf-8")).hexdigest() != payload["skill_hash"]:
                        _fail(
                            ErrorCode.CATALOG_WRITE_FAILED,
                            f"skill text for {ref} did not survive the write",
                        )
                try:
                    os.rename(scratch, target)
                except OSError:
                    # A concurrent publisher may have created the same version
                    # (identical content, identical hash). That is success.
                    if not target.exists():
                        raise
            except CatalogError:
                self._discard(scratch)
                raise
            except (OSError, ValueError) as exc:
                self._discard(scratch)
                raise CatalogError(
                    ErrorCode.CATALOG_WRITE_FAILED,
                    f"catalog write failed: {type(exc).__name__}",
                ) from exc
        self._repoint(version, revision)
        return version

    @staticmethod
    def _discard(path: Path) -> None:
        """Remove only our own scratch directory, never a published version."""
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass

    def _repoint(self, version: str, revision: int) -> None:
        payload = {
            "revision": revision,
            "version": version,
            "published_at": self._clock(),
        }
        self._root.mkdir(parents=True, exist_ok=True)
        temp = self._root / f".{CURRENT_FILE}.{uuid.uuid4().hex}.tmp"
        try:
            temp.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8",
            )
            os.replace(temp, self._current)
        except OSError as exc:
            try:
                temp.unlink()
            except OSError:
                pass
            raise CatalogError(
                ErrorCode.CATALOG_WRITE_FAILED,
                f"could not publish catalog pointer: {type(exc).__name__}",
            ) from exc

    def _check_revision(self, expected_revision: Any) -> int:
        current = self.revision()
        if type(expected_revision) is not int or expected_revision != current:
            _fail(
                ErrorCode.REVISION_CONFLICT,
                f"expected revision {expected_revision} but current is {current}",
                detail={"current_revision": current},
            )
        return current

    def save(
        self,
        record: ActionRecord | Mapping[str, Any],
        expected_revision: int,
    ) -> CatalogSnapshot:
        """Create or update one description. Never touches the EX action."""
        if isinstance(record, Mapping):
            record = ActionRecord.parse(dict(record))
        else:
            record.validate()
        with self._lock:
            current = self._check_revision(expected_revision)
            existing = {item.action_ref: item for item in self.records()}
            previous = existing.get(record.action_ref)
            if previous is not None:
                unchanged = replace(record, revision=previous.revision)
                if unchanged == previous:
                    # Byte-identical save: republishing would only churn the
                    # version and invalidate snapshots that are still valid.
                    return self.snapshot()
                # A real edit is a new revision of the same entry.
                record = replace(record, revision=previous.revision + 1)
            existing[record.action_ref] = record
            self._publish(existing, revision=current + 1)
            return self.snapshot()

    def delete(self, action_ref: str, expected_revision: int) -> CatalogSnapshot:
        """Remove a *description*; the EX action itself is untouched."""
        if not isinstance(action_ref, str) or not _ACTION_REF_RE.match(action_ref):
            _fail(ErrorCode.INVALID_ACTION, "invalid action_ref")
        with self._lock:
            current = self._check_revision(expected_revision)
            existing = {item.action_ref: item for item in self.records()}
            if action_ref not in existing:
                _fail(ErrorCode.ACTION_NOT_FOUND, f"unknown action {action_ref}")
            del existing[action_ref]
            # The new version simply omits the record. The old version stays on
            # disk because a saved task snapshot may still reference its text.
            self._publish(existing, revision=current + 1)
            return self.snapshot()

    # -- snapshot ---------------------------------------------------------

    def snapshot(
        self,
        *,
        capability: CapabilityView | None = None,
        settings_hash: str = "",
    ) -> CatalogSnapshot:
        """Freeze the catalog for one planning turn."""
        with self._lock:
            records = self.records()
            current = self._read_current()
            version = current.get("version") or ""
            revision = current.get("revision", 0)
            entries: list[CatalogEntry] = []
            for record in records:
                if capability is not None:
                    availability, reason = capability.availability_of(record)
                elif record.action_id is None:
                    availability, reason = "described_only", "no EX action bound"
                elif record.enabled:
                    availability, reason = "unknown", "no capability snapshot"
                else:
                    availability, reason = "unavailable", "disabled"
                entries.append(
                    CatalogEntry(record=record, availability=availability, reason=reason)
                )
            return CatalogSnapshot(
                version=version,
                revision=revision if type(revision) is int else 0,
                entries=tuple(entries),
                settings_hash=settings_hash,
                capability_version=(
                    f"{capability.ex_session}:{capability.catalog_version}"
                    if capability is not None
                    else ""
                ),
            )

    # -- controlled import ------------------------------------------------

    def import_skill(
        self,
        action_ref: str,
        text: str,
        *,
        level: str = "skill",
        name: str = "",
        expected_revision: int | None = None,
    ) -> CatalogSnapshot:
        """Controlled import. Takes *text*, never a filesystem path."""
        if not isinstance(text, str):
            _fail(ErrorCode.INVALID_ACTION, "skill text must be a string")
        record = ActionRecord(
            action_ref=action_ref,
            level=level,
            name=name or action_ref,
            skill_markdown=text,
        ).validate()
        revision = self.revision() if expected_revision is None else expected_revision
        return self.save(record, revision)

    def skill_text_of(self, action_ref: str) -> str:
        """Published text for one entry. There is no path parameter by design."""
        for record in self.records():
            if record.action_ref == action_ref:
                return record.skill_markdown
        _fail(ErrorCode.ACTION_NOT_FOUND, f"unknown action {action_ref}")
        raise AssertionError("unreachable")

    def has_record(self, action_ref: str) -> bool:
        return any(record.action_ref == action_ref for record in self.records())

    def refs(self) -> tuple[str, ...]:
        return tuple(record.action_ref for record in self.records())
