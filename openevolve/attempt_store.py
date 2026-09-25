"""Append-only stores for rejected proposal attempts.

The run-directory store writes one bounded JSON object per line beneath the OpenEvolve output path::

    store = RunDirectoryAttemptStore(output_dir)
    store.append(attempt)
    assert store.get(attempt.attempt_id) == attempt

Records use an allow-listed schema from :mod:`openevolve.rejection`; arbitrary evaluator values are
never pickled or written to disk.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Protocol

from openevolve.rejection import RejectedAttempt

logger = logging.getLogger(__name__)


class AttemptStore(Protocol):
    """Storage interface for immutable rejected-attempt records.

    Implement this protocol to keep attempts in another durable backend while
    preserving append idempotence and parent-scoped lookup semantics.
    """

    def append(self, attempt: RejectedAttempt) -> None:
        """Persist an attempt once; duplicate IDs with identical content are idempotent."""
        ...

    def get(self, attempt_id: str) -> Optional[RejectedAttempt]:
        """Look up a persisted attempt by stable ID."""
        ...

    def for_parent(self, parent_id: str, *, limit: int) -> list[RejectedAttempt]:
        """Return the newest attempts for one admitted parent, up to ``limit`` records."""
        ...


class InMemoryAttemptStore:
    """Process-local store useful for short runs and API consumers.

    Example::

        store = InMemoryAttemptStore()
        store.append(attempt)
        recent = store.for_parent(attempt.parent_id, limit=1)
    """

    def __init__(self) -> None:
        """Create an empty store with deterministic append order."""
        self._attempts: dict[str, RejectedAttempt] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    def append(self, attempt: RejectedAttempt) -> None:
        """Append an attempt and reject conflicting duplicate IDs."""
        with self._lock:
            existing = self._attempts.get(attempt.attempt_id)
            if existing is not None:
                if existing != attempt:
                    raise ValueError(f"attempt ID collision for {attempt.attempt_id}")
                return
            self._attempts[attempt.attempt_id] = attempt
            self._order.append(attempt.attempt_id)

    def get(self, attempt_id: str) -> Optional[RejectedAttempt]:
        """Return one attempt or ``None`` when the ID is unknown."""
        with self._lock:
            return self._attempts.get(attempt_id)

    def for_parent(self, parent_id: str, *, limit: int) -> list[RejectedAttempt]:
        """Return the newest records for ``parent_id`` in append order."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            matches = [
                self._attempts[attempt_id]
                for attempt_id in self._order
                if self._attempts[attempt_id].parent_id == parent_id
            ]
            return matches[-limit:] if limit else []


class RunDirectoryAttemptStore(InMemoryAttemptStore):
    """Durable append-only JSONL ledger under a run's output directory.

    Example::

        store = RunDirectoryAttemptStore(output_dir)
        store.append(attempt)
    """

    _path_locks: dict[str, threading.RLock] = {}
    _path_locks_guard = threading.Lock()

    def __init__(self, run_directory: str | Path) -> None:
        """Load existing records from ``<run_directory>/attempts/rejected_attempts.jsonl``."""
        super().__init__()
        self.run_directory = Path(run_directory).expanduser().resolve()
        self.directory = self.run_directory / "attempts"
        self.path = self.directory / "rejected_attempts.jsonl"
        try:
            self.directory.resolve().relative_to(self.run_directory)
        except ValueError as exc:
            raise ValueError(
                "attempt store directory must remain within the run directory"
            ) from exc
        lock_key = str(self.path)
        with self._path_locks_guard:
            self._lock = self._path_locks.setdefault(lock_key, threading.RLock())
        if self.path.exists():
            try:
                self.path.resolve().relative_to(self.directory.resolve())
            except ValueError as exc:
                raise ValueError(
                    "attempt ledger file must remain within the attempts directory"
                ) from exc
            with self._lock:
                self._load_existing()

    def _load_existing(self) -> None:
        """Reload records and discard only an incomplete final append after a crash."""
        self._attempts.clear()
        self._order.clear()
        valid_length = 0
        with self.path.open("rb") as ledger:
            for line_number, raw_line in enumerate(ledger, start=1):
                if not raw_line.endswith(b"\n"):
                    logger.warning(
                        "Discarding incomplete final attempt ledger line %s", line_number
                    )
                    break
                valid_length += len(raw_line)
                if not raw_line.strip():
                    continue
                if len(raw_line) > 16 * 1024:
                    raise ValueError(f"attempt ledger line {line_number} exceeds 16 KiB")
                try:
                    value = json.loads(raw_line)
                    attempt = RejectedAttempt.from_dict(value)
                except (
                    KeyError,
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise ValueError(f"invalid attempt ledger line {line_number}: {exc}") from exc
                existing = self._attempts.get(attempt.attempt_id)
                if existing is not None and existing != attempt:
                    raise ValueError(f"conflicting duplicate attempt ID at line {line_number}")
                if existing is None:
                    self._attempts[attempt.attempt_id] = attempt
                    self._order.append(attempt.attempt_id)
        if valid_length != self.path.stat().st_size:
            with self.path.open("r+b") as ledger:
                ledger.truncate(valid_length)
                ledger.flush()
                os.fsync(ledger.fileno())

    def append(self, attempt: RejectedAttempt) -> None:
        """Durably append one complete record, using O_APPEND and fsync for crash visibility."""
        encoded = (
            json.dumps(
                attempt.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > 16 * 1024:
            raise ValueError("serialized attempt exceeds 16 KiB")
        with self._lock:
            # Another handle may have appended since this instance was constructed.
            if self.path.exists():
                self._load_existing()
            existing = self._attempts.get(attempt.attempt_id)
            if existing is not None:
                if existing != attempt:
                    raise ValueError(f"attempt ID collision for {attempt.attempt_id}")
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            no_follow = getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | no_follow, 0o600
            )
            try:
                if hasattr(os, "fchmod"):
                    os.fchmod(descriptor, 0o600)
                previous_length = os.lseek(descriptor, 0, os.SEEK_END)
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    os.ftruncate(descriptor, previous_length)
                    raise OSError("attempt ledger append was incomplete")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._attempts[attempt.attempt_id] = attempt
            self._order.append(attempt.attempt_id)
