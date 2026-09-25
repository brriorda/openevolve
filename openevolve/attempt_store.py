"""Append-only stores for rejected attempts and one-shot feedback claims.

The run-directory store writes one bounded JSON object per line beneath the OpenEvolve output path::

    store = RunDirectoryAttemptStore(output_dir)
    store.append(attempt)
    assert store.get(attempt.attempt_id) == attempt
    claim = store.claim_feedback(attempt.parent_id, "iteration-12")
    if claim is not None:
        store.complete_claim(claim.claim_id, "candidate-12", prompt_digest=None)

Records use an allow-listed schema from :mod:`openevolve.rejection`; arbitrary evaluator values are
never pickled or written to disk. Claims contain identifiers and prompt digests,
never candidate code or full prompts.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Mapping, Optional, Protocol

from openevolve.rejection import RejectedAttempt, eligible_for_deferred_feedback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedbackClaim:
    """One proposal's durable reservation of a rejected attempt.

    Example::

        claim = store.claim_feedback("parent-1", "iteration-12")
        if claim is not None:
            store.complete_claim(claim.claim_id, "child-12", prompt_digest)
    """

    claim_id: str
    attempt_id: str
    parent_id: str
    proposal_id: str
    state: Literal["reserved", "completed", "released"] = "reserved"
    outcome_id: Optional[str] = None
    prompt_digest: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the small, allow-listed claim ledger schema."""
        uuid.UUID(self.claim_id)
        if not self.attempt_id or len(self.attempt_id) > 128:
            raise ValueError("claim attempt_id must contain 1 to 128 characters")
        if not self.parent_id or len(self.parent_id) > 256:
            raise ValueError("claim parent_id must contain 1 to 256 characters")
        if not self.proposal_id or len(self.proposal_id) > 128:
            raise ValueError("claim proposal_id must contain 1 to 128 characters")
        if self.state not in {"reserved", "completed", "released"}:
            raise ValueError("unknown feedback claim state")
        if self.state == "completed":
            if not self.outcome_id or len(self.outcome_id) > 128:
                raise ValueError("completed claim requires a bounded outcome ID")
        elif self.outcome_id is not None or self.prompt_digest is not None:
            raise ValueError("only a completed claim may carry outcome metadata")
        if self.prompt_digest is not None and not re.fullmatch(r"[a-f0-9]{64}", self.prompt_digest):
            raise ValueError("claim prompt_digest must be a SHA-256 digest")

    def to_dict(self) -> dict[str, str | None]:
        """Return only stable, JSON-compatible claim fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FeedbackClaim":
        """Restore one state transition from a durable JSONL entry.

        Args:
            value: Parsed claim event with no arbitrary metadata.
        """
        if not isinstance(value, Mapping):
            raise TypeError("feedback claim must be a mapping")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"feedback claim contains unsupported fields: {sorted(unknown)}")
        return cls(**value)


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

    def recent(self, *, limit: int) -> list[RejectedAttempt]:
        """Return recent attempts across parents for the global comparator."""
        ...

    def claim_feedback(self, parent_id: str, proposal_id: str) -> Optional[FeedbackClaim]:
        """Reserve one eligible parent attempt before proposal dispatch."""
        ...

    def complete_claim(
        self, claim_id: str, outcome_id: str, prompt_digest: Optional[str]
    ) -> FeedbackClaim:
        """Mark a reservation consumed after its proposal outcome is durable."""
        ...

    def release_claim(self, claim_id: str) -> FeedbackClaim:
        """Make a reservation available again when proposal dispatch fails."""
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
        self._claims: dict[str, FeedbackClaim] = {}
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

    def recent(self, *, limit: int) -> list[RejectedAttempt]:
        """Return recent attempts in append order without changing their ownership.

        Args:
            limit: Maximum number of records to return.
        """
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            return (
                [self._attempts[attempt_id] for attempt_id in self._order[-limit:]] if limit else []
            )

    def _choose_claim(self, parent_id: str, proposal_id: str) -> Optional[FeedbackClaim]:
        """Select an unclaimed eligible attempt while the store lock is held."""
        for claim in self._claims.values():
            if claim.proposal_id == proposal_id and claim.state != "released":
                if claim.parent_id != parent_id:
                    raise ValueError("proposal ID is already claimed for another parent")
                return claim if claim.state == "reserved" else None
        unavailable = {
            claim.attempt_id
            for claim in self._claims.values()
            if claim.state in {"reserved", "completed"}
        }
        for attempt_id in reversed(self._order):
            attempt = self._attempts[attempt_id]
            if (
                attempt.parent_id == parent_id
                and attempt_id not in unavailable
                and eligible_for_deferred_feedback(attempt)
            ):
                return FeedbackClaim(
                    claim_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{attempt_id}:{proposal_id}")),
                    attempt_id=attempt_id,
                    parent_id=parent_id,
                    proposal_id=proposal_id,
                )
        return None

    def claim_feedback(self, parent_id: str, proposal_id: str) -> Optional[FeedbackClaim]:
        """Reserve the newest eligible attempt once for this parent and proposal.

        Args:
            parent_id: Selected admitted evolutionary parent.
            proposal_id: Stable proposal identity, such as ``iteration-12``.
        """
        with self._lock:
            claim = self._choose_claim(parent_id, proposal_id)
            if claim is not None:
                self._claims[claim.claim_id] = claim
            return claim

    def complete_claim(
        self, claim_id: str, outcome_id: str, prompt_digest: Optional[str]
    ) -> FeedbackClaim:
        """Consume one reservation after recording its accepted or rejected outcome.

        Args:
            claim_id: Reservation returned before proposal dispatch.
            outcome_id: Durable program or rejected-attempt identifier.
            prompt_digest: Digest of the prompt that received the feedback.
        """
        with self._lock:
            claim = self._claims[claim_id]
            completed = replace(
                claim, state="completed", outcome_id=outcome_id, prompt_digest=prompt_digest
            )
            if claim.state == "released" or (claim.state == "completed" and claim != completed):
                raise ValueError("feedback claim cannot complete with conflicting outcome")
            self._claims[claim_id] = completed
            return completed

    def release_claim(self, claim_id: str) -> FeedbackClaim:
        """Release one unconsumed reservation after failed proposal dispatch.

        Args:
            claim_id: Reservation that did not reach a worker.
        """
        with self._lock:
            claim = self._claims[claim_id]
            if claim.state == "completed":
                raise ValueError("completed feedback claim cannot be released")
            released = replace(claim, state="released")
            self._claims[claim_id] = released
            return released


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
        self.claim_path = self.directory / "feedback_claims.jsonl"
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
        if self.claim_path.exists():
            with self._lock:
                self._load_claims()

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

    def _load_claims(self) -> None:
        """Restore claim transitions and discard only an incomplete final append."""
        if self.claim_path.is_symlink():
            raise ValueError("feedback claim ledger must not be a symlink")
        try:
            self.claim_path.resolve().relative_to(self.directory.resolve())
        except ValueError as exc:
            raise ValueError("feedback claim ledger must remain within the run directory") from exc
        self._claims.clear()
        valid_length = 0
        with self.claim_path.open("rb") as ledger:
            for line_number, raw_line in enumerate(ledger, start=1):
                if not raw_line.endswith(b"\n"):
                    logger.warning("Discarding incomplete feedback claim line %s", line_number)
                    break
                valid_length += len(raw_line)
                if len(raw_line) > 2048:
                    raise ValueError(f"feedback claim line {line_number} exceeds 2 KiB")
                try:
                    claim = FeedbackClaim.from_dict(json.loads(raw_line))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid feedback claim line {line_number}: {exc}") from exc
                attempt = self._attempts.get(claim.attempt_id)
                if attempt is None or attempt.parent_id != claim.parent_id:
                    raise ValueError(f"feedback claim line {line_number} has no matching attempt")
                previous = self._claims.get(claim.claim_id)
                if previous is not None:
                    if (
                        previous.attempt_id != claim.attempt_id
                        or previous.parent_id != claim.parent_id
                        or previous.proposal_id != claim.proposal_id
                    ):
                        raise ValueError(f"feedback claim identity changed at line {line_number}")
                    if previous.state == "completed" and previous != claim:
                        raise ValueError(f"completed feedback claim changed at line {line_number}")
                    if previous.state == "released" and claim.state == "completed":
                        raise ValueError(f"released feedback claim completed at line {line_number}")
                self._claims[claim.claim_id] = claim
        if valid_length != self.claim_path.stat().st_size:
            with self.claim_path.open("r+b") as ledger:
                ledger.truncate(valid_length)
                ledger.flush()
                os.fsync(ledger.fileno())
        active_attempts: set[str] = set()
        for claim in self._claims.values():
            if claim.state in {"reserved", "completed"}:
                if claim.attempt_id in active_attempts:
                    raise ValueError("one attempt has multiple active feedback claims")
                active_attempts.add(claim.attempt_id)

    def _append_claim(self, claim: FeedbackClaim) -> None:
        """Fsync one bounded claim transition before changing in-memory state."""
        encoded = (
            json.dumps(claim.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
        if len(encoded) > 2048:
            raise ValueError("serialized feedback claim exceeds 2 KiB")
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.claim_path,
            os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            previous_length = os.lseek(descriptor, 0, os.SEEK_END)
            written = os.write(descriptor, encoded)
            if written != len(encoded):
                os.ftruncate(descriptor, previous_length)
                raise OSError("feedback claim append was incomplete")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _refresh_claims(self) -> None:
        """Observe transitions written by another handle before claiming feedback."""
        if self.path.exists():
            self._load_existing()
        if self.claim_path.exists():
            self._load_claims()

    def claim_feedback(self, parent_id: str, proposal_id: str) -> Optional[FeedbackClaim]:
        """Durably reserve one eligible parent attempt before dispatch.

        Args:
            parent_id: Selected admitted parent ID.
            proposal_id: Stable identity of the proposal about to use feedback.
        """
        with self._lock:
            self._refresh_claims()
            claim = self._choose_claim(parent_id, proposal_id)
            if claim is not None and self._claims.get(claim.claim_id) != claim:
                self._append_claim(claim)
                self._claims[claim.claim_id] = claim
            return claim

    def complete_claim(
        self, claim_id: str, outcome_id: str, prompt_digest: Optional[str]
    ) -> FeedbackClaim:
        """Durably record which proposal outcome consumed a reservation.

        Args:
            claim_id: Previously persisted reservation ID.
            outcome_id: Admitted program or rejected-attempt ID.
            prompt_digest: Hash of the prompt receiving the diagnosis.
        """
        with self._lock:
            self._refresh_claims()
            claim = self._claims[claim_id]
            completed = replace(
                claim, state="completed", outcome_id=outcome_id, prompt_digest=prompt_digest
            )
            if claim.state == "released" or (claim.state == "completed" and claim != completed):
                raise ValueError("feedback claim cannot complete with conflicting outcome")
            if claim != completed:
                self._append_claim(completed)
                self._claims[claim_id] = completed
            return completed

    def release_claim(self, claim_id: str) -> FeedbackClaim:
        """Durably release a reservation when dispatch did not produce a proposal.

        Args:
            claim_id: Previously persisted reservation ID.
        """
        with self._lock:
            self._refresh_claims()
            claim = self._claims[claim_id]
            if claim.state == "completed":
                raise ValueError("completed feedback claim cannot be released")
            released = replace(claim, state="released")
            if claim != released:
                self._append_claim(released)
                self._claims[claim_id] = released
            return released
