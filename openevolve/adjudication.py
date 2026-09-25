"""Durable pending-measurement state for explicit human adjudication.

An evaluator can return ``EvaluationNeedsAdjudication``. The run then raises
``AdjudicationRequired`` with its output directory. Resolve the request with
``resolve_adjudication(output_dir, request_id, disposition, outcome=...)`` and
restart ``run_evolution`` against the same output directory. Retry measures the
stored candidate code directly; it never asks the proposal model for a replacement.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from openevolve.rejection import (
    CandidateRejected,
    RejectionCategory,
    sanitize_rejection_content,
)


class AdjudicationRequired(RuntimeError):
    """Signal that a run paused with a durable unresolved measurement."""

    def __init__(self, request_id: str, output_dir: str | Path) -> None:
        """Expose the request and run directory needed to resolve and resume."""
        self.request_id = request_id
        self.output_dir = str(output_dir)
        super().__init__(f"Adjudication {request_id!r} is pending in {self.output_dir}")


class MeasurementRetryRequired(AdjudicationRequired):
    """Signal that a retryable failure left the exact candidate pending on disk."""

    def __init__(self, request_id: str, output_dir: str | Path) -> None:
        """Expose the saved measurement so it can be retried or terminated."""
        super().__init__(request_id, output_dir)
        self.args = (f"Measurement retry {request_id!r} is pending in {self.output_dir}",)


class AdjudicationStore:
    """Keep one pending candidate and resolution in a private run file.

    Example::

        store = AdjudicationStore(output_dir)
        pending = store.load()
    """

    def __init__(self, output_dir: str | Path) -> None:
        """Use the run directory as the root for pending measurement state."""
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.directory = self.output_dir / "adjudication"
        self.path = self.directory / "pending.json"

    def load(self) -> Optional[dict[str, Any]]:
        """Read pending state and verify the exact candidate code against its digest."""
        if not self.path.exists():
            return None
        if self.path.is_symlink():
            raise ValueError("adjudication state must not be a symlink")
        if self.path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("adjudication state exceeds size limit")
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or value.get("status") not in {
            "pending",
            "resolved",
        }:
            raise ValueError("unsupported adjudication state")
        if value["status"] == "resolved" and not isinstance(value.get("resolution"), dict):
            raise ValueError("resolved adjudication is missing its disposition")
        candidate = value["candidate"]
        digest = hashlib.sha256(candidate["code"].encode("utf-8")).hexdigest()
        if digest != candidate["candidate_hash"]:
            raise ValueError("pending candidate hash mismatch")
        if value["request"].get("candidate_id") != candidate["id"]:
            raise ValueError("pending candidate identity mismatch")
        return value

    def _write(self, value: Mapping[str, Any]) -> None:
        """Replace pending state atomically and sync its directory entry."""
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix="pending-", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(value, output, sort_keys=True, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, self.path)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def save_pending(
        self,
        request: Mapping[str, Any],
        candidate: Mapping[str, Any],
        iteration: int,
        checkpoint_path: str,
        *,
        replace_resolved: bool = False,
    ) -> None:
        """Persist a request before a paused run is exposed to its caller."""
        previous = self.load()
        if previous is not None and not (replace_resolved and previous["status"] == "resolved"):
            raise RuntimeError("another adjudication is already pending")
        if not request.get("request_id") or not candidate.get("code"):
            raise ValueError("adjudication requires a request ID and candidate code")
        pending_request = dict(request)
        if pending_request.get("candidate_id") not in (None, candidate["id"]):
            pending_request["evaluator_candidate_id"] = pending_request["candidate_id"]
        pending_request["candidate_id"] = candidate["id"]
        pending_candidate = dict(candidate)
        pending_candidate["candidate_hash"] = hashlib.sha256(
            candidate["code"].encode("utf-8")
        ).hexdigest()
        self._write(
            {
                "schema_version": 1,
                "status": "pending",
                "request": pending_request,
                "candidate": pending_candidate,
                "iteration": iteration,
                "checkpoint_path": checkpoint_path,
                "resolution": None,
            }
        )

    def resolve(
        self,
        request_id: str,
        disposition: str,
        outcome: Optional[Mapping[str, Any] | CandidateRejected] = None,
    ) -> None:
        """Record an allowed assign, retry, or terminate choice for a pending request."""
        value = self.load()
        if value is None or value["request"]["request_id"] != request_id:
            raise ValueError("unknown adjudication request")
        if value["status"] != "pending":
            raise ValueError("adjudication request has already been resolved")
        allowed = value["request"].get("allowed_dispositions", ["assign", "retry", "terminate"])
        if disposition not in allowed or disposition not in {
            "assign",
            "retry",
            "terminate",
        }:
            raise ValueError("disposition is not allowed for this request")
        assigned: Optional[dict[str, Any]] = None
        if disposition == "assign":
            if isinstance(outcome, CandidateRejected):
                rationale, evidence = sanitize_rejection_content(
                    outcome.rationale, outcome.evidence
                )
                assigned = {
                    "kind": "rejected",
                    "category": RejectionCategory(outcome.category).value,
                    "code": outcome.code,
                    "rationale": rationale,
                    "evidence": evidence,
                    "repairable": outcome.repairable,
                }
            elif isinstance(outcome, Mapping) and outcome:
                if not all(
                    isinstance(key, str)
                    and isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(score)
                    for key, score in outcome.items()
                ):
                    raise ValueError("assigned metrics must be finite numbers")
                assigned = {"kind": "accepted", "metrics": dict(outcome)}
            else:
                raise ValueError("assign requires numeric metrics or CandidateRejected")
        elif outcome is not None:
            raise ValueError("only assign accepts an outcome")
        value["status"] = "resolved"
        value["resolution"] = {"disposition": disposition, "outcome": assigned}
        self._write(value)

    def clear(self) -> None:
        """Remove completed pending state after its outcome has been committed."""
        self.path.unlink(missing_ok=True)


def resolve_adjudication(
    output_dir: str | Path,
    request_id: str,
    disposition: str,
    *,
    outcome: Optional[Mapping[str, Any] | CandidateRejected] = None,
) -> None:
    """Resolve a paused run before restarting it in the same output directory.

    Args:
        output_dir: Original OpenEvolve run directory containing pending state.
        request_id: ID exposed by ``AdjudicationRequired``.
        disposition: One of ``assign``, ``retry``, or ``terminate``.
        outcome: Numeric metrics or ``CandidateRejected`` when assigning.
    """
    AdjudicationStore(output_dir).resolve(request_id, disposition, outcome)
