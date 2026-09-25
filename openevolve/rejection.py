"""Typed candidate rejection records and evaluator-facing helpers.

Evaluators can opt into categorical rejection without encoding rejection as a score::

    from openevolve.rejection import reject_candidate

    def evaluate(program_path):
        if violates_interface(program_path):
            return reject_candidate(
                category="static_invalid",
                code="required_interface_missing",
                rationale="The candidate does not define `solve`.",
            )
        return {"combined_score": score(program_path)}

Rejection records contain bounded, allow-listed evidence. Raw prompts, responses, candidate code,
and arbitrary evaluator objects are not serialized into the attempt ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Union


DEFAULT_MAX_RATIONALE_BYTES = 4096
DEFAULT_MAX_EVIDENCE_BYTES = 4096
_CODE_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_REJECTION_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_BEARER_PATTERN = re.compile(r"(?i)(\bauthorization\s*[:=]\s*bearer\s+)[^\s,;]+")
_BARE_BEARER_PATTERN = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\b"
    r"(\s*[:=]\s*)[^\s,;]+"
)
_COMPACT_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:sk-[a-z0-9_-]{8,}|gh[porus]_[a-z0-9]{8,}|"
    r"AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{20,}|"
    r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\b"
)
_LONG_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./+=-])[A-Za-z0-9_./+=-]{16,}(?![A-Za-z0-9_./+=-])"
)
_URL_CREDENTIAL_PATTERN = re.compile(r"(?i)([a-z][a-z0-9+.-]*://)[^\s/@:]+:[^\s/@]+@")
_SAFE_EVIDENCE_KEYS = frozenset(
    {"exception_type", "line", "column", "validator_code", "stage", "attempted_change"}
)


class RejectionCategory(str, Enum):
    """Candidate-level rejection classes, separate from evaluator or infrastructure failures."""

    GENERATION_FORMAT_INVALID = "generation_format_invalid"
    STATIC_INVALID = "static_invalid"
    RUNTIME_CANDIDATE_FAILURE = "runtime_candidate_failure"
    INTEGRITY_REJECTED = "integrity_rejected"
    NOVELTY_REJECTED = "novelty_rejected"
    ADMISSION_POLICY_REJECTED = "admission_policy_rejected"
    UNKNOWN_CANDIDATE_REJECTION = "unknown_candidate_rejection"


class RejectionDisposition(str, Enum):
    """Initial disposition values for a rejected proposal attempt."""

    DISCARDED = "discarded"
    REPAIR_PENDING = "repair_pending"
    REPAIR_SUCCEEDED = "repair_succeeded"
    REPAIR_EXHAUSTED = "repair_exhausted"
    FEEDBACK_AVAILABLE = "feedback_available"


def _bounded_text(value: str, *, max_bytes: int) -> str:
    """Normalize and UTF-8-bound free text, dropping control characters except whitespace."""
    if not isinstance(value, str):
        raise TypeError("rejection text fields must be strings")
    normalized = unicodedata.normalize("NFC", value)
    # Evaluator rationale and evidence are untrusted and may echo provider credentials.
    normalized = _BEARER_PATTERN.sub(r"\1[REDACTED]", normalized)
    normalized = _BARE_BEARER_PATTERN.sub(r"\1[REDACTED]", normalized)
    normalized = _SECRET_PATTERN.sub(r"\1\2[REDACTED]", normalized)
    normalized = _URL_CREDENTIAL_PATTERN.sub(r"\1[REDACTED]@", normalized)
    normalized = _COMPACT_SECRET_PATTERN.sub("[REDACTED]", normalized)
    # A long mixed compact value may be an unfamiliar provider token. Favor redaction.
    normalized = _LONG_TOKEN_PATTERN.sub(
        lambda match: (
            "[REDACTED]"
            if any(char.isalpha() for char in match.group())
            and any(char.isdigit() for char in match.group())
            else match.group()
        ),
        normalized,
    )
    safe = "".join(
        char
        for char in normalized
        if char in "\n\t" or not unicodedata.category(char).startswith("C")
    )
    encoded = safe.encode("utf-8")
    if len(encoded) <= max_bytes:
        return safe
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def _safe_evidence(value: Mapping[str, Any], *, max_bytes: int) -> dict[str, Any]:
    """Copy only approved scalar evidence fields and enforce an aggregate encoded-size bound."""
    if not isinstance(value, Mapping):
        raise TypeError("rejection evidence must be a mapping")
    safe: dict[str, Any] = {}
    for key in sorted(value):
        if key not in _SAFE_EVIDENCE_KEYS:
            continue
        item = value[key]
        if isinstance(item, bool) or item is None:
            safe[key] = item
        elif isinstance(item, int):
            safe[key] = item
        elif isinstance(item, str):
            safe[key] = _bounded_text(item, max_bytes=min(max_bytes, 1024))
        # Containers, bytes, and arbitrary objects are intentionally omitted.
    while (
        safe
        and len(json.dumps(safe, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        > max_bytes
    ):
        safe.pop(next(reversed(safe)))
    return safe


def sanitize_rejection_content(
    rationale: str,
    evidence: Optional[Mapping[str, Any]],
    *,
    max_rationale_bytes: int = DEFAULT_MAX_RATIONALE_BYTES,
    max_evidence_bytes: int = DEFAULT_MAX_EVIDENCE_BYTES,
) -> tuple[str, dict[str, Any]]:
    """Normalize evaluator text before it enters a durable attempt record.

    Args:
        rationale: Human-readable reason supplied by an evaluator.
        evidence: Optional scalar details; keys outside the allow-list are dropped.
        max_rationale_bytes: Maximum encoded rationale size.
        max_evidence_bytes: Maximum encoded evidence size.

    Returns:
        Sanitized rationale and evidence suitable for persistence.
    """
    if max_rationale_bytes < 1 or max_evidence_bytes < 1:
        raise ValueError("rejection content byte limits must be positive")
    return (
        _bounded_text(rationale, max_bytes=max_rationale_bytes),
        _safe_evidence(evidence or {}, max_bytes=max_evidence_bytes),
    )


@dataclass(frozen=True)
class CandidateRejected:
    """Typed evaluator outcome requesting candidate-level rejection.

    Use :func:`reject_candidate` from an evaluator. The controller converts this outcome into an
    attempt record; until the admission PR lands, the legacy controller may still store a low-score
    program for backward-compatible selection behavior.
    """

    category: RejectionCategory
    code: str
    rationale: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    repairable: bool = False


@dataclass(frozen=True)
class CandidateAccepted:
    """Explicit accepted outcome wrapper for code paths that opt into tagged results.

    Example::

        CandidateAccepted(metrics={"combined_score": 0.75})
    """

    metrics: Mapping[str, float]
    artifacts: Mapping[str, str | bytes] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRetryableFailure:
    """Operational measurement failure that should retry the same candidate, not mutate lineage.

    Example::

        EvaluationRetryableFailure(code="provider_rate_limit", message="Retry after cooldown")
    """

    code: str
    message: str
    retry_after_seconds: Optional[float] = None
    auto_retry: bool = True
    candidate_id: Optional[str] = None
    candidate_hash: Optional[str] = None


@dataclass(frozen=True)
class EvaluationNeedsAdjudication:
    """Durable pause request for a measurement that needs explicit human resolution.

    Example::

        EvaluationNeedsAdjudication(request_id="judge-42", candidate_id="candidate-7")
    """

    request_id: str
    candidate_id: Optional[str] = None
    completed_receipt_ids: tuple[str, ...] = ()
    unresolved_item_revisions: tuple[str, ...] = ()
    allowed_dispositions: tuple[str, ...] = ("assign", "retry", "terminate")


@dataclass(frozen=True)
class RunFatalFailure:
    """Sanitized operational failure that must stop the run without scoring a candidate.

    Example::

        RunFatalFailure(code="provider_auth", message="Provider authorization failed")
    """

    code: str
    message: str


CandidateOutcome = Union[
    CandidateAccepted,
    CandidateRejected,
    EvaluationRetryableFailure,
    EvaluationNeedsAdjudication,
    RunFatalFailure,
]


@dataclass(frozen=True)
class RejectedAttempt:
    """Bounded, serializable record for a rejected proposal, outside normal program artifacts.

    Example::

        RejectedAttempt(
            iteration=12,
            parent_id="program-4",
            category="static_invalid",
            code="syntax_error",
            rationale="The candidate did not parse.",
        )

    ``parent_id`` identifies the admitted program that produced the proposal. The
    optional prompt and response fields contain digests only, never raw content.
    """

    iteration: int
    parent_id: str
    category: RejectionCategory
    code: str
    rationale: str
    evidence: Mapping[str, Any] = field(default_factory=dict)
    attempt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    inspiration_ids: tuple[str, ...] = ()
    target_island: Optional[int] = None
    candidate_hash: Optional[str] = None
    stage: str = "evaluator"
    repairable: bool = False
    repair_attempt: int = 0
    disposition: RejectionDisposition = RejectionDisposition.DISCARDED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config_fingerprint: Optional[str] = None
    prompt_digest: Optional[str] = None
    response_digest: Optional[str] = None
    provider_usage: Mapping[str, int | float] = field(default_factory=dict)
    proposal_model: Optional[str] = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate identifiers and normalize all untrusted fields to the bounded schema."""
        object.__setattr__(self, "category", RejectionCategory(self.category))
        object.__setattr__(self, "disposition", RejectionDisposition(self.disposition))
        if self.schema_version != 1:
            raise ValueError(f"unsupported rejected-attempt schema version: {self.schema_version}")
        if self.iteration < 0 or self.repair_attempt < 0:
            raise ValueError("iteration and repair_attempt must be non-negative")
        if not self.parent_id or len(self.parent_id) > 256:
            raise ValueError("parent_id must contain 1 to 256 characters")
        if not self.attempt_id or len(self.attempt_id) > 128:
            raise ValueError("attempt_id must contain 1 to 128 characters")
        if not _REJECTION_CODE_PATTERN.fullmatch(self.code):
            raise ValueError("rejection code must be a stable lowercase identifier")
        object.__setattr__(
            self, "rationale", _bounded_text(self.rationale, max_bytes=DEFAULT_MAX_RATIONALE_BYTES)
        )
        object.__setattr__(
            self,
            "evidence",
            _safe_evidence(self.evidence, max_bytes=DEFAULT_MAX_EVIDENCE_BYTES),
        )
        object.__setattr__(
            self, "inspiration_ids", tuple(str(item)[:256] for item in self.inspiration_ids[:32])
        )
        object.__setattr__(self, "stage", _bounded_text(self.stage, max_bytes=128))
        if self.candidate_hash is not None and not re.fullmatch(
            r"[a-f0-9]{64}", self.candidate_hash
        ):
            raise ValueError("candidate_hash must be a lowercase SHA-256 digest")
        if self.target_island is not None and self.target_island < 0:
            raise ValueError("target_island must be non-negative")
        if self.prompt_digest is not None and not re.fullmatch(r"[a-f0-9]{64}", self.prompt_digest):
            raise ValueError("prompt_digest must be a lowercase SHA-256 digest")
        if self.response_digest is not None and not re.fullmatch(
            r"[a-f0-9]{64}", self.response_digest
        ):
            raise ValueError("response_digest must be a lowercase SHA-256 digest")
        object.__setattr__(self, "provider_usage", _safe_usage(self.provider_usage))
        if self.proposal_model is not None:
            object.__setattr__(
                self, "proposal_model", _bounded_text(self.proposal_model, max_bytes=128)
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the schema as plain JSON-compatible values."""
        result = asdict(self)
        result["category"] = self.category.value
        result["disposition"] = self.disposition.value
        result["inspiration_ids"] = list(self.inspiration_ids)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RejectedAttempt":
        """Validate and restore one JSON object from the durable attempt ledger."""
        if not isinstance(value, Mapping):
            raise TypeError("attempt record must be a mapping")
        fields = {field_name for field_name in cls.__dataclass_fields__}
        unknown = set(value) - fields
        if unknown:
            raise ValueError(f"attempt record contains unsupported fields: {sorted(unknown)}")
        payload = dict(value)
        payload["category"] = RejectionCategory(payload["category"])
        payload["disposition"] = RejectionDisposition(payload.get("disposition", "discarded"))
        payload["inspiration_ids"] = tuple(payload.get("inspiration_ids", ()))
        return cls(**payload)


def _safe_usage(value: Mapping[str, int | float]) -> dict[str, int | float]:
    """Validate bounded token or call accounting without accepting arbitrary nested data."""
    if not isinstance(value, Mapping):
        raise TypeError("provider_usage must be a mapping")
    result: dict[str, int | float] = {}
    for key, amount in value.items():
        if not isinstance(key, str) or not _CODE_PATTERN.fullmatch(key):
            continue
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            continue
        if amount < 0 or (isinstance(amount, float) and not math.isfinite(amount)):
            raise ValueError("provider usage values must be finite and non-negative")
        result[key] = amount
    if len(result) > 32:
        raise ValueError("provider_usage supports at most 32 counters")
    return result


def reject_candidate(
    *,
    category: RejectionCategory | str,
    code: str,
    rationale: str,
    evidence: Optional[Mapping[str, Any]] = None,
    repairable: bool = False,
) -> CandidateRejected:
    """Create an explicit evaluator rejection without assigning a fitness score.

    Args:
        category: Candidate-level rejection category.
        code: Stable lowercase identifier for this failure mode.
        rationale: Human-readable explanation, redacted and byte bounded on return.
        evidence: Optional allow-listed scalar evidence for the attempt ledger.
        repairable: Whether a later policy may try to repair this proposal.

    Returns:
        A typed outcome that OpenEvolve records as a rejected attempt.
    """
    bounded_rationale, safe_evidence = sanitize_rejection_content(
        rationale,
        evidence,
        max_rationale_bytes=DEFAULT_MAX_RATIONALE_BYTES,
        max_evidence_bytes=DEFAULT_MAX_EVIDENCE_BYTES,
    )
    return CandidateRejected(
        category=RejectionCategory(category),
        code=code,
        rationale=bounded_rationale,
        evidence=safe_evidence,
        repairable=repairable,
    )


def digest_text(value: Optional[str]) -> Optional[str]:
    """Return a SHA-256 digest for text without retaining the text itself."""
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_mapping(value: Mapping[str, Any]) -> str:
    """Hash canonical JSON configuration data without persisting the settings themselves."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
