"""Configuration validation and resolution for rejection-memory experiment arms.

Example YAML::

    rejection_memory:
      policy: artifact_low_score
      store: run_directory

The baseline, discard-only, parent-next-once, and global-history policies are
implemented in this revision. Other arm values remain recognized, but preflight rejects them
until their behavior is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class RejectionPolicy(str, Enum):
    """Public rejection-memory arm selector."""

    ARTIFACT_LOW_SCORE = "artifact_low_score"
    DISCARD_ONLY = "discard_only"
    IMMEDIATE_REPAIR = "immediate_repair"
    PARENT_NEXT_ONCE = "parent_next_once"
    COMBINED = "combined"
    GLOBAL_HISTORY = "global_history"


@dataclass(frozen=True)
class ResolvedRejectionPolicy:
    """Immutable internal feature matrix derived from one public policy value.

    Use ``resolve_rejection_policy("discard_only")`` or another supported arm
    to obtain the feature matrix; callers should not construct it manually.
    """

    policy: RejectionPolicy
    record_attempt: bool
    admit_rejected_program: bool
    attach_rejection_artifact: bool
    immediate_repair: bool
    deferred_parent_delivery: bool
    global_delivery: bool


def resolve_rejection_policy(policy: str | RejectionPolicy) -> ResolvedRejectionPolicy:
    """Resolve an arm or fail before a run starts when it is unavailable.

    Args:
        policy: Public rejection-memory policy name or enum value.

    Returns:
        The enabled internal feature matrix for the selected policy.

    Raises:
        NotImplementedError: A recognized but unavailable policy was selected.
        ValueError: The policy name is unknown.
    """
    selected = RejectionPolicy(policy)
    if selected not in {
        RejectionPolicy.ARTIFACT_LOW_SCORE,
        RejectionPolicy.DISCARD_ONLY,
        RejectionPolicy.PARENT_NEXT_ONCE,
        RejectionPolicy.GLOBAL_HISTORY,
    }:
        raise NotImplementedError(
            f"rejection_memory.policy={selected.value!r} is not implemented in this OpenEvolve "
            "revision; use 'artifact_low_score', 'discard_only', 'parent_next_once', 'global_history', "
            "or a revision that supports this policy"
        )
    return ResolvedRejectionPolicy(
        policy=selected,
        record_attempt=True,
        admit_rejected_program=selected is RejectionPolicy.ARTIFACT_LOW_SCORE,
        attach_rejection_artifact=selected is RejectionPolicy.ARTIFACT_LOW_SCORE,
        immediate_repair=False,
        deferred_parent_delivery=selected is RejectionPolicy.PARENT_NEXT_ONCE,
        global_delivery=selected is RejectionPolicy.GLOBAL_HISTORY,
    )


def validate_rejection_memory_config(config: object) -> ResolvedRejectionPolicy:
    """Validate programmatically built config and resolve its selected arm.

    Args:
        config: Rejection-memory settings, normally ``Config.rejection_memory``.

    Returns:
        The resolved feature matrix for a valid configuration.
    """
    store = getattr(config, "store", None)
    if store not in {"run_directory", "in_memory"}:
        raise ValueError("rejection_memory.store must be 'run_directory' or 'in_memory'")
    for name in ("max_rationale_bytes", "max_evidence_bytes"):
        value = getattr(config, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 4096:
            raise ValueError(f"rejection_memory.{name} must be between 1 and 4096 bytes")
    for name in ("feedback_recent_k", "repair_max_attempts"):
        value = getattr(config, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"rejection_memory.{name} must be a non-negative integer")
    if not isinstance(getattr(config, "security_filter", None), bool):
        raise ValueError("rejection_memory.security_filter must be a boolean")
    if not config.security_filter:
        raise ValueError(
            "rejection_memory.security_filter must remain enabled for durable attempts"
        )
    penalty_score = getattr(config, "penalty_score", None)
    if (
        isinstance(penalty_score, bool)
        or not isinstance(penalty_score, (int, float))
        or not math.isfinite(penalty_score)
    ):
        raise ValueError("rejection_memory.penalty_score must be a finite number")
    repair_models = getattr(config, "repair_models", None)
    if not isinstance(repair_models, list) or not all(
        isinstance(model, str) and model for model in repair_models
    ):
        raise ValueError("rejection_memory.repair_models must be a list of non-empty strings")
    if not isinstance(getattr(config, "repair_diff_based", None), bool):
        raise ValueError("rejection_memory.repair_diff_based must be a boolean")
    resolved = resolve_rejection_policy(getattr(config, "policy", None))
    if resolved.deferred_parent_delivery:
        if store != "run_directory":
            raise ValueError("parent_next_once requires a durable run_directory attempt store")
        if config.feedback_recent_k < 1:
            raise ValueError("parent_next_once requires feedback_recent_k >= 1")
    if resolved.global_delivery and config.feedback_recent_k < 1:
        raise ValueError("global_history requires feedback_recent_k >= 1")
    return resolved
