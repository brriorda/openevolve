"""Offline smoke coverage for parent-scoped and global rejected-attempt context.

Run from the OpenEvolve checkout with
``pytest -q tests/test_parent_feedback_smoke.py``. The typed-rejection example
evaluator and controller are real; proposal replies are scripted, so the test
does not use TeamBench or contact a model provider.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pytest

from openevolve import process_parallel
from openevolve.attempt_store import RunDirectoryAttemptStore
from openevolve.config import load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.process_parallel import ProcessParallelController
from openevolve.prompt.sampler import PromptSampler
from openevolve.rejection import RejectedAttempt, RejectionCategory, digest_text


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "typed_rejection"
INVALID_PROPOSAL = "def wrong_name(value):\n    return value\n"
VALID_PROPOSALS = (
    "def square(value):\n    return value * value\n",
    "def square(value):\n    return value * value + 0.1\n",
)


def test_example_arms_vary_only_canonical_policy() -> None:
    """Keep the PR1–PR3 examples paired on one public feature field."""
    files = {
        "artifact_low_score": "config.yaml",
        "discard_only": "discard_only.yaml",
        "parent_next_once": "parent_next_once.yaml",
        "global_history": "global_history.yaml",
    }
    configs = {policy: load_config(str(EXAMPLE / name)) for policy, name in files.items()}
    baseline = asdict(configs["artifact_low_score"])
    for policy, config in configs.items():
        value = asdict(config)
        assert value["rejection_memory"]["policy"] == policy
        value["rejection_memory"]["policy"] = "artifact_low_score"
        assert value == baseline


class ScriptedModel:
    """Capture real prompt text while returning fixed offline proposals.

    Example::

        model = ScriptedModel([INVALID_PROPOSAL, VALID_PROPOSALS[0]])
        response, usage, name = await model.generate_with_receipt(messages=[...])
    """

    def __init__(self, responses: list[str]) -> None:
        """Store ordered responses and start an empty prompt log.

        Args:
            responses: Candidate source to return on each proposal call.
        """
        self.responses = responses
        self.prompts: list[dict[str, str]] = []

    async def generate_with_receipt(
        self, *, system_message: str, messages: list[dict[str, str]]
    ) -> tuple[str, dict[str, int], str]:
        """Capture the generated prompt and return one scripted response.

        Args:
            system_message: OpenEvolve system prompt.
            messages: OpenEvolve user messages.

        Returns:
            Candidate source, deterministic usage, and model name.
        """
        self.prompts.append({"system": system_message, "user": messages[0]["content"]})
        return (
            self.responses[len(self.prompts) - 1],
            {"proposal_calls": 1, "prompt_tokens": 4},
            "scripted-model",
        )


def _configured_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> tuple[ProcessParallelController, ProgramDatabase, ScriptedModel]:
    """Build one real controller with the typed-rejection example evaluator.

    Args:
        tmp_path: Fresh run directory.
        monkeypatch: Pytest patch fixture for worker components and sampling.
        policy: Example policy config file stem.

    Returns:
        Controller, seeded database, and scripted model.
    """
    config = load_config(str(EXAMPLE / f"{policy}.yaml"))
    config.language = "python"
    config.diff_based_evolution = False
    config.evaluator.cascade_evaluation = False
    config.evaluator.max_retries = 0
    config.database.num_islands = 1
    config.evaluator.parallel_evaluations = 1
    evaluator = Evaluator(config.evaluator, str(EXAMPLE / "evaluator.py"))
    seed_code = (EXAMPLE / "initial_program.py").read_text(encoding="utf-8")
    seed_metrics = asyncio.run(evaluator.evaluate_program(seed_code, "seed"))
    assert isinstance(seed_metrics, dict)
    database = ProgramDatabase(config.database)
    database.add(
        Program(id="seed", code=seed_code, language="python", metrics=seed_metrics),
        target_island=0,
    )
    model = ScriptedModel([INVALID_PROPOSAL, *VALID_PROPOSALS])
    monkeypatch.setattr(process_parallel, "_worker_config", config, raising=False)
    monkeypatch.setattr(
        process_parallel, "_worker_prompt_sampler", PromptSampler(config.prompt), raising=False
    )
    monkeypatch.setattr(process_parallel, "_worker_llm_ensemble", model, raising=False)
    monkeypatch.setattr(process_parallel, "_worker_evaluator", evaluator, raising=False)
    monkeypatch.setattr(process_parallel, "_lazy_init_worker_components", lambda: None)
    controller = ProcessParallelController(
        config, str(EXAMPLE / "evaluator.py"), database, output_dir=str(tmp_path)
    )
    return controller, database, model


def test_parent_feedback_reaches_one_later_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliver a rejected diagnosis once, without admitting its candidate.

    Args:
        tmp_path: Fresh directory for attempts, claims, and checkpoints.
        monkeypatch: Fixes the model replies and parent sampling.
    """
    controller, database, model = _configured_run(tmp_path, monkeypatch, "parent_next_once")
    seed = database.get("seed")
    assert seed is not None
    monkeypatch.setattr(database, "sample_from_island", lambda **_kwargs: (seed, []))

    def checkpoint(iteration: int) -> None:
        """Persist the outcome before a feedback claim is completed."""
        database.save(str(tmp_path / "checkpoints" / f"checkpoint_{iteration}"), iteration)

    with ThreadPoolExecutor(max_workers=1) as executor:
        controller.executor = executor
        for iteration in (1, 2, 3):
            asyncio.run(
                controller.run_evolution(
                    start_iteration=iteration,
                    max_iterations=1,
                    checkpoint_callback=checkpoint,
                )
            )

    attempts = RunDirectoryAttemptStore(tmp_path).for_parent("seed", limit=10)
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt.code == "square_function_missing"
    assert attempt.category is RejectionCategory.STATIC_INVALID
    assert all(program.code != INVALID_PROPOSAL for program in database.programs.values())
    assert len(model.prompts) == 3
    assert "Rejected attempts associated with this parent" not in model.prompts[0]["user"]
    assert attempt.attempt_id in model.prompts[1]["user"]
    assert "square_function_missing" in model.prompts[1]["user"]
    assert "Rejected attempts associated with this parent" not in model.prompts[2]["user"]

    claim_rows = [
        json.loads(line)
        for line in (tmp_path / "attempts" / "feedback_claims.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["state"] for row in claim_rows] == ["reserved", "completed"]
    assert all(row["attempt_id"] == attempt.attempt_id for row in claim_rows)
    assert claim_rows[-1]["proposal_id"] == "iteration-2"
    assert claim_rows[-1]["prompt_digest"] == digest_text(
        model.prompts[1]["system"] + "\n" + model.prompts[1]["user"]
    )
    assert claim_rows[-1]["outcome_id"] in database.programs
    assert (tmp_path / "checkpoints" / "checkpoint_2" / "metadata.json").is_file()
    assert RunDirectoryAttemptStore(tmp_path).claim_feedback("seed", "iteration-4") is None
    assert INVALID_PROPOSAL not in (tmp_path / "attempts" / "feedback_claims.jsonl").read_text(
        encoding="utf-8"
    )


def test_global_history_labels_another_parent_without_claiming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Show bounded global context while keeping the selected parent explicit.

    Args:
        tmp_path: Fresh global comparator run directory.
        monkeypatch: Selects a different parent than the rejected attempt's owner.
    """
    controller, database, model = _configured_run(tmp_path, monkeypatch, "global_history")
    other = Program(
        id="other-parent",
        code=database.get("seed").code,
        language="python",
        metrics={"combined_score": 0.1},
    )
    database.add(other, target_island=0)
    monkeypatch.setattr(database, "sample_from_island", lambda **_kwargs: (other, []))
    attempt = RejectedAttempt(
        iteration=0,
        parent_id="seed",
        category=RejectionCategory.STATIC_INVALID,
        code="square_function_missing",
        rationale="The candidate omitted square(value).",
    )
    controller.attempt_store.append(attempt)
    model.responses = [VALID_PROPOSALS[0]]
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller.executor = executor
        asyncio.run(controller.run_evolution(start_iteration=1, max_iterations=1))

    assert len(model.prompts) == 1
    assert "Recent rejected attempts across the run" in model.prompts[0]["user"]
    assert '"parent_id": "seed"' in model.prompts[0]["user"]
    assert attempt.attempt_id in model.prompts[0]["user"]
    assert database.get(attempt.attempt_id) is None
    assert not (tmp_path / "attempts" / "feedback_claims.jsonl").exists()


def test_parent_feedback_does_not_cross_parent_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave another parent's diagnosis out of the selected parent's prompt.

    Args:
        tmp_path: Fresh parent-scoped run directory.
        monkeypatch: Provides a single accepted scripted proposal.
    """
    controller, database, model = _configured_run(tmp_path, monkeypatch, "parent_next_once")
    controller.attempt_store.append(
        RejectedAttempt(
            iteration=0,
            parent_id="other-parent",
            category=RejectionCategory.STATIC_INVALID,
            code="other_parent_failure",
            rationale="This diagnosis belongs to another admitted parent.",
        )
    )
    model.responses = [VALID_PROPOSALS[0]]
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller.executor = executor
        asyncio.run(
            controller.run_evolution(
                start_iteration=1,
                max_iterations=1,
                checkpoint_callback=lambda iteration: database.save(
                    str(tmp_path / "checkpoints" / f"checkpoint_{iteration}"), iteration
                ),
            )
        )
    assert "other_parent_failure" not in model.prompts[0]["user"]
    assert not (tmp_path / "attempts" / "feedback_claims.jsonl").exists()


def test_durable_claim_blocks_concurrent_delivery_after_reopen(tmp_path: Path) -> None:
    """Reserve one attempt across store handles and a simulated restart.

    Args:
        tmp_path: Fresh durable attempt and claim ledger directory.
    """
    first = RunDirectoryAttemptStore(tmp_path)
    attempt = RejectedAttempt(
        iteration=1,
        parent_id="seed",
        category=RejectionCategory.STATIC_INVALID,
        code="invalid_interface",
        rationale="Missing required function.",
    )
    first.append(attempt)
    claim = first.claim_feedback("seed", "iteration-2")
    assert claim is not None
    assert RunDirectoryAttemptStore(tmp_path).claim_feedback("seed", "iteration-3") is None
    first.complete_claim(claim.claim_id, "accepted-child", digest_text("prompt"))
    assert RunDirectoryAttemptStore(tmp_path).claim_feedback("seed", "iteration-4") is None
