"""Regression tests for typed rejection outcomes and durable attempt storage.

Run from the OpenEvolve checkout with ``pytest tests/test_rejection_memory.py``.
Fixtures use one small candidate and a temporary run directory; no provider calls occur.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future
from dataclasses import asdict

import pytest

from openevolve.adjudication import (
    AdjudicationRequired,
    AdjudicationStore,
    MeasurementRetryRequired,
    resolve_adjudication,
)
from openevolve.attempt_store import RunDirectoryAttemptStore
from openevolve.config import Config, EvaluatorConfig, LLMModelConfig
from openevolve.controller import OpenEvolve
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.process_parallel import ProcessParallelController
from openevolve import process_parallel
from openevolve.rejection import (
    CandidateRejected,
    EvaluationNeedsAdjudication,
    EvaluationRetryableFailure,
    RejectedAttempt,
    RejectionCategory,
    RunFatalFailure,
    digest_text,
    reject_candidate,
)


@pytest.fixture
def sample_attempt() -> RejectedAttempt:
    """Provide a stable rejected proposal for serialization and ledger checks."""
    return RejectedAttempt(
        attempt_id="attempt-one",
        iteration=1,
        parent_id="parent-one",
        category=RejectionCategory.STATIC_INVALID,
        code="invalid_interface",
        stage="evaluator",
        rationale="Missing solve function",
        candidate_hash=digest_text("def solve(): pass"),
    )


def test_attempt_roundtrip_redaction_and_bounds(
    sample_attempt: RejectedAttempt,
) -> None:
    """Preserve the typed schema while excluding bare compact tokens and excess text."""
    restored = RejectedAttempt.from_dict(sample_attempt.to_dict())
    assert restored == sample_attempt
    raw_token = "sk-" + "a1" * 20
    rejected = reject_candidate(
        category="static_invalid",
        code="invalid_interface",
        rationale=f"token {raw_token} " + "x" * 5000,
        evidence={"attempted_change": raw_token, "raw_response": "must be dropped"},
    )
    attempt = RejectedAttempt(
        iteration=1,
        parent_id="parent-one",
        category=rejected.category,
        code=rejected.code,
        stage="evaluator",
        rationale=rejected.rationale,
        evidence=rejected.evidence,
    )
    encoded = str(attempt.to_dict())
    assert raw_token not in encoded
    assert "[REDACTED]" in encoded
    assert "raw_response" not in encoded
    assert len(attempt.rationale.encode("utf-8")) <= 4096


def test_ledger_idempotence_and_partial_tail(
    tmp_path, sample_attempt: RejectedAttempt
) -> None:
    """Deduplicate across handles and recover only a torn final JSONL append."""
    first = RunDirectoryAttemptStore(tmp_path)
    second = RunDirectoryAttemptStore(tmp_path)
    first.append(sample_attempt)
    second.append(sample_attempt)
    assert first.path.read_bytes().count(b"\n") == 1
    with first.path.open("ab") as ledger:
        ledger.write(b'{"incomplete":')
    reopened = RunDirectoryAttemptStore(tmp_path)
    assert reopened.get(sample_attempt.attempt_id) == sample_attempt
    assert reopened.path.read_bytes().count(b"\n") == 1
    assert reopened.path.read_bytes().endswith(b"\n")


@pytest.fixture
def evaluator(tmp_path) -> Evaluator:
    """Create a direct evaluator with a replaceable measurement function."""
    evaluation_file = tmp_path / "evaluate.py"
    evaluation_file.write_text(
        "def evaluate(path):\n    return {'combined_score': 0.0}\n"
    )
    return Evaluator(
        EvaluatorConfig(cascade_evaluation=False, max_retries=1),
        str(evaluation_file),
    )


@pytest.mark.parametrize(
    "outcome",
    [
        EvaluationNeedsAdjudication(request_id="request-one"),
        RunFatalFailure(code="provider_auth", message="Authorization failed"),
    ],
)
def test_operational_outcomes_never_become_scores(
    evaluator: Evaluator, outcome
) -> None:
    """Return pause and fatal outcomes unchanged rather than accepting a zero score."""

    async def measure(_path: str):
        """Return the chosen operational result from the direct evaluator."""
        return outcome

    evaluator._direct_evaluate = measure
    assert asyncio.run(evaluator.evaluate_program("pass", "candidate-one")) == outcome


def test_retry_and_zero_score_are_distinct(evaluator: Evaluator) -> None:
    """Retry the same measurement and retain explicit score-zero acceptance."""
    calls = 0

    async def measure(_path: str):
        """Return one transient failure followed by a legitimate zero score."""
        nonlocal calls
        calls += 1
        return (
            EvaluationRetryableFailure("temporary", "Retry", retry_after_seconds=0.0)
            if calls == 1
            else {"combined_score": 0.0}
        )

    evaluator._direct_evaluate = measure
    assert asyncio.run(evaluator.evaluate_program("pass", "candidate-one")) == {
        "combined_score": 0.0
    }
    assert calls == 2
    rejected = reject_candidate(
        category="static_invalid", code="invalid_interface", rationale="Missing solve"
    )

    async def reject(_path: str):
        """Return an explicit candidate rejection."""
        return rejected

    evaluator._direct_evaluate = reject
    assert asyncio.run(evaluator.evaluate_program("pass", "candidate-two")) == rejected


def test_provider_receipt_uses_reported_usage() -> None:
    """Keep actual provider token counters instead of a guessed call-only receipt."""

    class FakeModel:
        """Small provider stand-in exposing its latest response usage."""

        model = "example-model"
        last_usage = {"prompt_tokens": 12, "completion_tokens": 7}
        last_call_attempts = 2

        async def generate_with_context(self, _system, _messages):
            """Return a provider response without contacting a service."""
            return "proposal"

    ensemble = object.__new__(LLMEnsemble)
    ensemble._sample_model = lambda: FakeModel()
    response, usage, model = asyncio.run(ensemble.generate_with_receipt("system", []))
    assert (response, model) == ("proposal", "example-model")
    assert usage == {"prompt_tokens": 12, "completion_tokens": 7, "proposal_calls": 2}


@pytest.mark.parametrize(
    "response, expected_code",
    [
        ("", "no_valid_code"),
        ("def solve(): return 2", "interface_invalid"),
    ],
)
def test_worker_records_parser_and_evaluator_rejections(
    monkeypatch,
    response: str,
    expected_code: str,
) -> None:
    """Keep one attempt for either rejection stage with real usage and baseline artifacts."""
    from openevolve.database import Program

    class FakePrompt:
        """Return a deterministic prompt for worker accounting."""

        def build_prompt(self, **_kwargs):
            """Supply one small prompt."""
            return {"system": "system", "user": "user"}

    class FakeEnsemble:
        """Return a deterministic proposal and provider receipt."""

        async def generate_with_receipt(self, **_kwargs):
            """Supply the requested generated text."""
            return response, {"proposal_calls": 1, "prompt_tokens": 4}, "example-model"

    class FakeEvaluator:
        """Reject parsed code as an invalid interface."""

        async def evaluate_program(self, _code: str, _program_id: str):
            """Reject a candidate with a typed reason."""
            return reject_candidate(
                category="static_invalid",
                code="interface_invalid",
                rationale="Missing required interface",
            )

        def get_pending_artifacts(self, _program_id: str):
            """Return no prior artifacts."""
            return None

    config = Config()
    config.language = "python"
    config.diff_based_evolution = False
    config.rejection_memory.penalty_score = -0.4
    monkeypatch.setattr(process_parallel, "_worker_config", config, raising=False)
    monkeypatch.setattr(
        process_parallel, "_worker_prompt_sampler", FakePrompt(), raising=False
    )
    monkeypatch.setattr(
        process_parallel, "_worker_llm_ensemble", FakeEnsemble(), raising=False
    )
    monkeypatch.setattr(
        process_parallel, "_worker_evaluator", FakeEvaluator(), raising=False
    )
    monkeypatch.setattr(process_parallel, "_lazy_init_worker_components", lambda: None)
    parent = Program(
        id="parent-one", code="def solve(): return 1", metrics={"combined_score": 1.0}
    )
    snapshot = {
        "programs": {parent.id: parent.to_dict()},
        "artifacts": {},
        "current_island": 0,
        "islands": [[parent.id]],
        "sampling_island": 0,
        "feature_dimensions": [],
    }
    result = process_parallel._run_iteration_worker(1, snapshot, parent.id, [])
    assert result.outcome_type == "rejected"
    attempt = RejectedAttempt.from_dict(result.rejected_attempt_dict)
    assert attempt.code == expected_code
    assert attempt.parent_id == parent.id
    assert attempt.provider_usage == {"proposal_calls": 1, "prompt_tokens": 4}
    assert attempt.proposal_model == "example-model"
    if response:
        assert result.child_program_dict["metrics"] == {"combined_score": -0.4}
        assert json.loads(result.artifacts["rejection"])["code"] == "interface_invalid"
    else:
        assert result.child_program_dict is None
        assert attempt.candidate_hash is None


def test_append_failure_stops_iteration(
    tmp_path, sample_attempt: RejectedAttempt, monkeypatch
) -> None:
    """Propagate a durable-write error before another mutation can be scheduled."""
    config = Config()
    config.database.num_islands = 1
    config.evaluator.parallel_evaluations = 1
    controller = ProcessParallelController(
        config, "unused.py", object(), output_dir=str(tmp_path)
    )
    controller.executor = object()
    future = Future()
    future.set_result(
        process_parallel.SerializableResult(
            iteration=1,
            outcome_type="rejected",
            rejected_attempt_dict=sample_attempt.to_dict(),
        )
    )
    monkeypatch.setattr(controller, "_submit_iteration", lambda *_args: future)

    def fail_append(_attempt: RejectedAttempt) -> None:
        """Simulate a failed durable append."""
        raise OSError("disk full")

    monkeypatch.setattr(controller.attempt_store, "append", fail_append)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(controller.run_evolution(1, 1))


def test_adjudication_pauses_then_retries_exact_candidate(tmp_path) -> None:
    """Block proposal sampling until the saved code has been measured and admitted."""
    code = "def solve(): return 7"
    request = asdict(EvaluationNeedsAdjudication(request_id="judge-one"))
    candidate = {
        "id": "candidate-one",
        "code": code,
        "parent_id": "parent-one",
        "inspiration_ids": [],
        "target_island": 0,
        "changes_description": "change",
        "changes_summary": "change",
        "parent_island": 0,
        "generation": 1,
        "provider_usage": {"proposal_calls": 1},
        "proposal_model": "example-model",
    }
    store = AdjudicationStore(tmp_path)
    store.save_pending(request, candidate, 1, str(tmp_path / "checkpoint"))

    class FakeDatabase:
        """Store only the parent and a resumed candidate for this regression case."""

        def __init__(self) -> None:
            """Seed one admitted parent."""
            from openevolve.database import Program

            self.programs = {
                "parent-one": Program(
                    id="parent-one", code="pass", metrics={"combined_score": 1.0}
                )
            }

        def get(self, program_id: str):
            """Look up an admitted program."""
            return self.programs.get(program_id)

        def add(self, program, **_kwargs) -> None:
            """Admit the resolved candidate."""
            self.programs[program.id] = program

        def get_best_program(self):
            """Return the candidate after resolution."""
            return self.programs.get("candidate-one")

    class FakeEvaluator:
        """Record the exact code measured after an explicit retry disposition."""

        calls = []

        async def evaluate_program(self, program_code: str, program_id: str):
            """Accept the saved candidate with a deterministic score."""
            self.calls.append((program_code, program_id))
            return {"combined_score": 0.8}

        def get_pending_artifacts(self, _program_id: str):
            """Return no evaluator artifacts."""
            return None

    config = Config()
    config.language = "python"
    database = FakeDatabase()
    measurement = FakeEvaluator()
    controller = ProcessParallelController(
        config,
        "unused.py",
        database,
        output_dir=str(tmp_path),
        evaluator=measurement,
    )
    controller.executor = object()
    controller._submit_iteration = lambda *_args: pytest.fail(
        "proposal resampled before adjudication"
    )
    with pytest.raises(AdjudicationRequired):
        asyncio.run(controller.run_evolution(1, 0))
    assert database.get("candidate-one") is None
    resolve_adjudication(tmp_path, "judge-one", "retry")
    asyncio.run(
        controller.run_evolution(1, 0, checkpoint_callback=lambda _iteration: None)
    )
    assert measurement.calls == [(code, "candidate-one")]
    assert database.get("candidate-one").metrics["combined_score"] == 0.8
    assert store.load() is None


def test_adjudication_assign_and_terminate_are_explicit(tmp_path) -> None:
    """Persist categorical assignment or termination only after a valid user choice."""
    store = AdjudicationStore(tmp_path)
    candidate = {"id": "candidate-one", "code": "pass", "parent_id": "parent-one"}
    request = asdict(EvaluationNeedsAdjudication(request_id="judge-one"))
    store.save_pending(request, candidate, 1, str(tmp_path / "checkpoint"))
    with pytest.raises(ValueError, match="finite"):
        resolve_adjudication(
            tmp_path, "judge-one", "assign", outcome={"combined_score": float("nan")}
        )
    resolve_adjudication(
        tmp_path, "judge-one", "assign", outcome={"combined_score": 0.5}
    )
    assert store.load()["resolution"] == {
        "disposition": "assign",
        "outcome": {"kind": "accepted", "metrics": {"combined_score": 0.5}},
    }
    with pytest.raises(ValueError, match="already been resolved"):
        resolve_adjudication(tmp_path, "judge-one", "terminate")
    store.clear()
    store.save_pending(request, candidate, 1, str(tmp_path / "checkpoint"))
    resolve_adjudication(tmp_path, "judge-one", "terminate")
    assert store.load()["resolution"] == {"disposition": "terminate", "outcome": None}


def test_controller_auto_loads_pending_checkpoint_and_remeasures(tmp_path) -> None:
    """Resume a saved request through OpenEvolve without replacing its candidate."""
    from openevolve.database import Program

    initial = tmp_path / "initial.py"
    initial.write_text("def solve(): return 1\n")
    evaluation = tmp_path / "evaluate.py"
    evaluation.write_text("def evaluate(path):\n    return {'combined_score': 0.9}\n")
    config = Config()
    config.max_iterations = 0
    config.database.num_islands = 1
    config.evaluator.parallel_evaluations = 1
    config.evaluator.cascade_evaluation = False
    config.llm.models = [LLMModelConfig(name="unused-model", api_key="unused")]
    config.llm.evaluator_models = config.llm.models
    controller = OpenEvolve(
        str(initial), str(evaluation), config, output_dir=str(tmp_path / "run")
    )
    parent = Program(
        id="parent-one", code=initial.read_text(), metrics={"combined_score": 0.2}
    )
    controller.database.add(parent, iteration=0, target_island=0)
    controller._save_checkpoint(0)
    store = AdjudicationStore(controller.output_dir)
    candidate = {
        "id": "candidate-one",
        "code": "def solve(): return 7\n",
        "parent_id": parent.id,
        "inspiration_ids": [],
        "target_island": 0,
        "changes_description": "return 7",
        "changes_summary": "return 7",
        "parent_island": 0,
        "generation": 1,
        "provider_usage": {"proposal_calls": 1},
    }
    store.save_pending(
        asdict(EvaluationNeedsAdjudication(request_id="judge-one")),
        candidate,
        1,
        str(tmp_path / "run" / "checkpoints" / "checkpoint_0"),
    )
    resolve_adjudication(controller.output_dir, "judge-one", "retry")
    best = asyncio.run(controller.run(iterations=0))
    assert best is not None
    assert controller.database.get("candidate-one").code == candidate["code"]
    assert store.load() is None


def test_worker_adjudication_result_pauses_without_admission(
    tmp_path, monkeypatch
) -> None:
    """Persist one pending request and stop scheduling when a worker asks for adjudication."""
    config = Config()
    config.database.num_islands = 1
    config.evaluator.parallel_evaluations = 1

    class FakeDatabase:
        """Reject any candidate admission during an unresolved measurement."""

        last_iteration = 0

        def add(self, *_args, **_kwargs) -> None:
            """Fail if a pending candidate is admitted."""
            pytest.fail("unresolved candidate was admitted")

    controller = ProcessParallelController(
        config,
        "unused.py",
        FakeDatabase(),
        output_dir=str(tmp_path),
    )
    controller.executor = object()
    candidate = {
        "id": "candidate-one",
        "code": "pass",
        "parent_id": "parent-one",
    }
    future = Future()
    future.set_result(
        process_parallel.SerializableResult(
            iteration=1,
            outcome_type="needs_adjudication",
            pending_candidate_dict=candidate,
            operational_outcome_dict=asdict(
                EvaluationNeedsAdjudication(request_id="judge-one")
            ),
        )
    )
    submissions = []

    def submit(iteration: int, _island: int):
        """Allow only the initial worker submission."""
        submissions.append(iteration)
        if len(submissions) > 1:
            pytest.fail("new mutation submitted after adjudication")
        return future

    monkeypatch.setattr(controller, "_submit_iteration", submit)
    checkpoints = []

    def checkpoint(iteration: int) -> None:
        """Record the pause checkpoint callback."""
        checkpoints.append(iteration)

    with pytest.raises(AdjudicationRequired):
        asyncio.run(controller.run_evolution(1, 1, checkpoint_callback=checkpoint))
    assert submissions == [1]
    assert checkpoints == [0]
    assert controller.adjudication_store.load()["candidate"]["code"] == "pass"


def test_retryable_worker_result_preserves_candidate_for_resume(
    tmp_path, monkeypatch
) -> None:
    """Save the candidate after retry exhaustion instead of scheduling a replacement."""
    config = Config()
    config.database.num_islands = 1
    config.evaluator.parallel_evaluations = 1

    class FakeDatabase:
        """Expose a checkpoint iteration without admitting the pending candidate."""

        last_iteration = 0

    controller = ProcessParallelController(
        config,
        "unused.py",
        FakeDatabase(),
        output_dir=str(tmp_path),
    )
    controller.executor = object()
    future = Future()
    future.set_result(
        process_parallel.SerializableResult(
            iteration=1,
            outcome_type="retryable_failure",
            pending_candidate_dict={
                "id": "candidate-one",
                "code": "pass",
                "parent_id": "parent-one",
            },
            operational_outcome_dict={
                "code": "provider_rate_limit",
                "message": "Retry later",
            },
        )
    )
    submissions = []

    def submit(iteration: int, _island: int):
        """Fail if a second mutation is submitted after a retryable failure."""
        submissions.append(iteration)
        if len(submissions) > 1:
            pytest.fail("replacement mutation submitted")
        return future

    monkeypatch.setattr(controller, "_submit_iteration", submit)
    with pytest.raises(MeasurementRetryRequired):
        asyncio.run(
            controller.run_evolution(1, 1, checkpoint_callback=lambda _iteration: None)
        )
    pending = controller.adjudication_store.load()
    assert submissions == [1]
    assert pending["candidate"]["code"] == "pass"
    assert pending["request"]["allowed_dispositions"] == ["retry", "terminate"]
