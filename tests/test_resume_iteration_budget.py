"""Check that checkpoint and adjudication resumes retain the original child budget.

Run ``pytest -q tests/test_resume_iteration_budget.py`` from the OpenEvolve
checkout. All proposal work is mocked or uses completed in-memory futures.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import ProcessParallelController, SerializableResult


@pytest.mark.parametrize(
    ("checkpoint_iteration", "expected_start", "expected_count"),
    [(None, 1, 24), (0, 1, 24), (3, 4, 21), (23, 24, 1), (24, None, 0)],
)
def test_run_respects_total_child_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_iteration: int | None,
    expected_start: int | None,
    expected_count: int,
) -> None:
    """Resume only the missing child iterations, even after a rejected child.

    Args:
        tmp_path: Fresh directory for the seed, evaluator, and saved checkpoint.
        monkeypatch: Replaces evaluation and process startup with offline mocks.
        checkpoint_iteration: Last completed child in the saved database, if any.
        expected_start: First child to schedule, or None when the budget is spent.
        expected_count: Number of children still available under the target.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    seed_path = tmp_path / "seed.py"
    seed_path.write_text("def solve(value):\n    return value\n", encoding="utf-8")
    evaluator_path = tmp_path / "evaluator.py"
    evaluator_path.write_text("def evaluate(path):\n    return {'combined_score': 0.5}\n")
    config = Config()
    config.database.in_memory = True
    config.checkpoint_interval = 100
    evaluator = MagicMock()
    evaluator.evaluate_program = AsyncMock(return_value={"combined_score": 0.5})
    monkeypatch.setattr("openevolve.controller.Evaluator", lambda *_args, **_kwargs: evaluator)

    checkpoint_path = None
    if checkpoint_iteration is not None:
        database = ProgramDatabase(config.database)
        database.add(
            Program(
                id="seed",
                code=seed_path.read_text(encoding="utf-8"),
                language="python",
                metrics={"combined_score": 0.5},
                iteration_found=0,
            )
        )
        checkpoint_path = tmp_path / f"checkpoint_{checkpoint_iteration}"
        database.save(str(checkpoint_path), checkpoint_iteration)

    parallel = MagicMock()
    parallel.run_evolution = AsyncMock()
    parallel.shutdown_event.is_set.return_value = False
    parallel.early_stopping_triggered = False
    factory = MagicMock(return_value=parallel)
    monkeypatch.setattr("openevolve.controller.ProcessParallelController", factory)
    controller = OpenEvolve(
        str(seed_path), str(evaluator_path), config, output_dir=str(tmp_path / "run")
    )

    resume_path = str(checkpoint_path) if checkpoint_path else None
    result = asyncio.run(controller.run(iterations=24, checkpoint_path=resume_path))

    assert result is not None
    if expected_start is None:
        factory.assert_not_called()
        parallel.run_evolution.assert_not_called()
    else:
        parallel.run_evolution.assert_awaited_once()
        args = parallel.run_evolution.await_args.args
        assert args == (expected_start, expected_count, None)
    assert evaluator.evaluate_program.await_count == (checkpoint_iteration is None)


def test_resolved_pending_candidate_consumes_its_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not replace adjudicated child 4 with an unbudgeted child 25.

    Args:
        tmp_path: Fresh directory for run-owned state.
        monkeypatch: Replaces pending resolution and worker submission.
    """
    config = Config()
    config.database.in_memory = True
    config.database.num_islands = 1
    config.evaluator.parallel_evaluations = 1
    database = ProgramDatabase(config.database)
    database.add(
        Program(id="seed", code="pass", language="python", metrics={"combined_score": 0.5})
    )
    controller = ProcessParallelController(
        config, str(tmp_path / "evaluator.py"), database, output_dir=str(tmp_path)
    )
    controller.executor = object()
    monkeypatch.setattr(controller, "_resume_pending", AsyncMock(return_value=4))
    submitted: list[int] = []

    def submit(iteration: int, _island: int) -> Future[SerializableResult]:
        """Return an empty completed proposal without contacting a provider."""
        submitted.append(iteration)
        future: Future[SerializableResult] = Future()
        future.set_result(SerializableResult(iteration=iteration, outcome_type="empty"))
        return future

    monkeypatch.setattr(controller, "_submit_iteration", submit)
    asyncio.run(controller.run_evolution(start_iteration=4, max_iterations=21))
    assert submitted == list(range(5, 25))
