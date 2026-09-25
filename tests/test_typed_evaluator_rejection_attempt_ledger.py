"""One-iteration regression for typed evaluator rejection and attempt recording.

Run from the OpenEvolve checkout with
``pytest -q tests/test_typed_evaluator_rejection_attempt_ledger.py``. A scripted proposal uses the
real example evaluator and the real worker/controller path, with no API calls.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openevolve.attempt_store import RunDirectoryAttemptStore
from openevolve.config import load_config
from openevolve.database import Program, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.process_parallel import ProcessParallelController
from openevolve.prompt.sampler import PromptSampler
from openevolve.rejection import RejectionCategory
from openevolve import process_parallel


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "typed_rejection"
REJECTED_PROGRAM = "def wrong_name(value: float) -> float:\n    return value\n"


class ScriptedModel:
    """Return one invalid-interface proposal without contacting a provider.

    Example::

        response, usage, model = await ScriptedModel().generate_with_receipt()
    """

    def __init__(self) -> None:
        """Start an empty call counter for the one-proposal assertion."""
        self.calls = 0

    async def generate_with_receipt(self, **_kwargs: Any) -> tuple[str, dict[str, int], str]:
        """Return candidate code and a fixed accounting receipt.

        Args:
            **_kwargs: Prompt arguments supplied by the worker.

        Returns:
            Invalid candidate source, usage counters, and model name.
        """
        self.calls += 1
        return (
            REJECTED_PROGRAM,
            {"proposal_calls": 1, "prompt_tokens": 4},
            "scripted-model",
        )


def test_one_rejected_candidate_persists_attempt_and_keeps_baseline_admission(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Exercise proposal, evaluator, ledger, and baseline artifact in one iteration.

    Args:
        tmp_path: Fresh output directory supplied by pytest.
        monkeypatch: Isolates scripted worker components from other tests.
    """
    config = load_config(str(EXAMPLE / "config.yaml"))
    config.max_iterations = 1
    config.language = "python"
    config.diff_based_evolution = False
    config.evaluator.cascade_evaluation = False
    config.evaluator.max_retries = 0
    config.database.num_islands = 1
    config.evaluator.parallel_evaluations = 1
    evaluator = Evaluator(config.evaluator, str(EXAMPLE / "evaluator.py"))

    seed_code = (EXAMPLE / "initial_program.py").read_text(encoding="utf-8")
    seed_metrics = asyncio.run(evaluator.evaluate_program(seed_code, "seed"))
    assert isinstance(seed_metrics, dict) and seed_metrics["combined_score"] > 0

    database = ProgramDatabase(config.database)
    database.add(
        Program(id="seed", code=seed_code, language="python", metrics=seed_metrics),
        target_island=0,
    )
    model = ScriptedModel()
    monkeypatch.setattr(process_parallel, "_worker_config", config, raising=False)
    monkeypatch.setattr(
        process_parallel,
        "_worker_prompt_sampler",
        PromptSampler(config.prompt),
        raising=False,
    )
    monkeypatch.setattr(process_parallel, "_worker_llm_ensemble", model, raising=False)
    monkeypatch.setattr(process_parallel, "_worker_evaluator", evaluator, raising=False)
    monkeypatch.setattr(process_parallel, "_lazy_init_worker_components", lambda: None)

    controller = ProcessParallelController(
        config, str(EXAMPLE / "evaluator.py"), database, output_dir=str(tmp_path)
    )
    # The worker calls asyncio.run internally, so it needs its own thread/event loop.
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller.executor = executor
        asyncio.run(controller.run_evolution(start_iteration=1, max_iterations=1))

    ledger = tmp_path / "attempts" / "rejected_attempts.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert model.calls == 1
    assert len(rows) == 1
    attempt = rows[0]
    assert attempt["iteration"] == 1
    assert attempt["parent_id"] == "seed"
    assert attempt["category"] == RejectionCategory.STATIC_INVALID.value
    assert attempt["code"] == "square_function_missing"
    assert attempt["stage"] == "evaluator"
    assert attempt["provider_usage"] == {"proposal_calls": 1, "prompt_tokens": 4}
    assert attempt["proposal_model"] == "scripted-model"
    assert REJECTED_PROGRAM not in ledger.read_text(encoding="utf-8")

    # The baseline policy records rejection while preserving low-score admission.
    children = [program for program in database.programs.values() if program.parent_id == "seed"]
    assert len(children) == 1
    assert children[0].metrics["combined_score"] == config.rejection_memory.penalty_score
    artifacts = database.get_artifacts(children[0].id)
    assert json.loads(artifacts["rejection"])["code"] == "square_function_missing"
    assert len(RunDirectoryAttemptStore(tmp_path).for_parent("seed", limit=10)) == 1
