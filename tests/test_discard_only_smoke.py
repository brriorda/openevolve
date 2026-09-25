"""Exercise one discarded evaluator rejection through OpenEvolve's real controller.

Run from the OpenEvolve checkout with
``pytest -q tests/test_discard_only_smoke.py``. The scripted proposal uses the
typed-rejection example evaluator and makes no provider calls.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from openevolve import process_parallel
from openevolve.attempt_store import RunDirectoryAttemptStore
from openevolve.config import Config, load_config
from openevolve.database import Program, ProgramAdmission, ProgramDatabase
from openevolve.evaluator import Evaluator
from openevolve.process_parallel import ProcessParallelController
from openevolve.prompt.sampler import PromptSampler
from openevolve.rejection import RejectionCategory
from openevolve.rejection_policy import resolve_rejection_policy


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "typed_rejection"
REJECTED_PROGRAM = "def wrong_name(value: float) -> float:\n    return value\n"


class ScriptedModel:
    """Generate one invalid-interface proposal without contacting a provider.

    Example::

        response, usage, model = await ScriptedModel().generate_with_receipt()
    """

    def __init__(self, response: str = REJECTED_PROGRAM) -> None:
        """Initialize the response and proposal call counter.

        Args:
            response: Candidate source returned for each scripted proposal.
        """
        self.calls = 0
        self.response = response

    async def generate_with_receipt(self, **_kwargs: Any) -> tuple[str, dict[str, int], str]:
        """Return candidate code and a deterministic provider receipt.

        Args:
            **_kwargs: Prompt arguments supplied by the worker.

        Returns:
            Invalid candidate source, usage counters, and model name.
        """
        self.calls += 1
        return self.response, {"proposal_calls": 1, "prompt_tokens": 4}, "scripted-model"


def test_discard_only_records_attempt_without_population_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run one rejected proposal and inspect every population search structure.

    Args:
        tmp_path: Fresh directory for the attempt ledger and checkpoint.
        monkeypatch: Replaces provider and worker initialization with scripted values.
    """
    config = load_config(str(EXAMPLE / "discard_only.yaml"))
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
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller.executor = executor
        asyncio.run(controller.run_evolution(start_iteration=1, max_iterations=1))

    attempts = RunDirectoryAttemptStore(tmp_path).for_parent("seed", limit=10)
    assert model.calls == 1
    assert len(attempts) == 1
    assert attempts[0].code == "square_function_missing"
    assert attempts[0].stage == "evaluator"
    assert attempts[0].provider_usage == {"proposal_calls": 1, "prompt_tokens": 4}
    assert REJECTED_PROGRAM not in (tmp_path / "attempts" / "rejected_attempts.jsonl").read_text(
        encoding="utf-8"
    )
    assert database.last_iteration == 1
    assert set(database.programs) == {"seed"}
    assert database.islands == [{"seed"}]
    assert all(set(cells.values()) <= {"seed"} for cells in database.island_feature_maps)
    assert database.archive <= {"seed"}
    assert database.best_program_id == "seed"
    assert database.sample()[0].id == "seed"
    assert all(program.id == "seed" for program in database.sample()[1])
    assert set(controller._create_database_snapshot()["programs"]) == {"seed"}
    assert database.get_artifacts("seed") == {}

    checkpoint = tmp_path / "checkpoint"
    database.save(str(checkpoint), iteration=database.last_iteration)
    assert {path.stem for path in (checkpoint / "programs").glob("*.json")} == {"seed"}
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["last_iteration"] == 1


def test_discard_only_policy_and_database_guard() -> None:
    """Resolve categorical exclusion and refuse a rejected database admission."""
    policy = resolve_rejection_policy("discard_only")
    assert policy.record_attempt
    assert not policy.admit_rejected_program
    assert not policy.attach_rejection_artifact
    assert not policy.immediate_repair
    assert not policy.deferred_parent_delivery
    assert not policy.global_delivery

    config = Config()
    database = ProgramDatabase(config.database)
    program = Program(id="rejected", code="pass", metrics={"combined_score": 1.0})
    with pytest.raises(ValueError, match="discarded candidate"):
        database.add(program, iteration=1, admission=ProgramAdmission.DISCARDED)
    assert database.programs == {}
    assert database.last_iteration == 0
    assert not any(database.islands)
    assert not database.archive
    assert database.best_program_id is None


def test_duplicate_program_completion_cannot_replace_admitted_state() -> None:
    """Refuse a repeated candidate ID before it can change population state."""
    database = ProgramDatabase(Config().database)
    first = Program(id="candidate", code="pass", metrics={"combined_score": 0.5})
    assert database.add(first, iteration=1) == "candidate"
    original_islands = [set(island) for island in database.islands]
    original_feature_maps = [dict(cells) for cells in database.island_feature_maps]
    original_archive = set(database.archive)
    original_best = database.best_program_id

    duplicate = Program(id="candidate", code="raise RuntimeError", metrics={"combined_score": 1.0})
    with pytest.raises(ValueError, match="program ID already exists"):
        database.add(duplicate, iteration=2)

    assert database.get("candidate") is first
    assert database.last_iteration == 1
    assert database.islands == original_islands
    assert database.island_feature_maps == original_feature_maps
    assert database.archive == original_archive
    assert database.best_program_id == original_best


def test_zero_score_acceptance_remains_a_program(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admit an explicitly accepted score-zero child under discard-only policy.

    Args:
        tmp_path: Fresh directory for the controller output.
        monkeypatch: Installs a deterministic evaluator and proposal model.
    """

    class ZeroScoreEvaluator:
        """Accept every candidate with a real zero fitness score.

        Example::

            metrics = await ZeroScoreEvaluator().evaluate_program("pass", "candidate")
        """

        async def evaluate_program(self, _code: str, _program_id: str) -> dict[str, float]:
            """Return an accepted zero score for the proposed program."""
            return {"combined_score": 0.0}

        def get_pending_artifacts(self, _program_id: str) -> None:
            """Return no program-owned artifacts."""
            return None

    config = load_config(str(EXAMPLE / "discard_only.yaml"))
    config.language = "python"
    config.diff_based_evolution = False
    config.database.num_islands = 1
    config.evaluator.parallel_evaluations = 1
    database = ProgramDatabase(config.database)
    database.add(
        Program(id="seed", code="def square(value): return value", metrics={"combined_score": 0.5}),
        target_island=0,
    )
    model = ScriptedModel("def square(value):\n    return 0.0\n")
    monkeypatch.setattr(process_parallel, "_worker_config", config, raising=False)
    monkeypatch.setattr(
        process_parallel,
        "_worker_prompt_sampler",
        PromptSampler(config.prompt),
        raising=False,
    )
    monkeypatch.setattr(process_parallel, "_worker_llm_ensemble", model, raising=False)
    monkeypatch.setattr(process_parallel, "_worker_evaluator", ZeroScoreEvaluator(), raising=False)
    monkeypatch.setattr(process_parallel, "_lazy_init_worker_components", lambda: None)

    controller = ProcessParallelController(config, "unused.py", database, output_dir=str(tmp_path))
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller.executor = executor
        asyncio.run(controller.run_evolution(start_iteration=1, max_iterations=1))

    children = [program for program in database.programs.values() if program.parent_id == "seed"]
    assert model.calls == 1
    assert len(children) == 1
    assert children[0].metrics == {"combined_score": 0.0}
    assert children[0].id in database.islands[0]
    assert database.last_iteration == 1
    assert not (tmp_path / "attempts" / "rejected_attempts.jsonl").exists()


def test_novelty_refusal_uses_attempt_path_without_population_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discard a measured candidate when database novelty refuses admission.

    Args:
        tmp_path: Fresh output directory for the ledger and checkpoint.
        monkeypatch: Replaces the provider, evaluator, and novelty decision.
    """

    class AcceptingEvaluator:
        """Return an accepted measurement so novelty is the only refusal.

        Example::

            metrics = await AcceptingEvaluator().evaluate_program("pass", "candidate")
        """

        async def evaluate_program(self, _code: str, _program_id: str) -> dict[str, float]:
            """Return a positive candidate fitness score."""
            return {"combined_score": 0.8}

        def get_pending_artifacts(self, _program_id: str) -> dict[str, str]:
            """Return artifacts that must not attach to a refused program."""
            return {"evaluation": "candidate measurement"}

    config = load_config(str(EXAMPLE / "discard_only.yaml"))
    config.language = "python"
    config.diff_based_evolution = False
    config.database.num_islands = 1
    config.evaluator.parallel_evaluations = 1
    database = ProgramDatabase(config.database)
    database.add(
        Program(id="seed", code="def square(value): return value", metrics={"combined_score": 0.5}),
        target_island=0,
    )
    monkeypatch.setattr(database, "_is_novel", lambda _program, _island: False)
    model = ScriptedModel("def square(value):\n    return value * value\n")
    monkeypatch.setattr(process_parallel, "_worker_config", config, raising=False)
    monkeypatch.setattr(
        process_parallel, "_worker_prompt_sampler", PromptSampler(config.prompt), raising=False
    )
    monkeypatch.setattr(process_parallel, "_worker_llm_ensemble", model, raising=False)
    monkeypatch.setattr(process_parallel, "_worker_evaluator", AcceptingEvaluator(), raising=False)
    monkeypatch.setattr(process_parallel, "_lazy_init_worker_components", lambda: None)

    checkpoints: list[int] = []
    config.checkpoint_interval = 1
    controller = ProcessParallelController(config, "unused.py", database, output_dir=str(tmp_path))
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller.executor = executor
        asyncio.run(
            controller.run_evolution(
                start_iteration=1, max_iterations=1, checkpoint_callback=checkpoints.append
            )
        )

    attempts = RunDirectoryAttemptStore(tmp_path).for_parent("seed", limit=10)
    assert model.calls == 1
    assert len(attempts) == 1
    assert attempts[0].category is RejectionCategory.NOVELTY_REJECTED
    assert attempts[0].code == "novelty_check_failed"
    assert attempts[0].stage == "admission"
    assert attempts[0].provider_usage == {"proposal_calls": 1, "prompt_tokens": 4}
    assert database.last_iteration == 1
    assert checkpoints == [1]
    assert set(database.programs) == {"seed"}
    assert database.islands == [{"seed"}]
    assert all(set(cells.values()) <= {"seed"} for cells in database.island_feature_maps)
    assert database.archive <= {"seed"}
    assert database.best_program_id == "seed"
    assert database.sample()[0].id == "seed"
    assert set(controller._create_database_snapshot()["programs"]) == {"seed"}
    assert database.get_artifacts("seed") == {}
