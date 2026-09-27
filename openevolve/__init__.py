"""
OpenEvolve: An open-source implementation of AlphaEvolve
"""

from openevolve._version import __version__
from openevolve.config import Config
from openevolve.controller import OpenEvolve
from openevolve.api import (
    run_evolution,
    evolve_function,
    evolve_algorithm,
    evolve_code,
    EvolutionResult,
)
from openevolve.rejection import (
    CandidateAccepted,
    CandidateRejected,
    EvaluationNeedsAdjudication,
    EvaluationRetryableFailure,
    RejectionCategory,
    RejectionDisposition,
    RunFatalFailure,
    reject_candidate,
)
from openevolve.adjudication import AdjudicationRequired, MeasurementRetryRequired, resolve_adjudication

__all__ = [
    "Config",
    "OpenEvolve",
    "__version__",
    "run_evolution",
    "evolve_function",
    "evolve_algorithm",
    "evolve_code",
    "EvolutionResult",
    "CandidateAccepted",
    "CandidateRejected",
    "EvaluationNeedsAdjudication",
    "EvaluationRetryableFailure",
    "RejectionCategory",
    "RejectionDisposition",
    "RunFatalFailure",
    "reject_candidate",
    "AdjudicationRequired",
    "MeasurementRetryRequired",
    "resolve_adjudication",
]
