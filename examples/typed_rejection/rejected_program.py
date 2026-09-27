"""Invalid fixture for the typed-rejection example.

Pass this file to ``evaluator.evaluate`` as shown in README.md to see a
deterministic ``CandidateRejected`` result without calling an LLM.
"""


def wrong_name(value: float) -> float:
    """Return an input unchanged to demonstrate a missing-interface rejection.

    Args:
        value: Input number.

    Returns:
        The unchanged input.
    """
    return value
