"""Seed program for the typed-rejection example.

Run from the OpenEvolve repository root with the command in this directory's
README.md. OpenEvolve may edit the single EVOLVE-BLOCK below.
"""


# EVOLVE-BLOCK-START
def square(value: float) -> float:
    """Estimate the square of a number.

    Args:
        value: Input number.

    Returns:
        An initial, deliberately inaccurate estimate.
    """
    return value + value


# EVOLVE-BLOCK-END
