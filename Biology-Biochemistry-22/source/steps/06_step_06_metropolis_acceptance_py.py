"""
Calculate the discretization-aware Metropolis acceptance probability for a proposed protein sequence and return a numerical accept-or-reject decision.

Calculate the discretization-aware Metropolis acceptance probability for a proposed protein sequence and return a numerical accept-or-reject decision. The source procedure performs this correction after every Hamiltonian update because mapping the updated continuous position to a one-hot sequence can change its potential energy. For the current update, compare the Hamiltonian of the continuous position and momentum before the update with the Hamiltonian formed by the discretized post-update proposal and the updated momentum. The acceptance probability is the minimum of one and the exponential of the pre-update Hamiltonian minus the discretized post-update Hamiltonian. Accept the proposal when the supplied uniform draw is strictly smaller than this probability. Rejection prevents that discrete proposal from being collected, but it does not reverse or terminate the underlying continuous trajectory.

Returns
-------
tuple[float, int], the Metropolis acceptance probability and a binary decision where 1 means accepted and 0 means rejected
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def metropolis_acceptance(
    initial_potential: float,
    proposed_potential: float,
    initial_momentum: np.ndarray,
    final_momentum: np.ndarray,
    uniform_draw: float,
) -> tuple[float, int]:
    """Calculate one update's Metropolis probability and decision.

    Parameters
    ----------
    initial_potential : float
        Potential energy of the continuous state before the current update.
    proposed_potential : float
        Potential energy of the discretized post-update proposal.
    initial_momentum : np.ndarray
        Momentum array before the current Hamiltonian update.
    final_momentum : np.ndarray
        Momentum array after the current Hamiltonian update, with matching shape.
    uniform_draw : float
        Uniform variate for this update in the half-open interval ``[0, 1)``.

    Returns
    -------
    result : tuple[float, int]
        Acceptance probability and binary decision, where 1 means accepted.

    Raises
    ------
    ValueError
        If either potential or ``uniform_draw`` is not a finite scalar; if the
        momentum arrays have different shapes; if either momentum array is
        empty or contains non-finite values; or if ``uniform_draw`` is outside
        the interval ``[0, 1)``.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_metropolis_acceptance(
    initial_potential: float,
    proposed_potential: float,
    initial_momentum: np.ndarray,
    final_momentum: np.ndarray,
    uniform_draw: float,
) -> tuple[float, int]:
    """Return the stable Metropolis probability and decision."""

    initial_momentum_array = np.asarray(
        initial_momentum,
        dtype=np.float64,
    )

    final_momentum_array = np.asarray(
        final_momentum,
        dtype=np.float64,
    )

    scalar_values = (
        initial_potential,
        proposed_potential,
        uniform_draw,
    )

    if not all(
        np.isscalar(value) and np.isfinite(value)
        for value in scalar_values
    ):
        raise ValueError(
            "potential energies and uniform_draw must be finite scalars"
        )

    if (
        initial_momentum_array.shape
        != final_momentum_array.shape
    ):
        raise ValueError(
            "initial_momentum and final_momentum must have matching shapes"
        )

    if (
        initial_momentum_array.size == 0
        or not np.all(
            np.isfinite(initial_momentum_array)
        )
    ):
        raise ValueError(
            "initial_momentum must be non-empty and finite"
        )

    if not np.all(
        np.isfinite(final_momentum_array)
    ):
        raise ValueError(
            "final_momentum must contain finite values"
        )

    if not 0.0 <= float(uniform_draw) < 1.0:
        raise ValueError(
            "uniform_draw must lie in [0, 1)"
        )

    initial_kinetic = (
        0.5
        * np.sum(initial_momentum_array**2)
    )

    final_kinetic = (
        0.5
        * np.sum(final_momentum_array**2)
    )

    log_ratio = (
        float(initial_potential)
        + initial_kinetic
        - float(proposed_potential)
        - final_kinetic
    )

    probability = (
        1.0
        if log_ratio >= 0.0
        else float(np.exp(log_ratio))
    )

    accepted = int(
        float(uniform_draw) < probability
    )

    return probability, accepted

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return accepted, rejected, and probability-one tests."""
    return [
        {
            "setup": """import numpy as np
initial_potential = 0.058462278476133105
proposed_potential = 0.5629153335603464
initial_momentum = np.array([
    [-0.7901525000, -2.0346254818, 0.6033017469],
    [0.7442945299, -0.3096867999, 0.3673213729],
    [1.7103942915, 1.0607978401, 0.7076390208],
])
final_momentum = np.array([
    [0.3803222950, -2.2300194621, -0.5609133016],
    [-0.7366985570, 0.5099835120, -0.0304002259],
    [1.8004504986, 0.9969539751, 0.3082515997],
])
uniform_draw = 0.22452433073513123
""",
            "call": (
                "metropolis_acceptance("
                "initial_potential, proposed_potential, "
                "initial_momentum, final_momentum, "
                "uniform_draw)"
            ),
            "gold_call": (
                "_oracle_metropolis_acceptance("
                "initial_potential, proposed_potential, "
                "initial_momentum, final_momentum, "
                "uniform_draw)"
            ),
        },
        {
            "setup": """import numpy as np
initial_potential = 0.08277152245355257
proposed_potential = 2.2155195231797546
initial_momentum = np.array([
    [1.8117203538, -0.7290535593, -1.0856264862],
    [-0.4019110540, -1.1525924203, 1.5082245980],
    [0.8799003751, 0.6376331108, 1.4098711969],
])
final_momentum = np.array([
    [-1.9132710827, -0.7039197978, 0.9817374805],
    [0.0598004913, -0.9481679373, -1.4507826771],
    [0.9764979763, -0.4476376400, 1.3969381613],
])
uniform_draw = 0.914502769923609
""",
            "call": (
                "metropolis_acceptance("
                "initial_potential, proposed_potential, "
                "initial_momentum, final_momentum, "
                "uniform_draw)"
            ),
            "gold_call": (
                "_oracle_metropolis_acceptance("
                "initial_potential, proposed_potential, "
                "initial_momentum, final_momentum, "
                "uniform_draw)"
            ),
        },
        {
            "setup": """import numpy as np
initial_potential = 1.0
proposed_potential = 1.0
initial_momentum = np.zeros((1, 1), dtype=float)
final_momentum = np.zeros((1, 1), dtype=float)
uniform_draw = np.nextafter(1.0, 0.0)
""",
            "call": (
                "metropolis_acceptance("
                "initial_potential, proposed_potential, "
                "initial_momentum, final_momentum, "
                "uniform_draw)"
            ),
            "gold_call": (
                "_oracle_metropolis_acceptance("
                "initial_potential, proposed_potential, "
                "initial_momentum, final_momentum, "
                "uniform_draw)"
            ),
        },
    ]
