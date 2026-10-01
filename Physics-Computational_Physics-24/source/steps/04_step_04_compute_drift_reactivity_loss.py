"""
Determine the reactivity that circulation of the fuel removes from the two-region system, which is the reactivity at which the circulating configuration is exactly critical.

Because part of every delayed group decays while it is outside the core, a circulating-fuel system needs a positive static reactivity merely to stay critical, and that offset is what a drift-blind analysis mistakes for excess reactivity. The offset is a property of the precursor rows of the drift matrix alone and therefore does not depend on the reactivity at which the matrix was assembled.

Returns
-------
float: the dimensionless reactivity removed by circulation, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_drift_reactivity_loss(kinetics_matrix: np.ndarray, delayed_fraction: float,
                                  decay_constants: np.ndarray,
                                  generation_time: float) -> float:
    """Return the reactivity removed by the circulation of the fuel.

    Parameters
    ----------
    kinetics_matrix : np.ndarray
        Square drift matrix of shape (1 + 2 * n_groups, 1 + 2 * n_groups) with
        the state ordering of sub-problem 03. Its neutron row and column may
        have been assembled at any reactivity.
    delayed_fraction : float
        Total delayed-neutron fraction, strictly between zero and one.
    decay_constants : np.ndarray
        One-dimensional array of the decay constants of the delayed groups in
        inverse seconds, in the same group order used to assemble the matrix.
    generation_time : float
        Prompt neutron generation time in seconds (generation_time > 0).

    Returns
    -------
    reactivity_loss : float
        Dimensionless reactivity removed by circulation, as a native Python
        float.

    Raises
    ------
    ValueError
        If ``kinetics_matrix`` is not a finite two-dimensional square array
        whose size is one more than twice the number of delay groups, if
        ``decay_constants`` is not a non-empty one-dimensional array of finite
        strictly positive entries, if ``delayed_fraction`` is not a finite
        number strictly between zero and one, if ``generation_time`` is not a
        finite number greater than zero, if the precursor block of the matrix is
        singular, or if the resulting per-neutron precursor inventory is not
        strictly positive in every group and region.
    """
    return reactivity_loss  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_drift_reactivity_loss(kinetics_matrix: np.ndarray,
                                          delayed_fraction: float,
                                          decay_constants: np.ndarray,
                                          generation_time: float) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _finite(name, value):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    delayed_fraction = _finite("delayed_fraction", delayed_fraction)
    generation_time = _finite("generation_time", generation_time)
    if not 0.0 < delayed_fraction < 1.0:
        raise ValueError("delayed_fraction must lie strictly between zero and one")
    if generation_time <= 0.0:
        raise ValueError("generation_time must be greater than zero")

    lambdas = np.asarray(decay_constants, dtype=float)
    if lambdas.ndim != 1 or lambdas.size < 1:
        raise ValueError("decay_constants must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(lambdas)):
        raise ValueError("decay_constants must contain only finite entries")
    if np.any(lambdas <= 0.0):
        raise ValueError("decay_constants entries must be strictly positive")

    matrix = np.asarray(kinetics_matrix, dtype=float)
    n_groups = lambdas.size
    size = 1 + 2 * n_groups
    if matrix.ndim != 2 or matrix.shape != (size, size):
        raise ValueError("kinetics_matrix must be square of size 1 + 2 * n_groups")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("kinetics_matrix must contain only finite entries")

    # Hold the neutron population at unity and let the precursor rows relax:
    # the resulting inventory is the stationary precursor distribution per
    # neutron, and it is what the neutron balance has to be closed against.
    precursor_block = matrix[1:, 1:]
    fission_source = matrix[1:, 0]
    if abs(float(np.linalg.det(precursor_block))) <= 0.0:
        raise ValueError("the precursor block of kinetics_matrix is singular")
    try:
        inventory = np.linalg.solve(precursor_block, -fission_source)
    except np.linalg.LinAlgError as exc:
        raise ValueError("the precursor block of kinetics_matrix is singular") from exc
    if not np.all(np.isfinite(inventory)) or np.any(inventory <= 0.0):
        raise ValueError("the per-neutron precursor inventory must be strictly positive")

    # Only the in-core inventory feeds neutrons back into the chain, so the
    # delayed fraction the chain actually sees falls short of the nominal one.
    returned = float(lambdas @ inventory[:n_groups])
    return float(delayed_fraction - generation_time * returned)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the six-group testbed loop (normal scenario) ---
        {
            "setup": """import numpy as np
delayed_fraction = 0.0065
group_fractions = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
decay_constants = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
generation_time = 1.0e-3
tau_core, tau_excore = 7.5, 12.5
n = decay_constants.size
A = np.zeros((1 + 2 * n, 1 + 2 * n))
A[0, 0] = -delayed_fraction / generation_time
A[0, 1:1 + n] = decay_constants
A[1:1 + n, 0] = delayed_fraction * group_fractions / generation_time
for j in range(n):
    A[1 + j, 1 + j] = -(decay_constants[j] + 1.0 / tau_core)
    A[1 + j, 1 + n + j] = 1.0 / tau_excore
    A[1 + n + j, 1 + n + j] = -(decay_constants[j] + 1.0 / tau_excore)
    A[1 + n + j, 1 + j] = 1.0 / tau_core
""",
            "call": "round(compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time), 12)",
            "gold_call": "round(_oracle_compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time), 12)",
        },
        # --- Valid: the same loop assembled at a strongly negative reactivity, on
        #     which the answer must not depend ---
        {
            "setup": """import numpy as np
delayed_fraction = 0.0065
group_fractions = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
decay_constants = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
generation_time = 1.0e-3
tau_core, tau_excore = 7.5, 12.5
reactivity = -0.4
n = decay_constants.size
A = np.zeros((1 + 2 * n, 1 + 2 * n))
A[0, 0] = (reactivity - delayed_fraction) / generation_time
A[0, 1:1 + n] = decay_constants
A[1:1 + n, 0] = delayed_fraction * group_fractions / generation_time
for j in range(n):
    A[1 + j, 1 + j] = -(decay_constants[j] + 1.0 / tau_core)
    A[1 + j, 1 + n + j] = 1.0 / tau_excore
    A[1 + n + j, 1 + n + j] = -(decay_constants[j] + 1.0 / tau_excore)
    A[1 + n + j, 1 + j] = 1.0 / tau_core
""",
            "call": "round(compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time), 12)",
            "gold_call": "round(_oracle_compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time), 12)",
        },
        # --- Boundary: an almost stagnant loop, where circulation removes very
        #     little of the delayed fraction ---
        {
            "setup": """import numpy as np
delayed_fraction = 0.0065
group_fractions = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
decay_constants = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
generation_time = 1.0e-3
tau_core, tau_excore = 1.0e5, 1.5e5
n = decay_constants.size
A = np.zeros((1 + 2 * n, 1 + 2 * n))
A[0, 0] = -delayed_fraction / generation_time
A[0, 1:1 + n] = decay_constants
A[1:1 + n, 0] = delayed_fraction * group_fractions / generation_time
for j in range(n):
    A[1 + j, 1 + j] = -(decay_constants[j] + 1.0 / tau_core)
    A[1 + j, 1 + n + j] = 1.0 / tau_excore
    A[1 + n + j, 1 + n + j] = -(decay_constants[j] + 1.0 / tau_excore)
    A[1 + n + j, 1 + j] = 1.0 / tau_core
""",
            "call": "round(compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time), 12)",
            "gold_call": "round(_oracle_compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time), 12)",
        },
        # --- Edge: a single very long lived group in a fast loop, where almost
        #     the whole delayed fraction is lost to drift ---
        {
            "setup": """import numpy as np
delayed_fraction = 0.008
group_fractions = np.array([1.0])
decay_constants = np.array([0.005])
generation_time = 5.0e-4
tau_core, tau_excore = 0.5, 3.0
n = decay_constants.size
A = np.zeros((1 + 2 * n, 1 + 2 * n))
A[0, 0] = -delayed_fraction / generation_time
A[0, 1:1 + n] = decay_constants
A[1:1 + n, 0] = delayed_fraction * group_fractions / generation_time
for j in range(n):
    A[1 + j, 1 + j] = -(decay_constants[j] + 1.0 / tau_core)
    A[1 + j, 1 + n + j] = 1.0 / tau_excore
    A[1 + n + j, 1 + n + j] = -(decay_constants[j] + 1.0 / tau_excore)
    A[1 + n + j, 1 + j] = 1.0 / tau_core
""",
            "call": "round(compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time), 12)",
            "gold_call": "round(_oracle_compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time), 12)",
        },
        # --- Invalid: a matrix whose size does not match the group count ---
        {
            "setup": """import numpy as np
delayed_fraction = 0.0065
decay_constants = np.array([0.0124, 0.0305, 0.111])
generation_time = 1.0e-3
A = np.eye(5) * -1.0
def run_model():
    try:
        compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a precursor block of zeros, which admits no inventory ---
        {
            "setup": """import numpy as np
delayed_fraction = 0.0065
decay_constants = np.array([0.02, 0.5])
generation_time = 1.0e-3
A = np.zeros((5, 5))
A[0, 0] = -6.5
A[0, 1:3] = decay_constants
A[1:3, 0] = np.array([2.6, 3.9])
def run_model():
    try:
        compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_drift_reactivity_loss(A, delayed_fraction, decay_constants, generation_time)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
