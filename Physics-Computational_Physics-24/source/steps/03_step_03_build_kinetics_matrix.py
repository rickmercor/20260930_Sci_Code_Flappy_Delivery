"""
Assemble the drift matrix of the two-region perfectly-mixed point-kinetic model, whose state carries the neutron population together with one in-core and one ex-core precursor population per delayed group.

Tracking the precursor inventory of each delayed group separately in the core and in the ex-core loop replaces the retarded term of the classical circulating-fuel model by a pair of ordinary differential equations coupled through the two mean residence times. Only precursors that decay inside the core return a neutron to the chain, which is what makes the resulting matrix asymmetric between the two regions.

Returns
-------
np.ndarray of shape (1 + 2 * n_groups, 1 + 2 * n_groups), float: the drift matrix in inverse seconds.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_kinetics_matrix(reactivity: float, delayed_fraction: float,
                          group_fractions: np.ndarray, decay_constants: np.ndarray,
                          generation_time: float, tau_core: float,
                          tau_excore: float) -> np.ndarray:
    """Assemble the drift matrix of the two-region point-kinetic model.

    The state vector is ordered with the neutron population first, then the
    in-core precursor population of every delayed group in the order supplied,
    then the ex-core precursor population of every delayed group in the same
    order.

    Parameters
    ----------
    reactivity : float
        Dimensionless reactivity of the configuration; may be negative.
    delayed_fraction : float
        Total delayed-neutron fraction, strictly between zero and one.
    group_fractions : np.ndarray
        One-dimensional array of the delayed-group shares of the total delayed
        fraction; entries are strictly positive and sum to one.
    decay_constants : np.ndarray
        One-dimensional array of the same length holding the decay constant of
        each delayed group in inverse seconds; entries are strictly positive.
    generation_time : float
        Prompt neutron generation time in seconds (generation_time > 0).
    tau_core : float
        Mean residence time of a precursor in the core in seconds
        (tau_core > 0).
    tau_excore : float
        Mean residence time of a precursor in the ex-core loop in seconds
        (tau_excore > 0).

    Returns
    -------
    kinetics_matrix : np.ndarray
        Array of shape (1 + 2 * n_groups, 1 + 2 * n_groups) in inverse seconds,
        the drift matrix acting on the state vector described above.

    Raises
    ------
    ValueError
        If ``reactivity``, ``generation_time``, ``tau_core`` or ``tau_excore``
        is not a finite number, if ``generation_time``, ``tau_core`` or
        ``tau_excore`` is not greater than zero, if ``delayed_fraction`` is not
        a finite number strictly between zero and one, if ``group_fractions``
        and ``decay_constants`` are not non-empty one-dimensional finite arrays
        of equal length, if any entry of either is not strictly positive, or if
        the ``group_fractions`` entries do not sum to one within 1e-10.
    """
    return kinetics_matrix  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_kinetics_matrix(reactivity: float, delayed_fraction: float,
                                  group_fractions: np.ndarray,
                                  decay_constants: np.ndarray, generation_time: float,
                                  tau_core: float, tau_excore: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _finite(name, value):
        if not (isinstance(value, (int, float, np.floating, np.integer))
                and not isinstance(value, bool) and np.isfinite(value)):
            raise ValueError(f"{name} must be a finite number")
        return float(value)

    reactivity = _finite("reactivity", reactivity)
    delayed_fraction = _finite("delayed_fraction", delayed_fraction)
    generation_time = _finite("generation_time", generation_time)
    tau_core = _finite("tau_core", tau_core)
    tau_excore = _finite("tau_excore", tau_excore)
    if not 0.0 < delayed_fraction < 1.0:
        raise ValueError("delayed_fraction must lie strictly between zero and one")
    for name, value in (("generation_time", generation_time), ("tau_core", tau_core),
                        ("tau_excore", tau_excore)):
        if value <= 0.0:
            raise ValueError(f"{name} must be greater than zero")

    shares = np.asarray(group_fractions, dtype=float)
    lambdas = np.asarray(decay_constants, dtype=float)
    for name, array in (("group_fractions", shares), ("decay_constants", lambdas)):
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite entries")
        if np.any(array <= 0.0):
            raise ValueError(f"{name} entries must be strictly positive")
    if shares.size != lambdas.size:
        raise ValueError("group_fractions and decay_constants must have equal length")
    if abs(float(shares.sum()) - 1.0) > 1e-10:
        raise ValueError("group_fractions entries must sum to one")

    n_groups = shares.size
    size = 1 + 2 * n_groups
    matrix = np.zeros((size, size), dtype=float)
    core = slice(1, 1 + n_groups)
    excore = slice(1 + n_groups, size)

    # Neutron balance: prompt multiplication net of the delayed fraction, fed by
    # the decay of the precursors that are inside the core at that instant.
    matrix[0, 0] = (reactivity - delayed_fraction) / generation_time
    matrix[0, core] = lambdas

    # In-core precursors: born from fission, removed by decay and by being
    # carried out of the core, replenished by the returning ex-core inventory.
    matrix[core, 0] = delayed_fraction * shares / generation_time
    core_diagonal = -(lambdas + 1.0 / tau_core)
    excore_diagonal = -(lambdas + 1.0 / tau_excore)
    for j in range(n_groups):
        matrix[1 + j, 1 + j] = core_diagonal[j]
        matrix[1 + j, 1 + n_groups + j] = 1.0 / tau_excore
        # Ex-core precursors: no fission source out there, and their decay
        # releases a neutron outside the chain, so it only removes inventory.
        matrix[1 + n_groups + j, 1 + n_groups + j] = excore_diagonal[j]
        matrix[1 + n_groups + j, 1 + j] = 1.0 / tau_core

    return matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the six-group testbed at its operating reactivity (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
reactivity = -0.022820635778950265
delayed_fraction = 0.0065
group_fractions = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
decay_constants = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
generation_time = 1.0e-3
tau_core = 7.5
tau_excore = 12.5
""",
            "call": "sig(build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore), 1.0e2)",
            "gold_call": "sig(_oracle_build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore), 1.0e2)",
        },
        # --- Valid: the same data at zero reactivity, the reference assembly from
        #     which the circulating critical state is located ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
reactivity = 0.0
delayed_fraction = 0.0065
group_fractions = np.array([0.033, 0.219, 0.196, 0.395, 0.115, 0.042])
decay_constants = np.array([0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01])
generation_time = 1.0e-3
tau_core = 7.5
tau_excore = 12.5
""",
            "call": "sig(build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore), 1.0e2)",
            "gold_call": "sig(_oracle_build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore), 1.0e2)",
        },
        # --- Boundary: a single delayed group and equal residence times, which
        #     makes the two regions symmetric apart from the decay feedback ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
reactivity = -0.05
delayed_fraction = 0.0075
group_fractions = np.array([1.0])
decay_constants = np.array([0.08])
generation_time = 2.0e-4
tau_core = 4.0
tau_excore = 4.0
""",
            "call": "sig(build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore), 1.0e2)",
            "gold_call": "sig(_oracle_build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore), 1.0e2)",
        },
        # --- Edge: a nearly static loop, where the residence times are far longer
        #     than every precursor lifetime ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float)
    metadata = np.array([float(a.ndim), *(float(n) for n in a.shape)])
    return np.concatenate((metadata, a.ravel()))
reactivity = 0.001
delayed_fraction = 0.0065
group_fractions = np.array([0.4, 0.6])
decay_constants = np.array([0.02, 0.5])
generation_time = 1.0e-3
tau_core = 5.0e3
tau_excore = 7.0e3
""",
            "call": "sig(build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore), 1.0e1)",
            "gold_call": "sig(_oracle_build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore), 1.0e1)",
        },
        # --- Invalid: group shares that do not sum to one ---
        {
            "setup": """import numpy as np
reactivity = -0.02
delayed_fraction = 0.0065
group_fractions = np.array([0.2, 0.3, 0.4])
decay_constants = np.array([0.02, 0.1, 1.0])
generation_time = 1.0e-3
tau_core = 7.5
tau_excore = 12.5
def run_model():
    try:
        build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-positive core residence time ---
        {
            "setup": """import numpy as np
reactivity = -0.02
delayed_fraction = 0.0065
group_fractions = np.array([0.4, 0.6])
decay_constants = np.array([0.02, 0.5])
generation_time = 1.0e-3
tau_core = 0.0
tau_excore = 12.5
def run_model():
    try:
        build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_kinetics_matrix(reactivity, delayed_fraction, group_fractions, decay_constants, generation_time, tau_core, tau_excore)
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
