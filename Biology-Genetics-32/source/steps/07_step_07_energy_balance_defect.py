"""
Measure by how much a discrete trajectory fails to close the exact energy balance of the continuous evolution over the whole loading path.

An exact rate-independent evolution conserves energy: the work the environment supplies over an interval equals the change in stored energy plus everything irreversibly dissipated, with no slack. A discrete trajectory generally leaves a residual in that budget, and this residual is the consistency error worth monitoring because the scheme controls the energy balance rather than the state error itself. The discrete work is assembled at frozen state, each increment crediting the environment with the energy change the new loading would produce if the state did not move, exactly as in the incremental objective; the discrete dissipation is the accumulated one-homogeneous potential of the state increments. For a trajectory that satisfies the incremental-minimisation inequality at every step, the resulting defect is non-negative because retaining the previous state is an admissible competitor. This function also accepts arbitrary finite trajectory arrays that satisfy its public input contract; such arrays need not satisfy that inequality, so the returned defect is signed and may be positive or negative. Its magnitude measures failure of the discrete energy budget to close.

Returns
-------
float, the initial stored energy plus the frozen-state discrete work, less the final stored energy and the accumulated discrete dissipation, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================


def energy_balance_defect(loading: np.ndarray, trajectory: np.ndarray,
                          k: float = 1.0, a: float = 0.15,
                          rho: float = 0.10, b: float = None) -> float:
    """Measure the accumulated defect of the discrete energy balance.

    The stored energy of the state q under the interaction potential ell is
    the configurational free energy of the bistable landscape less q times
    ell. The work is credited at frozen state, so the increment between two
    consecutive nodes uses the state at the earlier node with the potential at
    both. The defect is the initial stored energy plus the accumulated work,
    less the final stored energy and the accumulated dissipation.

    Parameters
    ----------
    loading : np.ndarray
        Array of shape (n_nodes, 3) whose third column holds the interaction
        potential at each node.
    trajectory : np.ndarray
        Array of shape (n_nodes,) holding the state at each node.
    k : float
        Curvature of each well of the landscape, k > 0.
    a : float
        Distance from the barrier to the bottom of the repressed well, a >= 0.
    rho : float
        Dissipation threshold of the remodelling machinery, rho > 0.
    b : float or None
        Distance from the barrier to the bottom of the active well, b >= 0;
        None places it at a, which makes the landscape symmetric.

    Returns
    -------
    defect : float
        The accumulated energy-balance defect, as a native Python float.

    Raises
    ------
    ValueError
        If loading is not a two-dimensional array of shape (n_nodes, 3) with
        at least two nodes, if trajectory does not hold one state per node of
        that table, if either is not finite throughout, if k or rho is not a
        finite number strictly greater than zero, if a is not a finite
        non-negative number, or if b is neither None nor a finite
        non-negative number.
    """
    return defect  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np 

def _oracle_energy_balance_defect(loading: np.ndarray, trajectory: np.ndarray,
                                  k: float = 1.0, a: float = 0.15,
                                  rho: float = 0.10, b: float = None) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    table = np.asarray(loading, dtype=float)
    states = np.asarray(trajectory, dtype=float).ravel()
    if table.ndim != 2 or table.shape[1] != 3 or table.shape[0] < 2:
        raise ValueError("loading must be a 2D array of shape (n_nodes, 3) with n_nodes >= 2")
    if states.size != table.shape[0]:
        raise ValueError("trajectory must hold one state per node of the loading table")
    if not (np.all(np.isfinite(table)) and np.all(np.isfinite(states))):
        raise ValueError("loading and trajectory must be finite throughout")
    if not (isinstance(rho, (int, float, np.floating, np.integer))
            and np.isfinite(float(rho)) and float(rho) > 0.0):
        raise ValueError("rho must be a finite number strictly greater than zero")
    if not (isinstance(k, (int, float, np.floating, np.integer))
            and np.isfinite(float(k)) and float(k) > 0.0):
        raise ValueError("k must be a finite number strictly greater than zero")
    if not (isinstance(a, (int, float, np.floating, np.integer))
            and np.isfinite(float(a)) and float(a) >= 0.0):
        raise ValueError("a must be a finite non-negative number")
    if b is not None and not (isinstance(b, (int, float, np.floating, np.integer))
                              and np.isfinite(float(b)) and float(b) >= 0.0):
        raise ValueError("b must be None or a finite non-negative number")

    potential = table[:, 2]

    # The landscape of sub-problem 02, written out here rather than imported,
    # so that the defect this step reports is reproducible from this file
    # alone. Continuity at the barrier raises the far well by the amount by
    # which the two bottoms are unequally placed.
    curvature = float(k)
    repressed = float(a)
    active = repressed if b is None else float(b)
    centre = np.where(states <= 0.0, -repressed, active)
    level = 0.5 * curvature * (repressed ** 2 - active ** 2)
    offset = np.where(states <= 0.0, 0.0, level)
    free_energy = 0.5 * curvature * (states - centre) ** 2 + offset

    stored = free_energy - states * potential

    # The work of one increment is credited at frozen state, so the state is
    # held at the earlier node while the potential moves to the later one.
    work = float(np.sum(-states[:-1] * np.diff(potential)))

    # A one-homogeneous dissipation potential accumulates as the total
    # variation of the trajectory, weighted by the threshold.
    dissipated = float(rho) * float(np.sum(np.abs(np.diff(states))))

    return float((stored[0] + work) - (stored[-1] + dissipated))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the benchmark cycle at a coarse step ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 1.0, 51)
S = 2.5 * (1.0 - np.cos(2.0 * np.pi * t))
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
ell = loading[:, 2]
# A closed-form bistable response: locked at the near well until the
# potential reaches the threshold, then flowing, then locked on unloading.
peak = 0.15 + max(ell.max() - 0.10, 0.0)
q = np.where(ell <= 0.10, -0.15, 0.15 + ell - 0.10)
rising = np.arange(ell.size) <= int(np.argmax(ell))
q = np.where(rising, q, np.minimum(peak, 0.15 + ell + 0.10))
""",
            "call": "float(1.0e6 * energy_balance_defect(loading, q, 1.0, 0.15, 0.10))",
            "gold_call": "float(1.0e6 * _oracle_energy_balance_defect(loading, q, 1.0, 0.15, 0.10))",
        },
        # --- Valid: a trajectory frozen throughout, so nothing is dissipated ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 1.0, 41)
S = 2.5 * (1.0 - np.cos(2.0 * np.pi * t))
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
q = np.full(loading.shape[0], -0.15)
""",
            "call": "float(1.0e6 * energy_balance_defect(loading, q, 1.0, 0.15, 0.10))",
            "gold_call": "float(1.0e6 * _oracle_energy_balance_defect(loading, q, 1.0, 0.15, 0.10))",
        },
        # --- Valid: a trajectory carrying one large excursion across the barrier ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 1.0, 21)
S = 5.0 * t
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
q = np.where(loading[:, 2] <= 0.10, -0.15, 0.15 + loading[:, 2] - 0.10)
""",
            "call": "float(1.0e6 * energy_balance_defect(loading, q, 1.0, 0.15, 0.10))",
            "gold_call": "float(1.0e6 * _oracle_energy_balance_defect(loading, q, 1.0, 0.15, 0.10))",
        },
        # --- Valid: the asymmetric landscape, where the raised far well must be
        # carried into the stored energy on the far side of the barrier ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 1.0, 51)
S = 2.5 * (1.0 - np.cos(2.0 * np.pi * t))
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
ell = loading[:, 2]
peak = 0.075 + max(ell.max() - 0.0970, 0.0) / 1.07
q = np.where(ell <= 0.126425, -0.130, 0.075 + (ell - 0.0970) / 1.07)
rising = np.arange(ell.size) <= int(np.argmax(ell))
q = np.where(rising, q, np.minimum(peak, 0.075 + (ell + 0.0970) / 1.07))
""",
            "call": "float(1.0e6 * energy_balance_defect(loading, q, 1.07, 0.130, 0.0970, 0.075))",
            "gold_call": "float(1.0e6 * _oracle_energy_balance_defect(loading, q, 1.07, 0.130, 0.0970, 0.075))",
        },
        # --- Boundary: the shortest admissible table, a single increment ---
        {
            "setup": """import numpy as np
loading = np.array([[0.0, 0.0, 0.0], [0.5, 2.5, 0.45]])
q = np.array([-0.15, 0.5])
""",
            "call": "float(1.0e6 * energy_balance_defect(loading, q, 1.0, 0.15, 0.10))",
            "gold_call": "float(1.0e6 * _oracle_energy_balance_defect(loading, q, 1.0, 0.15, 0.10))",
        },
        # --- Edge: a landscape with no barrier, where both wells coincide ---
        {
            "setup": """import numpy as np
t = np.linspace(0.0, 1.0, 31)
S = 2.5 * (1.0 - np.cos(2.0 * np.pi * t))
loading = np.column_stack((t, S, 0.5 * (1.0 - np.exp(-S))))
q = np.maximum(0.0, loading[:, 2] - 0.10)
""",
            "call": "float(1.0e6 * energy_balance_defect(loading, q, 1.0, 0.0, 0.10))",
            "gold_call": "float(1.0e6 * _oracle_energy_balance_defect(loading, q, 1.0, 0.0, 0.10))",
        },
        # --- Invalid: a trajectory of the wrong length ---
        {
            "setup": """import numpy as np
loading = np.zeros((6, 3))
def run_model():
    try:
        energy_balance_defect(loading, np.zeros(5))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_energy_balance_defect(loading, np.zeros(5))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a table holding a single node, so no increment exists ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        energy_balance_defect(np.zeros((1, 3)), np.zeros(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_energy_balance_defect(np.zeros((1, 3)), np.zeros(1))
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a machinery with no resistance to remodelling ---
        {
            "setup": """import numpy as np
loading = np.zeros((4, 3))
def run_model():
    try:
        energy_balance_defect(loading, np.zeros(4), 1.0, 0.15, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_energy_balance_defect(loading, np.zeros(4), 1.0, 0.15, 0.0)
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
