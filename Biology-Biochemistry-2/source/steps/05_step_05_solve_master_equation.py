"""
Solve the main equation and read out the ensemble-averaged flux time course

Given the couplon generator in coordinate form, the ensemble-averaged flux is the exact expectation of the channel ensemble, free of sampling error: the occupation vector p solves dp/dt = Q p from the resting (all-closed) initial condition, and the flux course follows by contraction with the flux observable. The observable counts each open sensor-coupled channel as one unit of open-channel flux and each open contact-free channel as R_CV = 5 such units; inactivated channels contribute nothing. Time is evaluated on the pinned uniform (dt = 0.02 ms) grid over [0, 100] ms. The generator is constant over the pulse, so the integration is a single stiff, constant-linear solve. The solver settings used for the graded numbers (BDF, rtol = 1e-8, atol = 1e-11) are fixed here so every implementation of this step is compared against a solver whose numerics are pinned, not floating.

Returns
-------
flux_time_course : np.ndarray
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

def solve_master_equation(rows: np.ndarray, cols: np.ndarray, values: np.ndarray,
                          flux_per_state: np.ndarray) -> np.ndarray:
    """Ensemble-averaged flux time course from a couplon generator in coordinate form.

    Parameters
    ----------
    rows, cols, values : np.ndarray
        Coordinate form of the generator Q (from step 04): destination state, source state,
        and transition rate in ms^-1. Values at rows == cols hold the diagonal entries.
    flux_per_state : np.ndarray
        Shape (n_states,). The flux observable: the contribution of each global state to
        the ensemble flux (open V channels count 1, open C channels count R_CV = 5, every
        other state component counts 0).

    Returns
    -------
    np.ndarray
        Shape (5001,): the exact ensemble-averaged flux on the uniform 0.02 ms grid over
        [0, 100] ms, starting from the all-closed configuration (global state index 0).

    Raises
    ------
    ValueError
        If the coordinate arrays are inconsistent, contain non-finite values, if the flux
        vector does not match the inferred state space, or if the inference of the state
        space size fails.
    """
    flux_time_course = np.empty(5001, dtype=float)
    return flux_time_course  # placeholder to fill

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_master_equation(rows, cols, values, flux_per_state):
    """Reference implementation for solve_master_equation."""
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.sparse import csc_matrix

    DT, TPULSE = 0.02, 100.0  # ms, pinned grid (treatment's own resolution bookkeeping)
    RTOL, ATOL = 1e-8, 1e-11  # solver tolerances behind the measured graded numbers
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    values = np.asarray(values, dtype=float).ravel()
    flux_per_state = np.asarray(flux_per_state, dtype=float).ravel()
    if not (rows.size == cols.size == values.size):
        raise ValueError("coordinate arrays must be the same length")
    if not np.all(np.isfinite(values)):
        raise ValueError("generator entries must be finite")
    n_states = int(flux_per_state.size)
    if n_states == 0 or rows.max(initial=-1) >= n_states or cols.max(initial=-1) >= n_states:
        raise ValueError("flux vector does not cover the state space")
    if not np.all(np.isfinite(flux_per_state)):
        raise ValueError("flux observable must be finite")

    Q = csc_matrix((values, (rows, cols)), shape=(n_states, n_states))
    p0 = np.zeros(n_states)
    p0[0] = 1.0  # all-closed resting configuration is global state index 0
    ngrid = int(round(TPULSE / DT)) + 1
    tgrid = np.linspace(0.0, TPULSE, ngrid)
    sol = solve_ivp(lambda t, p: np.asarray(Q @ p), (0.0, TPULSE), p0, method="BDF",
                    t_eval=tgrid, jac=lambda t, p: Q, rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(f"BDF integration failed: {sol.message}")
    return flux_per_state @ sol.y

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    # Fixture: the 2-channel V/C dimer (states 0 = both closed, 1 = V-open, 2 = C-open,
    # 3 = both open), no contact energy. Rates follow the treatment law in ms^-1 at the
    # given pulse energy; the coordinate-form generator is pinned literally in each setup
    # so the step is tested standalone (no cross-step imports in evaluation scope).
    _PRE = (
        "import numpy as np\n"
        "def _dimer(a_v, b_v, a_c, b_c):\n"
        "    # edges: V toggles 0<->1 and 2<->3; C toggles 0<->2 and 1<->3\n"
        "    pairs = [(1, 0, a_v), (2, 0, a_c), (0, 1, b_v), (3, 1, a_c),\n"
        "             (0, 2, b_c), (3, 2, a_v), (1, 3, b_c), (2, 3, b_v)]\n"
        "    rows, cols, values = [], [], []\n"
        "    for dst, src, r in pairs:\n"
        "        rows.append(dst); cols.append(src); values.append(r)\n"
        "    exit_rates = np.zeros(4)\n"
        "    np.add.at(exit_rates, cols, values)\n"
        "    for s in range(4):\n"
        "        rows.append(s); cols.append(s); values.append(-exit_rates[s])\n"
        "    F = np.array([0.0, 1.0, 5.0, 6.0])  # open V counts 1, open C counts R_CV = 5\n"
        "    return np.array(rows), np.array(cols), np.array(values), F\n"
    )
    small = _PRE + (
        # Normal: mid-family eps_V = -2 kT: open = 0.001 e^{0.8*2}, close = 2.0 e^{-0.4}.
        "rows, cols, values, F = _dimer(0.004953032, 1.340640, 0.001, 2.0)\n"
    )
    quiescent = _PRE + (
        # Boundary: resting energy eps_V = 0: electrical factors are exactly 1.
        "rows, cols, values, F = _dimer(0.001, 2.0, 0.001, 2.0)\n"
    )
    saturated = _PRE + (
        # Edge: saturating eps_V = -14 kT: open = 0.001 e^{11.2}, close = 2.0 e^{-2.8}; the
        # ~600:1 rate contrast exercises the stiff solver tolerances.
        "rows, cols, values, F = _dimer(73.13044, 0.12162, 0.001, 2.0)\n"
    )
    invalid = "import numpy as np\n" + (
        "def run_model(rows, cols, values, F):\n"
        "    try:\n"
        "        solve_master_equation(rows, cols, values, F)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
        "def run_gold(rows, cols, values, F):\n"
        "    try:\n"
        "        _oracle_solve_master_equation(rows, cols, values, F)\n"
        "        return 0.0\n"
        "    except ValueError:\n"
        "        return 1.0\n"
        "    except Exception:\n"
        "        return 2.0\n"
    )
    return [
        # Normal: the V/C dimer at mid-family energy; compared elementwise on the grid.
        {"setup": small,
         "call": "solve_master_equation(rows, cols, values, F)",
         "gold_call": "_oracle_solve_master_equation(rows, cols, values, F)"},
        # Boundary: resting energy; the V channel opens rarely, flux stays near zero.
        {"setup": quiescent,
         "call": "solve_master_equation(rows, cols, values, F)",
         "gold_call": "_oracle_solve_master_equation(rows, cols, values, F)"},
        # Edge: saturating energy; fast V cycling against slow C opening excites stiffness.
        {"setup": saturated,
         "call": "solve_master_equation(rows, cols, values, F)",
         "gold_call": "_oracle_solve_master_equation(rows, cols, values, F)"},
    ] + [
        # Invalid: coordinate arrays of unequal length.
        {"setup": invalid + "F = np.zeros(4)\n",
         "call": "run_model(np.array([0, 1]), np.array([0]), np.array([1.0, 2.0]), F)",
         "gold_call": "run_gold(np.array([0, 1]), np.array([0]), np.array([1.0, 2.0]), F)"},
        # Invalid: flux vector shorter than the inferred state space.
        {"setup": invalid + "vv = np.zeros((5, 5)); vv[0, 1] = 1.0; vv[1, 0] = -1.0\n"
          "rows = np.array([1, 0]); cols = np.array([0, 1]); values = np.array([1.0, -1.0])\n"
          "F = np.zeros(3)\n",
         "call": "run_model(rows, cols, values, F)",
         "gold_call": "run_gold(rows, cols, values, F)"},
    ]
