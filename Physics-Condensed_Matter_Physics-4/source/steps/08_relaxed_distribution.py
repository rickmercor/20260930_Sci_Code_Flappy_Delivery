"""
Propagate the homogeneous occupation of the moire exciton states from a hot excitation window through the phonon master equation to a given time.

With spatial gradients neglected the Boltzmann equation reduces to a linear master equation for the state
occupations N: every state gains from the rates into it and loses its total out-scattering rate. After
optical excitation the population is spread uniformly over all states whose energy, measured from the lowest
energy in the supplied set, lies within half_width of e_center (inclusive); the initial occupations sum to one.
The occupation at time t_eval is the exact solution of the linear equation, not a time-stepped approximation.
The rates obey detailed balance only approximately, and a relaxation bottleneck can hold the distribution far
from the thermal one on the time scale asked for.

Returns
-------
occupation : np.ndarray -- (S,) occupations at t_eval.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def relaxed_distribution(rates: "np.ndarray", energies: "np.ndarray", e_center: float, half_width: float,
                         t_eval: float) -> "np.ndarray":
    """Return the state occupations at time t_eval after the excitation.

    Parameters
    ----------
    rates : np.ndarray
        (S, S) rates in 1/ps, rates[f, i] from state i into state f.
    energies : np.ndarray
        State energies in meV, any shape with S entries in the flattened state order.
    e_center : float
        Centre of the excitation window above the lowest energy, meV.
    half_width : float
        Half-width of the window, meV.
    t_eval : float
        Evaluation time, ps.

    Returns
    -------
    occupation : np.ndarray
        (S,) occupations at t_eval.

    Raises
    ------
    ValueError
        If no state lies inside the excitation window.
    """
    return occupation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_relaxed_distribution(rates: "np.ndarray", energies: "np.ndarray", e_center: float, half_width: float,
                                 t_eval: float) -> "np.ndarray":
    e = np.asarray(energies, dtype=float).reshape(-1)
    e = e - e.min()
    n0 = (np.abs(e - e_center) <= half_width).astype(float)
    if n0.sum() == 0.0:
        raise ValueError("no state inside the excitation window")
    n0 /= n0.sum()
    return expm((rates - np.diag(rates.sum(axis=0))) * t_eval) @ n0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    base = """import numpy as np
def make(s, seed):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.0, 0.3, size=(s, s)) * (rng.uniform(size=(s, s)) < 0.6)
    e = 100.0 + rng.uniform(0.0, 80.0, size=s)
    return r, e
"""
    return [
        {"setup": base + "r, e = make(12, 1)",
         "call": "relaxed_distribution(r, e, 40.0, 10.0, 5.0)",
         "gold_call": "_oracle_relaxed_distribution(r, e, 40.0, 10.0, 5.0)"},
        # t = 0 returns the normalised window; the window edge is inclusive
        {"setup": base + "r, e = make(6, 2)\ne = np.array([[10.0, 13.0, 16.0], [19.0, 22.0, 25.0]])",
         "call": "relaxed_distribution(r, e, 6.0, 3.0, 0.0)",
         "gold_call": "_oracle_relaxed_distribution(r, e, 6.0, 3.0, 0.0)"},
        # long time with a slow bottleneck channel
        {"setup": base + "r, e = make(8, 3)\nr[:, 4:] *= 1e-3",
         "call": "relaxed_distribution(r, e, 60.0, 25.0, 300.0)",
         "gold_call": "_oracle_relaxed_distribution(r, e, 60.0, 25.0, 300.0)"},
        {"setup": base + """r, e = make(4, 5)
def run(fn):
    try:
        fn(r, np.array([0.0, 1.0, 2.0, 3.0]), 50.0, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
""",
         "call": "run(relaxed_distribution)",
         "gold_call": "run(_oracle_relaxed_distribution)"},
    ]
