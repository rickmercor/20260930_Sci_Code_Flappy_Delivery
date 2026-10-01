"""
Propagate the probability that a chain pulled at a constant rate is intact, given its scission and re-formation barriers along the end-to-end distance.

Under a prescribed extension history, reversible scission makes the intact probability relax toward a moving equilibrium, and at small extensions, that relaxation is many orders of magnitude faster than the loading.

Returns
-------
np.ndarray: intact probability at each grid node under the interval-wise exact update, starting from 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_intact_probability(y_grid: "np.ndarray", scission_barrier: "np.ndarray", healing_barrier: "np.ndarray", n_segments: int, loading_rate: float) -> "np.ndarray":
    """Return the intact probability at every node of a pulling protocol.

    The end-to-end distance grows at the constant reduced rate ``loading_rate``
    (Kuhn lengths per inverse attempt frequency) through the nodes of
    ``y_grid``. An intact chain breaks with the total scission rate of its
    ``n_segments`` equivalent segments, and a broken chain re-forms with the
    re-formation rate of the bond that broke. Both rates are Arrhenius with unit
    attempt frequency over the barriers (k_B T) given at the nodes, and the
    chain is intact at the first node.

    Across each interval both rate coefficients are held at the geometric means
    of their two endpoint values, the equilibrium intact probability is taken
    to vary linearly between its values at the two endpoints (each computed from
    that endpoint's own rates), and the probability is advanced by the exact
    solution of the resulting linear equation. The update must remain exact when
    an interval spans many relaxation lengths.

    Parameters
    ----------
    y_grid : np.ndarray
        Strictly increasing one-dimensional grid with at least two nodes.
    scission_barrier : np.ndarray
        Scission barrier at each node, same shape as ``y_grid``.
    healing_barrier : np.ndarray
        Re-formation barrier at each node, same shape as ``y_grid``.
    n_segments : int
        Number of equivalent segments that can break, at least 1.
    loading_rate : float
        Reduced loading rate, positive.

    Returns
    -------
    intact_probability : np.ndarray
        Intact probability at each node, same shape as ``y_grid``.

    Raises
    ------
    ValueError
        If ``y_grid`` is not strictly increasing with at least two nodes, if the
        barrier arrays do not match its shape, if ``n_segments`` is not a
        positive integer, or if ``loading_rate`` is not positive.
    """
    return intact_probability

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_integrate_intact_probability(y_grid: "np.ndarray", scission_barrier: "np.ndarray", healing_barrier: "np.ndarray", n_segments: int, loading_rate: float) -> "np.ndarray":
    """Reference implementation (exponential update with a linearly moving equilibrium)."""
    import numpy as np

    y = np.asarray(y_grid, dtype=float)
    e_s = np.asarray(scission_barrier, dtype=float)
    e_h = np.asarray(healing_barrier, dtype=float)
    if y.ndim != 1 or y.size < 2 or not np.all(np.diff(y) > 0.0):
        raise ValueError("y_grid must be strictly increasing with at least two nodes")
    if e_s.shape != y.shape or e_h.shape != y.shape:
        raise ValueError("barrier arrays must match y_grid")
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)) or n_segments < 1:
        raise ValueError("n_segments must be a positive integer")
    if not loading_rate > 0.0:
        raise ValueError("loading_rate must be positive")
    k_s = n_segments * np.exp(-e_s)
    k_h = np.exp(-e_h)
    p_eq = k_h / (k_s + k_h)
    width = np.diff(y)
    decay_length = (np.sqrt(k_s[:-1] * k_s[1:]) + np.sqrt(k_h[:-1] * k_h[1:])) * width / loading_rate
    survive = np.exp(-decay_length)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Averaging weight of a linearly moving target, (1 - exp(-L)) / L, with its small-L limit.
        lag = np.where(decay_length > 1e-8, -np.expm1(-decay_length) / decay_length, 1.0 - 0.5 * decay_length)
    p = np.empty_like(y)
    p[0] = 1.0
    for i in range(width.size):
        p[i + 1] = p_eq[i + 1] + (p[i] - p_eq[i]) * survive[i] - (p_eq[i + 1] - p_eq[i]) * lag[i]
    return p

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    setup = """import numpy as np
def _vector(value, size):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.all(np.isfinite(a)):
        raise AssertionError("Expected a finite vector with the documented shape")
    return a
y = np.linspace(0.5, 20.0, 391)
e_s = 31.0 - 0.02 * y ** 2
e_h = 1.0 + 0.09 * y ** 2
"""
    status = """import numpy as np
def run_model():
    try:
        integrate_intact_probability(np.array([1.0, 0.5]), np.array([30.0, 30.0]), np.array([2.0, 2.0]), 21, 1e-12)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_integrate_intact_probability(np.array([1.0, 0.5]), np.array([30.0, 30.0]), np.array([2.0, 2.0]), 21, 1e-12)
        return 0
    except ValueError:
        return 1
"""
    return [
        {
            "setup": setup,
            "call": "_vector(integrate_intact_probability(y.copy(), e_s.copy(), e_h.copy(), 21, 1e-12), 391)",
            "gold_call": "_vector(_oracle_integrate_intact_probability(y.copy(), e_s.copy(), e_h.copy(), 21, 1e-12), 391)",
        },
        {
            "setup": setup,
            "call": "_vector(integrate_intact_probability(y.copy(), e_s.copy(), e_h.copy(), 21, 1e-19), 391)",
            "gold_call": "_vector(_oracle_integrate_intact_probability(y.copy(), e_s.copy(), e_h.copy(), 21, 1e-19), 391)",
        },
        {
            "setup": setup,
            "call": "_vector(integrate_intact_probability(y.copy(), e_s - 25.0, e_h + 40.0, 21, 3.0), 391)",
            "gold_call": "_vector(_oracle_integrate_intact_probability(y.copy(), e_s - 25.0, e_h + 40.0, 21, 3.0), 391)",
        },
        {
            "setup": setup,
            "call": "_vector(integrate_intact_probability(np.array([12.0, 12.5]), np.array([28.0, 27.5]), np.array([14.0, 14.9]), 21, 2e-8), 2)",
            "gold_call": "_vector(_oracle_integrate_intact_probability(np.array([12.0, 12.5]), np.array([28.0, 27.5]), np.array([14.0, 14.9]), 21, 2e-8), 2)",
        },
        {
            "setup": status,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
