"""
Compute the unnormalized first moments of tension and extension at the last scission event of a pulled reversible chain, together with the total final-scission probability and the expected number of scission events.

When broken chains can re-form, only a scission that is never undone marks rupture, so the rupture statistics weight each scission by the chance that no re-formation follows it.

Returns
-------
np.ndarray: shape (4,) unnormalized first moments of tension and extension at the final scission, total final-scission probability, expected number of scission events.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_final_scission_statistics(y_grid: "np.ndarray", intact_probability: "np.ndarray", scission_barrier: "np.ndarray", healing_barrier: "np.ndarray", tension: "np.ndarray", n_segments: int, loading_rate: float) -> "np.ndarray":
    """Return statistics of the final (never re-formed) scission event.

    The chain is pulled at the constant reduced ``loading_rate`` through the
    nodes of ``y_grid``, with ``intact_probability`` from
    ``integrate_intact_probability`` and the same Arrhenius kinetics: the total
    scission rate of ``n_segments`` equivalent segments and the re-formation
    rate of the broken bond, over barriers in k_B T with unit attempt frequency.
    A scission at a node is final when the broken chain does not re-form at any
    later point of the grid; re-formation is impossible beyond the last node,
    where any probability still intact breaks as a point mass.

    Every integral over the grid, including the accumulated re-formation
    exposure between a node and the last node, uses the trapezoidal rule on the
    nodes, and the point mass at the last node enters all four results.

    Parameters
    ----------
    y_grid : np.ndarray
        Strictly increasing one-dimensional grid with at least two nodes.
    intact_probability : np.ndarray
        Intact probability at each node.
    scission_barrier : np.ndarray
        Scission barrier at each node.
    healing_barrier : np.ndarray
        Re-formation barrier at each node.
    tension : np.ndarray
        Reduced tension of the intact chain at each node.
    n_segments : int
        Number of equivalent segments that can break, at least 1.
    loading_rate : float
        Reduced loading rate, positive.

    Returns
    -------
    statistics : np.ndarray
        Shape (4,) array ``[unnormalized first moment of tension at the final
        scission, unnormalized first moment of end-to-end distance at the final
        scission, total probability of a final scission, expected number of
        scission events per pull]``. Divide either first moment by the returned
        total probability to obtain the corresponding conditional mean when
        that probability is not one.

    Raises
    ------
    ValueError
        If ``y_grid`` is not strictly increasing with at least two nodes, if any
        other array does not match its shape, if ``n_segments`` is not a
        positive integer, or if ``loading_rate`` is not positive.
    """
    return statistics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_final_scission_statistics(y_grid: "np.ndarray", intact_probability: "np.ndarray", scission_barrier: "np.ndarray", healing_barrier: "np.ndarray", tension: "np.ndarray", n_segments: int, loading_rate: float) -> "np.ndarray":
    """Reference implementation (trapezoidal survival exposure and moments)."""
    import numpy as np

    y = np.asarray(y_grid, dtype=float)
    arrays = [np.asarray(a, dtype=float) for a in (intact_probability, scission_barrier, healing_barrier, tension)]
    if y.ndim != 1 or y.size < 2 or not np.all(np.diff(y) > 0.0):
        raise ValueError("y_grid must be strictly increasing with at least two nodes")
    if any(a.shape != y.shape for a in arrays):
        raise ValueError("all arrays must match y_grid")
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)) or n_segments < 1:
        raise ValueError("n_segments must be a positive integer")
    if not loading_rate > 0.0:
        raise ValueError("loading_rate must be positive")
    p, e_s, e_h, force = arrays
    width = np.diff(y)

    def _trapezoid(values):
        return float(np.sum(0.5 * (values[:-1] + values[1:]) * width))

    scission_density = p * n_segments * np.exp(-e_s) / loading_rate
    healing_density = np.exp(-e_h) / loading_rate
    # Re-formation exposure from each node to the end of the grid.
    pieces = 0.5 * (healing_density[:-1] + healing_density[1:]) * width
    exposure = np.concatenate([np.cumsum(pieces[::-1])[::-1], [0.0]])
    final_density = scission_density * np.exp(-exposure)
    tail = p[-1]
    return np.array([
        _trapezoid(force * final_density) + tail * force[-1],
        _trapezoid(y * final_density) + tail * y[-1],
        _trapezoid(final_density) + tail,
        _trapezoid(scission_density) + tail,
    ])

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
def _propagate(y, e_s, e_h, n, rate):
    k_s, k_h = n * np.exp(-e_s), np.exp(-e_h)
    p = np.ones_like(y)
    for i in range(y.size - 1):
        ks = 0.5 * (k_s[i] + k_s[i + 1])
        kh = 0.5 * (k_h[i] + k_h[i + 1])
        target = kh / (ks + kh)
        p[i + 1] = target + (p[i] - target) * np.exp(
            -(ks + kh) * (y[i + 1] - y[i]) / rate
        )
    return p
y = np.linspace(0.5, 20.0, 1951)
e_s = 31.0 - 0.02 * y ** 2
e_h = 1.0 + 0.09 * y ** 2
f = 0.1 * y + 0.002 * y ** 2
"""
    status = """import numpy as np
def run_model():
    try:
        compute_final_scission_statistics(np.array([0.5, 1.0]), np.array([1.0, 0.9]), np.array([30.0, 30.0]), np.array([2.0, 2.0]), np.array([0.1, 0.2]), 21, 0.0)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_compute_final_scission_statistics(np.array([0.5, 1.0]), np.array([1.0, 0.9]), np.array([30.0, 30.0]), np.array([2.0, 2.0]), np.array([0.1, 0.2]), 21, 0.0)
        return 0
    except ValueError:
        return 1
"""
    return [
        {
            "setup": setup + "p = _propagate(y, e_s, e_h, 21, 1e-12)\n",
            "call": "_vector(compute_final_scission_statistics(y.copy(), p.copy(), e_s.copy(), e_h.copy(), f.copy(), 21, 1e-12), 4)",
            "gold_call": "_vector(_oracle_compute_final_scission_statistics(y.copy(), p.copy(), e_s.copy(), e_h.copy(), f.copy(), 21, 1e-12), 4)",
        },
        {
            "setup": setup + "p = _propagate(y, e_s, e_h + 700.0, 21, 1e-12)\n",
            "call": "_vector(compute_final_scission_statistics(y.copy(), p.copy(), e_s.copy(), e_h + 700.0, f.copy(), 21, 1e-12), 4)",
            "gold_call": "_vector(_oracle_compute_final_scission_statistics(y.copy(), p.copy(), e_s.copy(), e_h + 700.0, f.copy(), 21, 1e-12), 4)",
        },
        {
            "setup": setup + "p = _propagate(y, e_s, e_h, 21, 1e-16)\n",
            "call": "_vector(compute_final_scission_statistics(y.copy(), p.copy(), e_s.copy(), e_h.copy(), f.copy(), 21, 1e-16), 4)",
            "gold_call": "_vector(_oracle_compute_final_scission_statistics(y.copy(), p.copy(), e_s.copy(), e_h.copy(), f.copy(), 21, 1e-16), 4)",
        },
        {
            "setup": setup,
            "call": "_vector(compute_final_scission_statistics(np.array([3.0, 3.25]), np.array([0.8, 0.3]), np.array([24.0, 23.0]), np.array([9.0, 10.0]), np.array([0.5, 0.6]), 7, 1e-9), 4)",
            "gold_call": "_vector(_oracle_compute_final_scission_statistics(np.array([3.0, 3.25]), np.array([0.8, 0.3]), np.array([24.0, 23.0]), np.array([9.0, 10.0]), np.array([0.5, 0.6]), 7, 1e-9), 4)",
        },
        {
            "setup": status,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
