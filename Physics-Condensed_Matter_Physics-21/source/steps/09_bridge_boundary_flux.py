"""
Assemble the full construction for one dense-matter bridge and report the energy density transported across its boundaries during smoothing.

The construction runs end to end: the allowed region between two endpoint states is mapped, a uniform measure on it induces marginals in the chemical potential and in the density, prescribed quantiles of those marginals place intermediate states, recursive subdivision produces a bridge with structure on every resolved scale, and a conservative diffusion imposes a correlation length that is a fixed fraction of the density.




The quantity of interest is the accumulated boundary energy flux. Inside the domain the smoothing conserves the energy-density difference exactly, so any change must have crossed the two ends, where the chemical potential is pinned to the endpoint values. The accumulated flow therefore measures how far the smoothed bridge has drifted from exact thermodynamic consistency, and it depends on the entire construction through the boundary gradients and their evolution during smoothing.




The peak squared speed of sound of the smoothed bridge is reported alongside it as a diagnostic, since a construction that is thermodynamically well behaved but acausal would be of no use.

Returns
-------
np.ndarray, shape (2,), [flux in GeV fm^-3, max interior c_s**2]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bridge_boundary_flux(beta_low: "np.ndarray", beta_high: "np.ndarray", depth: int,
                         quantiles: "np.ndarray", correlation_fraction: float,
                         n_grid_points: int, n_tau_steps: int) -> "np.ndarray":
    '''Run the bridge construction and report its boundary flux and peak sound speed.

    Parameters
    ----------
    beta_low, beta_high : np.ndarray
        Finite shape (3,) endpoints (mu, n, p) in GeV, fm^-3 and GeV fm^-3.
    depth : int
        Integer refinement depth in [0, 12].
    quantiles : np.ndarray
        Finite shape (3,) with every entry in [0, 1].
    correlation_fraction : float
        Finite positive fractional correlation length.
    n_grid_points : int
        Integer number of uniformly spaced density grid points, at least 3,
        spanning the two endpoint densities inclusively.
    n_tau_steps : int
        Integer number of uniform flow-time steps, at least 1.

    Returns
    -------
    summary : np.ndarray
        Finite shape (2,). Element 0 is the accumulated boundary energy flux in
        GeV fm^-3, and element 1 is the largest squared speed of sound of the
        smoothed bridge, taken over the interior grid points only.

        The refined states are interpolated linearly in chemical potential
        against density onto the uniform grid to form the initial condition,
        which is then diffused for unit flow time.

    Raises
    ------
    ValueError
        If n_grid_points is not an integer of at least 3, or if a preceding
        step rejects its inputs, or if the summary is not finite.
    '''
    return summary

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bridge_boundary_flux(beta_low: "np.ndarray", beta_high: "np.ndarray", depth: int,
                                 quantiles: "np.ndarray", correlation_fraction: float,
                                 n_grid_points: int, n_tau_steps: int) -> "np.ndarray":
    states = _oracle_self_similar_refine(beta_low, beta_high, depth, quantiles)
    if isinstance(n_grid_points, bool) or not isinstance(n_grid_points, (int, np.integer)):
        raise ValueError("n_grid_points must be an integer")
    points = int(n_grid_points)
    if points < 3:
        raise ValueError("n_grid_points must be at least 3")
    grid = np.linspace(states[0, 1], states[-1, 1], points)
    initial = np.interp(grid, states[:, 1], states[:, 0])
    evolved = _oracle_diffuse_chemical_potential(grid, initial, correlation_fraction,
                                                 n_tau_steps)
    speeds = _oracle_sound_speed_squared(grid, evolved[1:])
    summary = np.array([evolved[0], float(np.max(speeds[1:-1]))], dtype=float)
    if not np.all(np.isfinite(summary)):
        raise ValueError("the construction summary must be finite")
    return summary

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    bench = ("import numpy as np\n"
             "low = np.array([1.00, 0.32, 0.018])\n"
             "high = np.array([2.60, 4.80, 3.500])\n"
             "q = np.array([0.5, 0.5, 0.5])\n"
             "depth = 6\nc = 0.20\npoints = 201\nsteps = 4000\n")
    call = "bridge_boundary_flux(low.copy(), high.copy(), depth, q.copy(), c, points, steps)"
    gold = "_oracle_bridge_boundary_flux(low.copy(), high.copy(), depth, q.copy(), c, points, steps)"
    guard = (f'def run_model():\n    try:\n        {call}\n        return 0\n'
             '    except ValueError:\n        return 1\n'
             f'def run_oracle():\n    try:\n        {gold}\n        return 0\n'
             '    except ValueError:\n        return 1\n')
    return [
        # --- Normal: the benchmark construction ---
        {"setup": bench, "call": call, "gold_call": gold, "tol": 1e-09},
        # --- Boundary: depth zero, the straight chord with no bridge structure ---
        {"setup": bench.replace("depth = 6", "depth = 0"),
         "call": call, "gold_call": gold, "tol": 1e-09},
        # --- Normal: off-median quantiles, biasing every insertion ---
        {"setup": bench.replace("0.5, 0.5, 0.5", "0.35, 0.65, 0.40").replace("depth = 6", "depth = 4"),
         "call": call, "gold_call": gold, "tol": 1e-09},
        # --- Normal: weaker smoothing, retaining more of the bridge structure ---
        {"setup": bench.replace("depth = 6", "depth = 4").replace("c = 0.20", "c = 0.10"),
         "call": call, "gold_call": gold, "tol": 1e-09},
        # --- Normal: a different endpoint pair ---
        {"setup": ("import numpy as np\n"
                   "low = np.array([1.05, 0.40, 0.025])\n"
                   "high = np.array([2.40, 4.00, 2.800])\n"
                   "q = np.array([0.5, 0.5, 0.5])\n"
                   "depth = 4\nc = 0.20\npoints = 201\nsteps = 4000\n"),
         "call": call, "gold_call": gold, "tol": 1e-09},
        # --- Invalid: too few grid points must raise ValueError ---
        {"setup": bench.replace("points = 201", "points = 2") + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
        # --- Invalid: a refinement depth beyond the allowed range must raise ValueError ---
        {"setup": bench.replace("depth = 6", "depth = 20") + guard,
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
