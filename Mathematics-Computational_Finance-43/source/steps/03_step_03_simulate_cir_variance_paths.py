"""
Simulate one variance factor by iterating its finite-interval conditional transition.

Sampling the CIR noncentral-chi-square law preserves nonnegativity without Euler truncation or reflection.

Returns
-------
np.ndarray, nonnegative variance paths as a float array with shape (n_steps + 1, n_paths).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_cir_variance_paths(
    initial_variance: float,
    transition_parameters: "np.ndarray",
    n_steps: int,
    n_paths: int,
    seed: int,
) -> "np.ndarray":
    r"""Return variance paths including the initial row.

    Parameters
    ----------
    initial_variance : float
        Finite nonnegative initial variance.
    transition_parameters : numpy.ndarray
        Scale, degrees of freedom, and noncentrality multiplier.
    n_steps, n_paths, seed : int
        Positive counts and deterministic random seed.

    Returns
    -------
    paths : numpy.ndarray
        Nonnegative array of shape ``(n_steps + 1, n_paths)``.

    Raises
    ------
    ValueError
        If the transition state or simulation controls are invalid.
    """
    return paths

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_simulate_cir_variance_paths(
    initial_variance: float, transition_parameters: "np.ndarray",
    n_steps: int, n_paths: int, seed: int,
) -> "np.ndarray":
    params = np.asarray(transition_parameters, dtype=float)
    if params.shape != (3,) or not np.all(np.isfinite(params)):
        raise ValueError("transition_parameters must be a finite length-three array")
    if params[0] <= 0.0 or params[1] <= 0.0 or params[2] < 0.0:
        raise ValueError("transition scale and degrees must be positive")
    if initial_variance < 0.0 or not np.isfinite(initial_variance):
        raise ValueError("initial_variance must be finite and nonnegative")
    for value in (n_steps, n_paths, seed):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts and seed must be integers")
    if n_steps < 1 or n_paths < 1 or seed < 0:
        raise ValueError("counts must be positive and seed nonnegative")
    if params[2] < 0.0:
        raise ValueError("noncentrality multiplier must be nonnegative")
    rng = np.random.default_rng(int(seed))
    paths = np.empty((n_steps + 1, n_paths), dtype=float)
    paths[0] = float(initial_variance)
    for step in range(n_steps):
        noncentrality = params[2] * paths[step]
        draw = rng.noncentral_chisquare(params[1], noncentrality)
        paths[step + 1] = params[0] * draw
    return paths

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge test specifications."""
    return [
        {"setup": "import numpy as np\np=np.array([.002,4.,300.]); w=np.arange(32,dtype=float).reshape(4,8)+1",
         "call": "float(np.sum(simulate_cir_variance_paths(.2,p,3,8,17)*w))",
         "gold_call": "float(np.sum(_oracle_simulate_cir_variance_paths(.2,p,3,8,17)*w))"},
        {"setup": "import numpy as np\np=np.array([.01,2.5,0.])",
         "call": "float(np.sum(simulate_cir_variance_paths(0.,p,1,5,0)))",
         "gold_call": "float(np.sum(_oracle_simulate_cir_variance_paths(0.,p,1,5,0)))"},
        {"setup": "import numpy as np\np=np.array([.01,3.,20.])",
         "call": "float(np.sum(simulate_cir_variance_paths(.1,p,4,9,23)))",
         "gold_call": "float(np.sum(_oracle_simulate_cir_variance_paths(.1,p,4,9,23)))"},
    ]
