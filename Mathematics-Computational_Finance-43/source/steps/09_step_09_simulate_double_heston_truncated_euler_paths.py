"""
Simulate the paper's truncated-Euler comparator for the fixed double-Heston model.

The comparator truncates each Euler variance update at zero and couples its shock to the corresponding asset-return shock.

Returns
-------
np.ndarray, spot and two nonnegative variance paths with shape (3, n_steps + 1, n_paths).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_double_heston_truncated_euler_paths(
    n_paths: int, n_steps: int, seed_v1: int, seed_v2: int, seed_price: int,
) -> "np.ndarray":
    r"""Return spot and variance paths from the source Euler comparator.

    Use the fixed double-Heston parameters in the task. At each step, draw the
    factor-one and factor-two variance normals from their dedicated streams,
    then the corresponding orthogonal normals from the price stream. Apply
    positive truncation after each variance update and standard Euler to spot,
    using current nonnegative variances and the correlated normal pairs.

    Parameters
    ----------
    n_paths, n_steps : int
        Positive path and interval counts.
    seed_v1, seed_v2, seed_price : int
        Nonnegative deterministic stream seeds.

    Returns
    -------
    paths : numpy.ndarray
        Float array of shape ``(3, n_steps + 1, n_paths)`` ordered as spot,
        first variance, and second variance.

    Raises
    ------
    ValueError
        If counts are not positive integers or seeds are not nonnegative integers.
    """
    return paths

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_simulate_double_heston_truncated_euler_paths(
    n_paths: int, n_steps: int, seed_v1: int, seed_v2: int, seed_price: int,
) -> "np.ndarray":
    values = (n_paths, n_steps, seed_v1, seed_v2, seed_price)
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in values):
        raise ValueError("counts and seeds must be integers")
    if n_paths < 1 or n_steps < 1 or min(seed_v1, seed_v2, seed_price) < 0:
        raise ValueError("counts must be positive and seeds nonnegative")
    kappa = np.array([0.9, 1.2]); theta = np.array([0.1, 0.15])
    gamma = np.array([0.1, 0.2]); rho = np.array([-0.5, -0.5])
    dt = 0.25 / n_steps
    variance_rngs = [np.random.default_rng(seed_v1), np.random.default_rng(seed_v2)]
    price_rng = np.random.default_rng(seed_price)
    paths = np.empty((3, n_steps + 1, n_paths), dtype=float)
    paths[:, 0] = np.array([61.9, 0.2, 0.49])[:, None]
    for step in range(n_steps):
        zv = [rng.standard_normal(n_paths) for rng in variance_rngs]
        zs = [price_rng.standard_normal(n_paths) for _ in range(2)]
        variances = paths[1:, step]
        correlated = [rho[j] * zv[j] + np.sqrt(1.0 - rho[j] ** 2) * zs[j]
                      for j in range(2)]
        return_shock = sum(np.sqrt(variances[j] * dt) * correlated[j]
                           for j in range(2))
        paths[0, step + 1] = paths[0, step] * (1.0 + 0.03 * dt + return_shock)
        for j in range(2):
            proposal = (variances[j] + kappa[j] * (theta[j] - variances[j]) * dt
                        + gamma[j] * np.sqrt(variances[j] * dt) * zv[j])
            paths[j + 1, step + 1] = np.maximum(proposal, 0.0)
    return paths

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return small-grid, one-step, and alternate-stream scalar projections."""
    return [
        {"setup": "import numpy as np\nw=np.arange(60,dtype=float).reshape(3,4,5)+1",
         "call": "float(np.sum(simulate_double_heston_truncated_euler_paths(5,3,11,23,37)*w))",
         "gold_call": "float(np.sum(_oracle_simulate_double_heston_truncated_euler_paths(5,3,11,23,37)*w))"},
        {"setup": "import numpy as np",
         "call": "float(np.sum(simulate_double_heston_truncated_euler_paths(7,1,0,0,0)))",
         "gold_call": "float(np.sum(_oracle_simulate_double_heston_truncated_euler_paths(7,1,0,0,0)))"},
        {"setup": "import numpy as np",
         "call": "float(np.sum(simulate_double_heston_truncated_euler_paths(9,4,101,103,107)**2))",
         "gold_call": "float(np.sum(_oracle_simulate_double_heston_truncated_euler_paths(9,4,101,103,107)**2))"},
    ]
