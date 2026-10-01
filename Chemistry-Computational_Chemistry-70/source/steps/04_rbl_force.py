"""
Implement a single-evaluation force estimator that reproduces RBMD 2.0's

random batch list (RBL) method: for each particle, sum the exact

Lennard-Jones force from all neighbors inside a core cutoff, add a

reweighted stochastic estimate of the force from a randomly sampled subset of

the neighbors in an outer shell, and apply the momentum-conserving

correction that removes the net spurious force the sampling would otherwise

introduce.

RBMD 2.0 accelerates short-range force evaluation by replacing the full

neighbor sum with a core-shell decomposition: neighbors within a core cutoff

r_c are summed exactly, while neighbors in the shell r_c < r <= r_s are

represented by a uniformly sampled p-element subset, reweighted by the

shell's true cardinality divided by the sample size (Eq. 10-11). Because this

reweighted stochastic sum is unbiased in expectation but not exactly momentum

conserving for a single random draw, a correction that subtracts the mean

sampled force across all particles is applied (Eq. 12). This routine

evaluates the resulting force once, for one configuration and one draw from

the random stream; Step 5 will call it repeatedly across an integration

trajectory using a single continuing pseudo-random stream.

Returns
-------
np.ndarray of shape (N, 3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rbl_force(positions: np.ndarray, box_length: float, core_cutoff: float,
               cutoff: float, batch_size: int, rng: np.random.Generator,
               epsilon: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with N >= 2, particle positions.
    box_length : float
        Side length of the cubic periodic box.
    core_cutoff : float
        Core cutoff r_c; neighbors within this distance are always included
        exactly.
    cutoff : float
        Full interaction cutoff r_s at which the underlying shifted-force
        Lennard-Jones law vanishes; must exceed `core_cutoff`.
    batch_size : int
        Number of shell neighbors to sample per particle when the shell is
        larger than `batch_size`.
    rng : np.random.Generator
        Random number generator (e.g. from `np.random.default_rng(seed)`)
        used for shell sampling. Its internal state is advanced by this call.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.

    Returns
    -------
    forces : np.ndarray
        Array of shape (N, 3), the momentum-conserving RBL force estimate on
        each particle.


    Raises
    ------
    ValueError
        If `positions` does not have shape (N, 3) with N >= 2, or contains
        non-finite values; if `box_length`, `core_cutoff`, `epsilon`, or
        `sigma` is not finite and > 0; if `cutoff` is not finite, >
        `core_cutoff`, and < `box_length` / 2; if `batch_size` is not a
        positive integer; or if `rng` is not an instance of
        `np.random.Generator`.
    """
    forces = np.zeros_like(positions)  # placeholder
    return forces

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_rbl_force(positions: np.ndarray, box_length: float, core_cutoff: float,
                       cutoff: float, batch_size: int, rng: np.random.Generator,
                       epsilon: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """Reference implementation."""

    import numpy as np

    def _pair_force_magnitude(r):
        sr6 = (sigma / r) ** 6
        sr12 = sr6 * sr6
        F = 24.0 * epsilon / r * (2.0 * sr12 - sr6)
        src6 = (sigma / cutoff) ** 6
        src12 = src6 * src6
        Fc = 24.0 * epsilon / cutoff * (2.0 * src12 - src6)
        return F - Fc

    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (N, 3) with N >= 2.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values.")
    if not np.isfinite(box_length) or box_length <= 0:
        raise ValueError("box_length must be finite and > 0.")
    if not np.isfinite(core_cutoff) or core_cutoff <= 0:
        raise ValueError("core_cutoff must be finite and > 0.")
    if not np.isfinite(cutoff) or cutoff <= core_cutoff or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > core_cutoff, and < box_length / 2.")
    if not (isinstance(batch_size, (int, np.integer)) and not isinstance(batch_size, bool) and batch_size > 0):
        raise ValueError("batch_size must be a positive integer.")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an instance of np.random.Generator.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")

    n = positions.shape[0]
    diff = positions[:, None, :] - positions[None, :, :]
    diff = diff - box_length * np.round(diff / box_length)
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)

    force_star = np.zeros((n, 3))
    for i in range(n):
        d_row = dist[i]
        core_idx = np.where(d_row <= core_cutoff)[0]
        shell_idx = np.where((d_row > core_cutoff) & (d_row <= cutoff))[0]

        contribution = np.zeros(3)
        if core_idx.size > 0:
            d_core = d_row[core_idx]
            f_core = _pair_force_magnitude(d_core)
            rhat = diff[i, core_idx] / d_core[:, None]
            contribution += (f_core[:, None] * rhat).sum(axis=0)

        n_shell = shell_idx.size
        if n_shell > 0:
            if n_shell <= batch_size:
                sample = shell_idx
                weight = 1.0
            else:
                sample = rng.choice(shell_idx, size=batch_size, replace=False)
                weight = n_shell / batch_size
            d_shell = d_row[sample]
            f_shell = _pair_force_magnitude(d_shell)
            rhat = diff[i, sample] / d_shell[:, None]
            contribution += weight * (f_shell[:, None] * rhat).sum(axis=0)

        force_star[i] = contribution

    forces = force_star - force_star.sum(axis=0, keepdims=True) / n
    return forces

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal scenario: 8 particles, shell larger than batch_size, so
            # stochastic subsampling is actually exercised.
            "setup": (
                "import numpy as np\n"
                "positions = np.array([\n"
                "    [1.0, 1.0, 1.0],\n"
                "    [1.8, 1.0, 1.0],\n"
                "    [1.0, 1.8, 1.0],\n"
                "    [1.0, 1.0, 1.8],\n"
                "    [2.4, 1.2, 1.1],\n"
                "    [1.1, 2.4, 1.3],\n"
                "    [1.3, 1.1, 2.4],\n"
                "    [2.2, 2.2, 2.2],\n"
                "])\n"
                "box_length = 10.0\n"
                "core_cutoff = 1.0\n"
                "cutoff = 2.5\n"
                "batch_size = 3\n"
                "rng_call = np.random.default_rng(42)\n"
                "rng_gold = np.random.default_rng(42)\n"
            ),
            "call": "rbl_force(positions, box_length, core_cutoff, cutoff, batch_size, rng_call)",
            "gold_call": "_oracle_rbl_force(positions, box_length, core_cutoff, cutoff, batch_size, rng_gold)",
        },
        {
            # Boundary case: the shell has exactly `batch_size` members, so
            # the exact-sum (weight = 1) branch is taken with no sampling.
            "setup": (
                "import numpy as np\n"
                "positions = np.array([\n"
                "    [1.0, 1.0, 1.0],\n"
                "    [2.5, 1.0, 1.0],\n"
                "    [1.0, 2.5, 1.0],\n"
                "])\n"
                "box_length = 10.0\n"
                "core_cutoff = 1.0\n"
                "cutoff = 3.0\n"
                "batch_size = 2\n"
                "rng_call = np.random.default_rng(7)\n"
                "rng_gold = np.random.default_rng(7)\n"
            ),
            "call": "rbl_force(positions, box_length, core_cutoff, cutoff, batch_size, rng_call)",
            "gold_call": "_oracle_rbl_force(positions, box_length, core_cutoff, cutoff, batch_size, rng_gold)",
        },
        {
            # Edge case: one particle lies beyond `cutoff` from every other
            # particle, so its RBL estimate before the momentum correction is
            # exactly zero.
            "setup": (
                "import numpy as np\n"
                "positions = np.array([\n"
                "    [1.0, 1.0, 1.0],\n"
                "    [1.6, 1.0, 1.0],\n"
                "    [1.0, 1.6, 1.0],\n"
                "    [8.0, 8.0, 8.0],\n"
                "])\n"
                "box_length = 20.0\n"
                "core_cutoff = 1.0\n"
                "cutoff = 2.0\n"
                "batch_size = 4\n"
                "rng_call = np.random.default_rng(3)\n"
                "rng_gold = np.random.default_rng(3)\n"
            ),
            "call": "rbl_force(positions, box_length, core_cutoff, cutoff, batch_size, rng_call)",
            "gold_call": "_oracle_rbl_force(positions, box_length, core_cutoff, cutoff, batch_size, rng_gold)",
        },
    ]
