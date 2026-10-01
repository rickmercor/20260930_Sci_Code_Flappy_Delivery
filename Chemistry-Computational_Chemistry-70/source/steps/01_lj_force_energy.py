"""
Implement a function that computes the net force on every particle and the total potential energy of an N-particle configuration, using a finite-range, force-shifted Lennard-Jones potential evaluated under the minimum-image convention in a cubic periodic box. Pairs separated (by minimum image) by more than the cutoff contribute zero force and zero energy; pairs within the cutoff use the shifted-force form so that both energy and force go to zero continuously at r = r_c.

Both RBMD 2.0's random batch list (RBL) estimator and DINaMo's physics-informed

trajectory solver are built on the same pairwise interaction: a finite-range,

force-shifted Lennard-Jones potential evaluated under the minimum-image

convention of a cubic periodic box. Every later step in this problem (initial-

condition construction and relaxation, the exact velocity-Verlet reference,

the RBL stochastic estimator, and the Newton/energy residuals used to train

the DINaMo-style solver) calls this same routine, so it must be implemented

once, exactly, and shared. The force-shifted form




    U_sf(r) = U_LJ(r) - U_LJ(r_c) + (r - r_c) * F_LJ(r_c),   r <= r_c

    F_sf(r) = F_LJ(r) - F_LJ(r_c),                            r <= r_c




(with U_LJ(r) = 4*epsilon*((sigma/r)**12 - (sigma/r)**6) and F_LJ(r) = -dU_LJ/dr)

is chosen because both papers require energy and force to vanish smoothly at

the cutoff rather than truncating discontinuously.

Returns
-------
np.ndarray, float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def lj_force_energy(positions: np.ndarray, box_length: float, cutoff: float,
                     epsilon: float = 1.0, sigma: float = 1.0) -> tuple:
    """
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with N >= 2, giving the Cartesian coordinates of
        each particle. Values need not already be wrapped into the primary box.
    box_length : float
        Side length of the cubic periodic box.
    cutoff : float
        Interaction cutoff r_c for the shifted-force potential.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.

    Returns
    -------
    forces : np.ndarray
        Array of shape (N, 3), the net shifted-force Lennard-Jones force on
        each particle.
    potential_energy : float
        Total shifted-force Lennard-Jones potential energy of the
        configuration, as a native Python float.


    Raises
    ------
    ValueError
        If `positions` does not have shape (N, 3) with N >= 2, or contains
        non-finite values; if `box_length`, `epsilon`, or `sigma` is not
        finite and > 0; or if `cutoff` is not finite, > 0, and <
        `box_length` / 2.
    """
    return np.zeros_like(positions), 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_lj_force_energy(positions: np.ndarray, box_length: float, cutoff: float,
                             epsilon: float = 1.0, sigma: float = 1.0) -> tuple:
    """Reference implementation."""

    import numpy as np

    def _min_image_pair_data(pos, box):
        """Local helper: minimum-image pairwise displacement vectors and distances."""
        diff = pos[:, None, :] - pos[None, :, :]
        diff = diff - box * np.round(diff / box)
        dist = np.linalg.norm(diff, axis=-1)
        return diff, dist

    positions = np.asarray(positions, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ValueError("positions must have shape (N, 3) with N >= 2.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values.")
    if not np.isfinite(box_length) or box_length <= 0:
        raise ValueError("box_length must be finite and > 0.")
    if not np.isfinite(cutoff) or cutoff <= 0 or cutoff >= box_length / 2:
        raise ValueError("cutoff must be finite, > 0, and < box_length / 2.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and > 0.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and > 0.")

    n = positions.shape[0]
    diff, dist = _min_image_pair_data(positions, box_length)

    iu, ju = np.triu_indices(n, k=1)
    d = dist[iu, ju]
    mask = d <= cutoff

    forces = np.zeros_like(positions)
    potential_energy = 0.0

    if np.any(mask):
        d_in = d[mask]
        dvec_in = diff[iu, ju][mask]

        sr6 = (sigma / d_in) ** 6
        sr12 = sr6 * sr6
        U = 4.0 * epsilon * (sr12 - sr6)
        F = 24.0 * epsilon / d_in * (2.0 * sr12 - sr6)

        src6 = (sigma / cutoff) ** 6
        src12 = src6 * src6
        Uc = 4.0 * epsilon * (src12 - src6)
        Fc = 24.0 * epsilon / cutoff * (2.0 * src12 - src6)

        Usf = U - Uc + (d_in - cutoff) * Fc
        Fsf = F - Fc

        rhat = dvec_in / d_in[:, None]
        fij = Fsf[:, None] * rhat

        ii = iu[mask]
        jj = ju[mask]
        np.add.at(forces, ii, fij)
        np.add.at(forces, jj, -fij)
        potential_energy = float(np.sum(Usf))

    return forces, potential_energy

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            # Normal scenario: 5 particles at generic, well-separated positions.
            "setup": (
                "import numpy as np\n"
                "positions = np.array([\n"
                "    [0.5, 0.5, 0.5],\n"
                "    [1.6, 0.7, 0.4],\n"
                "    [0.9, 1.8, 1.1],\n"
                "    [2.3, 2.1, 0.6],\n"
                "    [1.2, 1.3, 2.4],\n"
                "])\n"
                "box_length = 5.0\n"
                "cutoff = 2.0\n"
            ),
            "call": "lj_force_energy(positions, box_length, cutoff)",
            "gold_call": "_oracle_lj_force_energy(positions, box_length, cutoff)",
        },
        {
            # Boundary case: two particles separated by exactly the cutoff
            # distance, where the shifted-force potential and force are
            # defined to vanish exactly.
            "setup": (
                "import numpy as np\n"
                "box_length = 10.0\n"
                "cutoff = 2.5\n"
                "positions = np.array([\n"
                "    [1.0, 1.0, 1.0],\n"
                "    [1.0 + cutoff, 1.0, 1.0],\n"
                "])\n"
            ),
            "call": "lj_force_energy(positions, box_length, cutoff)",
            "gold_call": "_oracle_lj_force_energy(positions, box_length, cutoff)",
        },
        {
            # Edge case: two particles in the strongly repulsive regime,
            # well inside the cutoff, with non-default epsilon and sigma.
            "setup": (
                "import numpy as np\n"
                "box_length = 8.0\n"
                "cutoff = 3.0\n"
                "positions = np.array([\n"
                "    [4.0, 4.0, 4.0],\n"
                "    [4.0 + 0.85, 4.0, 4.0],\n"
                "])\n"
                "epsilon = 1.5\n"
                "sigma = 1.0\n"
            ),
            "call": "lj_force_energy(positions, box_length, cutoff, epsilon, sigma)",
            "gold_call": "_oracle_lj_force_energy(positions, box_length, cutoff, epsilon, sigma)",
        },
    ]
