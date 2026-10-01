"""
Implement correlation_indices, which measures the dynamic and nondynamic electron
correlation of a closed-shell molecule from the natural-orbital occupations of its
one-particle density matrix.

These indices distinguish electron correlation distributed across many orbitals from correlation associated with nearly degenerate electronic configurations.

Returns
-------
np.ndarray [I_D, I_ND, I_T] of shape (3,), dimensionless
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def correlation_indices(density: "np.ndarray", n_electrons: int) -> "np.ndarray":
    '''Per-electron dynamic, nondynamic and total correlation indices.

    Parameters
    ----------
    density : np.ndarray
        Spin-summed one-particle density matrix of a closed-shell state in an orthonormal
        orbital basis, symmetric, shape (n_mo, n_mo). Its eigenvalues (spatial natural
        occupations, nominally between 0 and 2; small non-positive eigenvalues may occur and
        are skipped) are shared equally by the two spins, so each spatial occupation eta
        contributes two natural spin orbitals of occupation eta / 2.
    n_electrons : int
        Number of electrons N, positive and even.

    Returns
    -------
    indices : np.ndarray
        Array [I_D, I_ND, I_T] of shape (3,), each dimensionless. Natural spin orbitals
        whose occupation is not strictly positive are left out of the sums. The
        per-electron normalization is fixed by one reference state: a two-electron
        density whose two spatial natural orbitals each hold one electron (all four
        natural spin orbitals at occupation 1/2) has I_D = 0 and I_ND = I_T = 1/2.

    Raises
    ------
    ValueError
        If density is not a square symmetric matrix, n_electrons is not a positive even
        integer, or a spatial occupation exceeds 2 by more than 1e-10.
    '''
    return indices

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_correlation_indices(density: "np.ndarray", n_electrons: int) -> "np.ndarray":
    density = np.asarray(density, dtype=float)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density must be a square matrix")
    if not np.allclose(density, density.T, rtol=0.0, atol=1e-10):
        raise ValueError("density must be symmetric")
    if int(n_electrons) != n_electrons or n_electrons <= 0 or int(n_electrons) % 2:
        raise ValueError("n_electrons must be a positive even integer")
    spatial = np.linalg.eigvalsh(0.5 * (density + density.T))
    if np.max(spatial) > 2.0 + 1e-10:
        raise ValueError("a natural occupation exceeds 2")
    spin = np.concatenate([spatial, spatial]) / 2.0
    spin = np.clip(spin[spin > 0.0], 0.0, 1.0)
    fluct = spin * (1.0 - spin)
    i_nd = np.sum(fluct) / n_electrons
    i_t = np.sum(np.sqrt(fluct)) / (2.0 * n_electrons)
    return np.array([i_t - i_nd, i_nd, i_t])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    invalid = """
def run_model():
    try:
        correlation_indices(dc(density), n_electrons)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_correlation_indices(dc(density), n_electrons)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Typical: MP2-like natural occupations of water near equilibrium (6-31G), in a rotated orbital basis ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
occupations = np.array([
    1.9999566042, 1.9899111209, 1.9817609489, 1.9767371543,
    1.9745703151, 0.0226215748, 0.0216150142, 0.0175562668,
    0.0103936822, 0.0021837842, 0.0019422965, 0.0004416274,
    0.0003096106,
])
q, _ = np.linalg.qr(np.random.default_rng(21).normal(size=(13, 13)))
density = q @ np.diag(occupations) @ q.T
density = 0.5 * (density + density.T)
n_electrons = 10
""",
            "call": "correlation_indices(dc(density), n_electrons)",
            "gold_call": "_oracle_correlation_indices(dc(density), n_electrons)",
            "tol": 1e-10,
        },
        # --- Typical: the same molecule with both bonds doubled (larger nondynamic share), rotated basis ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
occupations = np.array([
    1.9999566416, 1.9851815034, 1.9743421749, 1.8308961455,
    1.8267283522, 0.1755331412, 0.1610407455, 0.0240618465,
    0.0111140364, 0.0055282851, 0.0040760264, 0.0007760862,
    0.0007650151,
])
q, _ = np.linalg.qr(np.random.default_rng(22).normal(size=(13, 13)))
density = q @ np.diag(occupations) @ q.T
density = 0.5 * (density + density.T)
n_electrons = 10
""",
            "call": "correlation_indices(dc(density), n_electrons)",
            "gold_call": "_oracle_correlation_indices(dc(density), n_electrons)",
            "tol": 1e-10,
        },
        # --- Boundary: a rotated density with a half-filled spatial pair, occupations close to
        #     2 and 0, and a slightly negative eigenvalue that must be skipped ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
occupations = np.array([1.9999, 1.97, 1.0, 1.0, 0.03, 1e-4, -2e-6])
q, _ = np.linalg.qr(np.random.default_rng(11).normal(size=(7, 7)))
density = q @ np.diag(occupations) @ q.T
density = 0.5 * (density + density.T)
n_electrons = 6
""",
            "call": "correlation_indices(dc(density), n_electrons)",
            "gold_call": "_oracle_correlation_indices(dc(density), n_electrons)",
            "tol": 1e-10,
        },
        # --- Edge: a single-determinant density (all occupations 0 or 2) has zero indices ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
density = np.diag([2.0, 2.0, 0.0, 0.0])
n_electrons = 4
""",
            "call": "correlation_indices(dc(density), n_electrons)",
            "gold_call": "_oracle_correlation_indices(dc(density), n_electrons)",
            "tol": 1e-12,
        },
        # --- Invalid: a spatial occupation above 2 ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
density = np.diag([2.3, 1.7, 0.0])
n_electrons = 4
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: an odd electron count ---
        {
            "setup": """import numpy as np
from copy import deepcopy as dc
density = np.diag([2.0, 1.0, 0.0])
n_electrons = 3
""" + invalid,
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
