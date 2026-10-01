"""
Assemble the consistent tangent operator of Eq. (30) from the eigenstrain directions and their activity. The source builds it through the reduction it names in the text under Eq. (30) rather than by inverting the full iteration matrix, and it gives that reason explicitly. A direction whose strength was not exceeded is handled by a specific, stated substitution in both factors of that reduction. The small stabilising modulus appears on the reduced diagonal.

Eq. (25) defines the tangent as the strain derivative of the stress at fixed history, and Eq. (26) writes the single-direction case in closed form, which shows what the stabilising modulus does to it. Eq. (30) generalises that to several facets. The source reports that forming the reduction this way converged in every case it tried while the direct inverse did not.

Returns
-------
A (6, 6) float64 consistent tangent operator in the same layout as the elastic operator.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def consistent_tangent(dirs: "np.ndarray", active: "np.ndarray", E: float, nu: float, kappa_t: float) -> "np.ndarray":
    """Assemble the consistent tangent operator of Eq. (30) from the eigenstrain directions and
    their activity, through the reduction the source names in the text under Eq. (30), with
    an inactive direction handled by the substitution stated there.

    Args:
        dirs: An (m, 6) array whose rows are the eigenstrain directions, in the source's
            six-component tensor layout.
        active: An (m,) array of activity flags, 1.0 for a direction whose strength was
            exceeded and 0.0 otherwise.
        E: Young's modulus, in GPa.
        nu: Poisson ratio.
        kappa_t: The small stabilising modulus factor of the return mapping, non-negative.

    Returns:
        A (6, 6) float64 consistent tangent operator in the same layout as the elastic
        operator.

    Raises:
        ValueError: If dirs is not a two-dimensional array with six columns, if active does
            not carry exactly one flag per direction, if kappa_t is negative or not finite,
            or if E or nu lies outside the domain accepted by the elastic operator step.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_consistent_tangent(dirs: "np.ndarray", active: "np.ndarray", E: float, nu: float, kappa_t: float) -> "np.ndarray":
    dirs = np.asarray(dirs, dtype=float)
    active = np.asarray(active, dtype=float)
    if dirs.ndim != 2 or dirs.shape[1] != 6:
        raise ValueError("dirs must be a (m, 6) array of directions")
    if active.shape != (dirs.shape[0],):
        raise ValueError("active must carry one flag per direction")
    if kappa_t < 0.0 or not np.isfinite(kappa_t):
        raise ValueError("kappa_t must be a non-negative finite factor")
    blk0 = _oracle_elastic_operator(E, nu)
    D = blk0[:6, :]
    K = float(blk0[6, 0])
    g = np.array(dirs, dtype=float).T.copy()
    m = g.shape[1]
    S = g.T @ D @ g + kappa_t * K * np.eye(m)
    for i in range(m):
        if active[i] <= 0.0:
            g[:, i] = 0.0
            S[i, :] = 0.0
            S[:, i] = 0.0
            S[i, i] = 1.0
    return D - D @ g @ np.linalg.solve(S, g.T @ D)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndirs = np.array([[0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0], [0.4, -0.3, -0.1, 0.0, 0.0, 0.8602325267042627]])\nactive = np.array([1.0, 1.0])\nE = 200.0\nnu = 0.3\nkappa_t = 1.0e-9\n',
         'call': 'consistent_tangent(dirs, active, E, nu, kappa_t)',
         'gold_call': '_oracle_consistent_tangent(dirs, active, E, nu, kappa_t)'},
        {'setup': 'import numpy as np\ndirs = np.array([[-0.3333333333333333, -0.3333333333333333, -0.3333333333333333, 0.0, 0.0, 0.0], [0.1, -0.5, 0.4, 0.0, 0.0, 0.7615773105863909]])\nactive = np.array([0.0, 1.0])\nE = 200.0\nnu = 0.3\nkappa_t = 1.0e-9\n',
         'call': 'consistent_tangent(dirs, active, E, nu, kappa_t)',
         'gold_call': '_oracle_consistent_tangent(dirs, active, E, nu, kappa_t)'},
        {'setup': 'import numpy as np\ndirs = np.array([[0.5, 0.4, 0.1, 0.2, 0.0, 0.7416198487095663], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])\nactive = np.array([1.0, 0.0])\nE = 70.0\nnu = 0.22\nkappa_t = 1.0e-6\n',
         'call': 'consistent_tangent(dirs, active, E, nu, kappa_t)',
         'gold_call': '_oracle_consistent_tangent(dirs, active, E, nu, kappa_t)'},
        {'setup': 'import numpy as np\n# invalid input: three activity flags for two directions must raise ValueError\ndirs = np.array([[0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 0.0, 0.0, 0.0], [0.4, -0.3, -0.1, 0.0, 0.0, 0.8602325267042627]])\nactive = np.array([1.0, 1.0, 0.0])\nE = 200.0\nnu = 0.3\nkappa_t = 1.0e-9\ndef _catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n',
         'call': '_catches_value_error(lambda: consistent_tangent(dirs, active, E, nu, kappa_t))',
         'gold_call': '_catches_value_error(lambda: _oracle_consistent_tangent(dirs, active, E, nu, kappa_t))'},
    ]
