"""
Anharmonic force field rewritten in local X-H bond coordinates: the cubic and quartic Hamiltonian coefficients of every canonical monomial, split into pure X-H, mixed and pure-bath classes, with the diagonal bond terms removed and small coefficients screened out.

Quantum-chemistry programs report the cubic and quartic force field in normal coordinates, one derivative of the potential per canonical index tuple i <= j <= k (or i <= j <= k <= l), so that the potential is the usual Taylor series with those derivatives. The doorway treatment rewrites the X-H block in local bond coordinates xi_a = sum_i C_ai q_i, where q_i runs over the X-H normal coordinates, and uses q_i = sum_a A_ia xi_a with A the Moore-Penrose inverse of C. Bath coordinates are left as they are. The rewritten potential is again a sum over canonical monomials, now in the local and bath coordinates, and each monomial carries one Hamiltonian coefficient.

Terms that contain a single bond coordinate raised to the third or fourth power are dropped, because the Morse potential of that bond already contains its diagonal anharmonicity. Every other monomial is assigned to a class by its coordinates: pure X-H (only bond coordinates), mixed (at least one bond and at least one bath coordinate) or pure bath. A coefficient is kept only if its magnitude reaches the screening threshold of its class and order; this screening is applied to the Hamiltonian coefficient of the monomial, after the transformation. All vibrational coordinates are dimensionless. For a harmonic mode of frequency omega (cm^-1) the Hamiltonian is (omega / 2)(-d^2/dq^2 + q^2), and a local X-H bond of harmonic frequency omega and anharmonicity x has the Hamiltonian omega [ -(1/2) d^2/dxi^2 + (1 / (4x)) (1 - exp(-sqrt(2x) xi))^2 ] in its own dimensionless coordinate xi.

Returns
-------
numpy.ndarray of shape (L, 7): rows (order, a1, a2, a3, a4, coefficient in cm^-1, class) of the retained local-coordinate cubic and quartic monomials
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import itertools
import math

import numpy as np


def local_force_field(C: np.ndarray, cubic: np.ndarray, quartic: np.ndarray, n_bath: int, thresholds: np.ndarray) -> np.ndarray:
    '''Local-coordinate cubic and quartic Hamiltonian coefficients.

    Parameters
    ----------
    C : numpy.ndarray
        Matrix of shape (N_x, M_x) with xi_a = sum_i C[a, i] q_i, a over the N_x
        local bonds and i over the M_x X-H normal coordinates.
    cubic : numpy.ndarray
        Shape (n3, 4), rows (i, j, k, F) with 0-based normal-coordinate indices
        i <= j <= k (0..M_x-1 the X-H normal modes, M_x..M_x+n_bath-1 the bath
        modes) and F = d^3V / dq_i dq_j dq_k in cm^-1. Each tuple appears once.
    quartic : numpy.ndarray
        Shape (n4, 5), rows (i, j, k, l, F) with i <= j <= k <= l and
        F = d^4V / dq_i dq_j dq_k dq_l in cm^-1. Each tuple appears once.
    n_bath : int
        Number of bath modes.
    thresholds : numpy.ndarray
        Six positive screening thresholds in cm^-1, in the order (pure X-H cubic,
        pure X-H quartic, mixed cubic, mixed quartic, pure bath cubic, pure bath
        quartic). A coefficient G is kept when |G| >= threshold.

    Returns
    -------
    terms : numpy.ndarray
        Shape (L, 7). Row (r, a1, a2, a3, a4, G, cls): order r (3 or 4), local-frame
        coordinate indices a1 <= ... <= ar (0..N_x-1 the bonds, N_x..N_x+n_bath-1
        the bath modes in their input order), a4 = -1 for r = 3, G the coefficient
        in cm^-1 of the monomial xi_a1 ... xi_ar in the potential, and cls = 0 for
        pure X-H, 1 for mixed, 2 for pure bath. All cubic rows come first, then
        the quartic rows, each block in lexicographic order of its indices. An
        empty result has shape (0, 7).

    Raises
    ------
    ValueError
        If C is not two-dimensional, n_bath is negative, thresholds does not hold
        six positive numbers, or an index tuple is not integer, not canonical
        (non-decreasing), out of range or repeated.
    '''
    return terms

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import itertools
import math

import numpy as np


def _oracle_local_force_field(C: np.ndarray, cubic: np.ndarray, quartic: np.ndarray, n_bath: int, thresholds: np.ndarray) -> np.ndarray:
    C = np.array(C, dtype=float)
    if C.ndim != 2:
        raise ValueError("C must be a matrix")
    n_x = C.shape[0]
    n_bath = int(n_bath)
    if n_bath < 0:
        raise ValueError("n_bath must be non-negative")
    thr = np.array(thresholds, dtype=float).ravel()
    if thr.size != 6 or np.any(thr <= 0.0):
        raise ValueError("thresholds must hold six positive numbers")
    n_modes = C.shape[1] + n_bath
    n_all = n_x + n_bath
    A = np.linalg.pinv(C)
    T = np.zeros((n_modes, n_all))
    T[:C.shape[1], :n_x] = A
    T[C.shape[1]:, n_x:] = np.eye(n_bath)
    rows = []
    for arr, r in ((cubic, 3), (quartic, 4)):
        arr = np.array(arr, dtype=float).reshape(-1, r + 1)
        phi = np.zeros((n_modes,) * r)
        seen = set()
        for row in arr:
            idx = tuple(int(round(v)) for v in row[:r])
            if any(abs(v - round(v)) > 1e-12 for v in row[:r]):
                raise ValueError("indices must be integers")
            if list(idx) != sorted(idx) or idx[0] < 0 or idx[-1] >= n_modes:
                raise ValueError("index tuples must be canonical and in range")
            if idx in seen:
                raise ValueError("duplicate index tuple")
            seen.add(idx)
            for perm in set(itertools.permutations(idx)):
                phi[perm] = row[r]
        if r == 3:
            psi = np.einsum('ijk,ia,jb,kc->abc', phi, T, T, T, optimize=True)
        else:
            psi = np.einsum('ijkl,ia,jb,kc,ld->abcd', phi, T, T, T, T, optimize=True)
        for tup in itertools.combinations_with_replacement(range(n_all), r):
            n_loc = sum(1 for u in tup if u < n_x)
            if n_loc == r and len(set(tup)) == 1:
                continue
            mult = 1
            for u in set(tup):
                mult *= math.factorial(tup.count(u))
            coef = psi[tup] / mult
            cls = 0 if n_loc == r else (2 if n_loc == 0 else 1)
            if abs(coef) < thr[2 * cls + (r - 3)]:
                continue
            rows.append([r] + list(tup) + [-1] * (4 - r) + [coef, cls])
    if not rows:
        return np.zeros((0, 7))
    return np.array(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the benchmark force field with three bonds and four bath modes ---
        {
            "setup": """import numpy as np
omega_loc = np.array([3050.0, 3420.0, 3660.0])
x_loc = np.array([0.0205, 0.0215, 0.0225])
delta_loc = np.array([0.12, -0.08, 0.05])
C = np.array([[0.93, 0.25, 0.12], [-0.21, 0.95, 0.20], [0.10, -0.24, 0.97]])
p_xh = np.array([20.0, -12.0, 8.0])
omega_bath = np.array([1605.0, 1460.0, 1180.0, 760.0])
delta_bath = np.array([0.62, 0.35, 0.48, 0.85])
p_bath = np.array([45.0, 20.0, -15.0, 10.0])
cubic = np.array([[0, 0, 1, -180.0], [0, 1, 2, 95.0], [1, 1, 2, -120.0], [0, 2, 2, 60.0], [0, 4, 4, 195.0], [1, 4, 4, -140.0],
                  [2, 3, 4, 105.0], [0, 3, 5, 68.0], [1, 5, 6, 45.0], [2, 6, 6, -38.0], [0, 5, 5, 60.0], [3, 4, 6, 35.0],
                  [4, 5, 6, 18.0], [5, 6, 6, -22.0], [3, 6, 6, 5.5], [4, 6, 6, 5.5], [3, 4, 5, 2.5], [5, 5, 6, 5.0]])
quartic = np.array([[0, 0, 1, 1, 40.0], [0, 1, 2, 2, -25.0], [0, 0, 4, 4, 27.0], [1, 1, 3, 3, -18.0], [0, 1, 4, 6, 14.0],
                    [2, 2, 5, 5, 10.0], [3, 3, 4, 4, 36.0], [4, 4, 4, 4, 60.0], [5, 5, 6, 6, 48.0], [3, 4, 5, 6, 14.0], [4, 4, 4, 6, 66.0]])
thresholds = np.array([1.0, 1.0, 3.0, 3.0, 3.0, 10.0])
""",
            "call": "local_force_field(C.copy(), cubic.copy(), quartic.copy(), 4, thresholds.copy())",
            "gold_call": "_oracle_local_force_field(C.copy(), cubic.copy(), quartic.copy(), 4, thresholds.copy())",
            "tol": 1e-08,
        },
        # --- Normal: two bonds built from three X-H normal modes (non-square C), two bath modes ---
        {
            "setup": """import numpy as np
C = np.array([[0.80, 0.45, 0.20], [-0.30, 0.70, 0.55]])
cubic = np.array([[0, 0, 1, -140.0], [0, 1, 2, 75.0], [2, 2, 2, 110.0], [0, 3, 3, 150.0], [1, 3, 4, 50.0], [2, 4, 4, -30.0],
                  [3, 3, 4, 14.0], [3, 4, 4, 4.0]])
quartic = np.array([[0, 0, 1, 1, 36.0], [1, 1, 2, 2, -20.0], [0, 0, 3, 3, 24.0], [1, 2, 3, 4, 12.0], [3, 3, 4, 4, 44.0],
                    [4, 4, 4, 4, 90.0]])
thresholds = np.array([0.8, 0.8, 2.5, 2.5, 3.0, 10.0])
""",
            "call": "local_force_field(C.copy(), cubic.copy(), quartic.copy(), 2, thresholds.copy())",
            "gold_call": "_oracle_local_force_field(C.copy(), cubic.copy(), quartic.copy(), 2, thresholds.copy())",
            "tol": 1e-08,
        },
        # --- Boundary: bonds identical to the normal modes (C = identity), so only the diagonal removal and the screening act ---
        {
            "setup": """import numpy as np
C = np.eye(3)
cubic = np.array([[0, 0, 1, -140.0], [0, 1, 2, 75.0], [2, 2, 2, 110.0], [0, 3, 3, 150.0], [1, 3, 4, 50.0], [2, 4, 4, -30.0],
                  [3, 3, 4, 14.0], [3, 4, 4, 4.0]])
quartic = np.array([[0, 0, 1, 1, 36.0], [1, 1, 2, 2, -20.0], [0, 0, 3, 3, 24.0], [1, 2, 3, 4, 12.0], [3, 3, 4, 4, 44.0],
                    [4, 4, 4, 4, 90.0]])
thresholds = np.array([0.8, 0.8, 2.5, 2.5, 3.0, 10.0])
""",
            "call": "local_force_field(C.copy(), cubic.copy(), quartic.copy(), 2, thresholds.copy())",
            "gold_call": "_oracle_local_force_field(C.copy(), cubic.copy(), quartic.copy(), 2, thresholds.copy())",
            "tol": 1e-08,
        },
        # --- Edge: no bath modes and a pure X-H force field ---
        {
            "setup": """import numpy as np
C = np.array([[0.9, 0.35], [-0.2, 1.1]])
cubic = np.array([[0, 0, 0, -300.0], [0, 0, 1, 120.0], [1, 1, 1, 80.0]])
quartic = np.array([[0, 0, 1, 1, 30.0], [1, 1, 1, 1, 50.0]])
thresholds = np.array([0.5, 0.5, 1.0, 1.0, 1.0, 1.0])
""",
            "call": "local_force_field(C.copy(), cubic.copy(), quartic.copy(), 0, thresholds.copy())",
            "gold_call": "_oracle_local_force_field(C.copy(), cubic.copy(), quartic.copy(), 0, thresholds.copy())",
            "tol": 1e-08,
        },
        # --- Edge: screening thresholds above every coefficient, so no term survives ---
        {
            "setup": """import numpy as np
C = np.array([[0.80, 0.45, 0.20], [-0.30, 0.70, 0.55]])
cubic = np.array([[0, 0, 1, -140.0], [0, 1, 2, 75.0], [2, 2, 2, 110.0], [0, 3, 3, 150.0], [1, 3, 4, 50.0], [2, 4, 4, -30.0],
                  [3, 3, 4, 14.0], [3, 4, 4, 4.0]])
quartic = np.array([[0, 0, 1, 1, 36.0], [1, 1, 2, 2, -20.0], [0, 0, 3, 3, 24.0], [1, 2, 3, 4, 12.0], [3, 3, 4, 4, 44.0],
                    [4, 4, 4, 4, 90.0]])
thresholds = np.array([1e4, 1e4, 1e4, 1e4, 1e4, 1e4])
""",
            "call": "local_force_field(C.copy(), cubic.copy(), quartic.copy(), 2, thresholds.copy())",
            "gold_call": "_oracle_local_force_field(C.copy(), cubic.copy(), quartic.copy(), 2, thresholds.copy())",
            "tol": 1e-08,
        },
    ]
