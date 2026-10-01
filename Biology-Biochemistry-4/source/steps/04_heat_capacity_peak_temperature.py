"""
Find the effective temperature kT* inside the open interval (kT_lo, kT_hi) at which the global spanning-tree heat capacity C(kT) = Var(E)/(kT)^2 of the contact graph is maximal, where the contact weights, the tree energy and the exact tree-ensemble variance are those of the earlier steps, rebuilt at every temperature considered. The maximum is unique in the interval. Return kT* to an absolute accuracy of 1e-10 angstrom; a method whose result is limited by the precision of function values alone does not reach that accuracy.

The framework reads a peak of the heat capacity as the temperature at which the dominant spanning-tree microstates reorganise cooperatively, the combinatorial analogue of a phase transition, and it is the natural operating point at which to compare the channels of a wild type and its mutant. Locating the peak precisely requires the stationarity condition of C with respect to temperature, which involves the third cumulant of the tree energy; the same determinantal structure that gives the variance exactly gives that cumulant exactly.

Returns
-------
float, the peak temperature kT* in angstrom, accurate to 1e-10.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def heat_capacity_peak_temperature(coords: "np.ndarray", r_c: float, kT_lo: float, kT_hi: float) -> float:
    """Find the effective temperature kT* inside the open interval (kT_lo, kT_hi) at which the global spanning-tree heat capacity C(kT) = Var(E)/(kT)^2 of the contact graph is maximal, where the contact weights, the tree energy and the exact tree-ensemble variance are those of the earlier steps, rebuilt at every temperature considered. The maximum is unique in the interval. Return kT* to an absolute accuracy of 1e-10 angstrom; a method whose result is limited by the precision of function values alone does not reach that accuracy.

    Parameters
    ----------
    coords : np.ndarray
        C-alpha coordinates in angstrom, shape (N, 3).
    r_c : float
        Contact cutoff in angstrom.
    kT_lo : float
        Lower end of the search interval (angstrom), > 0.
    kT_hi : float
        Upper end of the search interval (angstrom), > kT_lo.

    Returns
    -------
    kT_peak : float
        Temperature of the unique interior maximum of C(kT), to 1e-10.

    Raises
    ------
    ValueError
        If coords is not a finite (N, 3) array with N >= 3, r_c is not positive, the interval is not 0 < kT_lo < kT_hi, or C has no interior maximum bracketed by the interval.
    """
    return kT_peak

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.linalg import det, slogdet


def _check_coords(coords):
    X = np.asarray(coords, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 3 or not np.all(np.isfinite(X)):
        raise ValueError("coords must be a finite (N, 3) array with N >= 3")
    return X


def _check_positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + " must be a finite positive number")
    return x


def _check_laplacian(L):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite square matrix of size >= 3")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):
        raise ValueError("L must be a symmetric Laplacian with zero row sums")
    return L


def _edges_from_laplacian(L):
    """Lexicographic (i < j) edge list, lengths and weights recovered from the weighted Laplacian."""
    n = L.shape[0]
    ii, jj = np.where(np.triu(L, 1) < 0.0)
    edges = list(zip(ii.tolist(), jj.tolist()))
    if not edges:
        raise ValueError("L has no edges")
    w = -L[ii, jj]
    return edges, w


def _transfer_current(L, edges, w):
    """Symmetric transfer-current matrix K_ab = sqrt(w_a w_b) Y(e_a, e_b), eqs. (8)-(9). The edge vectors chi_e are
    orthogonal to the constant null vector of L, so any generalised inverse gives the same Y as the Moore-Penrose
    pseudoinverse; the grounded inverse of the reduced Laplacian is used because it is well conditioned and exact."""
    n = L.shape[0]
    Kp = np.zeros((n, n))
    Kp[1:, 1:] = np.linalg.inv(L[1:, 1:])
    chi = np.zeros((len(edges), n))
    for a, (i, j) in enumerate(edges):
        chi[a, i] = 1.0
        chi[a, j] = -1.0
    Y = chi @ Kp @ chi.T
    return Y * np.sqrt(np.outer(w, w))


def _tree_energy_cumulants(X, r_c, kT):
    """Exact (Var(E), kappa_3) of the tree energy at temperature kT from the determinantal edge-indicator structure:
    Var = d^T Cov d with Cov = -K*K off the diagonal and p(1-p) on it; kappa_3 = sum d^3 p - 3 sum d_e^2 d_f K_ef^2
    + 2 sum d_e d_f d_g K_ef K_fg K_ge (= -dVar/dbeta)."""
    L = _oracle_contact_laplacian(X, r_c, kT)
    edges, w = _edges_from_laplacian(L)
    d = -kT * np.log(w)
    K = _transfer_current(L, edges, w)
    p = np.diag(K)
    K2 = K ** 2
    cov = -K2
    np.fill_diagonal(cov, p * (1.0 - p))
    var = float(d @ cov @ d)
    k3 = float(np.sum(d ** 3 * p) - 3.0 * np.sum((d ** 2)[:, None] * d[None, :] * K2)
               + 2.0 * (d @ (K * (K @ np.diag(d) @ K)) @ d))
    return var, k3


def _peak_residual(X, r_c, kT):
    """g(kT) = kappa_3 - 2 kT Var(E): positive below the heat-capacity peak, negative above it (dC/dbeta = beta^2 g / kT)."""
    var, k3 = _tree_energy_cumulants(X, r_c, kT)
    return k3 - 2.0 * kT * var


def _oracle_heat_capacity_peak_temperature(coords: "np.ndarray", r_c: float, kT_lo: float, kT_hi: float) -> float:
    """The effective temperature kT* in (kT_lo, kT_hi) at which the global spanning-tree heat capacity C(kT) = Var(E)/(kT)^2
    is maximal, to 1e-12. Stationarity of C in beta = 1/kT gives 2 beta Var(E) - beta^2 kappa_3 = 0, i.e. kappa_3 = 2 kT Var(E),
    with kappa_3 = -dVar/dbeta the third cumulant of the tree energy, evaluated exactly (see _tree_energy_cumulants). The root
    of g(kT) = kappa_3 - 2 kT Var(E) is bracketed by its sign change and polished by safeguarded Newton steps."""
    X = _check_coords(coords)
    r_c = _check_positive(r_c, "r_c")
    lo, hi = float(kT_lo), float(kT_hi)
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValueError("kT_lo and kT_hi must be finite with 0 < kT_lo < kT_hi")
    g_lo, g_hi = _peak_residual(X, r_c, lo), _peak_residual(X, r_c, hi)
    if not (g_lo > 0.0 > g_hi):
        raise ValueError("the heat capacity has no interior maximum bracketed by [kT_lo, kT_hi]")
    a, b = lo, hi
    x = 0.5 * (a + b)
    for _ in range(200):
        gx = _peak_residual(X, r_c, x)
        if gx > 0.0:
            a = x
        else:
            b = x
        h = 1e-6 * x
        dg = (_peak_residual(X, r_c, x + h) - _peak_residual(X, r_c, x - h)) / (2.0 * h)
        x_new = x - gx / dg if dg != 0.0 else 0.5 * (a + b)
        if not (a < x_new < b):
            x_new = 0.5 * (a + b)
        if abs(x_new - x) < 1e-13 or (b - a) < 1e-13:
            x = x_new
            break
        x = x_new
    return float(x)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT_lo, kT_hi = 7.8, 0.2, 5.0\n",
            "call": "heat_capacity_peak_temperature(X, r_c, kT_lo, kT_hi)",
            "gold_call": "_oracle_heat_capacity_peak_temperature(X, r_c, kT_lo, kT_hi)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nXm = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [8.0, -0.3, 7.9], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nX = Xm\nr_c, kT_lo, kT_hi = 7.8, 0.2, 5.0\n",
            "call": "heat_capacity_peak_temperature(X, r_c, kT_lo, kT_hi)",
            "gold_call": "_oracle_heat_capacity_peak_temperature(X, r_c, kT_lo, kT_hi)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [5.5, 3.4, 0.0], [3.8, 6.8, 0.0], [0.0, 6.8, 0.5], [-2.5, 3.4, 1.0], [1.9, 3.4, 4.0]])\nr_c, kT_lo, kT_hi = 5.5, 0.1, 10.0\n",
            "call": "heat_capacity_peak_temperature(X, r_c, kT_lo, kT_hi)",
            "gold_call": "_oracle_heat_capacity_peak_temperature(X, r_c, kT_lo, kT_hi)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\ndef run_model():\n    try:\n        heat_capacity_peak_temperature(X, 7.8, 2.0, 5.0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_heat_capacity_peak_temperature(X, 7.8, 2.0, 5.0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
