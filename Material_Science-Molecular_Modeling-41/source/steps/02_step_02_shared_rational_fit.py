"""
Fit matrix samples using a common scalar barycentric denominator.

A shared barycentric denominator fits the coupled velocity process. Conjugate-closed support and weights keep the resulting dynamics real.

Returns
-------
Tuple (indices,weights): integer (p,) and complex (p,) arrays in     support insertion order. 2<=p<=max_support, ||weights||_2=1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def shared_rational_fit(z: "np.ndarray", values: "np.ndarray", tolerance: float, max_support: int) -> "tuple[np.ndarray, np.ndarray]":
    """Fit matrix samples using a common scalar barycentric denominator.

    Parameters
    ----------
    z : complex array (g,)
        Counterclockwise uniform circle rho*exp(2*pi*i*l/g), even g>=8,
        rho>1. Circle agreement tolerance is 1e-10 in absolute entries.
    values : complex array (g,d,d), d>=1
        Conjugate symmetry values[(-l)%g]=conj(values[l]), tolerance 1e-9.
    tolerance : finite real scalar > 0
        Maximum Frobenius residual allowed over the entire grid.
    max_support : integer, 2<=max_support<=g-2
        Maximum number of support points, including conjugate partners.

    Contract
    --------
    Use the source's matrix-valued rational fit with a shared denominator.
    Initial support indices are [0,g//2]. Support and weights are
    conjugate paired, and real supports have real weights. Normalize
    the weight vector to Euclidean norm one; a common sign is immaterial.
    The convergence measure is the largest Frobenius residual on the grid.
    Nonsupport conjugate orbits are scored by their mean residual. Among
    orbits within 1e-12*max(1,largest_score), choose the smallest index
    in 0,...,g//2, inserted before its distinct conjugate partner.
    Any minimizing weights are accepted through the fitted function.

    Returns
    -------
    result
        Tuple (indices,weights): integer (p,) and complex (p,) arrays in
        support insertion order. 2<=p<=max_support, ||weights||_2=1.

    Raises
    ------
    ValueError
        Nonfinite inputs; invalid shapes, circle, conjugacy, tolerance or
        support budget; or residual tolerance not reached within budget.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_shared_rational_fit(z: "np.ndarray", values: "np.ndarray", tolerance: float, max_support: int) -> "tuple[np.ndarray, np.ndarray]":
    """Fit matrix samples using a common scalar barycentric denominator.

    Parameters
    ----------
    z : complex array (g,)
        Counterclockwise uniform circle rho*exp(2*pi*i*l/g), even g>=8,
        rho>1. Circle agreement tolerance is 1e-10 in absolute entries.
    values : complex array (g,d,d), d>=1
        Conjugate symmetry values[(-l)%g]=conj(values[l]), tolerance 1e-9.
    tolerance : finite real scalar > 0
        Maximum Frobenius residual allowed over the entire grid.
    max_support : integer, 2<=max_support<=g-2
        Maximum number of support points, including conjugate partners.

    Contract
    --------
    Use the source's matrix-valued rational fit with a shared denominator.
    Initial support indices are [0,g//2]. Support and weights are
    conjugate paired, and real supports have real weights. Normalize
    the weight vector to Euclidean norm one; a common sign is immaterial.
    The convergence measure is the largest Frobenius residual on the grid.
    Nonsupport conjugate orbits are scored by their mean residual. Among
    orbits within 1e-12*max(1,largest_score), choose the smallest index
    in 0,...,g//2, inserted before its distinct conjugate partner.
    Any minimizing weights are accepted through the fitted function.

    Returns
    -------
    result
        Tuple (indices,weights): integer (p,) and complex (p,) arrays in
        support insertion order. 2<=p<=max_support, ||weights||_2=1.

    Raises
    ------
    ValueError
        Nonfinite inputs; invalid shapes, circle, conjugacy, tolerance or
        support budget; or residual tolerance not reached within budget.
    """
    import numpy as np
    z = np.asarray(z, dtype=complex)
    v = np.asarray(values, dtype=complex)
    if z.ndim != 1 or len(z) < 8 or len(z) % 2 or (v.ndim != 3) or (v.shape[0] != len(z)) or (v.shape[1] < 1) or (v.shape[1] != v.shape[2]) or (not np.isfinite(z).all()) or (not np.isfinite(v).all()):
        raise ValueError('samples')
    g = len(z)
    rho = z[0].real
    if rho <= 1 or np.max(np.abs(z - rho * np.exp(2j * np.pi * np.arange(g) / g))) > 1e-10:
        raise ValueError('circle')
    if np.max(np.abs(v[-np.arange(g) % g] - v.conj())) > 1e-09:
        raise ValueError('conjugacy')
    if np.iscomplexobj(tolerance) or np.ndim(tolerance) != 0 or (not np.isfinite(tolerance)) or (tolerance <= 0):
        raise ValueError('tolerance')
    if isinstance(max_support, (bool, np.bool_)) or not isinstance(max_support, (int, np.integer)) or (not 2 <= max_support <= g - 2):
        raise ValueError('budget')
    s = [0, g // 2]
    while True:
        rest = np.array([l for l in range(g) if l not in s])
        p = len(s)
        U = np.zeros((p, p), complex)
        seen = set()
        col = 0
        for j, k in enumerate(s):
            if j in seen:
                continue
            partner = s.index(-k % g)
            if partner == j:
                U[j, col] = 1
                col += 1
            else:
                U[j, col] = U[partner, col] = 1 / np.sqrt(2)
                U[j, col + 1] = 1j / np.sqrt(2)
                U[partner, col + 1] = -1j / np.sqrt(2)
                seen.add(partner)
                col += 2
            seen.add(j)
        L = ((v[rest, None] - v[np.array(s)][None]) / (z[rest, None, None, None] - z[np.array(s)][None, :, None, None])).transpose(0, 2, 3, 1).reshape(-1, p)
        LU = L @ U
        _, _, vh = np.linalg.svd(np.vstack([LU.real, LU.imag]), full_matrices=False)
        w = U @ vh[-1]
        C = 1 / (z[rest, None] - z[np.array(s)][None])
        with np.errstate(divide='ignore', invalid='ignore'):
            fitted = np.einsum('ij,jab->iab', C * w, v[s]) / (C @ w)[:, None, None]
        scores = np.zeros(g)
        scores[rest] = np.linalg.norm(v[rest] - fitted, axis=(1, 2))
        scores[~np.isfinite(scores)] = np.inf
        if np.max(scores) <= tolerance:
            return (np.array(s, dtype=int), w)
        orbits = [k for k in range(g // 2 + 1) if k not in s]
        score = np.array([(scores[k] + scores[-k % g]) / 2 for k in orbits])
        largest = np.max(score)
        if np.isinf(largest):
            k = orbits[int(np.flatnonzero(np.isinf(score))[0])]
        else:
            k = orbits[int(np.flatnonzero(score >= largest - 1e-12 * max(1.0, largest))[0])]
        new = [k] if k == -k % g else [k, -k % g]
        if len(s) + len(new) > max_support:
            raise ValueError('budget exhausted')
        s.extend(new)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    common_0 = """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def true_drift(case=0):
    if case == 2:
        return (np.array([[0.0, 1.4, 0.2], [-1.4, -0.8, 1.1], [-0.2, -1.1, -1.2]]), 1)
    B = np.array([[1.2, 0.25], [-0.35, 1.05], [0.7, -0.55], [0.4, 0.8]])
    J = np.array([[0, 1.3, -0.2, 0.4], [-1.3, 0, 0.65, -0.3], [0.2, -0.65, 0, 0.9], [-0.4, 0.3, -0.9, 0.0]])
    A = np.block([[np.zeros((2, 2)), B.T], [-B, J - np.diag([0.6, 1.1, 0.85, 1.4])]])
    return (A * (0.8 if case == 1 else 1.0), 2)

def stage_data(case=0):
    A, d = true_drift(case)
    tau = 0.4
    Y = np.array([expm(k * tau * A)[:d, :d] for k in range(128)])
    z = 1.4 * np.exp(2j * np.pi * np.arange(64) / 64)
    F = np.einsum('kab,lk->lab', Y, z[:, None] ** (-np.arange(len(Y)) - 1))
    rates, V = np.linalg.eig(A)
    W = np.linalg.inv(V)
    residues = np.array([np.outer(V[:d, j], W[j, :d]) for j in range(len(A))])
    return (A, d, Y, z, F, rates, residues)

def rational_check(result, z, F):
    s, w = result
    s = np.asarray(s)
    w = np.asarray(w)
    if s.ndim != 1 or w.shape != s.shape or len(s) < 2 or (len(s) > 18) or (s.dtype.kind not in 'iu') or (len(set(s)) != len(s)) or (s[0] != 0) or (s[1] != len(z) // 2):
        return 0
    if np.any(s < 0) or np.any(s >= len(z)) or (not np.isfinite(w).all()) or (abs(np.linalg.norm(w) - 1) > 1e-08):
        return 0
    for j, k in enumerate(s):
        partner = np.flatnonzero(s == -k % len(z))
        if len(partner) != 1 or abs(w[j].conjugate() - w[partner[0]]) > 1e-08:
            return 0
    rest = np.array([k for k in range(len(z)) if k not in s])
    C = 1 / (z[rest, None] - z[s][None])
    if np.any(abs(C @ w) < 1e-15):
        return 0
    r = np.einsum('ij,jab->iab', C * w, F[s]) / (C @ w)[:, None, None]
    return int(np.max(np.linalg.norm(r - F[rest], axis=(1, 2))) <= 2e-11)
'Deterministic synthetic probe data; not measurements reported in the article.'
"""
    common_1 = """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def error_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0

def true_drift(case=0):
    if case == 2:
        return (np.array([[0.0, 1.4, 0.2], [-1.4, -0.8, 1.1], [-0.2, -1.1, -1.2]]), 1)
    B = np.array([[1.2, 0.25], [-0.35, 1.05], [0.7, -0.55], [0.4, 0.8]])
    J = np.array([[0, 1.3, -0.2, 0.4], [-1.3, 0, 0.65, -0.3], [0.2, -0.65, 0, 0.9], [-0.4, 0.3, -0.9, 0.0]])
    A = np.block([[np.zeros((2, 2)), B.T], [-B, J - np.diag([0.6, 1.1, 0.85, 1.4])]])
    return (A * (0.8 if case == 1 else 1.0), 2)

def stage_data(case=0):
    A, d = true_drift(case)
    tau = 0.4
    Y = np.array([expm(k * tau * A)[:d, :d] for k in range(128)])
    z = 1.4 * np.exp(2j * np.pi * np.arange(64) / 64)
    F = np.einsum('kab,lk->lab', Y, z[:, None] ** (-np.arange(len(Y)) - 1))
    rates, V = np.linalg.eig(A)
    W = np.linalg.inv(V)
    residues = np.array([np.outer(V[:d, j], W[j, :d]) for j in range(len(A))])
    return (A, d, Y, z, F, rates, residues)
'Deterministic synthetic probe data; not measurements reported in the article.'
A, d, Y, z, F, rates, gamma = stage_data()
"""
    return [
        {
            'setup': common_0 + """A, d, Y, z, F, rates, gamma = stage_data(0)
""",
            'call': 'rational_check(shared_rational_fit(copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(18)), z, F)',
            'gold_call': 'rational_check(_oracle_shared_rational_fit(copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(18)), z, F)',
        },
        {
            'setup': common_0 + """A, d, Y, z, F, rates, gamma = stage_data(1)
""",
            'call': 'rational_check(shared_rational_fit(copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(18)), z, F)',
            'gold_call': 'rational_check(_oracle_shared_rational_fit(copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(18)), z, F)',
        },
        {
            'setup': common_0 + """A, d, Y, z, F, rates, gamma = stage_data(2)
""",
            'call': 'rational_check(shared_rational_fit(copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(18)), z, F)',
            'gold_call': 'rational_check(_oracle_shared_rational_fit(copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(18)), z, F)',
        },
        {
            'setup': common_1,
            'call': 'error_code(shared_rational_fit, copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(2))',
            'gold_call': 'error_code(_oracle_shared_rational_fit, copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(2))',
        },
        {
            'setup': common_1 + """F[1, 0, 0] += 0.1j
""",
            'call': 'error_code(shared_rational_fit, copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(18))',
            'gold_call': 'error_code(_oracle_shared_rational_fit, copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(1e-11), copy.deepcopy(18))',
        },
        {
            'setup': common_1,
            'call': 'error_code(shared_rational_fit, copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(0.0), copy.deepcopy(18))',
            'gold_call': 'error_code(_oracle_shared_rational_fit, copy.deepcopy(z), copy.deepcopy(F), copy.deepcopy(0.0), copy.deepcopy(18))',
        },
    ]
