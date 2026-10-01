"""
Fit matrix exponential residues with inertial short-time constraints.

Inertial equilibrium velocities have a normalized covariance, zero initial slope and symmetric second derivative. The corresponding residue constraints are coupled across matrix entries.

Returns
-------
Complex array (r,d,d), ordered exactly as rates. Dimensionless. On exact-data fixtures, tests compare the sampled curve with Y (max abs error 2e-7) and check all three constraints (max abs residual 1e-7). Noisy-data tests compare the constrained minimizer with rtol=2e-8 and atol=2e-8.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def constrained_residues(Y: "np.ndarray", tau: float, rates: "np.ndarray") -> "np.ndarray":
    """Fit matrix exponential residues with inertial short-time constraints.

    Parameters
    ----------
    Y : finite real array (n,d,d), n>=4, d>=1
    tau : positive finite real scalar
    rates : finite complex array (r,), r>=1
        Distinct stable rates, conjugate closed, with |Im(rate)|<pi/tau.
        Real means |Im(rate)|<=1e-8. Pair and distinctness tolerance 1e-7.

    Contract
    --------
    Return the least-squares matrix amplitudes of the source's inertial
    exponential reconstruction, using every supplied lag and matrix
    entry with equal weight. The fitted curve has identity zero-lag
    covariance, zero initial slope and symmetric curvature at zero.
    Real rates have real residues and paired rates conjugate residues.
    The relative null-space rank threshold is 1e-12. Individual residue
    matrices need not be symmetric or positive.

    Returns
    -------
    result
        Complex array (r,d,d), ordered exactly as rates. Dimensionless.
        On exact-data fixtures, tests compare the sampled curve with Y
        (max abs error 2e-7) and check all three constraints
        (max abs residual 1e-7). Noisy-data fits compare the minimizer
        with rtol=2e-8 and atol=2e-8.

    Raises
    ------
    ValueError
        Complex Y, invalid shapes or nonfinite data, invalid tau, unstable,
        repeated, unpaired or out-of-band rates; inconsistent constraints
        (maximum constraint residual >1e-8); or a rank-deficient constrained fit.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_constrained_residues(Y: "np.ndarray", tau: float, rates: "np.ndarray") -> "np.ndarray":
    """Fit matrix exponential residues with inertial short-time constraints.

    Parameters
    ----------
    Y : finite real array (n,d,d), n>=4, d>=1
    tau : positive finite real scalar
    rates : finite complex array (r,), r>=1
        Distinct stable rates, conjugate closed, with |Im(rate)|<pi/tau.
        Real means |Im(rate)|<=1e-8. Pair and distinctness tolerance 1e-7.

    Contract
    --------
    Return the least-squares matrix amplitudes of the source's inertial
    exponential reconstruction, using every supplied lag and matrix
    entry with equal weight. The fitted curve has identity zero-lag
    covariance, zero initial slope and symmetric curvature at zero.
    Real rates have real residues and paired rates conjugate residues.
    The relative null-space rank threshold is 1e-12. Individual residue
    matrices need not be symmetric or positive.

    Returns
    -------
    result
        Complex array (r,d,d), ordered exactly as rates. Dimensionless.
        On exact-data fixtures, tests compare the sampled curve with Y
        (max abs error 2e-7) and check all three constraints
        (max abs residual 1e-7). Noisy-data fits compare the minimizer
        with rtol=2e-8 and atol=2e-8.

    Raises
    ------
    ValueError
        Complex Y, invalid shapes or nonfinite data, invalid tau, unstable,
        repeated, unpaired or out-of-band rates; inconsistent constraints
        (maximum constraint residual >1e-8); or a rank-deficient constrained fit.
    """
    import numpy as np
    from scipy.linalg import null_space
    if np.iscomplexobj(Y):
        raise ValueError('real Y')
    y = np.asarray(Y, float)
    lam = np.asarray(rates, complex)
    if y.ndim != 3 or len(y) < 4 or y.shape[1] < 1 or (y.shape[1] != y.shape[2]) or (not np.isfinite(y).all()) or (lam.ndim != 1) or (not len(lam)) or (not np.isfinite(lam).all()):
        raise ValueError('data')
    if np.iscomplexobj(tau) or np.ndim(tau) != 0 or (not np.isfinite(tau)) or (tau <= 0):
        raise ValueError('tau')
    if np.any(lam.real >= 0) or np.any(abs(lam.imag) >= np.pi / tau):
        raise ValueError('rates domain')
    if len(lam) > 1 and np.min(abs(lam[:, None] - lam[None, :] + np.eye(len(lam)) * 10000000000.0)) < 1e-07:
        raise ValueError('repeated rates')
    basis = []
    groups = []
    used = set()
    for i, a in enumerate(lam):
        if i in used:
            continue
        if abs(a.imag) <= 1e-08:
            basis.append((a.real, 0))
            groups.append((i, None))
            used.add(i)
        elif a.imag > 0:
            j = int(np.argmin(abs(lam - a.conjugate())))
            if j == i or abs(lam[j] - a.conjugate()) > 1e-07:
                raise ValueError('unpaired rates')
            basis.extend([(a, 1), (a, 2)])
            groups.append((i, j))
            used.update([i, j])
    if len(used) != len(lam):
        raise ValueError('unpaired rates')
    t = np.arange(len(y)) * tau
    columns = []
    ders = []
    for a, kind in basis:
        e = np.exp(t * a)
        columns.append(e.real if kind == 0 else 2 * e.real if kind == 1 else -2 * e.imag)
        ders.append([float(np.real(a ** k)) if kind == 0 else 2 * (a ** k).real if kind == 1 else -2 * (a ** k).imag for k in range(3)])
    X = np.array(columns).T
    der = np.array(ders).T
    p = len(basis)
    d = y.shape[1]
    design = np.kron(X, np.eye(d * d))
    C = []
    target = []
    for order in [0, 1]:
        for a in range(d):
            for b in range(d):
                row = np.zeros((p, d, d))
                row[:, a, b] = der[order]
                C.append(row.ravel())
                target.append(float(order == 0 and a == b))
    for a in range(d):
        for b in range(a + 1, d):
            row = np.zeros((p, d, d))
            row[:, a, b] = der[2]
            row[:, b, a] = -der[2]
            C.append(row.ravel())
            target.append(0.0)
    C = np.array(C)
    target = np.array(target)
    x0 = np.linalg.lstsq(C, target, rcond=1e-12)[0]
    if np.max(abs(C @ x0 - target)) > 1e-08:
        raise ValueError('inconsistent constraints')
    Z = null_space(C, rcond=1e-12)
    if Z.shape[1]:
        u, _, rank, _ = np.linalg.lstsq(design @ Z, y.ravel() - design @ x0, rcond=1e-12)
        if rank < Z.shape[1]:
            raise ValueError('unidentifiable fit')
        x0 = x0 + Z @ u
    coeff = x0.reshape(p, d, d)
    gamma = np.zeros((len(lam), d, d), complex)
    k = 0
    for i, j in groups:
        if j is None:
            gamma[i] = coeff[k]
            k += 1
        else:
            gamma[i] = coeff[k] + 1j * coeff[k + 1]
            gamma[j] = gamma[i].conj()
            k += 2
    return gamma

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

def residue_check(gamma, rates, Y):
    d = Y.shape[1]
    g = np.asarray(gamma)
    if g.shape != (len(rates), d, d) or not np.isfinite(g).all():
        return 0
    curve = np.einsum('kj,jab->kab', np.exp(0.4 * np.arange(len(Y))[:, None] * rates), g)
    g0 = g.sum(axis=0)
    g1 = np.einsum('j,jab->ab', rates, g)
    g2 = np.einsum('j,jab->ab', rates ** 2, g)
    return int(np.max(abs(curve - Y)) < 2e-07 and np.max(abs(g0 - np.eye(d))) < 1e-07 and (np.max(abs(g1)) < 1e-07) and (np.max(abs(g2 - g2.T)) < 1e-07))
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
            'call': 'residue_check(constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)), rates, Y)',
            'gold_call': 'residue_check(_oracle_constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)), rates, Y)',
        },
        {
            'setup': common_0 + """A, d, Y, z, F, rates, gamma = stage_data(1)
""",
            'call': 'residue_check(constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)), rates, Y)',
            'gold_call': 'residue_check(_oracle_constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)), rates, Y)',
        },
        {
            'setup': common_0 + """A, d, Y, z, F, rates, gamma = stage_data(2)
""",
            'call': 'residue_check(constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)), rates, Y)',
            'gold_call': 'residue_check(_oracle_constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)), rates, Y)',
        },
        {
            'setup': common_1,
            'call': 'error_code(constrained_residues, copy.deepcopy(Y.astype(complex)), copy.deepcopy(0.4), copy.deepcopy(rates))',
            'gold_call': 'error_code(_oracle_constrained_residues, copy.deepcopy(Y.astype(complex)), copy.deepcopy(0.4), copy.deepcopy(rates))',
        },
        {
            'setup': common_1,
            'call': 'error_code(constrained_residues, copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(np.array([-1.0])))',
            'gold_call': 'error_code(_oracle_constrained_residues, copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(np.array([-1.0])))',
        },
        {
            'setup': common_1,
            'call': 'error_code(constrained_residues, copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(np.r_[rates, rates[0]]))',
            'gold_call': 'error_code(_oracle_constrained_residues, copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(np.r_[rates, rates[0]]))',
        },
        {
            'setup': """import copy
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
'Deterministic synthetic probe data; not measurements reported in the article.'
A, d, Y, z, F, rates, gamma = stage_data()
Y = Y.copy()
perturb = 0.0001 * np.sin(np.arange(Y.size).reshape(Y.shape) * 0.73)
perturb[0] = 0
Y += perturb

def constraint_check(g):
    g = np.asarray(g)
    g0 = g.sum(axis=0)
    g1 = np.einsum('j,jab->ab', rates, g)
    g2 = np.einsum('j,jab->ab', rates ** 2, g)
    return int(g.shape == gamma.shape and np.isfinite(g).all() and (np.max(abs(g0 - np.eye(d))) < 1e-07) and (np.max(abs(g1)) < 1e-07) and (np.max(abs(g2 - g2.T)) < 1e-07))
""",
            'call': 'constraint_check(constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)))',
            'gold_call': 'constraint_check(_oracle_constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)))',
        },
        {
            'setup': """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def residue_check(gamma, rates, Y):
    d = Y.shape[1]
    g = np.asarray(gamma)
    if g.shape != (len(rates), d, d) or not np.isfinite(g).all():
        return 0
    curve = np.einsum('kj,jab->kab', np.exp(0.4 * np.arange(len(Y))[:, None] * rates), g)
    g0 = g.sum(axis=0)
    g1 = np.einsum('j,jab->ab', rates, g)
    g2 = np.einsum('j,jab->ab', rates ** 2, g)
    return int(np.max(abs(curve - Y)) < 2e-07 and np.max(abs(g0 - np.eye(d))) < 1e-07 and (np.max(abs(g1)) < 1e-07) and (np.max(abs(g2 - g2.T)) < 1e-07))
'Deterministic synthetic probe data; not measurements reported in the article.'
A = np.array([[0.0, 1.0], [-1.0, -3.0]])
rates, V = np.linalg.eig(A)
Y = np.array([expm(0.4 * k * A)[:1, :1] for k in range(12)])
""",
            'call': 'residue_check(constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)), rates, Y)',
            'gold_call': 'residue_check(_oracle_constrained_residues(copy.deepcopy(Y), copy.deepcopy(0.4), copy.deepcopy(rates)), rates, Y)',
        },
        {
            'setup': common_0 + """A, d, Y, z, F, rates, gamma = stage_data(0)
rng = np.random.default_rng(4141)
Y = Y.copy()
Y[1:] += 0.001*rng.standard_normal(Y[1:].shape)*np.exp(-0.02*np.arange(1,len(Y)))[:,None,None]
""",
            'call': 'constrained_residues(copy.deepcopy(Y.copy()), copy.deepcopy(0.4), copy.deepcopy(rates.copy()))',
            'gold_call': '_oracle_constrained_residues(copy.deepcopy(Y.copy()), copy.deepcopy(0.4), copy.deepcopy(rates.copy()))',
            'tol': 2e-08,
        },
    ]
