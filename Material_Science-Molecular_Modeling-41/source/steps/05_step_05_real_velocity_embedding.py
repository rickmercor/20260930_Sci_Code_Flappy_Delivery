"""
Construct a real minimal residue realization with velocity first.

Rank-reduced real residue factors produce a finite state model. A basis transformation puts the physical velocities in the leading coordinates.

Returns
-------
Real drift matrix A, shape (q,q), q>d, inverse-time units.     Its correlation is exp(A*t)[:d,:d].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def real_velocity_embedding(rates: "np.ndarray", residues: "np.ndarray", rank_tolerance: float) -> "np.ndarray":
    """Construct a real minimal residue realization with velocity first.

    Parameters
    ----------
    rates : finite complex array (r,), stable and conjugate closed
    residues : finite complex array (r,d,d)
        Conjugate closed with rates, maximum entry tolerance 1e-7.
        Real rates have real residues. sum residues=I within 1e-7.
    rank_tolerance : finite real scalar in (0,1)

    Contract
    --------
    Return a real residue realization with the normalized velocities
    first and the retained auxiliary coordinates following them.
    Retain each residue singular value s strictly above
    rank_tolerance*max(1,s_max). Conjugate pairs use real rotation blocks
    with the positive-imaginary representative first; equivalent real
    hidden-coordinate bases are accepted. The retained zero-lag
    correlation must agree with the identity within 1e-6.
    Tests compare the correlation and basis identities, not individual
    entries in an arbitrary hidden-coordinate basis.

    Returns
    -------
    result
        Real drift matrix A, shape (q,q), q>d, inverse-time units.
        Its correlation is exp(A*t)[:d,:d].

    Raises
    ------
    ValueError
        Invalid shapes, nonfinite inputs, unstable/unpaired/repeated
        rates (pair tolerance 1e-7), inconsistent residue conjugacy or
        normalization, invalid rank_tolerance, retained q<=d, or singular X.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_real_velocity_embedding(rates: "np.ndarray", residues: "np.ndarray", rank_tolerance: float) -> "np.ndarray":
    """Construct a real minimal residue realization with velocity first.

    Parameters
    ----------
    rates : finite complex array (r,), stable and conjugate closed
    residues : finite complex array (r,d,d)
        Conjugate closed with rates, maximum entry tolerance 1e-7.
        Real rates have real residues. sum residues=I within 1e-7.
    rank_tolerance : finite real scalar in (0,1)

    Contract
    --------
    Return a real residue realization with the normalized velocities
    first and the retained auxiliary coordinates following them.
    Retain each residue singular value s strictly above
    rank_tolerance*max(1,s_max). Conjugate pairs use real rotation blocks
    with the positive-imaginary representative first; equivalent real
    hidden-coordinate bases are accepted. The retained zero-lag
    correlation must agree with the identity within 1e-6.
    Tests compare the correlation and basis identities, not individual
    entries in an arbitrary hidden-coordinate basis.

    Returns
    -------
    result
        Real drift matrix A, shape (q,q), q>d, inverse-time units.
        Its correlation is exp(A*t)[:d,:d].

    Raises
    ------
    ValueError
        Invalid shapes, nonfinite inputs, unstable/unpaired/repeated
        rates (pair tolerance 1e-7), inconsistent residue conjugacy or
        normalization, invalid rank_tolerance, retained q<=d, or singular X.
    """
    import numpy as np
    from scipy.linalg import block_diag, null_space
    lam = np.asarray(rates, complex)
    gamma = np.asarray(residues, complex)
    if lam.ndim != 1 or not len(lam) or gamma.ndim != 3 or (gamma.shape[0] != len(lam)) or (gamma.shape[1] < 1) or (gamma.shape[1] != gamma.shape[2]) or (not np.isfinite(lam).all()) or (not np.isfinite(gamma).all()) or np.any(lam.real >= 0):
        raise ValueError('residues')
    if np.iscomplexobj(rank_tolerance) or np.ndim(rank_tolerance) != 0 or (not np.isfinite(rank_tolerance)) or (not 0 < rank_tolerance < 1):
        raise ValueError('rank tolerance')
    if len(lam) > 1 and np.min(abs(lam[:, None] - lam[None, :] + np.eye(len(lam)) * 10000000000.0)) < 1e-07:
        raise ValueError('repeated rates')
    d = gamma.shape[1]
    if np.max(abs(gamma.sum(axis=0) - np.eye(d))) > 1e-07:
        raise ValueError('normalization')
    blocks = []
    outputs = []
    inputs = []
    seen = set()
    for i, a in enumerate(lam):
        if i in seen:
            continue
        if abs(a.imag) <= 1e-08:
            if np.max(abs(gamma[i].imag)) > 1e-07:
                raise ValueError('real residue')
            g = gamma[i].real
            pair = False
            seen.add(i)
        else:
            if a.imag < 0:
                continue
            j = int(np.argmin(abs(lam - a.conjugate())))
            if i == j or abs(lam[j] - a.conjugate()) > 1e-07 or np.max(abs(gamma[j] - gamma[i].conj())) > 1e-07:
                raise ValueError('conjugacy')
            g = gamma[i]
            pair = True
            seen.update([i, j])
        u, s, vh = np.linalg.svd(g)
        keep = s > rank_tolerance * max(1.0, s[0])
        k = int(keep.sum())
        if not k:
            continue
        C = u[:, keep] * np.sqrt(s[keep])
        B = np.sqrt(s[keep])[:, None] * vh[keep]
        if not pair:
            blocks.append(a.real * np.eye(k))
            outputs.append(C.real)
            inputs.append(B.real)
        else:
            blocks.append(np.block([[a.real * np.eye(k), -a.imag * np.eye(k)], [a.imag * np.eye(k), a.real * np.eye(k)]]))
            outputs.append(np.concatenate([2 * C.real, -2 * C.imag], axis=1))
            inputs.append(np.concatenate([B.real, B.imag], axis=0))
    if len(seen) != len(lam) or not blocks:
        raise ValueError('unpaired or empty realization')
    J = block_diag(*blocks)
    Cbar = np.concatenate(outputs, axis=1)
    Bbar = np.concatenate(inputs, axis=0)
    if len(J) <= d or np.max(abs(Cbar @ Bbar - np.eye(d))) > 1e-06:
        raise ValueError('retained normalization')
    X = np.vstack([Cbar, null_space(Bbar.T, rcond=1e-12).T])
    try:
        A = np.linalg.solve(X.T, (X @ J).T).T
    except np.linalg.LinAlgError as e:
        raise ValueError('singular realization') from e
    return A

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    common_0 = """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def near(actual, expected, atol=2e-06, rtol=2e-06):
    a = np.asarray(actual)
    b = np.asarray(expected)
    return int(a.shape == b.shape and np.isfinite(a).all() and np.allclose(a, b, atol=atol, rtol=rtol))

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
            'setup': common_0 + """
def embedding_check(A, rates, gamma, d):
    A = np.asarray(A)
    if A.shape != (len(rates), len(rates)) or np.iscomplexobj(A) or (not np.isfinite(A).all()) or (np.max(np.linalg.eigvals(A).real) >= 0):
        return 0
    t = np.array([0.0, 0.17, 0.82, 1.9, 3.1])
    expected = np.einsum('kj,jab->kab', np.exp(t[:, None] * rates), gamma)
    actual = np.array([expm(x * A)[:d, :d] for x in t])
    return near(actual, expected, atol=2e-07, rtol=2e-07)
'Deterministic synthetic probe data; not measurements reported in the article.'
A, d, Y, z, F, rates, gamma = stage_data(0)
""",
            'call': 'embedding_check(real_velocity_embedding(copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1e-07)), rates, gamma, d)',
            'gold_call': 'embedding_check(_oracle_real_velocity_embedding(copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1e-07)), rates, gamma, d)',
        },
        {
            'setup': common_0 + """
def embedding_check(A, rates, gamma, d):
    A = np.asarray(A)
    if A.shape != (len(rates), len(rates)) or np.iscomplexobj(A) or (not np.isfinite(A).all()) or (np.max(np.linalg.eigvals(A).real) >= 0):
        return 0
    t = np.array([0.0, 0.17, 0.82, 1.9, 3.1])
    expected = np.einsum('kj,jab->kab', np.exp(t[:, None] * rates), gamma)
    actual = np.array([expm(x * A)[:d, :d] for x in t])
    return near(actual, expected, atol=2e-07, rtol=2e-07)
'Deterministic synthetic probe data; not measurements reported in the article.'
A, d, Y, z, F, rates, gamma = stage_data(1)
""",
            'call': 'embedding_check(real_velocity_embedding(copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1e-07)), rates, gamma, d)',
            'gold_call': 'embedding_check(_oracle_real_velocity_embedding(copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1e-07)), rates, gamma, d)',
        },
        {
            'setup': common_0 + """
def embedding_check(A, rates, gamma, d):
    A = np.asarray(A)
    if A.shape != (len(rates), len(rates)) or np.iscomplexobj(A) or (not np.isfinite(A).all()) or (np.max(np.linalg.eigvals(A).real) >= 0):
        return 0
    t = np.array([0.0, 0.17, 0.82, 1.9, 3.1])
    expected = np.einsum('kj,jab->kab', np.exp(t[:, None] * rates), gamma)
    actual = np.array([expm(x * A)[:d, :d] for x in t])
    return near(actual, expected, atol=2e-07, rtol=2e-07)
'Deterministic synthetic probe data; not measurements reported in the article.'
A, d, Y, z, F, rates, gamma = stage_data(2)
""",
            'call': 'embedding_check(real_velocity_embedding(copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1e-07)), rates, gamma, d)',
            'gold_call': 'embedding_check(_oracle_real_velocity_embedding(copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1e-07)), rates, gamma, d)',
        },
        {
            'setup': common_1,
            'call': 'error_code(real_velocity_embedding, copy.deepcopy(rates), copy.deepcopy(2 * gamma), copy.deepcopy(1e-07))',
            'gold_call': 'error_code(_oracle_real_velocity_embedding, copy.deepcopy(rates), copy.deepcopy(2 * gamma), copy.deepcopy(1e-07))',
        },
        {
            'setup': common_1,
            'call': 'error_code(real_velocity_embedding, copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1.0))',
            'gold_call': 'error_code(_oracle_real_velocity_embedding, copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1.0))',
        },
        {
            'setup': common_0 + """'Deterministic synthetic probe data; not measurements reported in the article.'
A, d, Y, z, F, rates, gamma = stage_data(2)
rates = np.r_[rates, -3.1]
gamma = np.concatenate([gamma, np.zeros((1, d, d))])
""",
            'call': 'near(expm(0.4 * real_velocity_embedding(copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1e-07)))[:d, :d], Y[1], atol=2e-07, rtol=2e-07)',
            'gold_call': 'near(expm(0.4 * _oracle_real_velocity_embedding(copy.deepcopy(rates), copy.deepcopy(gamma), copy.deepcopy(1e-07)))[:d, :d], Y[1], atol=2e-07, rtol=2e-07)',
        },
    ]
