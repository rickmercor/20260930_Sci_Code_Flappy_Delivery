"""
Extract stable continuous exponents of a barycentric denominator.

Stable discrete poles determine decaying continuous modes only after a logarithm branch and an alias band have been fixed.

Returns
-------
Complex array (r,) of stable rates in the specified order, r>=1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def continuous_poles(support: "np.ndarray", weights: "np.ndarray", tau: float) -> "np.ndarray":
    """Extract stable continuous exponents of a barycentric denominator.

    Parameters
    ----------
    support, weights : finite complex arrays (p,), p>=2
        Distinct support points and nonzero weights; same shape.
    tau : positive finite real scalar
        Correlation sampling interval in reduced time units.

    Contract
    --------
    Return the stable continuous exponents of the supplied denominator.
    Retain discrete poles only when 1e-12<abs(z)<1-1e-10; discard negative
    real poles. Real means |Im(z)|<=1e-8; conjugate pairing tolerance is
    1e-7. Average each pair with the conjugate of its partner.
    Use the principal logarithm and the open band |Im(lambda)|<pi/tau.
    Return real rates first, in increasing real part; then conjugate
    pairs sorted by the positive member's (real part, imaginary part),
    with the positive member first. Rates have inverse-time units.

    Returns
    -------
    result
        Complex array (r,) of stable rates in the specified order, r>=1.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite values, duplicate support within 1e-12,
        zero weight, invalid tau, no retained poles, or unpaired pole.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_continuous_poles(support: "np.ndarray", weights: "np.ndarray", tau: float) -> "np.ndarray":
    """Extract stable continuous exponents of a barycentric denominator.

    Parameters
    ----------
    support, weights : finite complex arrays (p,), p>=2
        Distinct support points and nonzero weights; same shape.
    tau : positive finite real scalar
        Correlation sampling interval in reduced time units.

    Contract
    --------
    Return the stable continuous exponents of the supplied denominator.
    Retain discrete poles only when 1e-12<abs(z)<1-1e-10; discard negative
    real poles. Real means |Im(z)|<=1e-8; conjugate pairing tolerance is
    1e-7. Average each pair with the conjugate of its partner.
    Use the principal logarithm and the open band |Im(lambda)|<pi/tau.
    Return real rates first, in increasing real part; then conjugate
    pairs sorted by the positive member's (real part, imaginary part),
    with the positive member first. Rates have inverse-time units.

    Returns
    -------
    result
        Complex array (r,) of stable rates in the specified order, r>=1.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite values, duplicate support within 1e-12,
        zero weight, invalid tau, no retained poles, or unpaired pole.
    """
    import numpy as np
    from scipy.linalg import eig
    s = np.asarray(support, complex)
    w = np.asarray(weights, complex)
    if s.ndim != 1 or len(s) < 2 or w.shape != s.shape or (not np.isfinite(s).all()) or (not np.isfinite(w).all()) or np.any(abs(w) == 0):
        raise ValueError('support and weights')
    if np.any(np.abs(s[:, None] - s[None, :] + np.eye(len(s))) < 1e-12):
        raise ValueError('duplicate support')
    if np.iscomplexobj(tau) or np.ndim(tau) != 0 or (not np.isfinite(tau)) or (tau <= 0):
        raise ValueError('tau')
    p = len(s)
    M = np.zeros((p + 1, p + 1), complex)
    M[0, 1:] = w
    M[1:, 0] = 1
    M[1:, 1:] = np.diag(s)
    N = np.eye(p + 1)
    N[0, 0] = 0
    z = eig(M, N, right=False)
    z = z[np.isfinite(z) & (abs(z) > 1e-12) & (abs(z) < 1 - 1e-10)]
    real = []
    positive = []
    used = set()
    for i, a in enumerate(z):
        if i in used:
            continue
        if abs(a.imag) <= 1e-08:
            if a.real > 0:
                real.append(np.log(a.real) / tau)
            used.add(i)
            continue
        choices = [j for j in range(len(z)) if j != i and j not in used]
        if not choices:
            raise ValueError('unpaired pole')
        j = min(choices, key=lambda k: abs(z[k] - a.conjugate()))
        if abs(z[j] - a.conjugate()) > 1e-07:
            raise ValueError('unpaired pole')
        a = (a + z[j].conjugate()) / 2
        if a.imag < 0:
            a = a.conjugate()
        positive.append(np.log(a) / tau)
        used.update([i, j])
    out = list(sorted(real))
    for a in sorted(positive, key=lambda x: (x.real, x.imag)):
        out.extend([a, a.conjugate()])
    if not out:
        raise ValueError('no stable poles')
    return np.array(out, complex)

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
'Deterministic synthetic probe data; not measurements reported in the article.'
"""
    return [
        {
            'setup': common_0 + """A, d, Y, z, F, rates, gamma = stage_data(0)
s = z[[0, 32, 7, 57, 15, 49, 23, 41]][:len(rates) + 1]
poles = np.exp(0.4 * rates)
w = np.array([np.prod(a - poles) / np.prod(a - np.delete(s, j)) for j, a in enumerate(s)])
expected = []
for a in sorted([x.real for x in rates if abs(x.imag) < 1e-08]):
    expected.append(complex(a))
for a in sorted([x for x in rates if x.imag > 1e-08], key=lambda x: (x.real, x.imag)):
    expected.extend([a, a.conjugate()])
""",
            'call': 'near(continuous_poles(copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4)), expected, atol=1e-07, rtol=1e-07)',
            'gold_call': 'near(_oracle_continuous_poles(copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4)), expected, atol=1e-07, rtol=1e-07)',
        },
        {
            'setup': common_0 + """A, d, Y, z, F, rates, gamma = stage_data(1)
s = z[[0, 32, 7, 57, 15, 49, 23, 41]][:len(rates) + 1]
poles = np.exp(0.4 * rates)
w = np.array([np.prod(a - poles) / np.prod(a - np.delete(s, j)) for j, a in enumerate(s)])
expected = []
for a in sorted([x.real for x in rates if abs(x.imag) < 1e-08]):
    expected.append(complex(a))
for a in sorted([x for x in rates if x.imag > 1e-08], key=lambda x: (x.real, x.imag)):
    expected.extend([a, a.conjugate()])
""",
            'call': 'near(continuous_poles(copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4)), expected, atol=1e-07, rtol=1e-07)',
            'gold_call': 'near(_oracle_continuous_poles(copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4)), expected, atol=1e-07, rtol=1e-07)',
        },
        {
            'setup': common_0 + """A, d, Y, z, F, rates, gamma = stage_data(2)
s = z[[0, 32, 7, 57, 15, 49, 23, 41]][:len(rates) + 1]
poles = np.exp(0.4 * rates)
w = np.array([np.prod(a - poles) / np.prod(a - np.delete(s, j)) for j, a in enumerate(s)])
expected = []
for a in sorted([x.real for x in rates if abs(x.imag) < 1e-08]):
    expected.append(complex(a))
for a in sorted([x for x in rates if x.imag > 1e-08], key=lambda x: (x.real, x.imag)):
    expected.extend([a, a.conjugate()])
""",
            'call': 'near(continuous_poles(copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4)), expected, atol=1e-07, rtol=1e-07)',
            'gold_call': 'near(_oracle_continuous_poles(copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4)), expected, atol=1e-07, rtol=1e-07)',
        },
        {
            'setup': """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def near(actual, expected, atol=2e-06, rtol=2e-06):
    a = np.asarray(actual)
    b = np.asarray(expected)
    return int(a.shape == b.shape and np.isfinite(a).all() and np.allclose(a, b, atol=atol, rtol=rtol))
'Deterministic synthetic probe data; not measurements reported in the article.'
s = np.array([1.4, -1.4, 1.4j, -1.4j, 1.4 * np.exp(0.6j), 1.4 * np.exp(-0.6j)])
poles = np.array([0.5, 0.8, 1.0, 0.0, -0.4])
w = np.array([np.prod(a - poles) / np.prod(a - np.delete(s, j)) for j, a in enumerate(s)])
""",
            'call': 'near(continuous_poles(copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4)), np.log([0.5, 0.8]) / 0.4, atol=1e-07, rtol=1e-07)',
            'gold_call': 'near(_oracle_continuous_poles(copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4)), np.log([0.5, 0.8]) / 0.4, atol=1e-07, rtol=1e-07)',
        },
        {
            'setup': common_1 + """s = np.array([2.0, 2.0])
w = np.ones(2)
""",
            'call': 'error_code(continuous_poles, copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4))',
            'gold_call': 'error_code(_oracle_continuous_poles, copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4))',
        },
        {
            'setup': common_1 + """s = np.array([2.0, -2.0])
w = np.array([1.0, 1.0])
""",
            'call': 'error_code(continuous_poles, copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4))',
            'gold_call': 'error_code(_oracle_continuous_poles, copy.deepcopy(s), copy.deepcopy(w), copy.deepcopy(0.4))',
        },
    ]
