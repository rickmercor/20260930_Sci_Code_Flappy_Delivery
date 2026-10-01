"""
Form the normalized joint first-beam spin and hard-color density h[p,q,a,b] defined explicitly in the main problem.

For coefficients $c[p,r,a]$ and positive beam density $S[r,s]$, form $h_{\rm raw}[p,q,a,b]=\sum_{r,s}c[p,r,a]S[r,s]c^*[q,s,b]$. Normalize by $\sum_{p,a,b}h_{\rm raw}[p,p,a,b]G[b,a]$, using the complete $3\times3$ Gram matrix in the main statement. Retain all coherences.

Returns
-------
Complex normalized (2,2,3,3) density with axes p,q,a,b.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def hard_spin_color_density(coefficients: 'ArrayLike',
                            beam_density: 'ArrayLike') -> 'np.ndarray':
    """Form the normalized joint spin-color density ``h[p,q,a,b]``.

    Parameters
    ----------
    coefficients : array-like
        Nonzero finite real or complex values convertible with ``np.asarray``
        to shape ``(2, 2, 3)``, with axes ``p, r, a`` in ordered basis
        ``D1, D2, D3``.
    beam_density : array-like
        Finite real symmetric positive-definite values convertible with
        ``np.asarray`` to shape ``(2, 2)``. Their positive scale is irrelevant
        and unit trace is not required.

    Returns
    -------
    density : numpy.ndarray
        Complex normalized array of shape ``(2, 2, 3, 3)`` with axes
        ``p, q, a, b``, retaining every coherence.

    Raises
    ------
    ValueError
        If either input cannot be converted to the required numeric shape,
        contains booleans, strings, objects, or nonfinite values, if the
        amplitude is zero, or if the beam density is complex, zero, asymmetric,
        singular, or not positive definite.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
def _joint_has_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (list, tuple)):
        return any(_joint_has_bool(v) for v in value)
    return isinstance(value, np.ndarray) and value.dtype.kind == 'b'
def _joint_array(value, shape, complex_ok=False):
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid numeric array') from exc
    if (_joint_has_bool(value) or raw.dtype.kind not in ('iufc' if complex_ok else 'iuf') or raw.shape != shape):
        raise ValueError('invalid numeric type or shape')
    out = np.asarray(raw, dtype=complex if complex_ok else float)
    if not np.all(np.isfinite(out)):
        raise ValueError('nonfinite input')
    return out
def _oracle_hard_spin_color_density(coefficients: 'ArrayLike',
                                    beam_density: 'ArrayLike') -> 'np.ndarray':
    from fractions import Fraction as F
    c = _joint_array(coefficients, (2, 2, 3), True)
    beam = _joint_array(beam_density, (2, 2))
    if not np.array_equal(beam, beam.T):
        raise ValueError('beam density must be real symmetric')
    s = [[F(float(v)) for v in row] for row in beam]
    if s[0][0] <= 0 or s[0][0]*s[1][1]-s[0][1]**2 <= 0:
        raise ValueError('beam density must be positive definite')
    # Exact products of the supplied binary floats avoid intermediate overflow
    # and scale-dependent loss in this small, normalized positive quadratic form.
    h = {}
    for p, q, a, b in np.ndindex(2, 2, 3, 3):
        real = imag = F(0)
        for r, t in np.ndindex(2, 2):
            u, v = c[p, r, a], c[q, t, b]
            ur, ui, vr, vi = map(lambda x: F(float(x)), (u.real, u.imag, v.real, v.imag))
            real += s[r][t]*(ur*vr+ui*vi)
            imag += s[r][t]*(ui*vr-ur*vi)
        h[p, q, a, b] = real, imag
    gram = [[F(64, 9), F(-8, 9), F(-8, 9)],
            [F(-8, 9), F(64, 9), F(1, 9)],
            [F(-8, 9), F(1, 9), F(64, 9)]]
    norm = sum(h[p, p, a, b][0]*gram[b][a]
               for p, a, b in np.ndindex(2, 3, 3))
    if norm <= 0:
        raise ValueError('zero hard amplitude')
    out = np.empty((2, 2, 3, 3), complex)
    for index, (real, imag) in h.items():
        out[index] = complex(float(real/norm), float(imag/norm))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup_1 = """import numpy as np
C = np.array([[[1 + 0j, 0.3j, 0.4 + 0j], [0.2 + 0.1j, -0 - 0.5j, 0.1 + 0j]], [[0.3 + 0j, 0.4 + 0.2j, -0 - 0.2j], [-0 - 0.4j, 0.1 + 0j, 0.6 + 0.2j]]], dtype=complex)

def _checked_numeric(value, shape):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in 'iufc':
        raise ValueError('unexpected numerical result type or shape')
    if not np.all(np.isfinite(result)):
        raise ValueError('nonfinite result')
    return np.stack((result.real, result.imag), axis=0)

def _independent(value):
    return np.array(value, copy=True)
"""

    setup_2 = """import numpy as np
def rejects_value_error(fn,*args,**kwargs):
    try:
        fn(*args,**kwargs)
    except ValueError:
        return 1.0
    return 0.0

def _independent(value):
    return np.array(value, copy=True)
"""

    setup_3 = """import numpy as np
def rejects_value_error(fn,*args,**kwargs):
    try:
        fn(*args,**kwargs)
    except ValueError:
        return 1.0
    return 0.0

C=np.array([[[(1+0j), 0.3j, (0.4+0j)], [(0.2+0.1j), (-0-0.5j), (0.1+0j)]], [[(0.3+0j), (0.4+0.2j), (-0-0.2j)], [(-0-0.4j), (0.1+0j), (0.6+0.2j)]]],dtype=complex)

def _independent(value):
    return np.array(value, copy=True)
"""

    return [
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(_independent(C), [[0.55, 0.04], [0.04, 0.45]]), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(_independent(C), [[0.55, 0.04], [0.04, 0.45]]), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(_independent(C)[::-1], [[0.4, -0.17], [-0.17, 0.8]]), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(_independent(C)[::-1], [[0.4, -0.17], [-0.17, 0.8]]), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(_independent(C) * (0.2 + 0.8j), [[2.0, 0.3], [0.3, 0.1]]), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(_independent(C) * (0.2 + 0.8j), [[2.0, 0.3], [0.3, 0.1]]), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(_independent(C) * 1e-06, np.eye(2) * 10000.0), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(_independent(C) * 1e-06, np.eye(2) * 10000.0), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(_independent(C) * 1000000.0, np.eye(2) * 0.0001), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(_independent(C) * 1000000.0, np.eye(2) * 0.0001), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(np.ones((2, 2, 3)), np.eye(2)), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(np.ones((2, 2, 3)), np.eye(2)), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(_independent(C).conj() + 0.1j, [[0.03, 0.01], [0.01, 0.8]]), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(_independent(C).conj() + 0.1j, [[0.03, 0.01], [0.01, 0.8]]), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(hard_spin_color_density, np.zeros((2, 2, 3)), np.eye(2))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, np.zeros((2, 2, 3)), np.eye(2))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(hard_spin_color_density, np.ones((2, 3)), np.eye(2))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, np.ones((2, 3)), np.eye(2))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(hard_spin_color_density, np.ones((2, 2, 3), dtype=bool), np.eye(2))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, np.ones((2, 2, 3), dtype=bool), np.eye(2))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(hard_spin_color_density, np.ones((2, 2, 3), dtype=object), np.eye(2))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, np.ones((2, 2, 3), dtype=object), np.eye(2))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(hard_spin_color_density, np.full((2, 2, 3), np.nan), np.eye(2))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, np.full((2, 2, 3), np.nan), np.eye(2))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(hard_spin_color_density, np.full((2, 2, 3), np.inf), np.eye(2))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, np.full((2, 2, 3), np.inf), np.eye(2))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(hard_spin_color_density, np.full((2, 2, 3), -np.inf), np.eye(2))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, np.full((2, 2, 3), -np.inf), np.eye(2))',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), np.zeros((2, 2)))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), np.zeros((2, 2)))',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), -np.eye(2))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), -np.eye(2))',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), np.ones((2, 2)))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), np.ones((2, 2)))',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), [[1, 2], [2, 1]])',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), [[1, 2], [2, 1]])',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), [[1, 0.1], [0.2, 1]])',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), [[1, 0.1], [0.2, 1]])',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), np.eye(3))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), np.eye(3))',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), np.eye(2, dtype=complex))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), np.eye(2, dtype=complex))',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), np.eye(2, dtype=bool))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), np.eye(2, dtype=bool))',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), np.full((2, 2), np.nan))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), np.full((2, 2), np.nan))',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(hard_spin_color_density, _independent(C), np.full((2, 2), np.inf))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, _independent(C), np.full((2, 2), np.inf))',
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(_independent(C), np.diag([1000000.0, 1.0])), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(_independent(C), np.diag([1000000.0, 1.0])), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(_independent(C), np.diag([1.0, 1000000.0])), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(_independent(C), np.diag([1.0, 1000000.0])), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(hard_spin_color_density(_independent(C), np.array([[1.0, 1.0], [1.0, np.nextafter(1.0, 2.0)]])), (2, 2, 3, 3))',
            "gold_call": '_checked_numeric(_oracle_hard_spin_color_density(_independent(C), np.array([[1.0, 1.0], [1.0, np.nextafter(1.0, 2.0)]])), (2, 2, 3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(hard_spin_color_density, np.ones((2, 2, 3), dtype=str), np.eye(2))',
            "gold_call": 'rejects_value_error(_oracle_hard_spin_color_density, np.ones((2, 2, 3), dtype=str), np.eye(2))',
        },
    ]
