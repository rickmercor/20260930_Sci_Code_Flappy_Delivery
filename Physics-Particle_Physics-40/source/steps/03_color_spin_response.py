"""
Contract the hard density with the three ordered color-response matrices.

In the stated $D_1,D_2,D_3$ basis, the ordered source color trace and crossed-leg convention give $L_3=[[0,-1,0],[1,0,-1/8],[0,1/8,0]]$, $L_4=[[0,-1/8,7/4],[1/8,0,-3/8],[-7/4,3/8,0]]$, and $L_5=[[0,9/8,-7/4],[-9/8,0,1/2],[7/4,-1/2,0]]$. Contract $K_{jpq}=i\sum_{a,b}(L_j)_{ab}h_{pqab}$ in attachment $3,4,5$ order. Do not add a spin trace, source color average, Hermitian conjugate or new normalization.

Returns
-------
Complex (3,2,2) response in attachment 3,4,5 and x,y ket/bra order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def color_spin_response(hard_density: 'ArrayLike') -> 'np.ndarray':
    """Contract the hard density with the ordered color-response matrices.

    Parameters
    ----------
    hard_density : array-like
        Finite real or complex values convertible with ``np.asarray`` to shape
        ``(2, 2, 3, 3)``, with axes ``p, q, a, b`` in ordered basis
        ``D1, D2, D3``. Zero, unnormalized, and non-Hermitian inputs are accepted.

    Returns
    -------
    response : numpy.ndarray
        Complex array of shape ``(3, 2, 2)`` in attachment ``3, 4, 5`` and
        ``x, y`` ket/bra order.

    Raises
    ------
    ValueError
        If the input cannot be converted to the required numeric shape,
        contains booleans, strings, objects, or nonfinite values, or produces
        a result not representable as finite complex128 values.
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
def _joint_finite_fraction(value):
    try:
        answer = float(value)
    except OverflowError as exc:
        raise ValueError('result not representable as finite float64') from exc
    if not np.isfinite(answer):
        raise ValueError('result not representable as finite float64')
    return answer
def _oracle_color_spin_response(hard_density: 'ArrayLike') -> 'np.ndarray':
    from fractions import Fraction
    try:
        h = _joint_array(hard_density, (2, 2, 3, 3), True)
    except ValueError as exc:
        raise ValueError('invalid hard-density input') from exc
    real_l = [
        [[0, -1, 0], [1, 0, Fraction(-1, 8)], [0, Fraction(1, 8), 0]],
        [[0, Fraction(-1, 8), Fraction(7, 4)], [Fraction(1, 8), 0, Fraction(-3, 8)],
         [Fraction(-7, 4), Fraction(3, 8), 0]],
        [[0, Fraction(9, 8), Fraction(-7, 4)], [Fraction(-9, 8), 0, Fraction(1, 2)],
         [Fraction(7, 4), Fraction(-1, 2), 0]],
    ]
    out = np.empty((3, 2, 2), complex)
    for j, p, q in np.ndindex(3, 2, 2):
        real = imag = Fraction(0)
        for a, b in np.ndindex(3, 3):
            coefficient = real_l[j][a][b]
            real -= coefficient*Fraction(float(h[p, q, a, b].imag))
            imag += coefficient*Fraction(float(h[p, q, a, b].real))
        out[j, p, q] = complex(_joint_finite_fraction(real), _joint_finite_fraction(imag))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup_1 = """import numpy as np
H = (np.arange(36.0).reshape(2, 2, 3, 3) - 13) / 37 + 1j * (np.arange(36.0).reshape(2, 2, 3, 3)[::-1] + 2) / 41

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

H=np.zeros((2,2,3,3),complex)
H[0,0,0,2]=complex(0,np.finfo(float).max)

def _independent(value):
    return np.array(value, copy=True)
"""

    setup_4 = """import numpy as np
H = np.zeros((2, 2, 3, 3), complex)
H[0, 1, 0, 1] = 0.3 + 0.7j
H[1, 0, 2, 0] = -0.2 + 0.1j
H[0, 0, 1, 0] = 0.17 - 0.8j

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

    return [
        {
            "setup": setup_1,
            "call": '_checked_numeric(color_spin_response(_independent(H)), (3, 2, 2))',
            "gold_call": '_checked_numeric(_oracle_color_spin_response(_independent(H)), (3, 2, 2))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(color_spin_response(_independent(H).conj()), (3, 2, 2))',
            "gold_call": '_checked_numeric(_oracle_color_spin_response(_independent(H).conj()), (3, 2, 2))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(color_spin_response(_independent(H).transpose(1, 0, 3, 2)), (3, 2, 2))',
            "gold_call": '_checked_numeric(_oracle_color_spin_response(_independent(H).transpose(1, 0, 3, 2)), (3, 2, 2))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(color_spin_response(np.zeros((2, 2, 3, 3))), (3, 2, 2))',
            "gold_call": '_checked_numeric(_oracle_color_spin_response(np.zeros((2, 2, 3, 3))), (3, 2, 2))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(color_spin_response(np.arange(36.0).reshape(2, 2, 3, 3) / 7), (3, 2, 2))',
            "gold_call": '_checked_numeric(_oracle_color_spin_response(np.arange(36.0).reshape(2, 2, 3, 3) / 7), (3, 2, 2))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(color_spin_response(_independent(H) * (0.3 + 0.8j)), (3, 2, 2))',
            "gold_call": '_checked_numeric(_oracle_color_spin_response(_independent(H) * (0.3 + 0.8j)), (3, 2, 2))',
            "tol": 1e-08,
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(color_spin_response, np.ones((2, 2, 3)))',
            "gold_call": 'rejects_value_error(_oracle_color_spin_response, np.ones((2, 2, 3)))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(color_spin_response, np.ones((2, 2, 3, 3), dtype=bool))',
            "gold_call": 'rejects_value_error(_oracle_color_spin_response, np.ones((2, 2, 3, 3), dtype=bool))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(color_spin_response, np.ones((2, 2, 3, 3), dtype=object))',
            "gold_call": 'rejects_value_error(_oracle_color_spin_response, np.ones((2, 2, 3, 3), dtype=object))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(color_spin_response, np.full((2, 2, 3, 3), np.nan))',
            "gold_call": 'rejects_value_error(_oracle_color_spin_response, np.full((2, 2, 3, 3), np.nan))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(color_spin_response, np.full((2, 2, 3, 3), np.inf))',
            "gold_call": 'rejects_value_error(_oracle_color_spin_response, np.full((2, 2, 3, 3), np.inf))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(color_spin_response, np.full((2, 2, 3, 3), -np.inf))',
            "gold_call": 'rejects_value_error(_oracle_color_spin_response, np.full((2, 2, 3, 3), -np.inf))',
        },
        {
            "setup": setup_3,
            "call": 'rejects_value_error(color_spin_response, _independent(H))',
            "gold_call": 'rejects_value_error(_oracle_color_spin_response, _independent(H))',
        },
        {
            "setup": setup_4,
            "call": '_checked_numeric(color_spin_response(_independent(H)), (3, 2, 2))',
            "gold_call": '_checked_numeric(_oracle_color_spin_response(_independent(H)), (3, 2, 2))',
            "tol": 1e-08,
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(color_spin_response, np.zeros((2, 2, 3, 3)).astype(str))',
            "gold_call": 'rejects_value_error(_oracle_color_spin_response, np.zeros((2, 2, 3, 3)).astype(str))',
        },
    ]
