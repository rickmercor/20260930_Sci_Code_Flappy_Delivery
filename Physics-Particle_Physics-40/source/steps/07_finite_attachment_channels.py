"""
Contract the finite source tensor with the joint color-spin response.

For each attachment $j$, set $R_j=\operatorname{Re}(K_j)-\operatorname{tr}[\operatorname{Re}(K_j)]I_2/2$ and $U_j=a_0T_{0j}+a_{-1}(T_{1j}+T_{2j})$, where $(a_{-1},a_0)$ are the pole and finite prefactors from step 05. Return diagonal $D_j=200\sum_p(R_j)_{pp}(U_j)_{pp}$, coherence $C_j=200[(R_j)_{xy}(U_j)_{yx}+(R_j)_{yx}(U_j)_{xy}]$, and continued-orientation diagnostic $E_j=200a_{-1}\sum_{p,q}(R_j)_{pq}(T_{2j})_{qp}$. The factor $200$ includes percentage $100$ and the Hermitian-conjugate factor $2$ exactly once. $E_j$ is already in $D_j+C_j$ and is not added again.

Returns
-------
Real (3,3) array in attachment order and diagonal/coherence/diagnostic columns.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def finite_attachment_channels(response: 'ArrayLike',
                               angular: 'ArrayLike',
                               prefactors: 'ArrayLike') -> 'np.ndarray':
    """Contract the finite source tensor with the color-spin response.

    Parameters
    ----------
    response : array-like
        Finite real or complex values convertible with ``np.asarray`` to shape
        ``(3, 2, 2)`` in attachment ``3, 4, 5`` and ``x, y`` ket/bra order.
    angular : array-like
        Finite real values convertible with ``np.asarray`` to shape
        ``(3, 3, 2, 2)``, with ``T0/T1/T2`` on the second axis.
    prefactors : array-like
        Finite real values convertible with ``np.asarray`` to shape ``(2,)``
        and ordered as ``[a_minus1, a_zero]``.

    Returns
    -------
    channels : numpy.ndarray
        Real array of shape ``(3, 3)`` with attachment rows and diagonal,
        coherence, and continued-orientation diagnostic columns.

    Raises
    ------
    ValueError
        If an input has a wrong type or shape, contains booleans, strings,
        objects, complex values where real values are required, or nonfinite
        values, or if the result is not representable as finite float64 values.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
def _r11_has_bool(value):
    if isinstance(value,(bool,np.bool_)):
        return True
    if isinstance(value,(tuple,list)):
        return any(_r11_has_bool(x) for x in value)
    return isinstance(value,np.ndarray) and value.dtype.kind=='b'
def _r11_array(value,shape=None,complex_ok=False):
    try:
        a=np.asarray(value)
    except (TypeError,ValueError) as exc:
        raise ValueError('invalid numeric input') from exc
    if _r11_has_bool(value) or a.dtype.kind not in ('iufc' if complex_ok else 'iuf'):
        raise ValueError('invalid numeric type')
    if shape is not None and a.shape!=shape:
        raise ValueError('invalid shape')
    a=a.astype(complex if complex_ok else float)
    if not np.all(np.isfinite(a)):
        raise ValueError('nonfinite input')
    return a
def _r11_scalar(value,lo,hi):
    x=float(_r11_array(value,()))
    if not lo<=x<=hi:
        raise ValueError('outside public interval')
    return x
def _r11_angles(eta,phi,jet_eta,jet_phi):
    e=_r11_array(eta); p=_r11_array(phi)
    if e.ndim>2 or p.ndim>2 or not e.size or not p.size:
        raise ValueError('radiation input must be nonempty of dimension at most two')
    try:
        e,p=np.broadcast_arrays(e,p)
    except ValueError as exc:
        raise ValueError('radiation inputs must broadcast') from exc
    if np.any(abs(e)>.7) or np.any(abs(p)>2*np.pi):
        raise ValueError('radiation input outside domain')
    y=_r11_scalar(jet_eta,-2.5,2.5)
    if abs(y)<1:
        raise ValueError('hard direction outside separated domain')
    a=_r11_scalar(jet_phi,-2*np.pi,2*np.pi)
    return e,p,y,a
def _r11_st(t):
    return t-np.trace(t,axis1=-2,axis2=-1)[...,None,None]*np.eye(2)/2
def _r11_transverse(phi,jet_phi):
    u=np.stack([np.cos(phi),np.sin(phi)],axis=-1)
    v=np.array([np.cos(jet_phi),np.sin(jet_phi)])
    uu=np.einsum('...a,...b->...ab',u,u)
    cross=np.einsum('...a,b->...ab',u,v)+np.einsum('a,...b->...ab',v,u)
    return uu,cross
def _r11_fraction_float(value):
    try:
        x=float(value)
    except OverflowError as exc:
        raise ValueError('output not finite float64') from exc
    if not np.isfinite(x):
        raise ValueError('output not finite float64')
    return x
def _oracle_finite_attachment_channels(response: 'ArrayLike',
                                       angular: 'ArrayLike',
                                       prefactors: 'ArrayLike') -> 'np.ndarray':
    from fractions import Fraction as F
    try:
        k=_r11_array(response,(3,2,2),True)
        t=_r11_array(angular,(3,3,2,2))
        pref=_r11_array(prefactors,(2,))
    except ValueError as exc:
        raise ValueError('invalid attachment-channel input') from exc
    f0,f1=map(lambda x:F(float(x)),pref)
    out=np.empty((3,3))
    for j in range(3):
        kk=[[F(float(k[j,p,q].real)) for q in range(2)] for p in range(2)]
        trace=kk[0][0]+kk[1][1]
        kk[0][0]-=trace/2;kk[1][1]-=trace/2
        tt=[[[F(float(t[j,b,p,q])) for q in range(2)] for p in range(2)] for b in range(3)]
        u=[[f1*tt[0][p][q]+f0*(tt[1][p][q]+tt[2][p][q]) for q in range(2)] for p in range(2)]
        diag=200*sum(kk[p][p]*u[p][p] for p in range(2))
        coherence=200*(kk[0][1]*u[1][0]+kk[1][0]*u[0][1])
        ev=200*f0*sum(kk[p][q]*tt[2][q][p] for p,q in np.ndindex(2,2))
        out[j]=[_r11_fraction_float(x) for x in [diag,coherence,ev]]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup_1 = """import numpy as np
K = (np.arange(12.0).reshape(3, 2, 2) - 3) / 11 + 1j * np.arange(12.0).reshape(3, 2, 2) / 19
T = (np.arange(36.0).reshape(3, 3, 2, 2) - 9) / 17

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
K=(np.arange(12.).reshape(3,2,2)-3)/11+1j*np.arange(12.).reshape(3,2,2)/19
T=(np.arange(36.).reshape(3,3,2,2)-9)/17

def rejects_value_error(fn,*args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0

def _independent(value):
    return np.array(value, copy=True)
"""

    return [
        {
            "setup": setup_1,
            "call": '_checked_numeric(finite_attachment_channels(_independent(K), _independent(T), [0.013, 0.047]), (3, 3))',
            "gold_call": '_checked_numeric(_oracle_finite_attachment_channels(_independent(K), _independent(T), [0.013, 0.047]), (3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(finite_attachment_channels(_independent(K).conj().transpose(0, 2, 1), _independent(T)[::-1], [0.03, -0.09]), (3, 3))',
            "gold_call": '_checked_numeric(_oracle_finite_attachment_channels(_independent(K).conj().transpose(0, 2, 1), _independent(T)[::-1], [0.03, -0.09]), (3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(finite_attachment_channels(_independent(K) * 2, _independent(T).transpose(0, 1, 3, 2), [-0.017, 0.022]), (3, 3))',
            "gold_call": '_checked_numeric(_oracle_finite_attachment_channels(_independent(K) * 2, _independent(T).transpose(0, 1, 3, 2), [-0.017, 0.022]), (3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(finite_attachment_channels(np.zeros((3, 2, 2)), _independent(T), [0.2, -0.1]), (3, 3))',
            "gold_call": '_checked_numeric(_oracle_finite_attachment_channels(np.zeros((3, 2, 2)), _independent(T), [0.2, -0.1]), (3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(finite_attachment_channels(_independent(K), np.zeros((3, 3, 2, 2)), [0.01, 0.04]), (3, 3))',
            "gold_call": '_checked_numeric(_oracle_finite_attachment_channels(_independent(K), np.zeros((3, 3, 2, 2)), [0.01, 0.04]), (3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(finite_attachment_channels(_independent(K), _independent(T), [0.0, 0.0]), (3, 3))',
            "gold_call": '_checked_numeric(_oracle_finite_attachment_channels(_independent(K), _independent(T), [0.0, 0.0]), (3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_1,
            "call": '_checked_numeric(finite_attachment_channels(_independent(K) * 1e+200, _independent(T) * 1e-200, [0.1, 0.7]), (3, 3))',
            "gold_call": '_checked_numeric(_oracle_finite_attachment_channels(_independent(K) * 1e+200, _independent(T) * 1e-200, [0.1, 0.7]), (3, 3))',
            "tol": 1e-08,
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, np.ones((2, 2)), _independent(T), [0.1, 0.2])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, np.ones((2, 2)), _independent(T), [0.1, 0.2])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), np.ones((2, 2, 2)), [0.1, 0.2])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), np.ones((2, 2, 2)), [0.1, 0.2])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T), [0.1])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T), [0.1])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, np.ones((3, 2, 2)) * 1e+308, _independent(T) * 1e+308, [1.0, 1.0])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, np.ones((3, 2, 2)) * 1e+308, _independent(T) * 1e+308, [1.0, 1.0])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T).astype(complex), [0.1, 0.2])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T).astype(complex), [0.1, 0.2])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, np.full(np.asarray(_independent(K)).shape, True), _independent(T), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, np.full(np.asarray(_independent(K)).shape, True), _independent(T), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, np.full(np.asarray(_independent(K)).shape, np.nan), _independent(T), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, np.full(np.asarray(_independent(K)).shape, np.nan), _independent(T), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, np.full(np.asarray(_independent(K)).shape, np.inf), _independent(T), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, np.full(np.asarray(_independent(K)).shape, np.inf), _independent(T), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, np.full(np.asarray(_independent(K)).shape, -np.inf), _independent(T), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, np.full(np.asarray(_independent(K)).shape, -np.inf), _independent(T), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, np.asarray(_independent(K)).astype(str), _independent(T), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, np.asarray(_independent(K)).astype(str), _independent(T), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, np.asarray(_independent(K), dtype=object), _independent(T), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, np.asarray(_independent(K), dtype=object), _independent(T), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, np.zeros((1, 1, 1, 1)), _independent(T), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, np.zeros((1, 1, 1, 1)), _independent(T), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), np.full(np.asarray(_independent(T)).shape, True), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), np.full(np.asarray(_independent(T)).shape, True), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), np.full(np.asarray(_independent(T)).shape, np.nan), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), np.full(np.asarray(_independent(T)).shape, np.nan), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), np.full(np.asarray(_independent(T)).shape, np.inf), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), np.full(np.asarray(_independent(T)).shape, np.inf), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), np.full(np.asarray(_independent(T)).shape, -np.inf), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), np.full(np.asarray(_independent(T)).shape, -np.inf), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), np.asarray(_independent(T)).astype(str), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), np.asarray(_independent(T)).astype(str), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), np.asarray(_independent(T), dtype=object), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), np.asarray(_independent(T), dtype=object), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), np.zeros((1, 1, 1, 1)), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), np.zeros((1, 1, 1, 1)), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), np.asarray(_independent(T), dtype=complex), [0.013, 0.047])',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), np.asarray(_independent(T), dtype=complex), [0.013, 0.047])',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T), np.full(np.asarray([0.013, 0.047]).shape, True))',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T), np.full(np.asarray([0.013, 0.047]).shape, True))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T), np.full(np.asarray([0.013, 0.047]).shape, np.nan))',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T), np.full(np.asarray([0.013, 0.047]).shape, np.nan))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T), np.full(np.asarray([0.013, 0.047]).shape, np.inf))',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T), np.full(np.asarray([0.013, 0.047]).shape, np.inf))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T), np.full(np.asarray([0.013, 0.047]).shape, -np.inf))',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T), np.full(np.asarray([0.013, 0.047]).shape, -np.inf))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T), np.asarray([0.013, 0.047]).astype(str))',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T), np.asarray([0.013, 0.047]).astype(str))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T), np.asarray([0.013, 0.047], dtype=object))',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T), np.asarray([0.013, 0.047], dtype=object))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T), np.zeros((1, 1, 1, 1)))',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T), np.zeros((1, 1, 1, 1)))',
        },
        {
            "setup": setup_2,
            "call": 'rejects_value_error(finite_attachment_channels, _independent(K), _independent(T), np.asarray([0.013, 0.047], dtype=complex))',
            "gold_call": 'rejects_value_error(_oracle_finite_attachment_channels, _independent(K), _independent(T), np.asarray([0.013, 0.047], dtype=complex))',
        },
    ]
