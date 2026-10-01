"""
Pack the linear, quadratic, and cubic magneto-optic orders.

Combine Step 01 and Step 02 packets with the cubic crystal's isotropic linear tensor K_ijk=epsilon_ijk K. For columns [M_T,M_L,M_P], the forward pair order (23,31,12) is K times those three columns, and the reverse pairs are their negatives. Preserve the exact six-pair order and place the linear, quadratic, and cubic packets on a new order axis. This separation is retained because the downstream Kerr cross-product must be truncated by total magnetization order. Require nonempty finite magnetization of shape (N,3), finite K, and finite quadratic and cubic packets of shape (N,6,2); raise ValueError otherwise.

Returns
-------
orders : np.ndarray, shape (N,3,6,2), float Axes event, order 1/2/3, pair (23,31,12,32,13,21), real/imag.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pack_magnetooptic_orders(magnetization: np.ndarray, k_linear: complex, quadratic_response: np.ndarray, cubic_response: np.ndarray) -> np.ndarray:
    '''Pack responses by magnetization order.

    Parameters
    ----------
    magnetization : np.ndarray, shape (N,3)
        Nonempty finite rows in M_T, M_L, M_P order.
    k_linear : complex
        Finite scalar linear magneto-optic coefficient.
    quadratic_response, cubic_response : np.ndarray, shape (N,6,2)
        Finite packets with matching event counts and final real/imag axis.

    Returns
    -------
    orders : np.ndarray, shape (N,3,6,2), float
        Axes event, order 1/2/3, pair (23,31,12,32,13,21), real/imag.

    Raises
    ------
    ValueError
        An input has an invalid shape, is nonfinite, or event counts do not match.'''
    return np.empty((0, 3, 6, 2), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _c_pack_magnetooptic_orders__packet(values):
    values = np.asarray(values, dtype=complex)
    return np.stack((values.real, values.imag), axis=-1).astype(float)

def _c_pack_magnetooptic_orders__unpacket(values, shape):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError('malformed real/imag packet')
    return array[..., 0] + 1j * array[..., 1]

def _c_pack_magnetooptic_orders__finite_complex(value, name):
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite scalar') from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f'{name} must be a finite scalar')
    return result

def _c_pack_magnetooptic_orders(magnetization: np.ndarray, k_linear: complex, quadratic_response: np.ndarray, cubic_response: np.ndarray) -> np.ndarray:
    M = np.asarray(magnetization, dtype=float)
    if M.ndim != 2 or M.shape[1] != 3 or M.shape[0] == 0:
        raise ValueError('magnetization must have nonempty shape (N,3)')
    if not np.all(np.isfinite(M)):
        raise ValueError('magnetization must be finite')
    count = M.shape[0]
    quadratic = _c_pack_magnetooptic_orders__unpacket(quadratic_response, (count, 6, 2))
    cubic = _c_pack_magnetooptic_orders__unpacket(cubic_response, (count, 6, 2))
    k_linear = _c_pack_magnetooptic_orders__finite_complex(k_linear, 'k_linear')
    mt, ml, mp = M.T
    forward = np.stack((k_linear * mt, k_linear * ml, k_linear * mp), axis=1)
    linear = np.concatenate((forward, -forward), axis=1)
    return _c_pack_magnetooptic_orders__packet(np.stack((linear, quadratic, cubic), axis=1))

def _oracle_pack_magnetooptic_orders(magnetization, k_linear, quadratic_response, cubic_response):
    return _c_pack_magnetooptic_orders(magnetization, k_linear, quadratic_response, cubic_response)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    exception_setup = 'def _value_error_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n'
    return [
        {
            "setup": "# Case: normal\nM=np.array([[0.2,0.6,-0.3],[-0.7,0.1,0.4],[0.3,-0.5,0.8]])\nq=_oracle_evaluate_rotated_qmoke_quadratic_response(M,.07-.02j,.03+.01j,'001',.23)\nc=_oracle_evaluate_rotated_cmoke_cubic_response(M,.004+.002j,-.001+.0003j,'001',.23)",
            "call": 'pack_magnetooptic_orders(M,.025-.014j,q,c)',
            "gold_call": '_oracle_pack_magnetooptic_orders(M,.025-.014j,q,c)',
        },
        {
            "setup": '# Case: boundary\n# Coverage: boundary (one all-zero packet row)\nM=np.zeros((1,3),dtype=float)\nq=np.zeros((1,6,2),dtype=float)\nc=np.zeros((1,6,2),dtype=float)',
            "call": 'pack_magnetooptic_orders(M,0j,q,c)',
            "gold_call": '_oracle_pack_magnetooptic_orders(M,0j,q,c)',
        },
        {
            "setup": "# Case: edge\n# Coverage: edge (small nonzero linear coefficient)\nM=np.array([[0.13,-0.71,0.22],[-0.42,0.33,-0.59],[0.81,0.14,0.07],[0.28,-0.19,0.63]])\nq=_oracle_evaluate_rotated_qmoke_quadratic_response(M,.05+.01j,-.02+.03j,'111',-.51)\nc=_oracle_evaluate_rotated_cmoke_cubic_response(M,-.003+.006j,.002-.001j,'111',-.51)",
            "call": 'pack_magnetooptic_orders(M,1e-6+0j,q,c)',
            "gold_call": '_oracle_pack_magnetooptic_orders(M,1e-6+0j,q,c)',
        },
        {
            "setup": '# Case: edge\nM=np.zeros((1,3))\nq=np.zeros((1,5,2))\nc=np.zeros((1,6,2))\n' + exception_setup,
            "call": '_value_error_code(pack_magnetooptic_orders, M, 0.1, q, c)',
            "gold_call": '_value_error_code(_oracle_pack_magnetooptic_orders, M, 0.1, q, c)',
        },
        {
            "setup": '# Case: edge\nM=np.zeros((1,3))\nq=np.zeros((1,6,2)); q[0,0,0]=np.nan\nc=np.zeros((1,6,2))\n' + exception_setup,
            "call": '_value_error_code(pack_magnetooptic_orders, M, 0.1, q, c)',
            "gold_call": '_value_error_code(_oracle_pack_magnetooptic_orders, M, 0.1, q, c)',
        },
        {
            "setup": '# Case: edge\nM=np.zeros((0,3))\nq=np.zeros((0,6,2))\nc=np.zeros((0,6,2))\n' + exception_setup,
            "call": '_value_error_code(pack_magnetooptic_orders, M, 0.1, q, c)',
            "gold_call": '_value_error_code(_oracle_pack_magnetooptic_orders, M, 0.1, q, c)',
        },
    ]
