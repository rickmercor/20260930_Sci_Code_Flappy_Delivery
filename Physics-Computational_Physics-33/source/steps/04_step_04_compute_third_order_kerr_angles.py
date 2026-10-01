"""
Evaluate the perturbative s- and p-polarized Kerr angles through cubic order.

Apply the paper's analytical thin-film relation. With total off-diagonal epsilon equal to the sum of orders 1,2,3, use Phi_s=A_s*(epsilon_yx-epsilon_yz*epsilon_zx/epsilon_d)+B_s*epsilon_zx and Phi_p=-A_p*(epsilon_xy-epsilon_zy*epsilon_xz/epsilon_d)+B_p*epsilon_xz. Truncate each product at total magnetization order three: retain 1x1, 1x2, and 2x1 only; exclude every factor involving the cubic packet and exclude 2x2. Return complex Kerr angles as real/imag packets, polarization order s,p. Require a nonempty finite packet of shape (N,3,6,2), finite scalar coefficients, and nonzero epsilon_d; raise ValueError otherwise.

Returns
-------
kerr : np.ndarray, shape (N,2,2), float Polarization order s,p, final axis real/imag.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_third_order_kerr_angles(order_packets: np.ndarray, epsilon_d: complex, a_s: complex, b_s: complex, a_p: complex, b_p: complex) -> np.ndarray:
    '''Return perturbative Kerr angles.

    Parameters
    ----------
    order_packets : np.ndarray, shape (N,3,6,2)
        Nonempty finite packets in order 1/2/3 and the specified pair order.
    epsilon_d : complex
        Finite nonzero scalar diagonal permittivity.
    a_s, b_s, a_p, b_p : complex
        Finite scalar optical weights for s and p polarization.

    Returns
    -------
    kerr : np.ndarray, shape (N,2,2), float
        Polarization order s,p, final axis real/imag.

    Raises
    ------
    ValueError
        An input has an invalid shape, is nonfinite, or epsilon_d is zero.'''
    return np.empty((0, 2, 2), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _c_compute_third_order_kerr_angles__packet(values):
    values = np.asarray(values, dtype=complex)
    return np.stack((values.real, values.imag), axis=-1).astype(float)

def _c_compute_third_order_kerr_angles__unpacket(values, shape):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError('malformed real/imag packet')
    return array[..., 0] + 1j * array[..., 1]

def _c_compute_third_order_kerr_angles__finite_complex(value, name):
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite scalar') from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f'{name} must be a finite scalar')
    return result

def _c_compute_third_order_kerr_angles(order_packets: np.ndarray, epsilon_d: complex, a_s: complex, b_s: complex, a_p: complex, b_p: complex) -> np.ndarray:
    raw = np.asarray(order_packets, dtype=float)
    if raw.ndim != 4 or raw.shape[1:] != (3, 6, 2) or raw.shape[0] == 0:
        raise ValueError('order_packets must have nonempty shape (N,3,6,2)')
    orders = _c_compute_third_order_kerr_angles__unpacket(raw, raw.shape)
    epsilon_d = _c_compute_third_order_kerr_angles__finite_complex(epsilon_d, 'epsilon_d')
    a_s = _c_compute_third_order_kerr_angles__finite_complex(a_s, 'a_s')
    b_s = _c_compute_third_order_kerr_angles__finite_complex(b_s, 'b_s')
    a_p = _c_compute_third_order_kerr_angles__finite_complex(a_p, 'a_p')
    b_p = _c_compute_third_order_kerr_angles__finite_complex(b_p, 'b_p')
    if epsilon_d == 0:
        raise ValueError('epsilon_d must be nonzero')
    linear, quadratic, cubic = (orders[:, 0], orders[:, 1], orders[:, 2])
    total = linear + quadratic + cubic
    cross_s = linear[:, 0] * linear[:, 1] + linear[:, 0] * quadratic[:, 1] + quadratic[:, 0] * linear[:, 1]
    cross_p = linear[:, 3] * linear[:, 4] + linear[:, 3] * quadratic[:, 4] + quadratic[:, 3] * linear[:, 4]
    phi_s = a_s * (total[:, 5] - cross_s / epsilon_d) + b_s * total[:, 1]
    phi_p = -a_p * (total[:, 2] - cross_p / epsilon_d) + b_p * total[:, 4]
    return _c_compute_third_order_kerr_angles__packet(np.stack((phi_s, phi_p), axis=1))

def _oracle_compute_third_order_kerr_angles(order_packets, epsilon_d, a_s, b_s, a_p, b_p):
    return _c_compute_third_order_kerr_angles(order_packets, epsilon_d, a_s, b_s, a_p, b_p)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    exception_setup = 'def _value_error_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n'
    return [
        {
            "setup": "# Case: normal\nM=np.array([[0.2,0.6,-0.3],[-0.7,0.1,0.4],[0.3,-0.5,0.8]])\nq=_oracle_evaluate_rotated_qmoke_quadratic_response(M,.07-.02j,.03+.01j,'001',.23)\nc=_oracle_evaluate_rotated_cmoke_cubic_response(M,.004+.002j,-.001+.0003j,'001',.23)\no=_oracle_pack_magnetooptic_orders(M,.025-.014j,q,c)",
            "call": 'compute_third_order_kerr_angles(o,-9.6+14.2j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j)',
            "gold_call": '_oracle_compute_third_order_kerr_angles(o,-9.6+14.2j,.73-.18j,.11+.06j,-.61+.27j,.16-.09j)',
        },
        {
            "setup": '# Case: boundary\n# Coverage: boundary (zero order packet and coefficients)\no=np.zeros((1,3,6,2),dtype=float)',
            "call": 'compute_third_order_kerr_angles(o,1+0j,0j,0j,0j,0j)',
            "gold_call": '_oracle_compute_third_order_kerr_angles(o,1+0j,0j,0j,0j,0j)',
        },
        {
            "setup": "# Case: edge\n# Coverage: edge (largest-finite epsilon_d)\nM=np.array([[0.13,-0.71,0.22],[-0.42,0.33,-0.59],[0.81,0.14,0.07],[0.28,-0.19,0.63]])\nq=_oracle_evaluate_rotated_qmoke_quadratic_response(M,.07-.02j,.03+.01j,'111',-.51)\nc=_oracle_evaluate_rotated_cmoke_cubic_response(M,.004+.002j,-.001+.0003j,'111',-.51)\no=_oracle_pack_magnetooptic_orders(M,.025-.014j,q,c)",
            "call": 'compute_third_order_kerr_angles(o,np.finfo(float).max+0j,.64+.11j,-.09+.04j,-.52-.13j,.14+.08j)',
            "gold_call": '_oracle_compute_third_order_kerr_angles(o,np.finfo(float).max+0j,.64+.11j,-.09+.04j,-.52-.13j,.14+.08j)',
        },
        {
            "setup": '# Case: edge\no=np.zeros((1,3,6,2))\n' + exception_setup,
            "call": '_value_error_code(compute_third_order_kerr_angles, o, 0j, 0.1, 0.2, 0.3, 0.4)',
            "gold_call": '_value_error_code(_oracle_compute_third_order_kerr_angles, o, 0j, 0.1, 0.2, 0.3, 0.4)',
        },
        {
            "setup": '# Case: edge\no=np.zeros((1,3,6,2)); o[0,0,0,0]=np.nan\n' + exception_setup,
            "call": '_value_error_code(compute_third_order_kerr_angles, o, 1j, 0.1, 0.2, 0.3, 0.4)',
            "gold_call": '_value_error_code(_oracle_compute_third_order_kerr_angles, o, 1j, 0.1, 0.2, 0.3, 0.4)',
        },
        {
            "setup": '# Case: edge\no=np.zeros((0,3,6,2))\n' + exception_setup,
            "call": '_value_error_code(compute_third_order_kerr_angles, o, 1j, 0.1, 0.2, 0.3, 0.4)',
            "gold_call": '_value_error_code(_oracle_compute_third_order_kerr_angles, o, 1j, 0.1, 0.2, 0.3, 0.4)',
        },
    ]
