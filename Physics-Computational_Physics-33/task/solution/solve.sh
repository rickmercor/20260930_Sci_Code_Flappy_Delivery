#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

from itertools import permutations

def _c_evaluate_rotated_cmoke_cubic_response__packet(values):
    values = np.asarray(values, dtype=complex)
    return np.stack((values.real, values.imag), axis=-1).astype(float)

def _c_evaluate_rotated_cmoke_cubic_response__finite_complex(value, name):
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite scalar') from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f'{name} must be a finite scalar')
    return result

def _c_evaluate_rotated_cmoke_cubic_response__sample_transform(crystal_cut, alpha):
    if crystal_cut not in ('001', '111'):
        raise ValueError('crystal_cut must be "001" or "111"')
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError('alpha must be a finite non-Boolean scalar')
    try:
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('alpha must be a finite non-Boolean scalar') from exc
    if not np.isfinite(alpha):
        raise ValueError('alpha must be a finite non-Boolean scalar')
    c, s = (np.cos(alpha), np.sin(alpha))
    rotate_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if crystal_cut == '001':
        return rotate_z
    root2, root3, root6 = (np.sqrt(2.0), np.sqrt(3.0), np.sqrt(6.0))
    to111 = np.array([[-root6 / 3.0, root6 / 6.0, root6 / 6.0], [0.0, -root2 / 2.0, root2 / 2.0], [root3 / 3.0, root3 / 3.0, root3 / 3.0]])
    return rotate_z @ to111

def _c_evaluate_rotated_cmoke_cubic_response(magnetization: np.ndarray, h123: complex, h125: complex, crystal_cut: str, alpha: float) -> np.ndarray:
    M = np.asarray(magnetization, dtype=float)
    if M.ndim != 2 or M.shape[1] != 3 or M.shape[0] == 0:
        raise ValueError('magnetization must have nonempty shape (N,3)')
    if not np.all(np.isfinite(M)):
        raise ValueError('magnetization must be finite')
    h123 = _c_evaluate_rotated_cmoke_cubic_response__finite_complex(h123, 'h123')
    h125 = _c_evaluate_rotated_cmoke_cubic_response__finite_complex(h125, 'h125')
    tensor = np.zeros((3, 3, 3, 3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if len({i, j, k}) != 3:
                    continue
                sign = 1.0 if (i, j, k) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)} else -1.0
                tensor[i, j, k, k, k] = sign * h123
                for ell in range(3):
                    if ell == k:
                        continue
                    for triple in set(permutations((k, ell, ell))):
                        tensor[i, j, triple[0], triple[1], triple[2]] = sign * h125
    transform = _c_evaluate_rotated_cmoke_cubic_response__sample_transform(crystal_cut, alpha)
    rotated = np.einsum('ia,jb,kc,ld,me,abcde->ijklm', transform, transform, transform, transform, transform, tensor, optimize=True)
    epsilon = np.einsum('ijklm,nk,nl,nm->nij', rotated, M, M, M, optimize=True)
    return _c_evaluate_rotated_cmoke_cubic_response__packet(np.stack([epsilon[:, i, j] for i, j in ((1, 2), (2, 0), (0, 1), (2, 1), (0, 2), (1, 0))], axis=1))

def evaluate_rotated_cmoke_cubic_response(magnetization, h123, h125, crystal_cut, alpha):
    return _c_evaluate_rotated_cmoke_cubic_response(magnetization, h123, h125, crystal_cut, alpha)

import numpy as np

def _c_evaluate_rotated_qmoke_quadratic_response__packet(values):
    values = np.asarray(values, dtype=complex)
    return np.stack((values.real, values.imag), axis=-1).astype(float)

def _c_evaluate_rotated_qmoke_quadratic_response__finite_complex(value, name):
    try:
        result = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite scalar') from exc
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f'{name} must be a finite scalar')
    return result

def _c_evaluate_rotated_qmoke_quadratic_response__sample_transform(crystal_cut, alpha):
    if crystal_cut not in ('001', '111'):
        raise ValueError('crystal_cut must be "001" or "111"')
    if isinstance(alpha, (bool, np.bool_)):
        raise ValueError('alpha must be a finite non-Boolean scalar')
    try:
        alpha = float(alpha)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('alpha must be a finite non-Boolean scalar') from exc
    if not np.isfinite(alpha):
        raise ValueError('alpha must be a finite non-Boolean scalar')
    c, s = (np.cos(alpha), np.sin(alpha))
    rotate_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if crystal_cut == '001':
        return rotate_z
    root2, root3, root6 = (np.sqrt(2.0), np.sqrt(3.0), np.sqrt(6.0))
    to111 = np.array([[-root6 / 3.0, root6 / 6.0, root6 / 6.0], [0.0, -root2 / 2.0, root2 / 2.0], [root3 / 3.0, root3 / 3.0, root3 / 3.0]])
    return rotate_z @ to111

def _c_evaluate_rotated_qmoke_quadratic_response(magnetization: np.ndarray, g_s: complex, two_g44: complex, crystal_cut: str, alpha: float) -> np.ndarray:
    M = np.asarray(magnetization, dtype=float)
    if M.ndim != 2 or M.shape[1] != 3 or M.shape[0] == 0:
        raise ValueError('magnetization must have nonempty shape (N,3)')
    if not np.all(np.isfinite(M)):
        raise ValueError('magnetization must be finite')
    g_s = _c_evaluate_rotated_qmoke_quadratic_response__finite_complex(g_s, 'g_s')
    two_g44 = _c_evaluate_rotated_qmoke_quadratic_response__finite_complex(two_g44, 'two_g44')
    g44 = 0.5 * two_g44
    tensor = np.zeros((3, 3, 3, 3), dtype=complex)
    for i in range(3):
        tensor[i, i, i, i] = g_s
    for i in range(3):
        for j in range(i + 1, 3):
            tensor[i, j, i, j] = g44
            tensor[i, j, j, i] = g44
            tensor[j, i, i, j] = g44
            tensor[j, i, j, i] = g44
    transform = _c_evaluate_rotated_qmoke_quadratic_response__sample_transform(crystal_cut, alpha)
    rotated = np.einsum('ia,jb,kc,ld,abcd->ijkl', transform, transform, transform, transform, tensor, optimize=True)
    epsilon = np.einsum('ijkl,nk,nl->nij', rotated, M, M, optimize=True)
    return _c_evaluate_rotated_qmoke_quadratic_response__packet(np.stack([epsilon[:, i, j] for i, j in ((1, 2), (2, 0), (0, 1), (2, 1), (0, 2), (1, 0))], axis=1))

def evaluate_rotated_qmoke_quadratic_response(magnetization, g_s, two_g44, crystal_cut, alpha):
    return _c_evaluate_rotated_qmoke_quadratic_response(magnetization, g_s, two_g44, crystal_cut, alpha)

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

def pack_magnetooptic_orders(magnetization, k_linear, quadratic_response, cubic_response):
    return _c_pack_magnetooptic_orders(magnetization, k_linear, quadratic_response, cubic_response)

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

def compute_third_order_kerr_angles(order_packets, epsilon_d, a_s, b_s, a_p, b_p):
    return _c_compute_third_order_kerr_angles(order_packets, epsilon_d, a_s, b_s, a_p, b_p)

import numpy as np

def _c_build_eight_directional_design(sample_angles: np.ndarray) -> np.ndarray:
    alpha = np.asarray(sample_angles, dtype=float)
    if alpha.ndim != 1 or alpha.size == 0 or (not np.all(np.isfinite(alpha))):
        raise ValueError('sample_angles must be a nonempty finite 1-D array')
    mu = np.arange(8, dtype=float) * (np.pi / 4.0)
    magnetization = np.stack((np.cos(mu), np.sin(mu), np.zeros(8)), axis=1)
    out = np.empty((alpha.size, 8, 4), dtype=float)
    out[:, :, 0] = alpha[:, None]
    out[:, :, 1:] = magnetization[None, :, :]
    return out

def build_eight_directional_design(sample_angles):
    return _c_build_eight_directional_design(sample_angles)

import numpy as np

def _c_separate_eight_directional_channels__packet(values):
    values = np.asarray(values, dtype=complex)
    return np.stack((values.real, values.imag), axis=-1).astype(float)

def _c_separate_eight_directional_channels__unpacket(values, shape):
    array = np.asarray(values, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError('malformed real/imag packet')
    return array[..., 0] + 1j * array[..., 1]

def _c_separate_eight_directional_channels(kerr_scan: np.ndarray) -> np.ndarray:
    scan = np.asarray(kerr_scan, dtype=float)
    if scan.ndim != 4 or scan.shape[1:] != (8, 2, 2) or scan.shape[0] == 0:
        raise ValueError('kerr_scan must have nonempty shape (A,8,2,2)')
    values = _c_separate_eight_directional_channels__unpacket(scan, scan.shape)
    channels = np.stack(((values[:, 2] - values[:, 6]) / 2.0, (values[:, 1] + values[:, 5] - values[:, 3] - values[:, 7]) / 2.0, (values[:, 0] + values[:, 4] - values[:, 2] - values[:, 6]) / 2.0, (values[:, 0] - values[:, 4]) / 2.0), axis=1)
    return _c_separate_eight_directional_channels__packet(channels)

def separate_eight_directional_channels(kerr_scan):
    return _c_separate_eight_directional_channels(kerr_scan)

import numpy as np

def compute_cmoke_crystal_advantage(epsilon_d: complex, k_linear: complex, g_s: complex, two_g44: complex, h123: complex, h125: complex, a_s: complex, b_s: complex, a_p: complex, b_p: complex, n_angles: int) -> float:
    if isinstance(n_angles, (bool, np.bool_)) or not isinstance(n_angles, (int, np.integer)):
        raise ValueError('n_angles must be an integer')
    n_angles = int(n_angles)
    if n_angles < 24 or n_angles % 12:
        raise ValueError('n_angles must be at least 24 and divisible by 12')
    beta = 2.0 * np.pi * np.arange(n_angles, dtype=float) / n_angles
    alpha = beta + 0.07 * np.sin(5.0 * beta + 0.17) + 0.03 * np.cos(7.0 * beta - 0.31)
    weights = 1.0 + 0.2 * np.cos(3.0 * beta + 0.4) + 0.1 * np.sin(8.0 * beta - 0.2)
    weight_sum = float(np.sum(weights))
    if not np.all(np.isfinite(alpha)) or not np.all(weights > 0.0) or weight_sum <= 0.0:
        raise ValueError('warped scan must have finite angles and positive weights')
    design = build_eight_directional_design(alpha)
    ratios = {}
    for cut, harmonic in (('001', 4), ('111', 3)):
        scan_rows = []
        for row in design:
            angle = float(row[0, 0])
            magnetization = row[:, 1:]
            cubic = evaluate_rotated_cmoke_cubic_response(magnetization, h123, h125, cut, angle)
            quadratic = evaluate_rotated_qmoke_quadratic_response(magnetization, g_s, two_g44, cut, angle)
            orders = pack_magnetooptic_orders(magnetization, k_linear, quadratic, cubic)
            scan_rows.append(compute_third_order_kerr_angles(orders, epsilon_d, a_s, b_s, a_p, b_p))
        separated = separate_eight_directional_channels(np.stack(scan_rows))
        complex_channels = separated[..., 0] + 1j * separated[..., 1]
        phase = np.exp(-1j * harmonic * alpha)
        coefficient = 2.0 / weight_sum * np.tensordot(weights * phase, complex_channels, axes=(0, 0))
        cubic_norm = float(np.linalg.norm(coefficient[[0, 3], :]))
        quadratic_norm = float(np.linalg.norm(coefficient[[1, 2], :]))
        absolute_fourier_scale = float(2.0 / weight_sum * np.linalg.norm(np.sum(weights[:, None, None] * np.abs(complex_channels), axis=0)))
        zero_tolerance = 512.0 * np.finfo(float).eps * absolute_fourier_scale
        if quadratic_norm <= zero_tolerance:
            raise ValueError('quadratic anisotropy norms must be nonzero')
        if cubic_norm <= zero_tolerance:
            cubic_norm = 0.0
        ratio = cubic_norm / quadratic_norm
        if not np.isfinite(ratio) or ratio < 0.0:
            raise ValueError('anisotropy ratios must be finite and nonnegative')
        ratios[cut] = ratio
    return float(np.round(np.log1p(ratios['111']) - np.log1p(ratios['001']), 10))
SCICODE_GOLD_EOF
