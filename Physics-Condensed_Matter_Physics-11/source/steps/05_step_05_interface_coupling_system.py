"""
The segment is bounded by two interfaces in space, and across each of them the displacement and the axial stress must be continuous at every instant. Because the exteriors are uniform and the interior is not, the two sides of each condition are written in different bases: outside, a sum of scattering orders each travelling at the wave speed of the unmodulated rod; inside, a sum over the basic modes of the previous step, each of which is itself a series in the harmonics. Projecting the conditions onto the time basis, which is orthogonal because the harmonic frequencies are distinct, turns each condition into one algebraic equation per retained order.

The count of retained orders is not free. The displacement condition involves the field alone and would balance at every order the expansion carries. The stress condition does not, because the stress inside the segment is the modulus times the gradient, and the modulus carries its own harmonic content: forming that product spreads each harmonic by up to P orders in each direction, so balancing the stress at order j calls on the interior field at orders j - P to j + P. Those lie inside the truncated window only when the magnitude of j does not exceed n_order - P. The mode-coupling order is therefore J = n_order - P, the scattering orders run from minus J to plus J, and only the basic modes of the same range are retained, which leaves 4 J + 2 interior amplitudes against 4 J + 2 exterior ones and a square system of 8 J + 4 equations. Matching at every order the expansion carries instead would call on harmonics the expansion does not hold and silently set them to zero.

Two conventions in the exterior have to be got right. A scattering order of order n sits at frequency omega + n omega_m, which for a sufficiently down-converted order is negative, and the wavenumber that keeps such an order travelling away from the segment is the signed ratio of that frequency to the wave speed rather than the positive root of its square. And the position of the second interface enters through more than the carrier: harmonic n of an interior mode advances with the shifted wavenumber, so the phase it accumulates across the segment is that of the mode plus n times the modulation wavenumber times the length. Those two contributions coincide only when the segment length is a whole number of modulation wavelengths, so a segment of any other length distinguishes an implementation that carries the shifted phase from one that does not.

The unknowns are ordered so that the solution of the next step can be sliced without ambiguity: the forward interior amplitudes indexed by basic-mode order, then the backward interior amplitudes in the same indexing, then the reflected amplitudes indexed by scattering order, then the transmitted amplitudes.

Phasor and amplitude convention: use the complex field convention exp[i(omega_n t - k_n x)], with omega_n = omega + n omega_m, and the common global coordinate x=0 at the left interface and x=segment_length at the right interface. In the left exterior, the reflected unknown R_n multiplies exp[i(omega_n t - k_n^- x)]; in the right exterior, the transmitted unknown T_n multiplies exp[i(omega_n t - k_n^+ x)], where k_n^- = -omega_n/c_0, k_n^+ = omega_n/c_0 and c_0 is the exterior wave speed. Thus the transmitted displacement at the right interface contains T_n exp(-i k_n^+ segment_length). Each interior amplitude multiplies the supplied mode vector in the same convention: its harmonic n carries exp[i(omega_n t - (kappa_s + n kappa_m)x)], where kappa_s is that supplied basic-mode wavenumber. The incident coefficient is one at x=0. Equivalent rescaling or reordering of the equation rows is permitted while the stated unknown ordering and amplitude definitions are retained.

Returns
-------
dict holding a complex128 matrix of shape (8 J + 4, 8 J + 4) and a complex128 rhs of shape (8 J + 4,) for a unit-amplitude incident wave, together with the native int j_order equal to J.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interface_coupling_system(
    forward_wavenumbers: np.ndarray,
    forward_modes: np.ndarray,
    backward_wavenumbers: np.ndarray,
    backward_modes: np.ndarray,
    coefficients: np.ndarray,
    omega: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
    segment_length: float,
) -> dict:
    """Assemble the square linear system that the two interface conditions impose on the interior and exterior amplitudes.

    Parameters
    ----------
    forward_wavenumbers : np.ndarray
        Forward basic-mode wavenumbers indexed by order.
    forward_modes : np.ndarray
        Their harmonic mode shapes.
    backward_wavenumbers : np.ndarray
        Backward basic-mode wavenumbers indexed by order.
    backward_modes : np.ndarray
        Their harmonic mode shapes.
    coefficients : np.ndarray
        One-dimensional modulus Fourier coefficients indexed by orders -P through +P,
        in that order, with odd length 2P+1 for any integer P >= 0. The single-cosine
        task instance has P=1 and therefore three entries. The P=0 one-entry limit is
        also supported by this function, including the N=0, J=0 one-mode limit.
    omega : float
        Driving angular frequency in radians per second.
    rho0 : float
        Mass density in kilogram per cubic metre.
    kappa_m : float
        Modulation wavenumber in radians per metre.
    omega_m : float
        Modulation angular frequency in radians per second.
    segment_length : float
        Length of the modulated segment in metres.

    Returns
    -------
    dict
        Under the keys matrix, rhs and j_order. The matrix entry is a complex128
        array of shape (8 J + 4, 8 J + 4) and the rhs entry a complex128 array of
        shape (8 J + 4,) for a unit-amplitude incident wave, with the unknowns
        ordered as the forward interior amplitudes, the backward interior
        amplitudes, the reflected amplitudes and the transmitted amplitudes. The
        j_order entry is the native int mode-coupling order J.

    Raises
    ------
    ValueError
        If the coefficients are not one-dimensional with odd length, if the mode-family
        shapes are inconsistent or not of odd side length, if any input is not finite,
        if omega, rho0 or segment_length is not above zero, if kappa_m or omega_m is
        zero, or if the truncation leaves no matchable order.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_interface_coupling_system(
    forward_wavenumbers: np.ndarray,
    forward_modes: np.ndarray,
    backward_wavenumbers: np.ndarray,
    backward_modes: np.ndarray,
    coefficients: np.ndarray,
    omega: float,
    rho0: float,
    kappa_m: float,
    omega_m: float,
    segment_length: float,
) -> dict:
    kf = np.asarray(forward_wavenumbers, dtype=np.complex128)
    kb = np.asarray(backward_wavenumbers, dtype=np.complex128)
    uf = np.asarray(forward_modes, dtype=np.complex128)
    ub = np.asarray(backward_modes, dtype=np.complex128)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    omega = float(omega)
    rho0 = float(rho0)
    kappa_m = float(kappa_m)
    omega_m = float(omega_m)
    segment_length = float(segment_length)

    if coefficients.ndim != 1 or coefficients.size % 2 != 1:
        raise ValueError("coefficients must be a one-dimensional array of odd length")
    p_order = (coefficients.size - 1) // 2
    if kf.ndim != 1 or kb.shape != kf.shape:
        raise ValueError("the two wavenumber families must be one-dimensional and of equal size")
    side = kf.size
    if side % 2 != 1:
        raise ValueError("each family must hold an odd number of basic modes")
    if uf.shape != (side, side) or ub.shape != (side, side):
        raise ValueError("each mode-shape array must be square and match its family")
    for arr in (kf, kb, uf, ub, coefficients):
        if not np.all(np.isfinite(arr)):
            raise ValueError("every input array must be finite")
    if not np.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and above zero")
    if not np.isfinite(rho0) or rho0 <= 0.0:
        raise ValueError("rho0 must be finite and above zero")
    if not np.isfinite(segment_length) or segment_length <= 0.0:
        raise ValueError("segment_length must be finite and above zero")
    if not np.isfinite(kappa_m) or kappa_m == 0.0:
        raise ValueError("kappa_m must be finite and not zero")
    if not np.isfinite(omega_m) or omega_m == 0.0:
        raise ValueError("omega_m must be finite and not zero")

    n_order = (side - 1) // 2
    j_order = n_order - p_order
    if j_order < 0:
        raise ValueError("the truncation leaves no matchable scattering order")

    e0 = float(coefficients[p_order])
    speed = np.sqrt(e0 / rho0)
    width = 2 * j_order + 1
    size = 4 * width
    matrix = np.zeros((size, size), dtype=np.complex128)
    rhs = np.zeros(size, dtype=np.complex128)

    def _amp_forward(s, n):
        return uf[n + n_order, s + n_order]

    def _amp_backward(s, n):
        return ub[n + n_order, s + n_order]

    def _shifted_forward(s, n):
        return kf[s + n_order] + n * kappa_m

    def _shifted_backward(s, n):
        return kb[s + n_order] + n * kappa_m

    def _exterior(n, sign):
        return sign * (omega + n * omega_m) / speed

    col_b = 0
    col_c = width
    col_r = 2 * width
    col_t = 3 * width
    incident_wavenumber = omega / speed

    row = 0
    for j in range(-j_order, j_order + 1):
        for s in range(-j_order, j_order + 1):
            matrix[row, col_b + s + j_order] -= _amp_forward(s, j)
            matrix[row, col_c + s + j_order] -= _amp_backward(s, j)
        matrix[row, col_r + j + j_order] += 1.0
        rhs[row] = -1.0 if j == 0 else 0.0
        row += 1

    for j in range(-j_order, j_order + 1):
        for s in range(-j_order, j_order + 1):
            for p in range(-p_order, p_order + 1):
                m = j - p
                if -n_order <= m <= n_order:
                    weight = coefficients[p + p_order]
                    matrix[row, col_b + s + j_order] -= weight * _shifted_forward(s, m) * _amp_forward(s, m)
                    matrix[row, col_c + s + j_order] -= weight * _shifted_backward(s, m) * _amp_backward(s, m)
        matrix[row, col_r + j + j_order] += e0 * _exterior(j, -1.0)
        rhs[row] = -e0 * incident_wavenumber if j == 0 else 0.0
        row += 1

    for j in range(-j_order, j_order + 1):
        for s in range(-j_order, j_order + 1):
            matrix[row, col_b + s + j_order] += _amp_forward(s, j) * np.exp(-1j * _shifted_forward(s, j) * segment_length)
            matrix[row, col_c + s + j_order] += _amp_backward(s, j) * np.exp(-1j * _shifted_backward(s, j) * segment_length)
        matrix[row, col_t + j + j_order] -= np.exp(-1j * _exterior(j, 1.0) * segment_length)
        row += 1

    for j in range(-j_order, j_order + 1):
        for s in range(-j_order, j_order + 1):
            phase_f = np.exp(-1j * _shifted_forward(s, j) * segment_length)
            phase_b = np.exp(-1j * _shifted_backward(s, j) * segment_length)
            for p in range(-p_order, p_order + 1):
                m = j - p
                if -n_order <= m <= n_order:
                    weight = coefficients[p + p_order]
                    matrix[row, col_b + s + j_order] += weight * _shifted_forward(s, m) * _amp_forward(s, m) * phase_f
                    matrix[row, col_c + s + j_order] += weight * _shifted_backward(s, m) * _amp_backward(s, m) * phase_b
        matrix[row, col_t + j + j_order] -= e0 * _exterior(j, 1.0) * np.exp(-1j * _exterior(j, 1.0) * segment_length)
        row += 1

    return {"matrix": matrix, "rhs": rhs, "j_order": int(j_order)}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
n = 2
side = 2 * n + 1
rng = np.random.default_rng(7)
KF = np.asarray(np.arange(-n, n + 1) * 1.7 + 3.0, dtype=complex)
KB = np.asarray(-np.arange(-n, n + 1) * 2.3 - 3.0, dtype=complex)
UF = np.asarray(rng.normal(size=(side, side)) + 1j * rng.normal(size=(side, side)), dtype=complex)
UB = np.asarray(rng.normal(size=(side, side)) + 1j * rng.normal(size=(side, side)), dtype=complex)
CO = np.array([0.15, 1.0, 0.15])

def capture_system(d):
    matrix = np.asarray(d['matrix'])
    rhs = np.asarray(d['rhs'])
    solution = np.linalg.solve(matrix, rhs)
    return np.concatenate((np.asarray(matrix.shape), np.asarray(rhs.shape),
                           [d['j_order']], solution.real, solution.imag))
""",
            'call': 'capture_system(interface_coupling_system(np.array(KF, copy=True), np.array(UF, copy=True), np.array(KB, copy=True), np.array(UB, copy=True), np.array(CO, copy=True), 15.0, 1.0, 10.0, 20.0, 0.5 * np.pi))',
            'gold_call': 'capture_system(_oracle_interface_coupling_system(np.array(KF, copy=True), np.array(UF, copy=True), np.array(KB, copy=True), np.array(UB, copy=True), np.array(CO, copy=True), 15.0, 1.0, 10.0, 20.0, 0.5 * np.pi))',
        },
        {
            "setup": """
import numpy as np
n = 1
side = 3
KF = np.asarray([1.0, 2.0, 3.0], dtype=complex)
KB = np.asarray([-3.0, -2.0, -1.0], dtype=complex)
UF = np.asarray(np.eye(3), dtype=complex)
UB = np.asarray(np.eye(3)[::-1], dtype=complex)
CO = np.array([0.0, 1.0, 0.0])

def capture_system(d):
    matrix = np.asarray(d['matrix'])
    rhs = np.asarray(d['rhs'])
    solution = np.linalg.solve(matrix, rhs)
    return np.concatenate((np.asarray(matrix.shape), np.asarray(rhs.shape),
                           [d['j_order']], solution.real, solution.imag))
""",
            'call': 'capture_system(interface_coupling_system(np.array(KF, copy=True), np.array(UF, copy=True), np.array(KB, copy=True), np.array(UB, copy=True), np.array(CO, copy=True), 5.0, 1.0, -4.0, 9.0, 1.25))',
            'gold_call': 'capture_system(_oracle_interface_coupling_system(np.array(KF, copy=True), np.array(UF, copy=True), np.array(KB, copy=True), np.array(UB, copy=True), np.array(CO, copy=True), 5.0, 1.0, -4.0, 9.0, 1.25))',
        },
        {
            "setup": """
import numpy as np
KF = np.asarray([1.0], dtype=complex)
KB = np.asarray([-1.0], dtype=complex)
UF = np.asarray([[1.0]], dtype=complex)
UB = np.asarray([[1.0]], dtype=complex)
CO = np.array([1.0])

def capture_system(d):
    matrix = np.asarray(d['matrix'])
    rhs = np.asarray(d['rhs'])
    solution = np.linalg.solve(matrix, rhs)
    return np.concatenate((np.asarray(matrix.shape), np.asarray(rhs.shape),
                           [d['j_order']], solution.real, solution.imag))
""",
            'call': 'capture_system(interface_coupling_system(np.array(KF, copy=True), np.array(UF, copy=True), np.array(KB, copy=True), np.array(UB, copy=True), np.array(CO, copy=True), 3.0, 1.0, 2.0, 7.0, 0.75))',
            'gold_call': 'capture_system(_oracle_interface_coupling_system(np.array(KF, copy=True), np.array(UF, copy=True), np.array(KB, copy=True), np.array(UB, copy=True), np.array(CO, copy=True), 3.0, 1.0, 2.0, 7.0, 0.75))',
        },
        {
            "setup": """
import numpy as np
n = 3
side = 2 * n + 1
rng = np.random.default_rng(29)
KF = np.asarray(np.linspace(-2.5, 8.5, side), dtype=complex)
KB = np.asarray(np.linspace(3.5, -9.5, side), dtype=complex)
UF = np.asarray(rng.normal(size=(side, side)) + 1j * rng.normal(size=(side, side)), dtype=complex)
UB = np.asarray(rng.normal(size=(side, side)) + 1j * rng.normal(size=(side, side)), dtype=complex)
CO = np.array([0.03, 0.11, 1.0, 0.11, 0.03])

def capture_system(d):
    matrix = np.asarray(d['matrix'])
    rhs = np.asarray(d['rhs'])
    solution = np.linalg.solve(matrix, rhs)
    return np.concatenate((np.asarray(matrix.shape), np.asarray(rhs.shape),
                           [d['j_order']], solution.real, solution.imag))
""",
            'call': 'capture_system(interface_coupling_system(np.array(KF, copy=True), np.array(UF, copy=True), np.array(KB, copy=True), np.array(UB, copy=True), np.array(CO, copy=True), 13.0, 1.3, -5.0, 8.0, 0.73))',
            'gold_call': 'capture_system(_oracle_interface_coupling_system(np.array(KF, copy=True), np.array(UF, copy=True), np.array(KB, copy=True), np.array(UB, copy=True), np.array(CO, copy=True), 13.0, 1.3, -5.0, 8.0, 0.73))',
        },
        {
            "setup": """
import numpy as np
KF = np.asarray([1.0, 2.0, 3.0], dtype=complex)
KB = np.asarray([-3.0, -2.0, -1.0], dtype=complex)
UF = np.asarray(np.eye(3), dtype=complex)
UB = np.asarray(np.eye(3), dtype=complex)
CO = np.array([0.15, 1.0, 0.15])
SHORT = np.asarray([1.0, 2.0], dtype=complex)
NAN = np.asarray([1.0, float('nan'), 3.0], dtype=complex)
RECT = np.asarray(np.ones((3, 4)), dtype=complex)
def verdict(fn, kf=KF, uf=UF, kb=KB, ub=UB, co=CO, w=5.0, r=1.0, km=4.0, wm=9.0, L=1.25):
    try:
        fn(np.array(kf, copy=True), np.array(uf, copy=True), np.array(kb, copy=True), np.array(ub, copy=True), np.array(co, copy=True), w, r, km, wm, L)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            'call': '(verdict(interface_coupling_system, kf=SHORT), verdict(interface_coupling_system, kf=NAN), verdict(interface_coupling_system, uf=RECT), verdict(interface_coupling_system, L=0.0), verdict(interface_coupling_system, km=0.0), verdict(interface_coupling_system, w=-1.0), verdict(interface_coupling_system))',
            'gold_call': '(verdict(_oracle_interface_coupling_system, kf=SHORT), verdict(_oracle_interface_coupling_system, kf=NAN), verdict(_oracle_interface_coupling_system, uf=RECT), verdict(_oracle_interface_coupling_system, L=0.0), verdict(_oracle_interface_coupling_system, km=0.0), verdict(_oracle_interface_coupling_system, w=-1.0), verdict(_oracle_interface_coupling_system))',
        },
    ]
