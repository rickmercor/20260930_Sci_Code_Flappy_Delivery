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
from math import sqrt, pi
from scipy.linalg import eigh

def _osc_states(n_max: int):
    """(n_x, n_y) pairs with n_x + n_y <= n_max: shells n = 0..n_max, n_x ascending within a shell."""
    return [(nx, n - nx) for n in range(n_max + 1) for nx in range(n + 1)]

def oscillator_basis(n_max: int) -> "np.ndarray":
    """Ordered two-dimensional oscillator basis: (N, 2) integer array of (n_x, n_y), N = (n_max+1)(n_max+2)/2."""
    if int(n_max) != n_max or n_max < 0: raise ValueError("n_max must be a non-negative integer")
    return np.array(_osc_states(n_max), dtype=int)

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _osc_states(n_max: int):
    """(n_x, n_y) pairs with n_x + n_y <= n_max: shells n = 0..n_max, n_x ascending within a shell."""
    return [(nx, n - nx) for n in range(n_max + 1) for nx in range(n + 1)]

def ladder_operators(n_max: int) -> "np.ndarray":
    """Annihilation operators a_x, a_y of the two oscillator coordinates in the ordered basis, shape (2, N, N)."""
    if int(n_max) != n_max or n_max < 0: raise ValueError("n_max must be a non-negative integer")
    st = _osc_states(n_max); idx = {s: i for i, s in enumerate(st)}; N = len(st)
    ax = np.zeros((N, N)); ay = np.zeros((N, N))
    for i, (nx, ny) in enumerate(st):
        if nx > 0: ax[idx[(nx - 1, ny)], i] = sqrt(nx)
        if ny > 0: ay[idx[(nx, ny - 1)], i] = sqrt(ny)
    return np.stack([ax, ay])

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _osc_states(n_max: int):
    """(n_x, n_y) pairs with n_x + n_y <= n_max: shells n = 0..n_max, n_x ascending within a shell."""
    return [(nx, n - nx) for n in range(n_max + 1) for nx in range(n + 1)]

def vibronic_operators(n_max: int) -> "np.ndarray":
    """Exact matrices of Q_+, Q_-, n_vib and l_vib on the retained basis, shape (4, N, N), complex:
    Q_+- = (x +- i y) with x = (a_x + a_x^+)/sqrt2, y = (a_y + a_y^+)/sqrt2, n_vib = a_x^+ a_x + a_y^+ a_y,
    l_vib = i (a_x a_y^+ - a_x^+ a_y), every element taken between retained states of the untruncated operators."""
    if int(n_max) != n_max or n_max < 0: raise ValueError("n_max must be a non-negative integer")
    big = ladder_operators(n_max + 2)
    st_big = _osc_states(n_max + 2); idx = {s: i for i, s in enumerate(st_big)}
    sel = [idx[s] for s in _osc_states(n_max)]
    ax, ay = big[0], big[1]
    x = (ax + ax.T) / sqrt(2.0); y = (ay + ay.T) / sqrt(2.0)
    Qp = x + 1j * y; Qm = x - 1j * y
    nvib = ax.T @ ax + ay.T @ ay
    lvib = 1j * (ax @ ay.T - ax.T @ ay)
    out = np.stack([Qp, Qm, nvib.astype(complex), lvib])
    return out[:, sel, :][:, :, sel]

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _apply(A, X):
    """A @ X, with A allowed to be the 1-D diagonal of a diagonal operator."""
    if A.ndim == 1: return A[:, None] * X if X.ndim > 1 else A * X
    return A @ X

def _kramers_g(H0: np.ndarray, Z: np.ndarray, extra: list):
    """Exact g-factor of the lowest Kramers doublet of H0 from the Zeeman operator Z (per mu_B B_z):
    eigenvalue splitting of the 2 x 2 projection of Z; also expectation values of the operators in `extra`
    in the doublet state of larger Zeeman eigenvalue. Z and the entries of `extra` may be dense matrices or
    the 1-D diagonals of diagonal operators."""
    E, C = eigh(H0, driver="evr", subset_by_index=[0, min(3, H0.shape[0] - 1)])
    if E[1] - E[0] > 1e-8: raise ValueError("the two lowest levels are not a degenerate Kramers doublet")
    if H0.shape[0] > 2 and E[2] - E[1] < 1e-8: raise ValueError("more than two degenerate lowest levels")
    P = C[:, :2]
    M = P.conj().T @ _apply(Z, P)
    w, U = np.linalg.eigh(M)
    v = P @ U[:, 1]
    vals = [float(np.real(np.vdot(v, _apply(A, v)))) for A in extra]
    return float(E[0]), float(w[1] - w[0]), vals, (E[2] - E[0] if H0.shape[0] > 2 else 0.0)

def static_model(xi: float, F_rho: float, particle: bool) -> "np.ndarray":
    """The source's static relativistic Jahn-Teller model (four states, frozen distortion):
    [E_0, g_exact, g_formula, <L_z>] with g_exact from the exact Kramers-doublet Zeeman splitting and
    g_formula = g_e -+ 2 xi / sqrt(xi^2 + 4 F_rho^2)."""
    if not (np.isfinite(xi) and xi > 0 and np.isfinite(F_rho) and F_rho > 0): raise ValueError("xi and F_rho must be positive")
    s = 1.0 if particle else -1.0
    Lz = np.diag([1.0, -1.0]); Sz = np.diag([0.5, -0.5]); I2 = np.eye(2)
    P1m = np.zeros((2, 2)); P1m[0, 1] = 1.0
    H0 = xi * np.kron(Lz, Sz) + F_rho * np.kron(P1m + P1m.T, I2)
    Z = s * np.kron(Lz, I2) + _GE() * np.kron(I2, Sz)
    E0, g, vals, _ = _kramers_g(H0.astype(complex), Z.astype(complex), [np.kron(Lz, I2).astype(complex)])
    gf = _GE() - s * 2.0 * xi / sqrt(xi * xi + 4.0 * F_rho * F_rho)
    return np.array([E0, g, gf, vals[0]], dtype=float)

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _MUB():
    """Bohr magneton in cm^-1 per tesla."""
    return 0.4668644814

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _check_pos(name, x):
    if not (np.isfinite(x) and x > 0): raise ValueError("%s must be positive" % name)

def _check_int(name, x, lo=0):
    if int(x) != x or x < lo: raise ValueError("%s must be an integer >= %d" % (name, lo))

def model_hamiltonian(xi: float, E_JT: float, hw: float, Bz: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Full Hamiltonian of the cavity-modified dynamic relativistic E x e Jahn-Teller model, shape (D, D),
    D = 4 N (n_ph + 1), ordering (m_l = +1, -1) x (m_s = up, down) x oscillator basis x photon number 0..n_ph."""
    _check_pos("xi", xi); _check_pos("hw", hw); _check_pos("hwc", hwc)
    if not (np.isfinite(E_JT) and E_JT >= 0): raise ValueError("E_JT must be non-negative")
    if not (np.isfinite(g_c) and g_c >= 0): raise ValueError("g_c must be non-negative")
    if not np.isfinite(Bz): raise ValueError("Bz must be finite")
    _check_int("n_max", n_max, 1); _check_int("n_ph", n_ph, 0)
    ops = vibronic_operators(n_max); N = ops.shape[1]
    s = 1.0 if particle else -1.0
    I2 = np.eye(2); IN = np.eye(N); IP = np.eye(n_ph + 1)
    Lz2 = np.diag([1.0, -1.0]); Sz2 = np.diag([0.5, -0.5])
    P1m = np.zeros((2, 2)); P1m[0, 1] = 1.0
    F = sqrt(2.0 * E_JT * hw)
    Hes = xi * np.kron(Lz2, Sz2) + _MUB() * Bz * (s * np.kron(Lz2, I2) + _GE() * np.kron(I2, Sz2))
    Hvib = hw * (ops[2] + IN)
    HJT = F * (np.kron(np.kron(P1m, I2), ops[0]) + np.kron(np.kron(P1m.T, I2), ops[1]))
    bph = np.diag(np.sqrt(np.arange(1, n_ph + 1)), 1)
    Hc = hwc * (bph.T @ bph)
    H = (np.kron(np.kron(Hes, IN), IP) + np.kron(np.kron(np.kron(I2, I2), Hvib), IP) + np.kron(HJT, IP)
         + np.kron(np.kron(np.kron(I2, I2), IN), Hc))
    if g_c > 0:
        sy = np.array([[0.0, -1j], [1j, 0.0]])
        H = H + 1j * g_c * np.kron(np.kron(np.kron(I2, sy), IN), bph.T - bph)
    H = np.asarray(H, dtype=complex)
    if np.abs(H - H.conj().T).max() > 1e-10: raise ValueError("Hamiltonian not Hermitian")
    return H

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _check_int(name, x, lo=0):
    if int(x) != x or x < lo: raise ValueError("%s must be an integer >= %d" % (name, lo))

def _apply(A, X):
    """A @ X, with A allowed to be the 1-D diagonal of a diagonal operator."""
    if A.ndim == 1: return A[:, None] * X if X.ndim > 1 else A * X
    return A @ X

def _model_operators(n_max: int, n_ph: int, particle: bool, which: str = "ZJLSV"):
    """Zeeman (per mu_B B_z), conserved J = l_vib - L_z/2, L_z, S_z and the cavity coupling operator
    i sigma_y (b^+ - b) on the full product space, ordering (m_l, m_s, vibration, photon). `which` selects
    which of Z, J, L_z, S_z, V to build; the others are returned as None. L_z, S_z and Z are diagonal in this
    ordering and are returned as their 1-D diagonals, to be applied with `_apply`; only J and V are dense."""
    ops = vibronic_operators(n_max); N = ops.shape[1]
    s = 1.0 if particle else -1.0
    nb = n_ph + 1
    blk = np.repeat(np.arange(4), N * nb)
    lz_d = np.where(blk < 2, 1.0, -1.0); sz_d = np.where(blk % 2 == 0, 0.5, -0.5)
    Lz = lz_d.astype(complex) if "L" in which else None
    Sz = sz_d.astype(complex) if "S" in which else None
    Z = (s * lz_d + _GE() * sz_d).astype(complex) if "Z" in which else None
    J = None
    if "J" in which:
        J = np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), ops[3]), np.eye(nb))
        J[np.diag_indices_from(J)] -= 0.5 * lz_d
    V = None
    if "V" in which:
        sy = np.array([[0.0, -1j], [1j, 0.0]]); bph = np.diag(np.sqrt(np.arange(1, n_ph + 1)), 1)
        V = 1j * np.kron(np.kron(np.kron(np.eye(2), sy), np.eye(N)), bph.T - bph)
    return Z, J, Lz, Sz, V

def vibronic_spectrum(xi: float, E_JT: float, hw: float, n_max: int, particle: bool, n_levels: int) -> "np.ndarray":
    """Lowest n_levels vibronic levels of the molecular model (no field, no cavity), shape (n_levels, 3):
    [E_k, |j_k|, <L_z S_z>_k] with |j_k| = sqrt(<J^2>) the magnitude of the conserved vibronic angular momentum
    J = l_vib - L_z/2 and <L_z S_z> the spin-orbit expectation value."""
    _check_int("n_levels", n_levels, 1)
    H = model_hamiltonian(xi, E_JT, hw, 0.0, 0.0, 1.0, n_max, 0, particle)
    if n_levels > H.shape[0]: raise ValueError("n_levels exceeds the basis size")
    Z, J, Lz, Sz, V = _model_operators(n_max, 0, particle, "JLS")
    E, C = eigh(H, driver="evr", subset_by_index=[0, n_levels - 1])
    J2 = J @ J; LS = Lz * Sz
    out = []
    for k in range(n_levels):
        v = C[:, k]
        out.append([E[k], sqrt(max(float(np.real(np.vdot(v, J2 @ v))), 0.0)), float(np.real(np.vdot(v, _apply(LS, v))))])
    return np.array(out, dtype=float)

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _check_pos(name, x):
    if not (np.isfinite(x) and x > 0): raise ValueError("%s must be positive" % name)

def _check_int(name, x, lo=0):
    if int(x) != x or x < lo: raise ValueError("%s must be an integer >= %d" % (name, lo))

def ham_reduction(E_JT: float, hw: float, n_max: int) -> "np.ndarray":
    """Spin-free linear E x e Jahn-Teller problem: [E_0, E_1 - E_0, p, q] with E_0 the ground vibronic doublet
    (|j| = 1/2), E_1 the first excited vibronic level (|j| = 3/2), and the Ham reduction factors p and q,
    the magnitudes of the eigenvalues of the 2 x 2 projections of L_z and of sigma_x = |1><-1| + |-1><1| on the
    ground doublet (q = (1 + p)/2 for the linear coupling)."""
    _check_pos("hw", hw)
    if not (np.isfinite(E_JT) and E_JT >= 0): raise ValueError("E_JT must be non-negative")
    _check_int("n_max", n_max, 1)
    ops = vibronic_operators(n_max); N = ops.shape[1]
    I2 = np.eye(2); IN = np.eye(N)
    P1m = np.zeros((2, 2)); P1m[0, 1] = 1.0
    F = sqrt(2.0 * E_JT * hw)
    H = np.kron(I2, hw * (ops[2] + IN)) + F * (np.kron(P1m, ops[0]) + np.kron(P1m.T, ops[1]))
    E, C = eigh(np.asarray(H, dtype=complex), driver="evr", subset_by_index=[0, 3])
    if E[1] - E[0] > 1e-8 or E[2] - E[1] < 1e-8: raise ValueError("the spin-free ground level is not an isolated vibronic doublet")
    P = C[:, :2]
    Lz = np.kron(np.diag([1.0, -1.0]), IN).astype(complex); sx = np.kron(P1m + P1m.T, IN).astype(complex)
    p = float(abs(np.linalg.eigvalsh(P.conj().T @ (Lz @ P))[1]))
    q = float(abs(np.linalg.eigvalsh(P.conj().T @ (sx @ P))[1]))
    return np.array([E[0], E[2] - E[0], p, q], dtype=float)

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh
from functools import lru_cache

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _apply(A, X):
    """A @ X, with A allowed to be the 1-D diagonal of a diagonal operator."""
    if A.ndim == 1: return A[:, None] * X if X.ndim > 1 else A * X
    return A @ X

def _kramers_g(H0: np.ndarray, Z: np.ndarray, extra: list):
    """Exact g-factor of the lowest Kramers doublet of H0 from the Zeeman operator Z (per mu_B B_z):
    eigenvalue splitting of the 2 x 2 projection of Z; also expectation values of the operators in `extra`
    in the doublet state of larger Zeeman eigenvalue. Z and the entries of `extra` may be dense matrices or
    the 1-D diagonals of diagonal operators."""
    E, C = eigh(H0, driver="evr", subset_by_index=[0, min(3, H0.shape[0] - 1)])
    if E[1] - E[0] > 1e-8: raise ValueError("the two lowest levels are not a degenerate Kramers doublet")
    if H0.shape[0] > 2 and E[2] - E[1] < 1e-8: raise ValueError("more than two degenerate lowest levels")
    P = C[:, :2]
    M = P.conj().T @ _apply(Z, P)
    w, U = np.linalg.eigh(M)
    v = P @ U[:, 1]
    vals = [float(np.real(np.vdot(v, _apply(A, v)))) for A in extra]
    return float(E[0]), float(w[1] - w[0]), vals, (E[2] - E[0] if H0.shape[0] > 2 else 0.0)

def _model_operators(n_max: int, n_ph: int, particle: bool, which: str = "ZJLSV"):
    """Zeeman (per mu_B B_z), conserved J = l_vib - L_z/2, L_z, S_z and the cavity coupling operator
    i sigma_y (b^+ - b) on the full product space, ordering (m_l, m_s, vibration, photon). `which` selects
    which of Z, J, L_z, S_z, V to build; the others are returned as None. L_z, S_z and Z are diagonal in this
    ordering and are returned as their 1-D diagonals, to be applied with `_apply`; only J and V are dense."""
    ops = vibronic_operators(n_max); N = ops.shape[1]
    s = 1.0 if particle else -1.0
    nb = n_ph + 1
    blk = np.repeat(np.arange(4), N * nb)
    lz_d = np.where(blk < 2, 1.0, -1.0); sz_d = np.where(blk % 2 == 0, 0.5, -0.5)
    Lz = lz_d.astype(complex) if "L" in which else None
    Sz = sz_d.astype(complex) if "S" in which else None
    Z = (s * lz_d + _GE() * sz_d).astype(complex) if "Z" in which else None
    J = None
    if "J" in which:
        J = np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), ops[3]), np.eye(nb))
        J[np.diag_indices_from(J)] -= 0.5 * lz_d
    V = None
    if "V" in which:
        sy = np.array([[0.0, -1j], [1j, 0.0]]); bph = np.diag(np.sqrt(np.arange(1, n_ph + 1)), 1)
        V = 1j * np.kron(np.kron(np.kron(np.eye(2), sy), np.eye(N)), bph.T - bph)
    return Z, J, Lz, Sz, V

@lru_cache(maxsize=256)
def _kramers_doublet(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool):
    """Cached core of the Kramers-doublet solve, returned as a plain tuple so no caller can mutate it.
    The calculation is pure in its arguments, and the audit of the last step asks for the same
    finite-coupling doublet twice per cavity energy: once directly, once through the cavity shift."""
    H0 = model_hamiltonian(xi, E_JT, hw, 0.0, g_c, hwc, n_max, n_ph, particle)
    Z, J, Lz, Sz, V = _model_operators(n_max, n_ph, particle, "ZLS")
    E0, g, vals, gap = _kramers_g(H0, Z, [Lz, Sz])
    return (E0, g, vals[0], vals[1], float(gap))

def kramers_g_factor(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Exact effective g-factor of the ground Kramers doublet of the full model at zero classical field:
    [E_0, g, <L_z>, <S_z>, E_2 - E_0], with <L_z>, <S_z> in the doublet state of larger Zeeman eigenvalue."""
    if int(n_ph) != n_ph or n_ph < 0: raise ValueError("n_ph must be a non-negative integer")
    return np.array(_kramers_doublet(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle), dtype=float)

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def cavity_shift(xi: float, E_JT: float, hw: float, g_c: float, hwc: float, n_max: int, n_ph: int, particle: bool) -> "np.ndarray":
    """Cavity-induced change of the exact g-factor at a finite coupling: [g(g_c) - g(0), kappa_fd, g(0)] with
    kappa_fd = (g(g_c) - g(0)) (hwc / g_c)^2 the finite-coupling estimate of the second-order coefficient."""
    if not (np.isfinite(g_c) and g_c > 0): raise ValueError("g_c must be positive")
    if int(n_ph) != n_ph or n_ph < 1: raise ValueError("n_ph must be a positive integer")
    g1 = kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, particle)[1]
    g0 = kramers_g_factor(xi, E_JT, hw, 0.0, hwc, n_max, 0, particle)[1]
    return np.array([g1 - g0, (g1 - g0) * (hwc / g_c) ** 2, g0], dtype=float)

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _apply(A, X):
    """A @ X, with A allowed to be the 1-D diagonal of a diagonal operator."""
    if A.ndim == 1: return A[:, None] * X if X.ndim > 1 else A * X
    return A @ X

def _model_operators(n_max: int, n_ph: int, particle: bool, which: str = "ZJLSV"):
    """Zeeman (per mu_B B_z), conserved J = l_vib - L_z/2, L_z, S_z and the cavity coupling operator
    i sigma_y (b^+ - b) on the full product space, ordering (m_l, m_s, vibration, photon). `which` selects
    which of Z, J, L_z, S_z, V to build; the others are returned as None. L_z, S_z and Z are diagonal in this
    ordering and are returned as their 1-D diagonals, to be applied with `_apply`; only J and V are dense."""
    ops = vibronic_operators(n_max); N = ops.shape[1]
    s = 1.0 if particle else -1.0
    nb = n_ph + 1
    blk = np.repeat(np.arange(4), N * nb)
    lz_d = np.where(blk < 2, 1.0, -1.0); sz_d = np.where(blk % 2 == 0, 0.5, -0.5)
    Lz = lz_d.astype(complex) if "L" in which else None
    Sz = sz_d.astype(complex) if "S" in which else None
    Z = (s * lz_d + _GE() * sz_d).astype(complex) if "Z" in which else None
    J = None
    if "J" in which:
        J = np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), ops[3]), np.eye(nb))
        J[np.diag_indices_from(J)] -= 0.5 * lz_d
    V = None
    if "V" in which:
        sy = np.array([[0.0, -1j], [1j, 0.0]]); bph = np.diag(np.sqrt(np.arange(1, n_ph + 1)), 1)
        V = 1j * np.kron(np.kron(np.kron(np.eye(2), sy), np.eye(N)), bph.T - bph)
    return Z, J, Lz, Sz, V

def _second_order_kappa(H0: np.ndarray, Z: np.ndarray, V: np.ndarray, hwc: float):
    """Exact second-order coefficient of the Kramers-doublet g-factor in the coupling g_c multiplying V:
    kappa = hwc^2 d^2 g / d g_c^2 / 2 ... expressed as g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4);
    also returns the second-order doublet energy shift coefficient w2 (E_0(g_c) = E_0 + w2 g_c^2 + ...).
    Z may be a dense matrix or the 1-D diagonal of a diagonal operator."""
    E, C = eigh(H0, driver="evr")
    if E[1] - E[0] > 1e-8 or E[2] - E[1] < 1e-8: raise ValueError("the lowest level is not an isolated Kramers doublet")
    E0 = E[0]; P0 = C[:, :2]; Q = C[:, 2:]; EQ = E[2:]
    if np.abs(P0.conj().T @ (V @ P0)).max() > 1e-10: raise ValueError("the coupling has matrix elements inside the doublet")
    Rd = 1.0 / (E0 - EQ)
    Vq = Q.conj().T @ (V @ P0)
    A1 = Q @ (Rd[:, None] * Vq)
    A2 = Q @ (Rd[:, None] * (Q.conj().T @ (V @ A1)))
    norm2 = -0.5 * np.sum(np.abs(Vq) ** 2 * (Rd ** 2)[:, None], axis=0)
    A2 = A2 + P0 * norm2[None, :]
    W = Vq.conj().T @ (Rd[:, None] * Vq)
    w2 = float(np.real(W[0, 0]))
    if abs(W[0, 0] - W[1, 1]) > 1e-9 * max(1.0, abs(w2)) or abs(W[0, 1]) > 1e-9 * max(1.0, abs(w2)):
        raise ValueError("the second-order doublet shift is not Kramers-degenerate")
    M0 = P0.conj().T @ _apply(Z, P0)
    M1 = P0.conj().T @ _apply(Z, A1) + A1.conj().T @ _apply(Z, P0)
    M2 = P0.conj().T @ _apply(Z, A2) + A2.conj().T @ _apply(Z, P0) + A1.conj().T @ _apply(Z, A1)
    w0, U = np.linalg.eigh(M0)
    m1 = U.conj().T @ M1 @ U; m2 = U.conj().T @ M2 @ U
    g0 = float(w0[1] - w0[0])
    c2 = float(np.real(m2[1, 1] - m2[0, 0]))
    if g0 > 1e-12: c2 += 2.0 * float(abs(m1[0, 1]) ** 2) / g0
    return g0, c2 * hwc * hwc, w2

def second_order_coefficient(xi: float, E_JT: float, hw: float, hwc: float, n_max: int, particle: bool) -> "np.ndarray":
    """Exact leading cavity correction of the g-factor of the dynamic model: [kappa, w2 hwc, g(0)] with
    g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4) and E_0(g_c) = E_0(0) + w2 g_c^2 + O(g_c^4), from second-order
    perturbation theory in the cavity coupling (one-photon intermediate states suffice)."""
    if not (np.isfinite(hwc) and hwc > 0): raise ValueError("hwc must be positive")
    H0 = model_hamiltonian(xi, E_JT, hw, 0.0, 0.0, hwc, n_max, 1, particle)
    Z, J, Lz, Sz, V = _model_operators(n_max, 1, particle, "ZV")
    g0, kappa, w2 = _second_order_kappa(H0, Z, V, hwc)
    return np.array([kappa, w2 * hwc, g0], dtype=float)

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _apply(A, X):
    """A @ X, with A allowed to be the 1-D diagonal of a diagonal operator."""
    if A.ndim == 1: return A[:, None] * X if X.ndim > 1 else A * X
    return A @ X

def _second_order_kappa(H0: np.ndarray, Z: np.ndarray, V: np.ndarray, hwc: float):
    """Exact second-order coefficient of the Kramers-doublet g-factor in the coupling g_c multiplying V:
    kappa = hwc^2 d^2 g / d g_c^2 / 2 ... expressed as g(g_c) = g(0) + kappa (g_c/hwc)^2 + O(g_c^4);
    also returns the second-order doublet energy shift coefficient w2 (E_0(g_c) = E_0 + w2 g_c^2 + ...).
    Z may be a dense matrix or the 1-D diagonal of a diagonal operator."""
    E, C = eigh(H0, driver="evr")
    if E[1] - E[0] > 1e-8 or E[2] - E[1] < 1e-8: raise ValueError("the lowest level is not an isolated Kramers doublet")
    E0 = E[0]; P0 = C[:, :2]; Q = C[:, 2:]; EQ = E[2:]
    if np.abs(P0.conj().T @ (V @ P0)).max() > 1e-10: raise ValueError("the coupling has matrix elements inside the doublet")
    Rd = 1.0 / (E0 - EQ)
    Vq = Q.conj().T @ (V @ P0)
    A1 = Q @ (Rd[:, None] * Vq)
    A2 = Q @ (Rd[:, None] * (Q.conj().T @ (V @ A1)))
    norm2 = -0.5 * np.sum(np.abs(Vq) ** 2 * (Rd ** 2)[:, None], axis=0)
    A2 = A2 + P0 * norm2[None, :]
    W = Vq.conj().T @ (Rd[:, None] * Vq)
    w2 = float(np.real(W[0, 0]))
    if abs(W[0, 0] - W[1, 1]) > 1e-9 * max(1.0, abs(w2)) or abs(W[0, 1]) > 1e-9 * max(1.0, abs(w2)):
        raise ValueError("the second-order doublet shift is not Kramers-degenerate")
    M0 = P0.conj().T @ _apply(Z, P0)
    M1 = P0.conj().T @ _apply(Z, A1) + A1.conj().T @ _apply(Z, P0)
    M2 = P0.conj().T @ _apply(Z, A2) + A2.conj().T @ _apply(Z, P0) + A1.conj().T @ _apply(Z, A1)
    w0, U = np.linalg.eigh(M0)
    m1 = U.conj().T @ M1 @ U; m2 = U.conj().T @ M2 @ U
    g0 = float(w0[1] - w0[0])
    c2 = float(np.real(m2[1, 1] - m2[0, 0]))
    if g0 > 1e-12: c2 += 2.0 * float(abs(m1[0, 1]) ** 2) / g0
    return g0, c2 * hwc * hwc, w2

def static_cavity(xi: float, F_rho: float, hwc: float, particle: bool) -> "np.ndarray":
    """The same second-order cavity coefficient for the source's static model (four states times the photon
    number 0, 1) against the source's own estimates: [kappa_static, kappa_source_weak, kappa_source_strong, g(0)]
    with kappa_source_weak = +-hwc/F_rho and kappa_source_strong = +-8 hwc F_rho^2/xi^3 (upper sign single-particle)."""
    if not (np.isfinite(xi) and xi > 0 and np.isfinite(F_rho) and F_rho > 0 and np.isfinite(hwc) and hwc > 0): raise ValueError("xi, F_rho and hwc must be positive")
    s = 1.0 if particle else -1.0
    Lz2 = np.diag([1.0, -1.0]); Sz2 = np.diag([0.5, -0.5]); I2 = np.eye(2); sy = np.array([[0.0, -1j], [1j, 0.0]])
    P1m = np.zeros((2, 2)); P1m[0, 1] = 1.0
    Hes = xi * np.kron(Lz2, Sz2) + F_rho * np.kron(P1m + P1m.T, I2)
    bph = np.diag([1.0], 1); IP = np.eye(2)
    H0 = np.kron(Hes, IP) + np.kron(np.eye(4), hwc * (bph.T @ bph))
    Z = np.kron(s * np.kron(Lz2, I2) + _GE() * np.kron(I2, Sz2), IP)
    V = 1j * np.kron(np.kron(I2, sy), bph.T - bph)
    g0, kappa, w2 = _second_order_kappa(H0.astype(complex), Z.astype(complex), V, hwc)
    return np.array([kappa, s * hwc / F_rho, s * 8.0 * hwc * F_rho * F_rho / xi ** 3, g0], dtype=float)

import numpy as np
from math import sqrt, pi
from scipy.linalg import eigh

def _GE():
    """Free-electron g-factor."""
    return 2.00231930436

def _check_pos(name, x):
    if not (np.isfinite(x) and x > 0): raise ValueError("%s must be positive" % name)

def _check_int(name, x, lo=0):
    if int(x) != x or x < lo: raise ValueError("%s must be an integer >= %d" % (name, lo))

def _model_operators(n_max: int, n_ph: int, particle: bool, which: str = "ZJLSV"):
    """Zeeman (per mu_B B_z), conserved J = l_vib - L_z/2, L_z, S_z and the cavity coupling operator
    i sigma_y (b^+ - b) on the full product space, ordering (m_l, m_s, vibration, photon). `which` selects
    which of Z, J, L_z, S_z, V to build; the others are returned as None. L_z, S_z and Z are diagonal in this
    ordering and are returned as their 1-D diagonals, to be applied with `_apply`; only J and V are dense."""
    ops = vibronic_operators(n_max); N = ops.shape[1]
    s = 1.0 if particle else -1.0
    nb = n_ph + 1
    blk = np.repeat(np.arange(4), N * nb)
    lz_d = np.where(blk < 2, 1.0, -1.0); sz_d = np.where(blk % 2 == 0, 0.5, -0.5)
    Lz = lz_d.astype(complex) if "L" in which else None
    Sz = sz_d.astype(complex) if "S" in which else None
    Z = (s * lz_d + _GE() * sz_d).astype(complex) if "Z" in which else None
    J = None
    if "J" in which:
        J = np.kron(np.kron(np.kron(np.eye(2), np.eye(2)), ops[3]), np.eye(nb))
        J[np.diag_indices_from(J)] -= 0.5 * lz_d
    V = None
    if "V" in which:
        sy = np.array([[0.0, -1j], [1j, 0.0]]); bph = np.diag(np.sqrt(np.arange(1, n_ph + 1)), 1)
        V = 1j * np.kron(np.kron(np.kron(np.eye(2), sy), np.eye(N)), bph.T - bph)
    return Z, J, Lz, Sz, V

def rjt_audit(xi: float, E_JT: float, hw: float, g_c: float, hwc_list: list, n_max: int, n_ph: int) -> "np.ndarray":
    """Orchestrate the chain. Row i >= 1 (per cavity frequency, 25 columns):
    [hwc, kappa_p, kappa_h, kappa_static_p, kappa_static_h, kappa_source_weak_p, kappa_source_weak_h,
     kappa_source_strong_p, dg_p(g_c), kappa_fd_p(g_c), <S_z>_p(g_c), w2 hwc (particle), g_p, g_h, g_static_p,
     g_static_h, g_formula_p, <L_z>_p, <L_z>_static_p, p_Ham, q_Ham, E_2 - E_0, |j_0|, D, max |[H, J]|];
    head row 0: [kappa_p for the first cavity frequency, followed by zeros]."""
    if not isinstance(hwc_list, (list, tuple, np.ndarray)) or len(hwc_list) == 0: raise ValueError("hwc_list must be non-empty")
    _check_pos("xi", xi); _check_pos("hw", hw); _check_pos("g_c", g_c)
    if not (np.isfinite(E_JT) and E_JT > 0): raise ValueError("E_JT must be positive")
    _check_int("n_max", n_max, 1); _check_int("n_ph", n_ph, 1)
    F_rho = 2.0 * E_JT
    basis = oscillator_basis(n_max)
    lad = ladder_operators(n_max)
    ops = vibronic_operators(n_max)
    if basis.shape[0] != lad.shape[1] or ops.shape[1] != lad.shape[1]: raise ValueError("basis and operator sizes disagree")
    Hfull = model_hamiltonian(xi, E_JT, hw, 0.0, g_c, float(hwc_list[0]), n_max, n_ph, True)
    Zf, Jf, Lzf, Szf, Vf = _model_operators(n_max, n_ph, True, "J")
    dim = float(Hfull.shape[0])
    jres = float(np.abs(Hfull @ Jf - Jf @ Hfull).max())
    if jres > 1e-6: raise ValueError("J = l_vib - L_z/2 is not conserved by the assembled Hamiltonian")
    stat_p = static_model(xi, F_rho, True); stat_h = static_model(xi, F_rho, False)
    gp = kramers_g_factor(xi, E_JT, hw, 0.0, float(hwc_list[0]), n_max, 0, True)
    gh = kramers_g_factor(xi, E_JT, hw, 0.0, float(hwc_list[0]), n_max, 0, False)
    spec = vibronic_spectrum(xi, E_JT, hw, n_max, True, 3)
    ham = ham_reduction(E_JT, hw, n_max)
    p_ham = ham[2]
    rows = []
    for hwc in hwc_list:
        hwc = float(hwc)
        if not (np.isfinite(hwc) and hwc > 0): raise ValueError("every cavity frequency must be positive")
        kp = second_order_coefficient(xi, E_JT, hw, hwc, n_max, True)
        kh = second_order_coefficient(xi, E_JT, hw, hwc, n_max, False)
        sp = static_cavity(xi, F_rho, hwc, True); sh = static_cavity(xi, F_rho, hwc, False)
        sh_p = cavity_shift(xi, E_JT, hw, g_c, hwc, n_max, n_ph, True)
        gc_p = kramers_g_factor(xi, E_JT, hw, g_c, hwc, n_max, n_ph, True)
        rows.append([hwc, kp[0], kh[0], sp[0], sh[0], sp[1], sh[1], sp[2], sh_p[0], sh_p[1], gc_p[3], kp[1],
                     gp[1], gh[1], stat_p[1], stat_h[1], stat_p[2], gp[2], stat_p[3], p_ham, ham[3], spec[2, 0] - spec[0, 0], spec[0, 1], dim, jres])
    head = np.zeros(len(rows[0])); head[0] = rows[0][1]
    return np.array([head] + rows, dtype=float)
SCICODE_GOLD_EOF
