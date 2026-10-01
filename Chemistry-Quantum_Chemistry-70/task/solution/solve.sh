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


def _ecg_exponents(exps):
    """Validate an (n, 3) exponent array and return it as floats."""
    e = np.asarray(exps, dtype=float)
    if e.ndim != 2 or e.shape[1] != 3 or e.shape[0] < 1:
        raise ValueError("exponents must have shape (n, 3) with n >= 1")
    if not np.all(np.isfinite(e)) or np.any(e < 0.0):
        raise ValueError("exponents must be finite and non-negative")
    if np.any(e[:, 0] * e[:, 1] + e[:, 2] * (e[:, 0] + e[:, 1]) <= 0.0):
        raise ValueError("every Gaussian must be square integrable")
    return e


def _ecg_blocks(eb, ek, kind, swap):
    """Pair quantities for bra exponents eb and ket exponents ek; swap exchanges the ket electrons."""
    if kind not in ("even", "odd"):
        raise ValueError("kind must be 'even' or 'odd'")
    if swap:
        ek = ek[:, [1, 0, 2]]
    ab, bb, cb = eb[:, 0][:, None], eb[:, 1][:, None], eb[:, 2][:, None]
    ak, bk, ck = ek[:, 0][None, :], ek[:, 1][None, :], ek[:, 2][None, :]
    a_mat = (2.0 * (ab + cb), -2.0 * cb, 2.0 * (bb + cb))
    b_mat = (2.0 * (ak + ck), -2.0 * ck, 2.0 * (bk + ck))
    c00, c01, c11 = a_mat[0] + b_mat[0], a_mat[1] + b_mat[1], a_mat[2] + b_mat[2]
    det = c00 * c11 - c01 ** 2
    sig = (c11 / det, -c01 / det, c00 / det)
    return {"A": a_mat, "B": b_mat, "S": sig, "N": (2.0 * np.pi) ** 3 * det ** -1.5,
            "dS": sig[0] * sig[2] - sig[1] ** 2}


def _odd_vectors(swap):
    """Coefficients of (r1, r2) in the z-coordinate prefactor of the bra and of the (possibly swapped) ket."""
    return (1.0, 0.0), ((0.0, 1.0) if swap else (1.0, 0.0))


def _quad(v, m, w):
    """v^T M w for a symmetric 2 x 2 array M stored as (M00, M01, M11)."""
    return v[0] * (m[0] * w[0] + m[1] * w[1]) + v[1] * (m[1] * w[0] + m[2] * w[1])


def _one_overlap(k, kind, swap):
    if kind == "even":
        return (-1.0 if swap else 1.0) * k["N"] * 2.0 * k["dS"]
    v, w = _odd_vectors(swap)
    return k["N"] * _quad(v, k["S"], w)


def ecg_overlap(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str) -> "np.ndarray":
    eb, ek = _ecg_exponents(exps_bra), _ecg_exponents(exps_ket)
    direct = _one_overlap(_ecg_blocks(eb, ek, kind, False), kind, False)
    exchanged = _one_overlap(_ecg_blocks(eb, ek, kind, True), kind, True)
    return np.asarray(2.0 * (direct - exchanged), dtype=float)

import numpy as np


def _sym_times(x, y):
    """Product X Y of two symmetric 2 x 2 arrays (00, 01, 11), returned as a general (00, 01, 10, 11) array."""
    return (x[0] * y[0] + x[1] * y[1], x[0] * y[1] + x[1] * y[2],
            x[1] * y[0] + x[2] * y[1], x[1] * y[1] + x[2] * y[2])


def _gen_times_sym(f, y):
    """Product F Y of a general (00, 01, 10, 11) array and a symmetric (00, 01, 11) array."""
    return (f[0] * y[0] + f[1] * y[1], f[0] * y[1] + f[1] * y[2],
            f[2] * y[0] + f[3] * y[1], f[2] * y[1] + f[3] * y[2])


def _gen_quad(v, f, w):
    """v^T F w for a general 2 x 2 array F stored as (00, 01, 10, 11)."""
    return v[0] * (f[0] * w[0] + f[1] * w[1]) + v[1] * (f[2] * w[0] + f[3] * w[1])


def _one_kinetic(k, kind, swap):
    a_mat, b_mat, sig = k["A"], k["B"], k["S"]
    asb = _gen_times_sym(_sym_times(a_mat, sig), b_mat)
    trace_asb = asb[0] + asb[3]
    if kind == "even":
        value = (5.0 * k["dS"] * trace_asb - k["dS"] * (a_mat[0] + a_mat[2] + b_mat[0] + b_mat[2])
                 + sig[0] + sig[2])
        return (-1.0 if swap else 1.0) * k["N"] * value
    v, w = _odd_vectors(swap)
    sa, sb, bs = _sym_times(sig, a_mat), _sym_times(sig, b_mat), _sym_times(b_mat, sig)
    sabs = _gen_times_sym(_gen_times_sym(sa, b_mat), sig)
    sbas = _gen_times_sym(_gen_times_sym(sb, a_mat), sig)
    value = (1.5 * trace_asb * _quad(v, sig, w) + 0.5 * _gen_quad(v, sabs, w) + 0.5 * _gen_quad(v, sbas, w)
             + 0.5 * (v[0] * w[0] + v[1] * w[1]) - 0.5 * _gen_quad(v, sa, w) - 0.5 * _gen_quad(v, bs, w))
    return k["N"] * value


def ecg_kinetic(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str) -> "np.ndarray":
    eb, ek = _ecg_exponents(exps_bra), _ecg_exponents(exps_ket)
    direct = _one_kinetic(_ecg_blocks(eb, ek, kind, False), kind, False)
    exchanged = _one_kinetic(_ecg_blocks(eb, ek, kind, True), kind, True)
    return np.asarray(2.0 * (direct - exchanged), dtype=float)

import numpy as np


def _one_coulomb(k, kind, swap, charge_z):
    sig = k["S"]
    total = 0.0
    for q, charge in (((1.0, 0.0), -charge_z), ((0.0, 1.0), -charge_z), ((1.0, -1.0), 1.0)):
        beta = _quad(q, sig, q)
        prefactor = charge * (2.0 / np.sqrt(np.pi)) * k["N"] / np.sqrt(2.0 * beta)
        if kind == "even":
            total = total + (-1.0 if swap else 1.0) * prefactor * (4.0 / 3.0) * k["dS"]
        else:
            v, w = _odd_vectors(swap)
            sq = (sig[0] * q[0] + sig[1] * q[1], sig[1] * q[0] + sig[2] * q[1])
            m0 = _quad(v, sig, w)
            m1 = (v[0] * sq[0] + v[1] * sq[1]) * (w[0] * sq[0] + w[1] * sq[1]) / beta
            total = total + prefactor * (m0 - m1 / 3.0)
    return total


def ecg_coulomb(exps_bra: "np.ndarray", exps_ket: "np.ndarray", kind: str, Z: float) -> "np.ndarray":
    charge_z = float(Z)
    if not charge_z > 0.0:
        raise ValueError("Z must be positive")
    eb, ek = _ecg_exponents(exps_bra), _ecg_exponents(exps_ket)
    direct = _one_coulomb(_ecg_blocks(eb, ek, kind, False), kind, False, charge_z)
    exchanged = _one_coulomb(_ecg_blocks(eb, ek, kind, True), kind, True, charge_z)
    return np.asarray(2.0 * (direct - exchanged), dtype=float)

import numpy as np


def _one_dipole(k, swap, gauge):
    _, w = _odd_vectors(swap)
    b00, b01, b11 = k["B"]
    if gauge == "length":
        moment = -(w[0] - w[1]) * k["dS"]
    else:
        moment = k["dS"] * (-b00 * w[1] + b01 * w[0] - b01 * w[1] + b11 * w[0])
    return k["N"] * moment


def ecg_dipole(exps_even: "np.ndarray", exps_odd: "np.ndarray", gauge: str) -> "np.ndarray":
    if gauge not in ("length", "velocity"):
        raise ValueError("gauge must be 'length' or 'velocity'")
    ee, eo = _ecg_exponents(exps_even), _ecg_exponents(exps_odd)
    direct = _one_dipole(_ecg_blocks(ee, eo, "odd", False), False, gauge)
    exchanged = _one_dipole(_ecg_blocks(ee, eo, "odd", True), True, gauge)
    return np.asarray(2.0 * (direct - exchanged), dtype=float)

import numpy as np
import scipy.linalg as spl


def rotated_levels(exps: "np.ndarray", kind: str, Z: float, theta: float, cut: float) -> "np.ndarray":
    charge = float(Z)
    angle = float(theta)
    drop = float(cut)
    if not charge > 0.0:
        raise ValueError("Z must be positive")
    if not np.isfinite(angle) or not 0.0 <= angle < 0.25 * np.pi:
        raise ValueError("theta must lie in [0, pi/4)")
    if not 0.0 < drop < 1.0:
        raise ValueError("cut must lie strictly between 0 and 1")
    overlap = ecg_overlap(exps, exps, kind)
    hamiltonian = (np.exp(-2.0j * angle) * ecg_kinetic(exps, exps, kind)
                   + np.exp(-1.0j * angle) * ecg_coulomb(exps, exps, kind, charge))
    s_val, s_vec = np.linalg.eigh(0.5 * (overlap + overlap.T))
    keep = s_val > drop * s_val.max()
    proj = s_vec[:, keep] / np.sqrt(s_val[keep])
    small = proj.T @ (0.5 * (hamiltonian + hamiltonian.T)) @ proj
    values, vectors = spl.eig(0.5 * (small + small.T))
    vectors = vectors / np.sqrt(np.sum(vectors * vectors, axis=0))
    order = np.lexsort((values.imag, values.real))
    values, vectors = values[order], vectors[:, order]
    coeff = proj @ vectors
    lead = np.argmax(np.abs(coeff), axis=0)
    pivot = coeff[lead, np.arange(coeff.shape[1])]
    flip = np.where(pivot.real != 0.0, np.sign(pivot.real), np.sign(pivot.imag))
    flip = np.where(flip == 0.0, 1.0, flip)
    coeff = coeff * flip
    return np.column_stack([values, coeff.T]).astype(complex)

import numpy as np


def bound_line_rates(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, cut: float) -> "np.ndarray":
    fine_structure = 7.2973525693e-3
    au_time_s = 2.4188843265857e-17
    even = rotated_levels(exps_even, "even", Z, 0.0, cut)
    odd = rotated_levels(exps_odd, "odd", Z, 0.0, cut)
    e_init = even[0, 0].real
    c_init = even[0, 1:].real
    e_final = odd[:, 0].real
    c_final = odd[:, 1:].real.T
    coupling = c_init @ ecg_dipole(exps_even, exps_odd, "length") @ c_final
    omega = e_init - e_final
    below = (e_final < -0.5 * float(Z) ** 2) & (omega > 0.0)
    rate = ((4.0 / 3.0) * fine_structure ** 3 * omega[below] ** 3
            * 2.0 * coupling[below] ** 2 / au_time_s)
    return np.column_stack([e_final[below], omega[below], rate]).astype(float)

import numpy as np


def dissociation_spectrum(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, theta: float, cut: float,
                                  energies: "np.ndarray") -> "np.ndarray":
    fine_structure = 7.2973525693e-3
    au_time_s = 2.4188843265857e-17
    grid = np.asarray(energies, dtype=float)
    if grid.ndim != 1 or grid.size < 1 or not np.all(np.isfinite(grid)):
        raise ValueError("energies must be a one-dimensional array of finite values")
    if not float(theta) > 0.0:
        raise ValueError("theta must be strictly positive")
    even = rotated_levels(exps_even, "even", Z, theta, cut)
    odd = rotated_levels(exps_odd, "odd", Z, theta, cut)
    e_init = even[0, 0]
    coupling = (even[0, 1:] @ (np.exp(1.0j * float(theta)) * ecg_dipole(exps_even, exps_odd, "length"))
                @ odd[:, 1:].T)
    weight = coupling ** 2
    values = odd[:, 0]
    dens = np.array([np.sum(weight / (values - e)).imag for e in grid])
    omega = e_init.real - grid
    spec = (4.0 / 3.0) * fine_structure ** 3 * omega ** 3 * (2.0 / np.pi) * dens / au_time_s
    return np.asarray(spec, dtype=float)

import numpy as np


def shake_fraction(exps_even: "np.ndarray", exps_odd: "np.ndarray", Z: float, theta: float, cut: float,
                           n_grid: int) -> float:
    points = int(n_grid)
    if points < 2:
        raise ValueError("n_grid must be at least 2")
    lines = bound_line_rates(exps_even, exps_odd, Z, cut)
    if lines.shape[0] == 0:
        raise ValueError("no bound odd-parity level lies below the initial state")
    threshold = -0.5 * float(Z) ** 2
    e_init = lines[0, 0] + lines[0, 1]
    grid = np.linspace(threshold, e_init, points)
    spec = dissociation_spectrum(exps_even, exps_odd, Z, theta, cut, grid)
    continuum = float(np.trapezoid(np.asarray(spec, dtype=float), grid))
    total = float(np.sum(lines[:, 2])) + continuum
    return float(1.0 - lines[0, 2] / total)
SCICODE_GOLD_EOF
