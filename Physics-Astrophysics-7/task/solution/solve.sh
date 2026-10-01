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


def exact_hubble(psi: 'np.ndarray', rho_other: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    psi = np.asarray(psi, dtype=complex)
    rho_other = np.asarray(rho_other, dtype=float)
    m = float(m); m_pl = float(m_pl)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    # First Friedmann equation with the exact axion density of Eq (21).
    tot = m * np.abs(psi) ** 2 + rho_other
    if np.any(tot < 0.0):
        raise ValueError("total background density is negative")
    return np.sqrt(tot / (3.0 * m_pl ** 2))

import numpy as np


def slow_hubble(psi_s: 'np.ndarray', rho_other_s: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    psi_s = np.asarray(psi_s, dtype=complex)
    rho_other_s = np.asarray(rho_other_s, dtype=float)
    m = float(m); m_pl = float(m_pl)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    a2 = np.abs(psi_s) ** 2
    # Eq (9).  The slow-mode Friedmann is NOT the exact one with slow variables
    # substituted: it carries a relativistic correction of its own.
    tot = (m * a2 + rho_other_s
           + (3.0 * a2 / (32.0 * m * m_pl ** 2)) * (m * a2 + 2.0 * rho_other_s))
    if np.any(tot < 0.0):
        raise ValueError("total slow-mode density is negative")
    return np.sqrt(tot / (3.0 * m_pl ** 2))

import numpy as np


def exact_rate(psi: 'np.ndarray', hubble: 'np.ndarray', m: float, t: 'np.ndarray') -> 'np.ndarray':
    psi = np.asarray(psi, dtype=complex)
    hubble = np.asarray(hubble, dtype=float)
    m = float(m); t = float(t) if isinstance(t, (int, float)) else np.asarray(t, dtype=float)
    if not (m > 0.0):
        raise ValueError("m must be strictly positive")
    # Eq (4).  Both terms carry 3/2; the second is conjugated and phase shifted.
    return -1.5 * hubble * psi + 1.5 * hubble * np.conj(psi) * np.exp(2j * m * t)

import numpy as np


def slow_rate(psi_s: 'np.ndarray', hubble_s: 'np.ndarray', rho_other_s: 'np.ndarray', p_other_s: 'np.ndarray', m: float, m_pl: float) -> 'np.ndarray':
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = np.asarray(hubble_s, dtype=float)
    rho_other_s = np.asarray(rho_other_s, dtype=float)
    p_other_s = np.asarray(p_other_s, dtype=float)
    m = float(m); m_pl = float(m_pl)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    a2 = np.abs(psi_s) ** 2
    # Eq (7): a damping term, a purely imaginary term, and a real damping correction.
    term1 = -1.5 * hubble_s * psi_s
    term2 = (3j / (16.0 * m * m_pl ** 2)) * psi_s * (3.0 * m * a2 + 2.0 * rho_other_s)
    term3 = -(9.0 / (32.0 * m ** 2 * m_pl ** 2)) * hubble_s * psi_s * (
        m * a2 + rho_other_s + p_other_s)
    return term1 + term2 + term3

import numpy as np


def matching_coefficients(
    psi_s: "np.ndarray",
    hubble_s: "np.ndarray",
    rho_other_s: "np.ndarray",
    p_other_s: "np.ndarray",
    m: float,
    m_pl: float,
    t: "np.ndarray",
) -> "np.ndarray":
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = np.asarray(hubble_s, dtype=float)
    rho_other_s = np.asarray(rho_other_s, dtype=float)
    p_other_s = np.asarray(p_other_s, dtype=float)
    m = float(m)
    m_pl = float(m_pl)
    t = float(t) if isinstance(t, (int, float)) else np.asarray(t, dtype=float)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    R = np.real(psi_s)
    imag_part = np.imag(psi_s)
    c2 = np.cos(2.0 * m * t)
    s2 = np.sin(2.0 * m * t)
    c4 = np.cos(4.0 * m * t)
    s4 = np.sin(4.0 * m * t)
    # Eq (38).
    a_sin = (3.0 * hubble_s / (4.0 * m)) * s2
    a_cos = (3.0 * hubble_s / (4.0 * m)) * c2
    w = rho_other_s + p_other_s
    # Eq (39).
    c_re = (3.0 / (64.0 * m * m_pl**2)) * (
        c4 * R**3
        + (4.0 * s2 - s4) * imag_part**3
        - (4.0 * s2 - 3.0 * s4) * R**2 * imag_part
        + (8.0 * c2 - 3.0 * c4) * R * imag_part**2
    ) + (3.0 / (16.0 * m**2 * m_pl**2)) * w * (c2 * R + s2 * imag_part)
    # Eq (40).
    c_im = (3.0 / (64.0 * m * m_pl**2)) * (
        (4.0 * s2 + s4) * R**3
        + c4 * imag_part**3
        - (8.0 * c2 + 3.0 * c4) * R**2 * imag_part
        - (4.0 * s2 + 3.0 * s4) * R * imag_part**2
    ) + (3.0 / (16.0 * m**2 * m_pl**2)) * w * (s2 * R - c2 * imag_part)
    return np.stack(np.broadcast_arrays(a_sin, a_cos, c_re, c_im), axis=0)

import numpy as np


def solve_matching_system(psi: complex, coeffs: 'np.ndarray') -> 'np.ndarray':
    psi = complex(np.asarray(psi, dtype=complex))
    c = np.asarray(coeffs, dtype=float)
    if c.shape != (4,):
        raise ValueError("coeffs must hold exactly four entries")
    a_sin, a_cos, c_re, c_im = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
    # Eq (41).  The matrix is symmetric but not diagonal: the two diagonal entries differ
    # in the sign they give the first coefficient, while both off-diagonal entries are
    # the negative of the second.
    mat = np.array([[1.0 + a_sin, -a_cos], [-a_cos, 1.0 - a_sin]], dtype=float)
    rhs = np.array([psi.real + c_re, psi.imag + c_im], dtype=float)
    if abs(np.linalg.det(mat)) < 1e-300:
        raise ValueError("the matching system is singular")
    sol = np.linalg.solve(mat, rhs)
    return np.array([float(sol[0]), float(sol[1])], dtype=float)

import numpy as np


def exact_perturbation_rate(
    dpsi: "np.ndarray",
    psi: "np.ndarray",
    hubble: "np.ndarray",
    hdot: "np.ndarray",
    a: "np.ndarray",
    m: float,
    k: float,
    t: "np.ndarray",
) -> "np.ndarray":
    dpsi = np.asarray(dpsi, dtype=complex)
    psi = np.asarray(psi, dtype=complex)
    hubble = np.asarray(hubble, dtype=float)
    hdot = np.asarray(hdot, dtype=float)
    a = np.asarray(a, dtype=float)
    m = float(m)
    k = float(k)
    t = float(t) if isinstance(t, (int, float)) else np.asarray(t, dtype=float)
    if not (m > 0.0):
        raise ValueError("m must be strictly positive")
    if np.any(a <= 0.0):
        raise ValueError("a must be strictly positive")
    # The Klein-Gordon equation (A2) fixes a negative metric term
    # outside the oscillatory bracket. The printed Eq (5) has a
    # sign error there; the conjugate metric term stays positive.
    e2 = np.exp(2j * m * t)
    q = 1j * k**2 / (2.0 * m * a**2)
    return (
        -(1.5 * hubble + q) * dpsi
        - 0.25 * psi * hdot
        + e2
        * ((1.5 * hubble - q) * np.conj(dpsi) + 0.25 * np.conj(psi) * hdot)
    )

import numpy as np


def slow_perturbation_rate(
    dpsi_s: "np.ndarray",
    psi_s: "np.ndarray",
    hubble_s: "np.ndarray",
    hdot_s: "np.ndarray",
    a_s: "np.ndarray",
    m: float,
    m_pl: float,
    k: float,
) -> "np.ndarray":
    dpsi_s = np.asarray(dpsi_s, dtype=complex)
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = np.asarray(hubble_s, dtype=float)
    hdot_s = np.asarray(hdot_s, dtype=float)
    a_s = np.asarray(a_s, dtype=float)
    m = float(m)
    m_pl = float(m_pl)
    k = float(k)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    if np.any(a_s <= 0.0):
        raise ValueError("a_s must be strictly positive")
    # Use the unstarred background square from the EFT mode derivation.
    # The printed Eq (8) conjugates it in error; Eq (17) fixes H_{-2}.
    return (
        -(1.5 * hubble_s + 1j * k**2 / (2.0 * m * a_s**2)) * dpsi_s
        - 0.25 * hdot_s * psi_s
        + (3j * hubble_s / (8.0 * m) + k**2 / (16.0 * m**2 * a_s**2))
        * psi_s
        * hdot_s
        + (3j * psi_s ** 2 / (16.0 * m_pl**2)) * np.conj(dpsi_s)
        + (
            9j * hubble_s**2 / (8.0 * m)
            + 3j * np.abs(psi_s) ** 2 / (8.0 * m_pl**2)
            + 1j * k**4 / (8.0 * m**3 * a_s**4)
        )
        * dpsi_s
    )

import numpy as np


def metric_slow_rates(psi_s: 'np.ndarray', dpsi_s: 'np.ndarray', eta_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, m_pl: float, k: float) -> 'np.ndarray':
    psi_s = np.asarray(psi_s, dtype=complex)
    dpsi_s = np.asarray(dpsi_s, dtype=complex)
    eta_s = np.asarray(eta_s, dtype=float)
    hubble_s = np.asarray(hubble_s, dtype=float)
    a_s = np.asarray(a_s, dtype=float)
    m = float(m); m_pl = float(m_pl); k = float(k)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    if np.any(a_s <= 0.0) or np.any(hubble_s <= 0.0):
        raise ValueError("a_s and hubble_s must be strictly positive")
    plus = np.conj(psi_s) * dpsi_s + psi_s * np.conj(dpsi_s)
    minus = np.conj(psi_s) * dpsi_s - psi_s * np.conj(dpsi_s)
    # Eq (10).  The second entry needs the first, so the order is forced.
    hdot_s = ((2.0 * m_pl ** 2 * k ** 2 / (a_s ** 2 * hubble_s)) * eta_s
              + (m / hubble_s) * np.real(plus)) / m_pl ** 2
    # Eq (11).
    etadot_s = ((1j * m / 4.0) * minus
                - (1.0 / 16.0) * np.abs(psi_s) ** 2 * hdot_s
                - (3.0 / 8.0) * hubble_s * plus
                - (1j * k ** 2 / (16.0 * m * a_s ** 2)) * minus) / m_pl ** 2
    return np.stack(np.broadcast_arrays(np.real(hdot_s), np.real(etadot_s)), axis=0)

import numpy as np


def perturbation_matching(
    psi_s: "np.ndarray",
    hubble_s: "np.ndarray",
    a_s: "np.ndarray",
    m: float,
    m_pl: float,
    k: float,
    t: float,
    dpsi: complex,
    hdot: float,
) -> "np.ndarray":
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = float(np.asarray(hubble_s, dtype=float))
    a_s = float(np.asarray(a_s, dtype=float))
    m = float(m)
    m_pl = float(m_pl)
    k = float(k)
    t = float(t)
    dpsi = complex(dpsi)
    hdot = float(hdot)
    if not (m > 0.0 and m_pl > 0.0):
        raise ValueError("m and m_pl must be strictly positive")
    if not (a_s > 0.0):
        raise ValueError("a_s must be strictly positive")
    R = float(np.real(psi_s))
    imag_part = float(np.imag(psi_s))
    c2 = np.cos(2.0 * m * t)
    s2 = np.sin(2.0 * m * t)
    # Eqs (46)-(49), (51), (52).
    b_plus = (3.0 * hubble_s / (4.0 * m)) * c2 + (
        k**2 / (4.0 * m**2 * a_s**2)
    ) * s2
    b_minus = (3.0 * hubble_s / (4.0 * m)) * s2 - (
        k**2 / (4.0 * m**2 * a_s**2)
    ) * c2
    d_plus = (1.0 / (8.0 * m)) * (c2 * R + s2 * imag_part)
    d_minus = (1.0 / (8.0 * m)) * (c2 * imag_part - s2 * R)
    e_plus = (3.0 / m_pl**2) * (imag_part * s2 + R * c2)
    e_minus = (3.0 / m_pl**2) * (imag_part * c2 - R * s2)
    # Eq (53).
    mat = np.array(
        [
            [1.0 + b_minus, -b_plus, -d_minus],
            [-b_plus, 1.0 - b_minus, -d_plus],
            [-e_minus, -e_plus, 1.0],
        ],
        dtype=float,
    )
    if abs(np.linalg.det(mat)) < 1e-14:
        raise ValueError("perturbation matching system is singular")
    rhs = np.array([dpsi.real, dpsi.imag, hdot], dtype=float)
    return np.linalg.solve(mat, rhs)

import numpy as np


def reconstruct_perturbation(dpsi_s: 'np.ndarray', psi_s: 'np.ndarray', hubble_s: 'np.ndarray', hdot_s: 'np.ndarray', a_s: 'np.ndarray', m: float, k: float, t: 'np.ndarray') -> 'np.ndarray':
    dpsi_s = np.asarray(dpsi_s, dtype=complex)
    psi_s = np.asarray(psi_s, dtype=complex)
    hubble_s = np.asarray(hubble_s, dtype=float)
    hdot_s = np.asarray(hdot_s, dtype=float)
    a_s = np.asarray(a_s, dtype=float)
    m = float(m); k = float(k)
    t = float(t) if isinstance(t, (int, float)) else np.asarray(t, dtype=float)
    if not (m > 0.0):
        raise ValueError("m must be strictly positive")
    if np.any(a_s <= 0.0):
        raise ValueError("a_s must be strictly positive")
    # Eq (16).  One correction only, unlike the four the background carries.
    e2 = np.exp(2j * m * t)
    return dpsi_s - ((3j * hubble_s / (4.0 * m) + k ** 2 / (4.0 * m ** 2 * a_s ** 2))
                     * np.conj(dpsi_s) + (1j * hdot_s / (8.0 * m)) * np.conj(psi_s)) * e2

import numpy as np


def slow_perturbation_observables(psi_s: 'np.ndarray', dpsi_s: 'np.ndarray', hdot_s: 'np.ndarray', hubble_s: 'np.ndarray', a_s: 'np.ndarray', m: float, k: float) -> 'np.ndarray':
    psi_s = np.asarray(psi_s, dtype=complex)
    dpsi_s = np.asarray(dpsi_s, dtype=complex)
    hdot_s = np.asarray(hdot_s, dtype=float)
    hubble_s = np.asarray(hubble_s, dtype=float)
    a_s = np.asarray(a_s, dtype=float)
    m = float(m); k = float(k)
    if not (m > 0.0):
        raise ValueError("m must be strictly positive")
    if np.any(a_s <= 0.0):
        raise ValueError("a_s must be strictly positive")
    plus = np.conj(psi_s) * dpsi_s + psi_s * np.conj(dpsi_s)
    minus = np.conj(psi_s) * dpsi_s - psi_s * np.conj(dpsi_s)
    # Eq (28).
    drho = m * np.real(plus)
    # Eq (29).  The slow pressure perturbation is NOT the slow part of Eq (24):
    # it is the same combination carrying the gradient factor instead.
    dp = (k ** 2 / (4.0 * m * a_s ** 2)) * np.real(plus)
    # Eq (30).
    du = np.real(-0.5j * minus + (3.0 * hubble_s / (4.0 * m)) * plus
                 + (1j * k ** 2 / (8.0 * m ** 2 * a_s ** 2)) * minus
                 + (1.0 / (8.0 * m)) * np.abs(psi_s) ** 2 * hdot_s)
    return np.stack(np.broadcast_arrays(drho, dp, du), axis=0)

import numpy as np


def axion_reconstruction_audit(
    refine: int, psi_init: complex = 0.312 + 0.0j
) -> "np.ndarray":
    refine = int(refine)
    if refine < 1:
        raise ValueError("refine must be a positive integer")
    # Evaluation configuration: GIVEN in the prompt.
    m = 1.0
    m_pl = 1.0
    rho_m0 = 0.126
    rho_r0 = 0.0432
    k = 0.5
    psi_i = complex(psi_init)
    dpsi_i = 1.0e-3 + 0.0j
    eta_i = 1.0e-3
    a_i = 1.0
    t_i, t_star, t_f = 0.0, 5.0, 65.0
    n_iter = 4
    n_exact = 26000 * refine
    n_slow = 2600 * refine
    n_sample = 400

    def _other(a):
        rho = rho_m0 * a**-3 + rho_r0 * a**-4
        p = rho_r0 * a**-4 / 3.0
        return rho, p

    def _exact_hdot(psi, dpsi, a, eta, h_rate):
        # 00 Einstein equation with the axion density perturbation of Eq (23).
        drho = float(np.real(m * (np.conj(psi) * dpsi + psi * np.conj(dpsi))))
        return (
            (2.0 * m_pl**2 * k**2 / (a**2 * h_rate)) * eta + drho / h_rate
        ) / m_pl**2

    def _exact_etadot(psi, dpsi, t):
        # 0i Einstein equation with the axion velocity perturbation of Eq (25).
        e = np.exp(-2j * m * t)
        du = 0.5j * (
            psi * np.conj(dpsi)
            - np.conj(psi) * dpsi
            + psi * dpsi * e
            - np.conj(psi) * np.conj(dpsi) / e
        )
        return float(np.real(-(m / 2.0) * du)) / m_pl**2

    def _exact_deriv(state, t):
        psi, dpsi = state[0], state[1]
        a, eta = state[2].real, state[4].real
        rho, _ = _other(a)
        h_rate = float(exact_hubble(psi, rho, m, m_pl))
        hd = _exact_hdot(psi, dpsi, a, eta, h_rate)
        ed = _exact_etadot(psi, dpsi, t)
        return np.array(
            [
                exact_rate(psi, h_rate, m, t),
                exact_perturbation_rate(
                    dpsi, psi, h_rate, hd, a, m, k, t
                ),
                complex(a * h_rate, 0.0),
                complex(hd, 0.0),
                complex(ed, 0.0),
            ]
        )

    def _slow_deriv(state, t):
        psi_s, dpsi_s = state[0], state[1]
        a_s, eta_s = state[2].real, state[4].real
        rho, p = _other(a_s)
        h_s = float(slow_hubble(psi_s, rho, m, m_pl))
        rates = metric_slow_rates(
            psi_s, dpsi_s, eta_s, h_s, a_s, m, m_pl, k
        )
        hd = float(rates[0])
        ed = float(rates[1])
        return np.array(
            [
                slow_rate(psi_s, h_s, rho, p, m, m_pl),
                slow_perturbation_rate(
                    dpsi_s, psi_s, h_s, hd, a_s, m, m_pl, k
                ),
                complex(a_s * h_s, 0.0),
                complex(hd, 0.0),
                complex(ed, 0.0),
            ]
        )

    def _rk4(deriv, state, t0, t1, n):
        h = (t1 - t0) / n
        t = t0
        for _ in range(n):
            k1 = deriv(state, t)
            k2 = deriv(state + 0.5 * h * k1, t + 0.5 * h)
            k3 = deriv(state + 0.5 * h * k2, t + 0.5 * h)
            k4 = deriv(state + h * k3, t + h)
            state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += h
        return state

    # Step 2: exact evolution of background and perturbations up to the
    # transition.
    st = np.array(
        [psi_i, dpsi_i, complex(a_i, 0.0), 0.0 + 0.0j, complex(eta_i, 0.0)]
    )
    n_to_star = int(round(n_exact * (t_star - t_i) / (t_f - t_i)))
    st_star = _rk4(_exact_deriv, st, t_i, t_star, n_to_star)

    a_st = st_star[2].real
    rho_st, p_st = _other(a_st)

    # Step 3a: background matching, iterated because the coefficients depend on
    # the
    # unknown through the slow-mode expansion rate.
    psi_s_star = complex(st_star[0])
    for _ in range(n_iter):
        h_guess = float(slow_hubble(psi_s_star, rho_st, m, m_pl))
        co = matching_coefficients(
            psi_s_star, h_guess, rho_st, p_st, m, m_pl, t_star
        )
        ms = solve_matching_system(st_star[0], co)
        psi_s_star = complex(ms[0], ms[1])

    # Step 3b: perturbation matching.  Its coefficients use the background slow
    # mode
    # just found, and it solves for the slow metric rate alongside the
    # perturbation.
    h_s_star = float(slow_hubble(psi_s_star, rho_st, m, m_pl))
    h_rate_ex = float(exact_hubble(st_star[0], rho_st, m, m_pl))
    hdot_ex = _exact_hdot(
        st_star[0], st_star[1], a_st, st_star[4].real, h_rate_ex
    )
    pm = perturbation_matching(
        psi_s_star,
        h_s_star,
        a_st,
        m,
        m_pl,
        k,
        t_star,
        complex(st_star[1]),
        hdot_ex,
    )
    dpsi_s_star = complex(pm[0], pm[1])
    hdot_s_star = float(pm[2])

    # Eq (19) inverted for the slow metric potential at the transition.
    e2 = np.exp(2j * m * t_star)
    eta_s_star = st_star[4].real - float(
        np.real(
            (
                np.conj(psi_s_star) * np.conj(dpsi_s_star) * e2
                + psi_s_star * dpsi_s_star / e2
            )
            / (8.0 * m * m_pl**2)
        )
    )

    # Step 4: EFT evolution to the end time, sampled, with Step 5
    # reconstruction.
    sst = np.array(
        [
            psi_s_star,
            dpsi_s_star,
            complex(a_st, 0.0),
            0.0 + 0.0j,
            complex(eta_s_star, 0.0),
        ]
    )
    est = st_star.copy()
    dt = (t_f - t_star) / n_sample
    n_e = max(1, int(round(n_exact * dt / (t_f - t_i))))
    n_s = max(1, int(round(n_slow * dt / (t_f - t_star))))
    errs = []
    t = t_star
    for _ in range(n_sample):
        est = _rk4(_exact_deriv, est, t, t + dt, n_e)
        sst = _rk4(_slow_deriv, sst, t, t + dt, n_s)
        t += dt
        a_s = sst[2].real
        rho, p = _other(a_s)
        h_s = float(slow_hubble(sst[0], rho, m, m_pl))
        rates = metric_slow_rates(
            sst[0], sst[1], sst[4].real, h_s, a_s, m, m_pl, k
        )
        rec = reconstruct_perturbation(
            sst[1], sst[0], h_s, float(rates[0]), a_s, m, k, t
        )
        errs.append(abs(complex(rec) - complex(est[1])))
    errs = np.asarray(errs, dtype=float)

    a_sf = sst[2].real
    rho_f, p_f = _other(a_sf)
    h_s_f = float(slow_hubble(sst[0], rho_f, m, m_pl))
    rates_f = metric_slow_rates(
        sst[0], sst[1], sst[4].real, h_s_f, a_sf, m, m_pl, k
    )
    obs_f = slow_perturbation_observables(
        sst[0], sst[1], float(rates_f[0]), h_s_f, a_sf, m, k
    )
    amp2_f = float(np.abs(sst[0]) ** 2)
    rho_a_s_f = m * amp2_f + 3.0 * (m * amp2_f + rho_f) * amp2_f / (
        16.0 * m * m_pl**2
    )
    delta_a_f = float(obs_f[0]) / rho_a_s_f

    return np.array(
        [
            float(errs.max()),
            float(errs.mean()),
            float(abs(dpsi_s_star)),
            hdot_s_star,
            float(abs(psi_s_star)),
            a_sf,
            delta_a_f,
            float(sst[4].real),
            h_s_f,
        ],
        dtype=float,
    )
SCICODE_GOLD_EOF
