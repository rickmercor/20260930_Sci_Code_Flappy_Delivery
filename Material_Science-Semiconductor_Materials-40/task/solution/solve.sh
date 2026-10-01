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
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _bessel_zeros(l, nmax):
    """first nmax positive zeros of the spherical Bessel function j_l, each to 1e-14"""
    _f = lambda x: spherical_jn(l, x)
    zs = []; xa = l + 1e-9; fa = _f(xa); step = 0.1
    while len(zs) < nmax:
        xb = xa + step; fb = _f(xb)
        if fa * fb < 0: zs.append(brentq(_f, xa, xb, xtol=1e-15, rtol=1e-15, maxiter=200))
        xa, fa = xb, fb
    return np.array(zs)

def box_spectrum(A: float, N_n: int, L_max: int) -> "np.ndarray":
    if not (A > 0) or not (isinstance(N_n, (int, np.integer)) and N_n >= 1) or not (isinstance(L_max, (int, np.integer)) and L_max >= 0):
        raise ValueError("A > 0, N_n >= 1 and L_max >= 0 required")
    Zt = np.zeros((L_max + 1, N_n))
    for l in range(L_max + 1): Zt[l] = _bessel_zeros(l, N_n)
    return Zt

import numpy as np
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _bessel_zeros(l, nmax):
    """first nmax positive zeros of the spherical Bessel function j_l, each to 1e-14"""
    _f = lambda x: spherical_jn(l, x)
    zs = []; xa = l + 1e-9; fa = _f(xa); step = 0.1
    while len(zs) < nmax:
        xb = xa + step; fb = _f(xb)
        if fa * fb < 0: zs.append(brentq(_f, xa, xb, xtol=1e-15, rtol=1e-15, maxiter=200))
        xa, fa = xb, fb
    return np.array(zs)

def _simpson_weights(r):
    n = len(r); h = r[1] - r[0]
    w = np.ones(n); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    return w * h / 3.0

def _radial_functions(A, N_n, L_max, N_r):
    """normalised radial box functions R_{nl}(r) = N j_l(z_{ln} r / A) on the Simpson grid of N_r points"""
    if not (isinstance(N_r, (int, np.integer)) and N_r >= 201 and N_r % 2 == 1): raise ValueError("N_r must be an odd integer >= 201")
    r = np.linspace(0.0, A, N_r); w = _simpson_weights(r)
    R = {}; Z = {}
    for l in range(L_max + 1):
        z = _bessel_zeros(l, N_n)
        for n in range(1, N_n + 1):
            f = spherical_jn(l, z[n - 1] * r / A)
            R[(n, l)] = f / sqrt(np.sum(w * f * f * r * r))
            Z[(n, l)] = z[n - 1]
    return r, w, R, Z

def _poisson_potentials(F, r, w, k):
    """Phi_k(r1) = int F(r2) r_<^k / r_>^(k+1) dr2 for every row of F (rows already carry the r2^2 weight)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = cumulative_simpson(F * r ** k, x=r, initial=0)
        rinv = np.where(r > 0, r ** (-(k + 1.0)), 0.0)
        outer_tot = cumulative_simpson(F * rinv, x=r, initial=0)
        outer = outer_tot[:, -1:] - outer_tot
        Phi = rinv * inner + r ** k * outer
    Phi[:, 0] = outer_tot[:, -1] if k == 0 else 0.0
    return Phi

def coulomb_integral(A: float, k: int, ne2: int, le2: int, ne1: int, le1: int, nh2: int, lh2: int, nh1: int, lh1: int, N_r: int) -> float:
    if not (A > 0) or k < 0: raise ValueError("A > 0 and k >= 0 required")
    if min(ne1, ne2, nh1, nh2) < 1 or min(le1, le2, lh1, lh2) < 0: raise ValueError("radial quantum numbers n >= 1 and l >= 0 required")
    N_n = max(ne1, ne2, nh1, nh2); L_max = max(le1, le2, lh1, lh2)
    r, w, R, Z = _radial_functions(A, N_n, L_max, N_r)
    Fe = (R[(ne2, le2)] * R[(ne1, le1)] * r * r)[None, :]
    Fh = (R[(nh2, lh2)] * R[(nh1, lh1)] * r * r)[None, :]
    Phi = _poisson_potentials(Fh, r, w, k)
    return float(np.sum(w * Fe[0] * Phi[0]))

import numpy as np
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _HB():
    """hbar^2/(2 m_0) in eV nm^2"""
    return 0.03809985

def _E2():
    """e^2/(4 pi eps_0) in eV nm"""
    return 1.439964

def _bessel_zeros(l, nmax):
    """first nmax positive zeros of the spherical Bessel function j_l, each to 1e-14"""
    _f = lambda x: spherical_jn(l, x)
    zs = []; xa = l + 1e-9; fa = _f(xa); step = 0.1
    while len(zs) < nmax:
        xb = xa + step; fb = _f(xb)
        if fa * fb < 0: zs.append(brentq(_f, xa, xb, xtol=1e-15, rtol=1e-15, maxiter=200))
        xa, fa = xb, fb
    return np.array(zs)

def _simpson_weights(r):
    n = len(r); h = r[1] - r[0]
    w = np.ones(n); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    return w * h / 3.0

def _radial_functions(A, N_n, L_max, N_r):
    """normalised radial box functions R_{nl}(r) = N j_l(z_{ln} r / A) on the Simpson grid of N_r points"""
    if not (isinstance(N_r, (int, np.integer)) and N_r >= 201 and N_r % 2 == 1): raise ValueError("N_r must be an odd integer >= 201")
    r = np.linspace(0.0, A, N_r); w = _simpson_weights(r)
    R = {}; Z = {}
    for l in range(L_max + 1):
        z = _bessel_zeros(l, N_n)
        for n in range(1, N_n + 1):
            f = spherical_jn(l, z[n - 1] * r / A)
            R[(n, l)] = f / sqrt(np.sum(w * f * f * r * r))
            Z[(n, l)] = z[n - 1]
    return r, w, R, Z

def _cg(j1, m1, j2, m2, J, M, _memo={}):
    """Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M> (Racah formula, integer arguments)"""
    if m1 + m2 != M or J < abs(j1 - j2) or J > j1 + j2 or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J: return 0.0
    key = (j1, m1, j2, m2, J, M)
    if key in _memo: return _memo[key]
    pre = (2 * J + 1) * factorial(J + j1 - j2) * factorial(J - j1 + j2) * factorial(j1 + j2 - J) / factorial(j1 + j2 + J + 1)
    pre *= factorial(J + M) * factorial(J - M) * factorial(j1 - m1) * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2)
    s = 0.0
    for k in range(0, j1 + j2 + J + 2):
        d = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if min(d) < 0: continue
        s += (-1) ** k / np.prod([float(factorial(x)) for x in d])
    _memo[key] = sqrt(pre) * s
    return _memo[key]

def _ylm_element(l2, m2, k, q, l1, m1):
    """<l2 m2| Y_kq |l1 m1> = sqrt((2 l1 + 1)(2 k + 1)/(4 pi (2 l2 + 1))) <l1 0 k 0|l2 0> <l1 m1 k q|l2 m2>"""
    if m2 != m1 + q: return 0.0
    return sqrt((2 * l1 + 1) * (2 * k + 1) / (4 * pi * (2 * l2 + 1))) * _cg(l1, 0, k, 0, l2, 0) * _cg(l1, m1, k, q, l2, m2)

def _angular_factor(le2, lh2, le1, lh1, k, L):
    """angular part of <(le2 lh2) L 0| P_k(cos theta_eh) |(le1 lh1) L 0> times (4 pi/(2k+1)) sum_q Y_kq(e) Y_kq^*(h)"""
    val = 0.0
    for me1 in range(-le1, le1 + 1):
        mh1 = -me1
        if abs(mh1) > lh1: continue
        c1 = _cg(le1, me1, lh1, mh1, L, 0)
        if c1 == 0.0: continue
        for me2 in range(-le2, le2 + 1):
            mh2 = -me2
            if abs(mh2) > lh2: continue
            c2 = _cg(le2, me2, lh2, mh2, L, 0)
            if c2 == 0.0: continue
            q = me2 - me1
            ge = _ylm_element(le2, me2, k, q, le1, me1)
            gh = _ylm_element(lh2, mh2, k, -q, lh1, mh1)
            val += c1 * c2 * ge * gh * (-1) ** q
    return 4.0 * pi / (2 * k + 1) * val

def _pair_basis(N_n, L_max, L):
    """coupled two-particle basis (n_e, l_e, n_h, l_h) of total angular momentum L, M = 0 and parity (-1)^L"""
    states = []
    for le in range(L_max + 1):
        for lh in range(L_max + 1):
            if (le + lh + L) % 2 or not (abs(le - lh) <= L <= le + lh): continue
            for ne in range(1, N_n + 1):
                for nh in range(1, N_n + 1):
                    states.append((ne, le, nh, lh))
    return states

def _poisson_potentials(F, r, w, k):
    """Phi_k(r1) = int F(r2) r_<^k / r_>^(k+1) dr2 for every row of F (rows already carry the r2^2 weight)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = cumulative_simpson(F * r ** k, x=r, initial=0)
        rinv = np.where(r > 0, r ** (-(k + 1.0)), 0.0)
        outer_tot = cumulative_simpson(F * rinv, x=r, initial=0)
        outer = outer_tot[:, -1:] - outer_tot
        Phi = rinv * inner + r ** k * outer
    Phi[:, 0] = outer_tot[:, -1] if k == 0 else 0.0
    return Phi

def _coulomb_matrix(A, eps, N_n, L_max, L, N_r, states, r, w, R):
    """matrix of -e^2/(4 pi eps eps_0 |r_e - r_h|) in the coupled basis (eV)"""
    cc = _E2() / eps
    N = len(states)
    idx = {s: i for i, s in enumerate(states)}
    lpairs = sorted(set((s[1], s[3]) for s in states))
    ang = {}
    Fcache = {}
    def _F(l2, l1):
        key = (l2, l1)
        if key not in Fcache:
            Fcache[key] = np.array([R[(n2, l2)] * R[(n1, l1)] * r * r for n2 in range(1, N_n + 1) for n1 in range(1, N_n + 1)])
        return Fcache[key]
    Phicache = {}
    def _Phi(l2, l1, k):
        key = (l2, l1, k)
        if key not in Phicache: Phicache[key] = _poisson_potentials(_F(l2, l1), r, w, k)
        return Phicache[key]
    V = np.zeros((N, N))
    for (le2, lh2) in lpairs:
        for (le1, lh1) in lpairs:
            if (le2, lh2) < (le1, lh1): continue
            block = np.zeros((N_n * N_n, N_n * N_n))
            for k in range(max(abs(le1 - le2), abs(lh1 - lh2)), min(le1 + le2, lh1 + lh2) + 1):
                if (le1 + le2 + k) % 2 or (lh1 + lh2 + k) % 2: continue
                a = _angular_factor(le2, lh2, le1, lh1, k, L)
                if a == 0.0: continue
                Rk = np.dot(_F(le2, le1) * w, _Phi(lh2, lh1, k).T)      # (Nn^2 e-pairs, Nn^2 h-pairs)
                # block index: row (ne2, nh2), col (ne1, nh1): Rk[(ne2,ne1),(nh2,nh1)]
                Rk4 = Rk.reshape(N_n, N_n, N_n, N_n)            # [ne2, ne1, nh2, nh1]
                block += a * Rk4.transpose(0, 2, 1, 3).reshape(N_n * N_n, N_n * N_n)
            rows = [idx[(ne, le2, nh, lh2)] for ne in range(1, N_n + 1) for nh in range(1, N_n + 1)]
            cols = [idx[(ne, le1, nh, lh1)] for ne in range(1, N_n + 1) for nh in range(1, N_n + 1)]
            V[np.ix_(rows, cols)] = -cc * block
            if (le2, lh2) != (le1, lh1): V[np.ix_(cols, rows)] = -cc * block.T
    return V

def _kinetic_diagonal(A, me, mh, states, Z):
    return np.array([_HB() / me * (Z[(ne, le)] / A) ** 2 + _HB() / mh * (Z[(nh, lh)] / A) ** 2 for (ne, le, nh, lh) in states])

def _check_pair(A, me, mh, eps, N_n, L_max):
    if not (A > 0 and me > 0 and mh > 0 and eps > 0): raise ValueError("A, me, mh, eps must be positive")
    if not (isinstance(N_n, (int, np.integer)) and N_n >= 1 and isinstance(L_max, (int, np.integer)) and L_max >= 1): raise ValueError("N_n >= 1 and L_max >= 1 required")

def pair_levels(A: float, me: float, mh: float, eps: float, N_n: int, L_max: int, L: int, N_r: int, m: int) -> "np.ndarray":
    _check_pair(A, me, mh, eps, N_n, L_max)
    if L not in (0, 1): raise ValueError("L must be 0 or 1")
    states = _pair_basis(N_n, L_max, L)
    if m < 1 or m > len(states): raise ValueError("m must lie in 1..len(basis)")
    r, w, R, Z = _radial_functions(A, N_n, L_max, N_r)
    H = np.diag(_kinetic_diagonal(A, me, mh, states, Z)) + _coulomb_matrix(A, eps, N_n, L_max, L, N_r, states, r, w, R)
    return eigh(H, eigvals_only=True)[:m]

import numpy as np
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _HB():
    """hbar^2/(2 m_0) in eV nm^2"""
    return 0.03809985

def _E2():
    """e^2/(4 pi eps_0) in eV nm"""
    return 1.439964

def _bessel_zeros(l, nmax):
    """first nmax positive zeros of the spherical Bessel function j_l, each to 1e-14"""
    _f = lambda x: spherical_jn(l, x)
    zs = []; xa = l + 1e-9; fa = _f(xa); step = 0.1
    while len(zs) < nmax:
        xb = xa + step; fb = _f(xb)
        if fa * fb < 0: zs.append(brentq(_f, xa, xb, xtol=1e-15, rtol=1e-15, maxiter=200))
        xa, fa = xb, fb
    return np.array(zs)

def _simpson_weights(r):
    n = len(r); h = r[1] - r[0]
    w = np.ones(n); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    return w * h / 3.0

def _radial_functions(A, N_n, L_max, N_r):
    """normalised radial box functions R_{nl}(r) = N j_l(z_{ln} r / A) on the Simpson grid of N_r points"""
    if not (isinstance(N_r, (int, np.integer)) and N_r >= 201 and N_r % 2 == 1): raise ValueError("N_r must be an odd integer >= 201")
    r = np.linspace(0.0, A, N_r); w = _simpson_weights(r)
    R = {}; Z = {}
    for l in range(L_max + 1):
        z = _bessel_zeros(l, N_n)
        for n in range(1, N_n + 1):
            f = spherical_jn(l, z[n - 1] * r / A)
            R[(n, l)] = f / sqrt(np.sum(w * f * f * r * r))
            Z[(n, l)] = z[n - 1]
    return r, w, R, Z

def _cg(j1, m1, j2, m2, J, M, _memo={}):
    """Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M> (Racah formula, integer arguments)"""
    if m1 + m2 != M or J < abs(j1 - j2) or J > j1 + j2 or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J: return 0.0
    key = (j1, m1, j2, m2, J, M)
    if key in _memo: return _memo[key]
    pre = (2 * J + 1) * factorial(J + j1 - j2) * factorial(J - j1 + j2) * factorial(j1 + j2 - J) / factorial(j1 + j2 + J + 1)
    pre *= factorial(J + M) * factorial(J - M) * factorial(j1 - m1) * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2)
    s = 0.0
    for k in range(0, j1 + j2 + J + 2):
        d = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if min(d) < 0: continue
        s += (-1) ** k / np.prod([float(factorial(x)) for x in d])
    _memo[key] = sqrt(pre) * s
    return _memo[key]

def _ylm_element(l2, m2, k, q, l1, m1):
    """<l2 m2| Y_kq |l1 m1> = sqrt((2 l1 + 1)(2 k + 1)/(4 pi (2 l2 + 1))) <l1 0 k 0|l2 0> <l1 m1 k q|l2 m2>"""
    if m2 != m1 + q: return 0.0
    return sqrt((2 * l1 + 1) * (2 * k + 1) / (4 * pi * (2 * l2 + 1))) * _cg(l1, 0, k, 0, l2, 0) * _cg(l1, m1, k, q, l2, m2)

def _angular_factor(le2, lh2, le1, lh1, k, L):
    """angular part of <(le2 lh2) L 0| P_k(cos theta_eh) |(le1 lh1) L 0> times (4 pi/(2k+1)) sum_q Y_kq(e) Y_kq^*(h)"""
    val = 0.0
    for me1 in range(-le1, le1 + 1):
        mh1 = -me1
        if abs(mh1) > lh1: continue
        c1 = _cg(le1, me1, lh1, mh1, L, 0)
        if c1 == 0.0: continue
        for me2 in range(-le2, le2 + 1):
            mh2 = -me2
            if abs(mh2) > lh2: continue
            c2 = _cg(le2, me2, lh2, mh2, L, 0)
            if c2 == 0.0: continue
            q = me2 - me1
            ge = _ylm_element(le2, me2, k, q, le1, me1)
            gh = _ylm_element(lh2, mh2, k, -q, lh1, mh1)
            val += c1 * c2 * ge * gh * (-1) ** q
    return 4.0 * pi / (2 * k + 1) * val

def _pair_basis(N_n, L_max, L):
    """coupled two-particle basis (n_e, l_e, n_h, l_h) of total angular momentum L, M = 0 and parity (-1)^L"""
    states = []
    for le in range(L_max + 1):
        for lh in range(L_max + 1):
            if (le + lh + L) % 2 or not (abs(le - lh) <= L <= le + lh): continue
            for ne in range(1, N_n + 1):
                for nh in range(1, N_n + 1):
                    states.append((ne, le, nh, lh))
    return states

def _poisson_potentials(F, r, w, k):
    """Phi_k(r1) = int F(r2) r_<^k / r_>^(k+1) dr2 for every row of F (rows already carry the r2^2 weight)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = cumulative_simpson(F * r ** k, x=r, initial=0)
        rinv = np.where(r > 0, r ** (-(k + 1.0)), 0.0)
        outer_tot = cumulative_simpson(F * rinv, x=r, initial=0)
        outer = outer_tot[:, -1:] - outer_tot
        Phi = rinv * inner + r ** k * outer
    Phi[:, 0] = outer_tot[:, -1] if k == 0 else 0.0
    return Phi

def _coulomb_matrix(A, eps, N_n, L_max, L, N_r, states, r, w, R):
    """matrix of -e^2/(4 pi eps eps_0 |r_e - r_h|) in the coupled basis (eV)"""
    cc = _E2() / eps
    N = len(states)
    idx = {s: i for i, s in enumerate(states)}
    lpairs = sorted(set((s[1], s[3]) for s in states))
    ang = {}
    Fcache = {}
    def _F(l2, l1):
        key = (l2, l1)
        if key not in Fcache:
            Fcache[key] = np.array([R[(n2, l2)] * R[(n1, l1)] * r * r for n2 in range(1, N_n + 1) for n1 in range(1, N_n + 1)])
        return Fcache[key]
    Phicache = {}
    def _Phi(l2, l1, k):
        key = (l2, l1, k)
        if key not in Phicache: Phicache[key] = _poisson_potentials(_F(l2, l1), r, w, k)
        return Phicache[key]
    V = np.zeros((N, N))
    for (le2, lh2) in lpairs:
        for (le1, lh1) in lpairs:
            if (le2, lh2) < (le1, lh1): continue
            block = np.zeros((N_n * N_n, N_n * N_n))
            for k in range(max(abs(le1 - le2), abs(lh1 - lh2)), min(le1 + le2, lh1 + lh2) + 1):
                if (le1 + le2 + k) % 2 or (lh1 + lh2 + k) % 2: continue
                a = _angular_factor(le2, lh2, le1, lh1, k, L)
                if a == 0.0: continue
                Rk = np.dot(_F(le2, le1) * w, _Phi(lh2, lh1, k).T)      # (Nn^2 e-pairs, Nn^2 h-pairs)
                # block index: row (ne2, nh2), col (ne1, nh1): Rk[(ne2,ne1),(nh2,nh1)]
                Rk4 = Rk.reshape(N_n, N_n, N_n, N_n)            # [ne2, ne1, nh2, nh1]
                block += a * Rk4.transpose(0, 2, 1, 3).reshape(N_n * N_n, N_n * N_n)
            rows = [idx[(ne, le2, nh, lh2)] for ne in range(1, N_n + 1) for nh in range(1, N_n + 1)]
            cols = [idx[(ne, le1, nh, lh1)] for ne in range(1, N_n + 1) for nh in range(1, N_n + 1)]
            V[np.ix_(rows, cols)] = -cc * block
            if (le2, lh2) != (le1, lh1): V[np.ix_(cols, rows)] = -cc * block.T
    return V

def _kinetic_diagonal(A, me, mh, states, Z):
    return np.array([_HB() / me * (Z[(ne, le)] / A) ** 2 + _HB() / mh * (Z[(nh, lh)] / A) ** 2 for (ne, le, nh, lh) in states])

def _check_pair(A, me, mh, eps, N_n, L_max):
    if not (A > 0 and me > 0 and mh > 0 and eps > 0): raise ValueError("A, me, mh, eps must be positive")
    if not (isinstance(N_n, (int, np.integer)) and N_n >= 1 and isinstance(L_max, (int, np.integer)) and L_max >= 1): raise ValueError("N_n >= 1 and L_max >= 1 required")

def _dipole_block(states1, states0, r, w, R):
    """<(states1) L=1, M=0| z_e - z_h |(states0) L=0> in nm"""
    def _ang_z(l2, l1, m):
        if abs(l1 - l2) != 1 or abs(m) > min(l1, l2): return 0.0
        l = min(l1, l2)
        return sqrt(((l + 1) ** 2 - m * m) / ((2 * l + 1) * (2 * l + 3)))
    radcache = {}
    def _rad(a2, a1):
        key = (a2, a1)
        if key not in radcache: radcache[key] = float(np.sum(w * R[a2] * R[a1] * r ** 3))
        return radcache[key]
    D = np.zeros((len(states1), len(states0)))
    angcache = {}
    def _angsum(le2, lh2, le1, lh1, which):
        key = (le2, lh2, le1, lh1, which)
        if key not in angcache:
            val = 0.0
            for m in range(-min(le1, lh1), min(le1, lh1) + 1):
                c1 = _cg(le1, m, lh1, -m, 0, 0); c2 = _cg(le2, m, lh2, -m, 1, 0)
                if c1 == 0.0 or c2 == 0.0: continue
                val += c1 * c2 * (_ang_z(le2, le1, m) if which == "e" else _ang_z(lh2, lh1, -m))
            angcache[key] = val
        return angcache[key]
    for j, (ne1, le1, nh1, lh1) in enumerate(states0):
        for i, (ne2, le2, nh2, lh2) in enumerate(states1):
            val = 0.0
            if (nh2, lh2) == (nh1, lh1) and abs(le2 - le1) == 1:
                val += _angsum(le2, lh2, le1, lh1, "e") * _rad((ne2, le2), (ne1, le1))
            if (ne2, le2) == (ne1, le1) and abs(lh2 - lh1) == 1:
                val -= _angsum(le2, lh2, le1, lh1, "h") * _rad((nh2, lh2), (nh1, lh1))
            D[i, j] = val
    return D

def thz_transitions(A: float, me: float, mh: float, eps: float, N_n: int, L_max: int, N_r: int, m: int) -> "np.ndarray":
    _check_pair(A, me, mh, eps, N_n, L_max)
    r, w, R, Z = _radial_functions(A, N_n, L_max, N_r)
    st0 = _pair_basis(N_n, L_max, 0); st1 = _pair_basis(N_n, L_max, 1)
    if m < 1 or m > len(st1): raise ValueError("m must lie in 1..len(L = 1 basis)")
    H0 = np.diag(_kinetic_diagonal(A, me, mh, st0, Z)) + _coulomb_matrix(A, eps, N_n, L_max, 0, N_r, st0, r, w, R)
    H1 = np.diag(_kinetic_diagonal(A, me, mh, st1, Z)) + _coulomb_matrix(A, eps, N_n, L_max, 1, N_r, st1, r, w, R)
    w0, v0 = eigh(H0); w1, v1 = eigh(H1)
    D = _dipole_block(st1, st0, r, w, R)
    d = np.dot(v1[:, :m].T, np.dot(D, v0[:, 0]))
    dE = w1[:m] - w0[0]
    out = np.zeros((m, 4))
    out[:, 0] = dE; out[:, 1] = d * d; out[:, 2] = dE * d * d; out[:, 3] = w0[0]
    return out

import numpy as np
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _HB():
    """hbar^2/(2 m_0) in eV nm^2"""
    return 0.03809985

def _bessel_zeros(l, nmax):
    """first nmax positive zeros of the spherical Bessel function j_l, each to 1e-14"""
    _f = lambda x: spherical_jn(l, x)
    zs = []; xa = l + 1e-9; fa = _f(xa); step = 0.1
    while len(zs) < nmax:
        xb = xa + step; fb = _f(xb)
        if fa * fb < 0: zs.append(brentq(_f, xa, xb, xtol=1e-15, rtol=1e-15, maxiter=200))
        xa, fa = xb, fb
    return np.array(zs)

def _simpson_weights(r):
    n = len(r); h = r[1] - r[0]
    w = np.ones(n); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    return w * h / 3.0

def _radial_functions(A, N_n, L_max, N_r):
    """normalised radial box functions R_{nl}(r) = N j_l(z_{ln} r / A) on the Simpson grid of N_r points"""
    if not (isinstance(N_r, (int, np.integer)) and N_r >= 201 and N_r % 2 == 1): raise ValueError("N_r must be an odd integer >= 201")
    r = np.linspace(0.0, A, N_r); w = _simpson_weights(r)
    R = {}; Z = {}
    for l in range(L_max + 1):
        z = _bessel_zeros(l, N_n)
        for n in range(1, N_n + 1):
            f = spherical_jn(l, z[n - 1] * r / A)
            R[(n, l)] = f / sqrt(np.sum(w * f * f * r * r))
            Z[(n, l)] = z[n - 1]
    return r, w, R, Z

def bare_response(A: float, me: float, mh: float, N_r: int) -> "np.ndarray":
    if not (A > 0 and me > 0 and mh > 0): raise ValueError("A, me, mh must be positive")
    r, w, R, Z = _radial_functions(A, 1, 1, N_r)
    dE = (Z[(1, 1)] ** 2 - Z[(1, 0)] ** 2) / A ** 2
    d2 = float(np.sum(w * R[(1, 1)] * R[(1, 0)] * r ** 3)) ** 2 / 3.0
    return np.array([_HB() / me * dE, _HB() / mh * dE, d2, d2])

import numpy as np
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _HB():
    """hbar^2/(2 m_0) in eV nm^2"""
    return 0.03809985

def _E2():
    """e^2/(4 pi eps_0) in eV nm"""
    return 1.439964

def _bessel_zeros(l, nmax):
    """first nmax positive zeros of the spherical Bessel function j_l, each to 1e-14"""
    _f = lambda x: spherical_jn(l, x)
    zs = []; xa = l + 1e-9; fa = _f(xa); step = 0.1
    while len(zs) < nmax:
        xb = xa + step; fb = _f(xb)
        if fa * fb < 0: zs.append(brentq(_f, xa, xb, xtol=1e-15, rtol=1e-15, maxiter=200))
        xa, fa = xb, fb
    return np.array(zs)

def _simpson_weights(r):
    n = len(r); h = r[1] - r[0]
    w = np.ones(n); w[1:-1:2] = 4.0; w[2:-1:2] = 2.0
    return w * h / 3.0

def _radial_functions(A, N_n, L_max, N_r):
    """normalised radial box functions R_{nl}(r) = N j_l(z_{ln} r / A) on the Simpson grid of N_r points"""
    if not (isinstance(N_r, (int, np.integer)) and N_r >= 201 and N_r % 2 == 1): raise ValueError("N_r must be an odd integer >= 201")
    r = np.linspace(0.0, A, N_r); w = _simpson_weights(r)
    R = {}; Z = {}
    for l in range(L_max + 1):
        z = _bessel_zeros(l, N_n)
        for n in range(1, N_n + 1):
            f = spherical_jn(l, z[n - 1] * r / A)
            R[(n, l)] = f / sqrt(np.sum(w * f * f * r * r))
            Z[(n, l)] = z[n - 1]
    return r, w, R, Z

def _cg(j1, m1, j2, m2, J, M, _memo={}):
    """Clebsch-Gordan coefficient <j1 m1 j2 m2 | J M> (Racah formula, integer arguments)"""
    if m1 + m2 != M or J < abs(j1 - j2) or J > j1 + j2 or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J: return 0.0
    key = (j1, m1, j2, m2, J, M)
    if key in _memo: return _memo[key]
    pre = (2 * J + 1) * factorial(J + j1 - j2) * factorial(J - j1 + j2) * factorial(j1 + j2 - J) / factorial(j1 + j2 + J + 1)
    pre *= factorial(J + M) * factorial(J - M) * factorial(j1 - m1) * factorial(j1 + m1) * factorial(j2 - m2) * factorial(j2 + m2)
    s = 0.0
    for k in range(0, j1 + j2 + J + 2):
        d = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if min(d) < 0: continue
        s += (-1) ** k / np.prod([float(factorial(x)) for x in d])
    _memo[key] = sqrt(pre) * s
    return _memo[key]

def _ylm_element(l2, m2, k, q, l1, m1):
    """<l2 m2| Y_kq |l1 m1> = sqrt((2 l1 + 1)(2 k + 1)/(4 pi (2 l2 + 1))) <l1 0 k 0|l2 0> <l1 m1 k q|l2 m2>"""
    if m2 != m1 + q: return 0.0
    return sqrt((2 * l1 + 1) * (2 * k + 1) / (4 * pi * (2 * l2 + 1))) * _cg(l1, 0, k, 0, l2, 0) * _cg(l1, m1, k, q, l2, m2)

def _angular_factor(le2, lh2, le1, lh1, k, L):
    """angular part of <(le2 lh2) L 0| P_k(cos theta_eh) |(le1 lh1) L 0> times (4 pi/(2k+1)) sum_q Y_kq(e) Y_kq^*(h)"""
    val = 0.0
    for me1 in range(-le1, le1 + 1):
        mh1 = -me1
        if abs(mh1) > lh1: continue
        c1 = _cg(le1, me1, lh1, mh1, L, 0)
        if c1 == 0.0: continue
        for me2 in range(-le2, le2 + 1):
            mh2 = -me2
            if abs(mh2) > lh2: continue
            c2 = _cg(le2, me2, lh2, mh2, L, 0)
            if c2 == 0.0: continue
            q = me2 - me1
            ge = _ylm_element(le2, me2, k, q, le1, me1)
            gh = _ylm_element(lh2, mh2, k, -q, lh1, mh1)
            val += c1 * c2 * ge * gh * (-1) ** q
    return 4.0 * pi / (2 * k + 1) * val

def _pair_basis(N_n, L_max, L):
    """coupled two-particle basis (n_e, l_e, n_h, l_h) of total angular momentum L, M = 0 and parity (-1)^L"""
    states = []
    for le in range(L_max + 1):
        for lh in range(L_max + 1):
            if (le + lh + L) % 2 or not (abs(le - lh) <= L <= le + lh): continue
            for ne in range(1, N_n + 1):
                for nh in range(1, N_n + 1):
                    states.append((ne, le, nh, lh))
    return states

def _poisson_potentials(F, r, w, k):
    """Phi_k(r1) = int F(r2) r_<^k / r_>^(k+1) dr2 for every row of F (rows already carry the r2^2 weight)"""
    with np.errstate(divide="ignore", invalid="ignore"):
        inner = cumulative_simpson(F * r ** k, x=r, initial=0)
        rinv = np.where(r > 0, r ** (-(k + 1.0)), 0.0)
        outer_tot = cumulative_simpson(F * rinv, x=r, initial=0)
        outer = outer_tot[:, -1:] - outer_tot
        Phi = rinv * inner + r ** k * outer
    Phi[:, 0] = outer_tot[:, -1] if k == 0 else 0.0
    return Phi

def _coulomb_matrix(A, eps, N_n, L_max, L, N_r, states, r, w, R):
    """matrix of -e^2/(4 pi eps eps_0 |r_e - r_h|) in the coupled basis (eV)"""
    cc = _E2() / eps
    N = len(states)
    idx = {s: i for i, s in enumerate(states)}
    lpairs = sorted(set((s[1], s[3]) for s in states))
    ang = {}
    Fcache = {}
    def _F(l2, l1):
        key = (l2, l1)
        if key not in Fcache:
            Fcache[key] = np.array([R[(n2, l2)] * R[(n1, l1)] * r * r for n2 in range(1, N_n + 1) for n1 in range(1, N_n + 1)])
        return Fcache[key]
    Phicache = {}
    def _Phi(l2, l1, k):
        key = (l2, l1, k)
        if key not in Phicache: Phicache[key] = _poisson_potentials(_F(l2, l1), r, w, k)
        return Phicache[key]
    V = np.zeros((N, N))
    for (le2, lh2) in lpairs:
        for (le1, lh1) in lpairs:
            if (le2, lh2) < (le1, lh1): continue
            block = np.zeros((N_n * N_n, N_n * N_n))
            for k in range(max(abs(le1 - le2), abs(lh1 - lh2)), min(le1 + le2, lh1 + lh2) + 1):
                if (le1 + le2 + k) % 2 or (lh1 + lh2 + k) % 2: continue
                a = _angular_factor(le2, lh2, le1, lh1, k, L)
                if a == 0.0: continue
                Rk = np.dot(_F(le2, le1) * w, _Phi(lh2, lh1, k).T)      # (Nn^2 e-pairs, Nn^2 h-pairs)
                # block index: row (ne2, nh2), col (ne1, nh1): Rk[(ne2,ne1),(nh2,nh1)]
                Rk4 = Rk.reshape(N_n, N_n, N_n, N_n)            # [ne2, ne1, nh2, nh1]
                block += a * Rk4.transpose(0, 2, 1, 3).reshape(N_n * N_n, N_n * N_n)
            rows = [idx[(ne, le2, nh, lh2)] for ne in range(1, N_n + 1) for nh in range(1, N_n + 1)]
            cols = [idx[(ne, le1, nh, lh1)] for ne in range(1, N_n + 1) for nh in range(1, N_n + 1)]
            V[np.ix_(rows, cols)] = -cc * block
            if (le2, lh2) != (le1, lh1): V[np.ix_(cols, rows)] = -cc * block.T
    return V

def _kinetic_diagonal(A, me, mh, states, Z):
    return np.array([_HB() / me * (Z[(ne, le)] / A) ** 2 + _HB() / mh * (Z[(nh, lh)] / A) ** 2 for (ne, le, nh, lh) in states])

def _check_pair(A, me, mh, eps, N_n, L_max):
    if not (A > 0 and me > 0 and mh > 0 and eps > 0): raise ValueError("A, me, mh, eps must be positive")
    if not (isinstance(N_n, (int, np.integer)) and N_n >= 1 and isinstance(L_max, (int, np.integer)) and L_max >= 1): raise ValueError("N_n >= 1 and L_max >= 1 required")

def _dipole_block(states1, states0, r, w, R):
    """<(states1) L=1, M=0| z_e - z_h |(states0) L=0> in nm"""
    def _ang_z(l2, l1, m):
        if abs(l1 - l2) != 1 or abs(m) > min(l1, l2): return 0.0
        l = min(l1, l2)
        return sqrt(((l + 1) ** 2 - m * m) / ((2 * l + 1) * (2 * l + 3)))
    radcache = {}
    def _rad(a2, a1):
        key = (a2, a1)
        if key not in radcache: radcache[key] = float(np.sum(w * R[a2] * R[a1] * r ** 3))
        return radcache[key]
    D = np.zeros((len(states1), len(states0)))
    angcache = {}
    def _angsum(le2, lh2, le1, lh1, which):
        key = (le2, lh2, le1, lh1, which)
        if key not in angcache:
            val = 0.0
            for m in range(-min(le1, lh1), min(le1, lh1) + 1):
                c1 = _cg(le1, m, lh1, -m, 0, 0); c2 = _cg(le2, m, lh2, -m, 1, 0)
                if c1 == 0.0 or c2 == 0.0: continue
                val += c1 * c2 * (_ang_z(le2, le1, m) if which == "e" else _ang_z(lh2, lh1, -m))
            angcache[key] = val
        return angcache[key]
    for j, (ne1, le1, nh1, lh1) in enumerate(states0):
        for i, (ne2, le2, nh2, lh2) in enumerate(states1):
            val = 0.0
            if (nh2, lh2) == (nh1, lh1) and abs(le2 - le1) == 1:
                val += _angsum(le2, lh2, le1, lh1, "e") * _rad((ne2, le2), (ne1, le1))
            if (ne2, le2) == (ne1, le1) and abs(lh2 - lh1) == 1:
                val -= _angsum(le2, lh2, le1, lh1, "h") * _rad((nh2, lh2), (nh1, lh1))
            D[i, j] = val
    return D

def minimal_scr(A: float, me: float, mh: float, eps: float, N_r: int) -> "np.ndarray":
    if not (A > 0 and me > 0 and mh > 0 and eps > 0): raise ValueError("A, me, mh, eps must be positive")
    _check_pair(A, me, mh, eps, 1, 1)
    r, w, R, Z = _radial_functions(A, 1, 1, N_r)
    st0 = _pair_basis(1, 1, 0); st1 = _pair_basis(1, 1, 1)
    H0 = np.diag(_kinetic_diagonal(A, me, mh, st0, Z)) + _coulomb_matrix(A, eps, 1, 1, 0, N_r, st0, r, w, R)
    H1 = np.diag(_kinetic_diagonal(A, me, mh, st1, Z)) + _coulomb_matrix(A, eps, 1, 1, 1, N_r, st1, r, w, R)
    w0, v0 = eigh(H0); w1, v1 = eigh(H1)
    D = _dipole_block(st1, st0, r, w, R)
    d = np.dot(v1.T, np.dot(D, v0[:, 0]))
    i_e = int(np.argmax(w1)); i_h = 1 - i_e
    EC = H1[0, 1]                       # <(1s)(1p)| V |(1p)(1s)> coupling, eV (negative)
    c = -EC * A * eps / _E2()     # |E_C| in units of e^2/(4 pi eps A)
    return np.array([w0[0], w1[i_h] - w0[0], w1[i_e] - w0[0], d[i_h] ** 2, d[i_e] ** 2, EC, c])

import numpy as np
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _HB():
    """hbar^2/(2 m_0) in eV nm^2"""
    return 0.03809985

def _E2():
    """e^2/(4 pi eps_0) in eV nm"""
    return 1.439964

def _check_pair(A, me, mh, eps, N_n, L_max):
    if not (A > 0 and me > 0 and mh > 0 and eps > 0): raise ValueError("A, me, mh, eps must be positive")
    if not (isinstance(N_n, (int, np.integer)) and N_n >= 1 and isinstance(L_max, (int, np.integer)) and L_max >= 1): raise ValueError("N_n >= 1 and L_max >= 1 required")

def _wcr_fd(A, me, mh, eps, N_g):
    """three-point finite-difference eigenvalues of the weak-confinement radial equation for l = 0 and 1 on the
    uniform grid of N_g intervals of [0, r_max], r_max = A min(me, mh)/mu, with u = r R and u(0) = u(r_max) = 0"""
    M = me + mh; mu = me * mh / M
    r_max = A * min(me, mh) / mu
    h = r_max / N_g
    x = h * np.arange(1, N_g)
    rho = mu * x / min(me, mh)
    chi = _HB() / M * (pi / (A - rho)) ** 2
    res = []
    for l in (0, 1):
        Vl = -_E2() / eps / x + _HB() / mu * l * (l + 1) / x ** 2 + chi
        diag = _HB() / mu * 2.0 / h ** 2 + Vl
        off = -_HB() / mu / h ** 2 * np.ones(N_g - 2)
        wl, vl = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
        u = vl[:, 0] / sqrt(np.sum(vl[:, 0] ** 2) * h)
        res.append((float(wl[0]), u, x, h))
    E1s, u1s, x, h = res[0]; E2p, u2p, _, _ = res[1]
    d = np.sum(u2p * u1s * x) * h / sqrt(3.0)
    return np.array([E1s, E2p, E2p - E1s, d * d])

def wcr_model(A: float, me: float, mh: float, eps: float, N_g: int) -> "np.ndarray":
    _check_pair(A, me, mh, eps, 1, 1)
    if not (isinstance(N_g, (int, np.integer)) and N_g >= 500): raise ValueError("N_g must be an integer >= 500")
    # the finite-difference error is O(h^2) with an O(h^4) remainder: two Richardson levels on N_g, 2 N_g, 4 N_g
    e1 = _wcr_fd(A, me, mh, eps, N_g); e2 = _wcr_fd(A, me, mh, eps, 2 * N_g); e4 = _wcr_fd(A, me, mh, eps, 4 * N_g)
    r12 = (4.0 * e2 - e1) / 3.0; r24 = (4.0 * e4 - e2) / 3.0
    return (16.0 * r24 - r12) / 15.0

import numpy as np
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _pair_basis(N_n, L_max, L):
    """coupled two-particle basis (n_e, l_e, n_h, l_h) of total angular momentum L, M = 0 and parity (-1)^L"""
    states = []
    for le in range(L_max + 1):
        for lh in range(L_max + 1):
            if (le + lh + L) % 2 or not (abs(le - lh) <= L <= le + lh): continue
            for ne in range(1, N_n + 1):
                for nh in range(1, N_n + 1):
                    states.append((ne, le, nh, lh))
    return states

def _renormalisation(A, me, mh, eps, N_n, L_max, N_r):
    """[dE_e, d2_e, dE_h, d2_h, E0, renorm_e, renorm_h, ratio] of the exact model: the electron-like resonance carries
    the largest weight dE |d|^2 and the hole-like one is the largest weight among the resonances below it"""
    tr = thz_transitions(A, me, mh, eps, N_n, L_max, N_r, len(_pair_basis(N_n, L_max, 1)))
    ie = int(np.argmax(tr[:, 2]))                      # electron-like: the largest resonance weight
    below = [k for k in range(ie) if tr[k, 0] < tr[ie, 0]]
    if not below: raise ValueError("no resonance below the dominant one")
    ih = int(below[int(np.argmax(tr[below, 2]))])        # hole-like: the largest weight below it
    bare = bare_response(A, me, mh, N_r)
    return np.array([tr[ie, 0], tr[ie, 1], tr[ih, 0], tr[ih, 1], tr[0, 3], tr[ie, 1] / bare[2], tr[ih, 1] / bare[3], tr[ih, 2] / tr[ie, 2]])

def independence_radius(me: float, mh: float, eps: float, N_n: int, L_max: int, N_r: int, A_lo: float, A_hi: float, level: float) -> float:
    if not (0 < A_lo < A_hi) or not (level > 1.0): raise ValueError("0 < A_lo < A_hi and level > 1 required")
    _f = lambda A: _renormalisation(A, me, mh, eps, N_n, L_max, N_r)[5] - level
    flo, fhi = _f(A_lo), _f(A_hi)
    if flo * fhi > 0: raise ValueError("the renormalisation level is not bracketed by [A_lo, A_hi]")
    return brentq(_f, A_lo, A_hi, xtol=1e-10, rtol=1e-12, maxiter=100)

import numpy as np
from math import sqrt, pi, factorial
from scipy.special import spherical_jn
from scipy.optimize import brentq
from scipy.integrate import cumulative_simpson
from scipy.linalg import eigh, eigh_tridiagonal

def _HB():
    """hbar^2/(2 m_0) in eV nm^2"""
    return 0.03809985

def _E2():
    """e^2/(4 pi eps_0) in eV nm"""
    return 1.439964

def _pair_basis(N_n, L_max, L):
    """coupled two-particle basis (n_e, l_e, n_h, l_h) of total angular momentum L, M = 0 and parity (-1)^L"""
    states = []
    for le in range(L_max + 1):
        for lh in range(L_max + 1):
            if (le + lh + L) % 2 or not (abs(le - lh) <= L <= le + lh): continue
            for ne in range(1, N_n + 1):
                for nh in range(1, N_n + 1):
                    states.append((ne, le, nh, lh))
    return states

def confinement_audit(me: float, mh: float, eps: float, radii: list, N_n: int, L_max: int, N_r: int, N_g: int, A_star: float, A_lo: float, A_hi: float, level: float) -> "np.ndarray":
    radii = [float(a) for a in radii]
    if len(radii) < 1 or min(radii) <= 0: raise ValueError("radii must be positive")
    if not (A_star > 0): raise ValueError("A_star must be positive")
    rows = []
    n1 = len(_pair_basis(N_n, L_max, 1))
    for A in list(radii) + [A_star]:
        tr = thz_transitions(A, me, mh, eps, N_n, L_max, N_r, n1)
        ie = int(np.argmax(tr[:, 2]))
        below = [k for k in range(ie) if tr[k, 0] < tr[ie, 0]]
        if not below: raise ValueError("no resonance below the dominant one")
        ih = int(below[int(np.argmax(tr[below, 2]))])
        E0 = float(pair_levels(A, me, mh, eps, N_n, L_max, 0, N_r, 1)[0])
        if abs(E0 - tr[0, 3]) > 1e-12: raise ValueError("ground-state energies of the two block solvers disagree")
        bare = bare_response(A, me, mh, N_r)
        mn = minimal_scr(A, me, mh, eps, N_r)
        wc = wcr_model(A, me, mh, eps, N_g)
        rows.append([A, E0, tr[ie, 0], tr[ie, 1], tr[ih, 0], tr[ih, 1], tr[ih, 2] / tr[ie, 2], bare[0], bare[1], bare[2], mn[2], mn[1], mn[4], mn[3], wc[2], wc[3],
                     tr[ie, 1] / bare[2], tr[ih, 1] / bare[3], tr[ie, 0] - bare[0]])
    A10 = independence_radius(me, mh, eps, N_n, L_max, N_r, A_lo, A_hi, level)
    mu = me * mh / (me + mh)
    aX = 2.0 * _HB() / mu * eps / _E2()
    A_est = 1.5 * _HB() * eps / _E2() * (1.0 / me - 1.0 / mh)   # source's Eq. (26): 3 pi hbar^2 eps/(e^2) (1/me - 1/mh)
    Z = box_spectrum(A_star, max(N_n, 1), max(L_max, 1))
    split = float(Z[1, 0] ** 2 - Z[0, 0] ** 2)                              # k_1p^2 - k_1s^2 in units of 1/A^2
    direct = float(coulomb_integral(A_star, 0, 1, 0, 1, 0, 1, 0, 1, 0, N_r)) * A_star   # 1s-1s direct integral in e^2/(4 pi eps eps_0 A)
    names = ["A", "E0", "dE_e", "d2_e", "dE_h", "d2_h", "ratio", "dE_e_bare", "dE_h_bare", "d2_bare", "dE_e_min", "dE_h_min", "d2_e_min", "d2_h_min", "dE_w", "d2_w", "renorm_e", "renorm_h", "shift_e"]
    table = np.zeros((1 + len(rows), len(names)))
    table[1:] = np.array(rows)
    star = rows[-1]
    wc0 = wcr_model(A_star, me, mh, np.inf, N_g)                     # Coulomb term omitted: the strongly confined limit of the model
    limit = float(wc0[2] / bare_response(A_star, me, mh, N_r)[0])
    table[0, :18] = [star[2], star[18], star[6], star[16], A10, minimal_scr(A_star, me, mh, eps, N_r)[6], aX, A_est, A_star, N_n, L_max, N_r, N_g, len(radii), split, direct,
                     limit, star[14] / star[2]]
    return table
SCICODE_GOLD_EOF
