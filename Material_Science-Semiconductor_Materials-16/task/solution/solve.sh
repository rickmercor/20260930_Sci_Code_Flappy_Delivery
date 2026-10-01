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

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def hopping_vectors(a: float, theta: float) -> "np.ndarray":
    if a <= 0:
        raise ValueError("lattice constant must be positive")
    if not (0.0 < theta < 0.5 * np.pi):
        raise ValueError("bond angle must lie strictly between 0 and pi/2")
    c, s = np.cos(theta), np.sin(theta)
    r3 = np.sqrt(3.0)
    # CONVENTION (source, Table I): bond length a/(sqrt(3) cos t), not a/cos t.
    b = a / (r3 * c)
    # LISTING ORDER, fixed by the problem statement rather than by the physics: every family
    # runs by increasing azimuthal angle about +x. Without it the six M-X bonds admit several
    # equally defensible listings and the step is not well posed.
    A = np.array([[r3 / 2 * c, 0.5 * c, s], [0.0, c, -s], [-r3 / 2 * c, 0.5 * c, s],
                  [-r3 / 2 * c, -0.5 * c, -s], [0.0, -c, s], [r3 / 2 * c, -0.5 * c, -s]]) * b
    # the three independent in-plane NN directions; their opposites are supplied by the
    # cosine sum in step 6 rather than being listed here.
    D = np.array([[1.0, 0.0, 0.0], [0.5, r3 / 2, 0.0], [-0.5, r3 / 2, 0.0]]) * a
    # CONVENTION (source, Fig. 2): next-nearest in-plane neighbour at a*sqrt(3), not a.
    C = np.array([[r3 / 2, 0.5, 0.0], [0.0, 1.0, 0.0], [-r3 / 2, 0.5, 0.0]]) * (a * r3)
    # CONVENTION (source, App. B): inter-plane vector carries 2*sin(theta); the printed
    # factor d = cos^2 + 4 sin^2 is the signature of that two.
    L = np.array([[r3 / 2 * c, 0.5 * c, -2 * s], [-r3 / 2 * c, 0.5 * c, -2 * s],
                  [0.0, -c, -2 * s]]) * b
    return np.vstack([A, D, C, L])

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _sk_pp(direction, v_sigma, v_pi):
    d = np.asarray(direction, dtype=float)
    n = np.linalg.norm(d)
    if d.shape != (3,) or n == 0.0:
        raise ValueError("direction must be a non-zero vector of length 3")
    u = d / n
    return np.outer(u, u) * (v_sigma - v_pi) + np.eye(3) * v_pi


def _sk_dd(direction, v_sigma, v_pi, v_delta):
    d = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(d)
    if d.shape != (3,) or nrm == 0.0:
        raise ValueError("direction must be a non-zero vector of length 3")
    l, m, n = d / nrm
    r3 = np.sqrt(3.0)
    ll, mm, nn = l * l, m * m, n * n
    S, P, D = v_sigma, v_pi, v_delta
    E = np.zeros((5, 5))
    E[0, 0] = (nn - 0.5 * (ll + mm)) ** 2 * S + 3 * nn * (ll + mm) * P + 0.75 * (ll + mm) ** 2 * D
    E[0, 1] = r3 / 2 * (ll - mm) * (nn - 0.5 * (ll + mm)) * S - r3 * nn * (ll - mm) * P \
        + r3 / 4 * (1 + nn) * (ll - mm) * D
    E[0, 2] = r3 * l * m * (nn - 0.5 * (ll + mm)) * S - 2 * r3 * l * m * nn * P \
        + r3 / 2 * l * m * (1 + nn) * D
    E[0, 3] = r3 * m * n * (nn - 0.5 * (ll + mm)) * S + r3 * m * n * (ll + mm - nn) * P \
        - r3 / 2 * m * n * (ll + mm) * D
    E[0, 4] = r3 * l * n * (nn - 0.5 * (ll + mm)) * S + r3 * l * n * (ll + mm - nn) * P \
        - r3 / 2 * l * n * (ll + mm) * D
    E[1, 1] = 0.75 * (ll - mm) ** 2 * S + (ll + mm - (ll - mm) ** 2) * P \
        + (nn + 0.25 * (ll - mm) ** 2) * D
    E[1, 2] = 1.5 * l * m * (ll - mm) * S - 2 * l * m * (ll - mm) * P + 0.5 * l * m * (ll - mm) * D
    E[1, 3] = 1.5 * m * n * (ll - mm) * S - m * n * (1 + 2 * (ll - mm)) * P \
        + m * n * (1 + 0.5 * (ll - mm)) * D
    E[1, 4] = 1.5 * l * n * (ll - mm) * S + l * n * (1 - 2 * (ll - mm)) * P \
        - l * n * (1 - 0.5 * (ll - mm)) * D
    E[2, 2] = 3 * ll * mm * S + (ll + mm - 4 * ll * mm) * P + (nn + ll * mm) * D
    E[2, 3] = 3 * l * mm * n * S + l * n * (1 - 4 * mm) * P + l * n * (mm - 1) * D
    E[2, 4] = 3 * ll * m * n * S + m * n * (1 - 4 * ll) * P + m * n * (ll - 1) * D
    E[3, 3] = 3 * mm * nn * S + (mm + nn - 4 * mm * nn) * P + (ll + mm * nn) * D
    E[3, 4] = 3 * m * nn * l * S + m * l * (1 - 4 * nn) * P + m * l * (nn - 1) * D
    E[4, 4] = 3 * nn * ll * S + (nn + ll - 4 * nn * ll) * P + (mm + nn * ll) * D
    return E + np.triu(E, 1).T


def _sk_pd(direction, v_sigma, v_pi):
    d = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(d)
    if d.shape != (3,) or nrm == 0.0:
        raise ValueError("direction must be a non-zero vector of length 3")
    l, m, n = d / nrm
    r3 = np.sqrt(3.0)
    ll, mm, nn = l * l, m * m, n * n
    S, P = v_sigma, v_pi
    E = np.zeros((3, 5))
    E[0, 0] = l * (nn - 0.5 * (ll + mm)) * S - r3 * l * nn * P
    E[1, 0] = m * (nn - 0.5 * (ll + mm)) * S - r3 * m * nn * P
    E[2, 0] = n * (nn - 0.5 * (ll + mm)) * S + r3 * n * (ll + mm) * P
    E[0, 1] = r3 / 2 * l * (ll - mm) * S + l * (1 - ll + mm) * P
    E[1, 1] = r3 / 2 * m * (ll - mm) * S - m * (1 + ll - mm) * P
    E[2, 1] = r3 / 2 * n * (ll - mm) * S - n * (ll - mm) * P
    E[0, 2] = r3 * ll * m * S + m * (1 - 2 * ll) * P
    E[1, 2] = r3 * mm * l * S + l * (1 - 2 * mm) * P
    E[2, 2] = r3 * l * m * n * S - 2 * l * m * n * P
    E[0, 3] = r3 * l * m * n * S - 2 * l * m * n * P
    E[1, 3] = r3 * mm * n * S + n * (1 - 2 * mm) * P
    E[2, 3] = r3 * nn * m * S + m * (1 - 2 * nn) * P
    E[0, 4] = r3 * ll * n * S + n * (1 - 2 * ll) * P
    E[1, 4] = r3 * l * m * n * S - 2 * l * m * n * P
    E[2, 4] = r3 * nn * l * S + l * (1 - 2 * nn) * P
    return E


def _onsite(p):
    # CONVENTION (source, shorthand after eq. 7, NOT eq. 7 read positionally): d1 is the
    # {dyz, dzx} doublet and d2 is {dx2-y2, dxy}. The positional reading is the reverse and
    # misses both published gaps.
    diag = np.empty(11)
    diag[0:3] = [p[3], p[3], p[4]]
    diag[3:8] = [p[0], p[2], p[2], p[1], p[1]]
    diag[8:11] = [p[3], p[3], p[4]]
    return np.diag(diag)


def bloch_hamiltonian(k: "np.ndarray", vectors: "np.ndarray", params: "np.ndarray", deriv: int) -> "np.ndarray":
    kk = np.asarray(k, dtype=float)
    v = np.asarray(vectors, dtype=float)
    p = np.asarray(params, dtype=float)
    if kk.shape != (3,):
        raise ValueError("k must be a vector of length 3")
    if v.shape != (15, 3):
        raise ValueError("vectors must have shape (15, 3)")
    if p.shape != (17,):
        raise ValueError("params must have exactly 17 entries")
    if deriv not in (-1, 0, 1, 2):
        raise ValueError("deriv must be -1, 0, 1 or 2")
    Vdds, Vddp, Vddd, Vpps, Vppp, Vpds, Vpdp = p[5:12]
    Kdds, Kddp, Kddd, Kpps, Kppp = p[12:17]
    # the on-site block carries no k, so it survives only in the matrix itself.
    H = _onsite(p).astype(complex) if deriv < 0 else np.zeros((11, 11), dtype=complex)

    # CONVENTION (source, eq. 8): half of each bond family is listed; the opposite members
    # return as the FACTOR TWO on the cosine.
    for rows, sk in ((slice(6, 9), (Vdds, Vddp, Vddd, Vpps, Vppp)),
                     (slice(9, 12), (Kdds, Kddp, Kddd, Kpps, Kppp))):
        for d in v[rows]:
            f = 2.0 * np.cos(kk @ d) if deriv < 0 else -2.0 * d[deriv] * np.sin(kk @ d)
            H[3:8, 3:8] += f * _sk_dd(d, sk[0], sk[1], sk[2])
            blk = f * _sk_pp(d, sk[3], sk[4])
            H[0:3, 0:3] += blk
            H[8:11, 8:11] += blk

    top = np.zeros((5, 3), dtype=complex)
    bot = np.zeros((5, 3), dtype=complex)
    for i, d in enumerate(v[0:6], start=1):
        ph = np.exp(1j * (kk @ d))
        if deriv >= 0:
            ph = 1j * d[deriv] * ph
        # CONVENTION (source, eq. 9 / App. A, where the matrices are 5x3): the M-X block is
        # metal-index-first, so the p-d block enters TRANSPOSED and carries the phase there.
        blk = ph * _sk_pd(d, Vpds, Vpdp).T
        # CONVENTION (source, Fig. 2 / eq. 9): the six M-X bonds alternate between the two
        # chalcogen planes, and in the listing order of step 1 the FIRST one reaches the top
        # plane. (Getting this backwards is the same error as the orientation above.)
        if i % 2 == 1:
            top += blk
        else:
            bot += blk
    H[3:8, 0:3] += top
    H[0:3, 3:8] += top.conj().T
    H[3:8, 8:11] += bot
    H[8:11, 3:8] += bot.conj().T

    tb = np.zeros((3, 3), dtype=complex)
    for d in v[12:15]:
        ph = np.exp(1j * (kk @ d))
        if deriv >= 0:
            ph = 1j * d[deriv] * ph
        # CONVENTION (source, App. B, sentence introducing r0): the coupling ACROSS the metal
        # plane uses the NEAREST-neighbour Vpp integrals, not the next-nearest Kpp pair.
        tb += ph * _sk_pp(d, Vpps, Vppp)
    H[0:3, 8:11] += tb
    H[8:11, 0:3] += tb.conj().T
    return H

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def brillouin_zone(a: float) -> "np.ndarray":
    if a <= 0:
        raise ValueError("lattice constant must be positive")
    r3 = np.sqrt(3.0)
    b1 = 2 * np.pi / a * np.array([1.0, -1.0 / r3, 0.0])
    b2 = 2 * np.pi / a * np.array([0.0, 2.0 / r3, 0.0])
    # CONVENTION: |M| = 2 pi / (sqrt(3) a) and |K| = 4 pi / (3 a).
    return np.vstack([b1, b2, np.zeros(3), 0.5 * b1, (2.0 * b1 + b2) / 3.0])

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def indirect_gap(vectors: "np.ndarray", reciprocal: "np.ndarray", params: "np.ndarray", n_grid: int, n_occupied: int) -> float:
    r = np.asarray(reciprocal, dtype=float)
    if r.shape != (5, 3):
        raise ValueError("reciprocal must have shape (5, 3)")
    if n_grid < 1:
        raise ValueError("n_grid must be a positive integer")
    if not (0 < n_occupied < 11):
        raise ValueError("n_occupied must lie strictly between 0 and 11")
    b1, b2 = r[0], r[1]
    top, bot = -np.inf, np.inf
    for i in range(n_grid):
        for j in range(n_grid):
            e = np.linalg.eigvalsh(bloch_hamiltonian(
                (i / n_grid) * b1 + (j / n_grid) * b2, vectors, params, -1))
            # extrema taken over the WHOLE grid independently: the gap is the INDIRECT one.
            top = max(top, e[n_occupied - 1])
            bot = min(bot, e[n_occupied])
    return bot - top

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def effective_mass(vectors: "np.ndarray", params: "np.ndarray", band: int, k_extremum: "np.ndarray", direction: "np.ndarray", dk: float, n_points: int) -> float:
    u = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(u)
    if u.shape != (3,) or nrm == 0.0:
        raise ValueError("direction must be a non-zero vector of length 3")
    if dk <= 0:
        raise ValueError("dk must be positive")
    if n_points < 3 or n_points % 2 == 0:
        raise ValueError("n_points must be an odd integer of at least 3")
    k0 = np.asarray(k_extremum, dtype=float)
    if k0.shape != (3,):
        raise ValueError("k_extremum must be a vector of length 3")
    u = u / nrm
    half = (n_points - 1) // 2
    t = dk * np.arange(-half, half + 1, dtype=float)
    e = np.array([np.linalg.eigvalsh(
        bloch_hamiltonian(k0 + s * u, vectors, params, -1))[band] for s in t])
    # symmetric nodes decouple the linear term, so the least-squares quadratic coefficient
    # is closed form; using it avoids BLAS-dependent drift from a least-squares solve.
    n = float(n_points)
    s2, s4 = (t ** 2).sum(), (t ** 4).sum()
    c2 = (n * (t ** 2 * e).sum() - s2 * e.sum()) / (n * s4 - s2 * s2)
    if c2 == 0.0:
        raise ValueError("the band is flat along this direction; no parabolic mass")
    # CONVENTION (source, Table V): masses quoted POSITIVE at both band edges.
    hbar2_over_me = 7.6199682  # hbar^2 / m_e in eV Angstrom^2
    return abs(hbar2_over_me / (2.0 * c2))

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def interband_transitions(vectors: "np.ndarray", reciprocal: "np.ndarray", params: "np.ndarray", n_grid: int, n_occupied: int, axis: int) -> "np.ndarray":
    r = np.asarray(reciprocal, dtype=float)
    if r.shape != (5, 3):
        raise ValueError("reciprocal must have shape (5, 3)")
    if n_grid < 1:
        raise ValueError("n_grid must be a positive integer")
    if not (0 < n_occupied < 11):
        raise ValueError("n_occupied must lie strictly between 0 and 11")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    b1, b2 = r[0], r[1]
    out = []
    for i in range(n_grid):
        for j in range(n_grid):
            k = (i / n_grid) * b1 + (j / n_grid) * b2
            w, u = np.linalg.eigh(bloch_hamiltonian(k, vectors, params, -1))
            vel = u.conj().T @ bloch_hamiltonian(k, vectors, params, axis) @ u
            # CONVENTION (source, eq. 17): only OCCUPIED -> EMPTY pairs appear; pairs inside
            # the occupied manifold destroy the gauge invariance.
            for n in range(n_occupied):
                for m in range(n_occupied, 11):
                    out.append((w[m] - w[n], abs(vel[n, m]) ** 2))
    return np.array(out, dtype=float)

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def quantum_metric(transitions: "np.ndarray", n_grid: int) -> float:
    t = np.asarray(transitions, dtype=float)
    if t.ndim != 2 or t.shape[1] != 2 or t.shape[0] == 0:
        raise ValueError("transitions must be a non-empty array of shape (N, 2)")
    if n_grid < 1:
        raise ValueError("n_grid must be a positive integer")
    if np.any(t[:, 0] <= 0.0):
        raise ValueError("every transition energy must be positive")
    # eq. (16): the sum over every listed transition divided by the NUMBER OF K POINTS --
    # not by the number of transitions, and not by the zone area.
    return float((t[:, 1] / t[:, 0] ** 2).sum() / (n_grid * n_grid))

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def optical_conductivity(omega: "np.ndarray", transitions: "np.ndarray", n_grid: int, eta: float) -> "np.ndarray":
    w = np.atleast_1d(np.asarray(omega, dtype=float))
    t = np.asarray(transitions, dtype=float)
    if t.ndim != 2 or t.shape[1] != 2 or t.shape[0] == 0:
        raise ValueError("transitions must be a non-empty array of shape (N, 2)")
    if n_grid < 1:
        raise ValueError("n_grid must be a positive integer")
    if eta <= 0:
        raise ValueError("the broadening must be positive")
    if np.any(t[:, 0] <= 0.0):
        raise ValueError("every transition energy must be positive")
    de = t[:, 0][:, None]
    sq = t[:, 1][:, None]
    out = np.zeros(w.shape[0])
    # CONVENTION (source, eq. 13): |v|^2 divided by the transition's OWN energy, not by the
    # photon energy; the delta becomes a normalised LORENTZIAN of half width eta.
    for s in range(0, de.shape[0], 2048):
        d, q = de[s:s + 2048], sq[s:s + 2048]
        out += ((q / d) * (eta / np.pi) / ((w[None, :] - d) ** 2 + eta ** 2)).sum(0)
    return np.pi * out / (n_grid * n_grid)

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def spectral_weight(omega: "np.ndarray", conductivity: "np.ndarray") -> float:
    w = np.asarray(omega, dtype=float)
    s = np.asarray(conductivity, dtype=float)
    if w.ndim != 1 or w.shape != s.shape or w.shape[0] < 2:
        raise ValueError("omega and conductivity must be matching 1-D arrays of at least two points")
    if np.any(w <= 0.0):
        raise ValueError("every photon energy must be positive")
    # CONVENTION: the window starts at a POSITIVE omega_min. A Lorentzian does not vanish at
    # zero frequency, so Re sigma / omega has a 1/omega tail and an integral from zero
    # diverges logarithmically with the grid; cutting below the onset removes the artefact.
    y = s / w
    # trapezoidal rule written out: np.trapezoid exists only from numpy 2.0 and np.trapz was
    # removed in that release, so neither name runs on both sides of the boundary.
    return float(np.sum(np.diff(w) * (y[1:] + y[:-1]) / 2.0))

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def tb_optics_report(a: float, theta: float, params: "np.ndarray", n_grid: int, n_occupied: int, eta: float,
                             omega_min: float, omega_max: float, n_omega: int, dk: float, n_points: int) -> "np.ndarray":
    if not (0.0 < omega_min < omega_max):
        raise ValueError("the frequency window must satisfy 0 < omega_min < omega_max")
    if n_omega < 2:
        raise ValueError("n_omega must be at least 2")
    # built once here and passed down, so every later step sees the same geometry and the
    # same zone rather than rebuilding them from a and theta.
    vec = hopping_vectors(a, theta)
    bz = brillouin_zone(a)
    G, M, K = bz[2], bz[3], bz[4]
    gap = indirect_gap(vec, bz, params, n_grid, n_occupied)
    # the lowest VERTICAL transition, which is the absorption edge and therefore what fixes
    # where the frequency window below may honestly begin.
    e0 = np.linalg.eigvalsh(bloch_hamiltonian(G, vec, params, -1))
    direct = e0[n_occupied] - e0[n_occupied - 1]
    mv1 = effective_mass(vec, params, n_occupied - 1, G, M - G, dk, n_points)
    mv2 = effective_mass(vec, params, n_occupied - 1, G, K - G, dk, n_points)
    mc1 = effective_mass(vec, params, n_occupied, M, G - M, dk, n_points)
    mc2 = effective_mass(vec, params, n_occupied, M, K - M, dk, n_points)
    # CONVENTION (source, Table V): the dos mass is the GEOMETRIC mean of the two principal
    # masses, not the arithmetic one.
    mdos = np.sqrt(mc1 * mc2)
    tr = interband_transitions(vec, bz, params, n_grid, n_occupied, 0)
    piG = np.pi * quantum_metric(tr, n_grid)
    w = np.linspace(omega_min, omega_max, n_omega)
    sig = optical_conductivity(w, tr, n_grid, eta)
    wopt = spectral_weight(w, sig)
    if piG <= 0.0:
        raise ValueError("the quantum metric vanished; the sum rule cannot be tested")
    return np.array([gap, direct, mv1, mv2, mc1, mc2, mdos, piG, wopt,
                     100.0 * (wopt - piG) / piG])
SCICODE_GOLD_EOF
