#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def pp_occupations_and_orbital_energies(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
    w_a: float,
    w_b: float,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    if len(h) != 4 or any(len(row) != 4 for row in h):
        raise ValueError("h must be a 4x4 table of one-electron integrals.")
    if len(v) != 4:
        raise ValueError("v must be a 4x4x4x4 table of two-electron integrals.")

    gaps = (w_a, w_b)
    eta = [(gaps[a] * gaps[a] + 1.0) ** 0.5 for a in (0, 1)]
    n = [0.0] * 4
    for a in (0, 1):
        for mu in (0, 1):
            n[2 * a + mu] = 1.0 + ((-1.0) ** mu) * gaps[a] / eta[a]

    bigj = [[v[p][p][q][q] for q in range(4)] for p in range(4)]
    bigk = [[v[p][q][q][p] for q in range(4)] for p in range(4)]
    bigl = [[v[p][q][p][q] for q in range(4)] for p in range(4)]
    bigg = [[2.0 * bigj[p][q] - bigk[p][q] for q in range(4)] for p in range(4)]

    eps = [0.0] * 4
    for a in (0, 1):
        for mu in (0, 1):
            p = 2 * a + mu
            total = h[p][p] + 0.5 * bigl[p][p]
            for g in (0, 1):
                if g == a:
                    continue
                for lam in (0, 1):
                    q = 2 * g + lam
                    total += 0.5 * bigg[p][q] * n[q]
            eps[p] = total
    return (eta[0], eta[1], n[0], n[1], n[2], n[3], eps[0], eps[1], eps[2], eps[3])

def pp_reference_energy(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
    w_a: float,
    w_b: float,
) -> tuple[float, float, float]:
    packed = pp_occupations_and_orbital_energies(h, v, w_a, w_b)
    eta = [packed[0], packed[1]]
    n = list(packed[2:6])
    eps = list(packed[6:10])

    bigj = [[v[p][p][q][q] for q in range(4)] for p in range(4)]
    bigk = [[v[p][q][q][p] for q in range(4)] for p in range(4)]
    bigl = [[v[p][q][p][q] for q in range(4)] for p in range(4)]
    bigg = [[2.0 * bigj[p][q] - bigk[p][q] for q in range(4)] for p in range(4)]

    energy = 0.0
    for p in range(4):
        energy += eps[p] * n[p]
    for a in (0, 1):
        energy -= bigl[2 * a][2 * a + 1] * (n[2 * a] * n[2 * a + 1]) ** 0.5
    for p in range(4):
        for q in range(p + 1, 4):
            if p // 2 == q // 2:
                continue
            energy -= 0.5 * bigg[p][q] * n[p] * n[q]

    gaps = (w_a, w_b)
    grad = []
    for a in (0, 1):
        numerator = gaps[a] * bigl[2 * a][2 * a + 1] + eps[2 * a] - eps[2 * a + 1]
        grad.append(numerator / eta[a] ** 3)
    return (energy, grad[0], grad[1])

def stationary_vbs_gaps(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, float, float]:
    if len(h) != 4 or len(v) != 4:
        raise ValueError("h and v must have the four-orbital shape.")
    bigl = [[v[p][q][p][q] for q in range(4)] for p in range(4)]

    gaps = [0.0, 0.0]
    for _ in range(500):
        packed = pp_occupations_and_orbital_energies(h, v, gaps[0], gaps[1])
        eps = list(packed[6:10])
        fresh = [(eps[2 * a + 1] - eps[2 * a]) / bigl[2 * a][2 * a + 1] for a in (0, 1)]
        shift = abs(fresh[0] - gaps[0]) + abs(fresh[1] - gaps[1])
        gaps = [0.5 * (gaps[0] + fresh[0]), 0.5 * (gaps[1] + fresh[1])]
        if shift < 1.0e-15:
            break
    for _ in range(200):
        packed = pp_occupations_and_orbital_energies(h, v, gaps[0], gaps[1])
        eps = list(packed[6:10])
        fresh = [(eps[2 * a + 1] - eps[2 * a]) / bigl[2 * a][2 * a + 1] for a in (0, 1)]
        shift = abs(fresh[0] - gaps[0]) + abs(fresh[1] - gaps[1])
        gaps = fresh
        if shift < 1.0e-16:
            break

    energy = pp_reference_energy(h, v, gaps[0], gaps[1])[0]
    return (gaps[0], gaps[1], energy)

def generalized_fock_elements(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, ...]:
    w_a, w_b, _energy = stationary_vbs_gaps(h, v)
    packed = pp_occupations_and_orbital_energies(h, v, w_a, w_b)
    eta = [packed[0], packed[1]]
    n = list(packed[2:6])

    direct = [[0.0] * 4 for _ in range(4)]
    pair = [[0.0] * 4 for _ in range(4)]
    for p in range(4):
        for q in range(4):
            if p != q and p // 2 != q // 2:
                direct[p][q] = n[p] * n[q]
    for p in range(4):
        pair[p][p] = n[p]
    for a in (0, 1):
        pair[2 * a][2 * a + 1] = -1.0 / eta[a]
        pair[2 * a + 1][2 * a] = -1.0 / eta[a]

    fock = [[0.0] * 4 for _ in range(4)]
    for p in range(4):
        for q in range(4):
            total = h[p][q] * n[p]
            for r in range(4):
                total += 0.5 * (2.0 * v[r][r][p][q] - v[r][q][p][r]) * direct[r][p]
                total += v[r][p][r][q] * pair[r][p]
            fock[p][q] = total
    return tuple(fock[p][q] for p in range(4) for q in range(4))

def pairing_intermediates(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, ...]:
    w_a, w_b, _energy = stationary_vbs_gaps(h, v)
    packed = pp_occupations_and_orbital_energies(h, v, w_a, w_b)
    eta = [packed[0], packed[1]]
    gaps = [w_a, w_b]

    bigj = [[v[p][p][q][q] for q in range(4)] for p in range(4)]
    bigk = [[v[p][q][q][p] for q in range(4)] for p in range(4)]
    bigl = [[v[p][q][p][q] for q in range(4)] for p in range(4)]
    bigg = [[2.0 * bigj[p][q] - bigk[p][q] for q in range(4)] for p in range(4)]

    t_same = []
    for p, q in ((0, 1), (2, 3)):
        t_same.append(bigj[p][q] + bigk[p][q] - 0.5 * bigl[p][p] - 0.5 * bigl[q][q])
    t_cross = []
    for p, q in ((0, 2), (0, 3), (1, 2), (1, 3)):
        t_cross.append(bigj[p][q] + bigk[p][q] - 0.5 * bigl[p][p] - 0.5 * bigl[q][q]
                       - 0.5 * bigg[p][q])

    g_dd = 0.0
    for mu in (0, 1):
        for nu in (0, 1):
            g_dd += ((-1.0) ** (mu + nu)) * bigg[mu][2 + nu]
    g_dd *= 0.5 * (gaps[0] / eta[0]) * (gaps[1] / eta[1])

    g_plain = 0.5 * (bigg[0][2] + bigg[0][3] + bigg[1][2] + bigg[1][3])
    g_single = []
    for p in (2, 3):
        g_single.append(0.5 * (gaps[0] / eta[0]) * (bigg[0][p] - bigg[1][p]))
    for p in (0, 1):
        g_single.append(0.5 * (gaps[1] / eta[1]) * (bigg[2][p] - bigg[3][p]))
    return (t_same[0], t_same[1], t_cross[0], t_cross[1], t_cross[2], t_cross[3],
            g_dd, g_plain, g_single[0], g_single[1], g_single[2], g_single[3])

def electron_transfer_channel(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
    mu: int,
    nu: int,
) -> tuple[float, float, float]:
    if mu not in (0, 1) or nu not in (0, 1):
        raise ValueError("mu and nu must each be 0 or 1.")
    w_a, w_b, e_ref = stationary_vbs_gaps(h, v)
    packed = pp_occupations_and_orbital_energies(h, v, w_a, w_b)
    eta = [packed[0], packed[1]]
    n = list(packed[2:6])
    eps = list(packed[6:10])
    fock_flat = generalized_fock_elements(h, v)
    fock = [[fock_flat[4 * p + q] for q in range(4)] for p in range(4)]
    inter = pairing_intermediates(h, v)

    bigj = [[v[p][p][q][q] for q in range(4)] for p in range(4)]
    bigk = [[v[p][q][q][p] for q in range(4)] for p in range(4)]
    bigl = [[v[p][q][p][q] for q in range(4)] for p in range(4)]
    bigg = [[2.0 * bigj[p][q] - bigk[p][q] for q in range(4)] for p in range(4)]
    gaps = [w_a, w_b]

    am = mu
    bn = 2 + nu
    am1 = 1 - mu
    bn1 = 2 + (1 - nu)
    t_cross = inter[2 + 2 * mu + nu]
    g_dd = inter[6]
    g_a_at_bn1 = inter[8 + (1 - nu)]
    g_b_at_am1 = inter[10 + (1 - mu)]

    d_ab = (eta[0] * bigl[0][1] + eta[1] * bigl[2][3] + t_cross
            + eps[bn1] - eps[am1] + bigg[2][3]
            + g_dd - g_a_at_bn1 + g_b_at_am1 - 0.5 * bigg[am1][bn1])
    d_ba = (eta[1] * bigl[2][3] + eta[0] * bigl[0][1] + t_cross
            + eps[am1] - eps[bn1] + bigg[0][1]
            + g_dd - g_b_at_am1 + g_a_at_bn1 - 0.5 * bigg[bn1][am1])

    norm = 1.0 + ((-1.0) ** (mu + nu + 1)) * (gaps[0] * gaps[1]) / (eta[0] * eta[1])
    twice = (n[am1] * n[bn] * (e_ref + d_ab) + n[am] * n[bn1] * (e_ref + d_ba)
             + (1.0 / (eta[0] * eta[1])) * (bigl[am1][bn1] + bigl[bn1][am1]))
    energy_plus = (0.5 * twice) / norm

    raw = (fock[am][bn] / n[am] + fock[bn][am] / n[bn]
           + 0.5 * (2.0 * v[am1][am1][bn][am] - v[am1][am][bn][am1]
                    - v[am][am][bn][am] + 2.0 * eta[0] * v[am][am1][bn][am1]) * n[am]
           + 0.5 * (2.0 * v[bn1][bn1][am][bn] - v[bn1][bn][am][bn1]
                    - v[bn][bn][am][bn] + 2.0 * eta[1] * v[bn][bn1][am][bn1]) * n[bn])
    coupling_unnormalized = raw / (2.0 * ((-1.0) ** (mu + nu + 1)) * eta[0] * eta[1])
    coupling_squared = coupling_unnormalized * coupling_unnormalized / norm
    return (norm, energy_plus - e_ref, coupling_squared)

def remaining_valence_channels(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> tuple[float, float, float]:
    w_a, w_b, _e_ref = stationary_vbs_gaps(h, v)
    packed = pp_occupations_and_orbital_energies(h, v, w_a, w_b)
    eta = [packed[0], packed[1]]
    n = list(packed[2:6])
    eps = list(packed[6:10])

    bigj = [[v[p][p][q][q] for q in range(4)] for p in range(4)]
    bigk = [[v[p][q][q][p] for q in range(4)] for p in range(4)]
    bigl = [[v[p][q][p][q] for q in range(4)] for p in range(4)]
    bigg = [[2.0 * bigj[p][q] - bigk[p][q] for q in range(4)] for p in range(4)]

    inter = pairing_intermediates(h, v)
    t_same = [inter[0], inter[1]]
    g_dd = inter[6]
    g_plain = inter[7]
    gs_tab = [[0.0, 0.0, inter[8], inter[9]], [inter[10], inter[11], 0.0, 0.0]]

    sen0 = 0.0
    sen2 = 0.0
    sen4 = 0.0

    # seniority zero: the double swap
    coupling = 0.0
    for mm in (0, 1):
        for nn in (0, 1):
            coupling += ((-1.0) ** (mm + nn)) * bigg[mm][2 + nn]
    coupling *= 0.5 / (eta[0] * eta[1])
    denom = 2.0 * eta[0] * bigl[0][1] + 2.0 * eta[1] * bigl[2][3] + 4.0 * g_dd
    sen0 += -(coupling * coupling) / denom

    # seniority zero: transfer of a whole pair between the subsystems
    for a, b in ((0, 1), (1, 0)):
        denom = (eta[a] * bigl[2 * a][2 * a + 1] + eta[b] * bigl[2 * b][2 * b + 1]
                 + eps[2 * b] + eps[2 * b + 1] - eps[2 * a] - eps[2 * a + 1]
                 + 2.0 * bigg[2 * b][2 * b + 1]
                 - g_plain + g_dd
                 - gs_tab[a][2 * b] - gs_tab[a][2 * b + 1]
                 + gs_tab[b][2 * a] + gs_tab[b][2 * a + 1])
        coupling = 0.0
        for mm in (0, 1):
            for nn in (0, 1):
                coupling += (((-1.0) ** (mm + nn + 1)) * bigl[2 * a + mm][2 * b + nn]
                             * (n[2 * a + mm] * n[2 * b + (1 - nn)]) ** 0.5)
        coupling *= 0.5
        sen0 += -(coupling * coupling) / denom

    # seniority two: a swap in one subsystem together with a split in the other
    for a, b in ((0, 1), (1, 0)):
        denom = (2.0 * eta[a] * bigl[2 * a][2 * a + 1]
                 + eta[b] * bigl[2 * b][2 * b + 1]
                 + t_same[b] + 2.0 * g_dd)
        prefactor = (n[2 * b] ** 0.5 - n[2 * b + 1] ** 0.5) / (2.0 * eta[a])
        coupling = 0.0
        for mm in (0, 1):
            coupling += ((-1.0) ** mm) * (
                2.0 * v[2 * a + mm][2 * a + mm][2 * b][2 * b + 1]
                - v[2 * a + mm][2 * b + 1][2 * b][2 * a + mm])
        coupling *= prefactor
        sen2 += -(coupling * coupling) / denom

    # seniority four: both subsystems left open
    denom4 = (eta[0] * bigl[0][1] + eta[1] * bigl[2][3]
              + t_same[0] + t_same[1] + g_dd)
    coupling = ((n[0] ** 0.5 - n[1] ** 0.5) * (n[2] ** 0.5 - n[3] ** 0.5) * v[0][1][2][3]
                - 0.5 * ((n[0] * n[2]) ** 0.5 + (n[1] * n[3]) ** 0.5) * v[0][3][2][1]
                + 0.5 * ((n[1] * n[2]) ** 0.5 + (n[0] * n[3]) ** 0.5) * v[1][3][2][0])
    sen4 += -(coupling * coupling) / denom4

    denom4b = (denom4 + bigk[0][2] + bigk[0][3] + bigk[1][2] + bigk[1][3]
               - 2.0 * bigk[0][1] - 2.0 * bigk[2][3])
    root3 = 3.0 ** 0.5
    coupling = (-(root3 / 2.0) * ((n[0] * n[2]) ** 0.5 + (n[1] * n[3]) ** 0.5) * v[0][3][2][1]
                - (root3 / 2.0) * ((n[1] * n[2]) ** 0.5 + (n[0] * n[3]) ** 0.5) * v[1][3][2][0])
    sen4 += -(coupling * coupling) / denom4b
    return (sen0 * 1000.0, sen2 * 1000.0, sen4 * 1000.0)

def en2_valence_correction(
    h: list[list[float]],
    v: list[list[list[list[float]]]],
) -> float:
    total = 0.0
    for mu in (0, 1):
        for nu in (0, 1):
            _norm, energy, coupling_squared = electron_transfer_channel(h, v, mu, nu)
            total += -1000.0 * coupling_squared / energy
    sen0, sen2, sen4 = remaining_valence_channels(h, v)
    total += sen0 + sen2 + sen4
    return round(total, 3)
SCICODE_GOLD_EOF
