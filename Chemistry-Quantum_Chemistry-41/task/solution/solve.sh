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
from scipy.special import gamma


def normalize_contraction(l: int, exponents: "np.ndarray", coefficients: "np.ndarray") -> "np.ndarray":
    if int(l) != l or l < 0:
        raise ValueError("l must be a non-negative integer")
    l = int(l)
    beta = np.asarray(exponents, dtype=float).ravel()
    coef = np.asarray(coefficients, dtype=float).ravel()
    if beta.size == 0 or beta.size != coef.size:
        raise ValueError("exponents and coefficients must be non-empty and of equal length")
    if np.any(beta <= 0.0):
        raise ValueError("exponents must be positive")
    if not np.any(coef != 0.0):
        raise ValueError("at least one contraction coefficient must be non-zero")
    g = gamma(l + 1.5)
    prim_norm = np.sqrt(2.0 * (2.0 * beta) ** (l + 1.5) / g)
    weights = coef * prim_norm
    radial_overlap = g / (2.0 * (beta[:, None] + beta[None, :]) ** (l + 1.5))
    return weights / np.sqrt(weights @ radial_overlap @ weights)

import numpy as np
from scipy.special import gamma, gammainc

def _solid_harmonics(l: int) -> "np.ndarray":
    """Solid-harmonic polynomials (rows) on the Cartesian monomials of _cartesian_list(l)."""
    if l == 0:
        return np.array([[1.0]])
    if l == 1:
        return np.eye(3)
    return np.array([
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    ])


def _cartesian_list(l: int) -> list:
    """Cartesian exponent triples (xx, xy, xz, yy, yz, zz order for l = 2)."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def _hermite_indices(L: int) -> list:
    return [(t, u, v) for t in range(L + 1) for u in range(L + 1 - t) for v in range(L + 1 - t - u)]


def _boys(n_max: int, t: "np.ndarray") -> "np.ndarray":
    """Boys functions F_0..F_n_max at the points t."""
    t = np.asarray(t, dtype=float)
    small = t < 1e-12
    t_safe = np.where(small, 1.0, t)
    out = np.empty((n_max + 1,) + t.shape)
    for n in range(n_max + 1):
        exact = gamma(n + 0.5) * gammainc(n + 0.5, t_safe) / (2.0 * t_safe ** (n + 0.5))
        out[n] = np.where(small, 1.0 / (2 * n + 1) - t / (2 * n + 3), exact)
    return out


def _hermite_1d(la: int, lb: int, x_ab: float, a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """E[i, j, t] (i <= la, j <= lb, t <= i + j) of a 1-D Gaussian product, per primitive pair."""
    p = a + b
    x_pa = -b / p * x_ab
    x_pb = a / p * x_ab
    e = np.zeros((la + 1, lb + 1, la + lb + 2) + a.shape)
    e[0, 0, 0] = np.exp(-a * b / p * x_ab * x_ab)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            for t in range(i + j + 1):
                if i > 0:
                    val = x_pa * e[i - 1, j, t] + (t + 1) * e[i - 1, j, t + 1]
                    if t > 0:
                        val = val + e[i - 1, j, t - 1] / (2.0 * p)
                else:
                    val = x_pb * e[i, j - 1, t] + (t + 1) * e[i, j - 1, t + 1]
                    if t > 0:
                        val = val + e[i, j - 1, t - 1] / (2.0 * p)
                e[i, j, t] = val
    return e[:, :, : la + lb + 1]


def _hermite_coulomb(L: int, alpha: "np.ndarray", x: "np.ndarray", y: "np.ndarray", z: "np.ndarray") -> dict:
    """Hermite Coulomb integrals R_{tuv}, t + u + v <= L, broadcast over the batch."""
    boys = _boys(L, alpha * (x * x + y * y + z * z))
    r = {(0, 0, 0, n): (-2.0 * alpha) ** n * boys[n] for n in range(L + 1)}
    for total in range(1, L + 1):
        for (t, u, v) in _hermite_indices(total):
            if t + u + v != total:
                continue
            for n in range(L - total + 1):
                if t > 0:
                    val = x * r[(t - 1, u, v, n + 1)]
                    if t > 1:
                        val = val + (t - 1) * r[(t - 2, u, v, n + 1)]
                elif u > 0:
                    val = y * r[(t, u - 1, v, n + 1)]
                    if u > 1:
                        val = val + (u - 1) * r[(t, u - 2, v, n + 1)]
                else:
                    val = z * r[(t, u, v - 1, n + 1)]
                    if v > 1:
                        val = val + (v - 1) * r[(t, u, v - 2, n + 1)]
                r[(t, u, v, n)] = val
    return {key[:3]: val for key, val in r.items() if key[3] == 0}


def _prepare_shells(coords: "np.ndarray", shells: list) -> list:
    prepared = []
    for shell in shells:
        atom, l, exponents, coefficients = shell
        if l not in (0, 1, 2):
            raise ValueError("only s, p and d shells (l = 0, 1, 2) are supported")
        if not (0 <= int(atom) < len(coords)) or int(atom) != atom:
            raise ValueError("shell refers to a missing atom")
        weights = normalize_contraction(l, exponents, coefficients)
        exponents = np.asarray(exponents, dtype=float).ravel()
        keep = weights != 0.0
        prepared.append({"centre": coords[int(atom)], "l": int(l), "exps": exponents[keep],
                         "w": weights[keep], "cart": _cartesian_list(int(l))})
    return prepared


def _pair_data(sa: dict, sb: dict) -> dict:
    a = np.repeat(sa["exps"], sb["exps"].size)
    b = np.tile(sb["exps"], sa["exps"].size)
    w = np.repeat(sa["w"], sb["exps"].size) * np.tile(sb["w"], sa["exps"].size)
    p = a + b
    centre_p = (a[:, None] * sa["centre"] + b[:, None] * sb["centre"]) / p[:, None]
    ab = sa["centre"] - sb["centre"]
    herm = [_hermite_1d(sa["l"], sb["l"] + 2, ab[k], a, b) for k in range(3)]
    L = sa["l"] + sb["l"]
    hidx = _hermite_indices(L)
    dens = np.zeros((len(sa["cart"]), len(sb["cart"]), len(hidx), p.size))
    for ci, (ix, iy, iz) in enumerate(sa["cart"]):
        for cj, (jx, jy, jz) in enumerate(sb["cart"]):
            for h, (t, u, v) in enumerate(hidx):
                if t <= ix + jx and u <= iy + jy and v <= iz + jz:
                    dens[ci, cj, h] = w * herm[0][ix, jx, t] * herm[1][iy, jy, u] * herm[2][iz, jz, v]
    return {"b": b, "w": w, "p": p, "P": centre_p, "herm": herm, "dens": dens, "hidx": hidx, "L": L}


def _one_electron_block(sa: dict, sb: dict, pair: dict, charges: "np.ndarray", coords: "np.ndarray") -> tuple:
    b, w, p, herm = pair["b"], pair["w"], pair["p"], pair["herm"]
    pref = (np.pi / p) ** 1.5

    def _s1(k, i, j):
        return herm[k][i, j, 0] if j >= 0 else np.zeros_like(p)

    def _t1(k, i, j):
        val = b * (2 * j + 1) * _s1(k, i, j) - 2.0 * b * b * _s1(k, i, j + 2)
        if j >= 2:
            val = val - 0.5 * j * (j - 1) * _s1(k, i, j - 2)
        return val

    na, nb = len(sa["cart"]), len(sb["cart"])
    s_blk = np.zeros((na, nb))
    t_blk = np.zeros((na, nb))
    for ci, lmn_a in enumerate(sa["cart"]):
        for cj, lmn_b in enumerate(sb["cart"]):
            sx, sy, sz = (_s1(k, lmn_a[k], lmn_b[k]) for k in range(3))
            s_blk[ci, cj] = np.sum(w * pref * sx * sy * sz)
            t_blk[ci, cj] = np.sum(w * pref * (_t1(0, lmn_a[0], lmn_b[0]) * sy * sz
                                                + sx * _t1(1, lmn_a[1], lmn_b[1]) * sz
                                                + sx * sy * _t1(2, lmn_a[2], lmn_b[2])))
    v_blk = np.zeros((na, nb))
    for z_c, centre_c in zip(charges, coords):
        pc = pair["P"] - centre_c
        r = _hermite_coulomb(pair["L"], p, pc[:, 0], pc[:, 1], pc[:, 2])
        r_mat = np.array([r[key] for key in pair["hidx"]])
        v_blk -= z_c * np.einsum("abhp,hp,p->ab", pair["dens"], r_mat, 2.0 * np.pi / p)
    return s_blk, t_blk + v_blk


def _eri_block(bra: dict, ket: dict) -> "np.ndarray":
    p = bra["p"][:, None]
    q = ket["p"][None, :]
    alpha = p * q / (p + q)
    pq = bra["P"][:, None, :] - ket["P"][None, :, :]
    r = _hermite_coulomb(bra["L"] + ket["L"], alpha, pq[..., 0], pq[..., 1], pq[..., 2])
    coupling = np.empty((len(bra["hidx"]), len(ket["hidx"])) + alpha.shape)
    for x, (t, u, v) in enumerate(bra["hidx"]):
        for y, (tt, uu, vv) in enumerate(ket["hidx"]):
            coupling[x, y] = (-1) ** (tt + uu + vv) * r[(t + tt, u + uu, v + vv)]
    coupling *= 2.0 * np.pi ** 2.5 / (p * q * np.sqrt(p + q))
    return np.einsum("abhp,hgpq,cdgq->abcd", bra["dens"], coupling, ket["dens"], optimize=True)


def compute_ao_integrals(charges: "np.ndarray", coords: "np.ndarray", shells: list) -> tuple:
    charges = np.asarray(charges, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    if charges.size != coords.shape[0]:
        raise ValueError("charges and coords must describe the same atoms")
    e_nuc = 0.0
    for i in range(charges.size):
        for j in range(i):
            distance = np.linalg.norm(coords[i] - coords[j])
            if distance == 0.0:
                raise ValueError("two nuclei coincide")
            e_nuc += charges[i] * charges[j] / distance
    prepared = _prepare_shells(coords, shells)
    offsets = np.cumsum([0] + [len(sh["cart"]) for sh in prepared])
    n_cart = offsets[-1]
    overlap = np.zeros((n_cart, n_cart))
    core = np.zeros((n_cart, n_cart))
    pairs = {}
    for i in range(len(prepared)):
        for j in range(i + 1):
            pair = _pair_data(prepared[i], prepared[j])
            pairs[(i, j)] = pair
            s_blk, h_blk = _one_electron_block(prepared[i], prepared[j], pair, charges, coords)
            rows, cols = slice(offsets[i], offsets[i + 1]), slice(offsets[j], offsets[j + 1])
            overlap[rows, cols], overlap[cols, rows] = s_blk, s_blk.T
            core[rows, cols], core[cols, rows] = h_blk, h_blk.T
    eri = np.zeros((n_cart, n_cart, n_cart, n_cart))
    keys = sorted(pairs)
    for n1, (i, j) in enumerate(keys):
        for (k, l) in keys[: n1 + 1]:
            blk = _eri_block(pairs[(i, j)], pairs[(k, l)])
            si, sj, sk, sl = (slice(offsets[x], offsets[x + 1]) for x in (i, j, k, l))
            eri[si, sj, sk, sl] = blk
            eri[sj, si, sk, sl] = blk.transpose(1, 0, 2, 3)
            eri[si, sj, sl, sk] = blk.transpose(0, 1, 3, 2)
            eri[sj, si, sl, sk] = blk.transpose(1, 0, 3, 2)
            eri[sk, sl, si, sj] = blk.transpose(2, 3, 0, 1)
            eri[sl, sk, si, sj] = blk.transpose(3, 2, 0, 1)
            eri[sk, sl, sj, si] = blk.transpose(2, 3, 1, 0)
            eri[sl, sk, sj, si] = blk.transpose(3, 2, 1, 0)
    n_sph = sum(2 * sh["l"] + 1 for sh in prepared)
    to_sph = np.zeros((n_cart, n_sph))
    col = 0
    for x, sh in enumerate(prepared):
        block = _solid_harmonics(sh["l"])
        to_sph[offsets[x]:offsets[x + 1], col:col + block.shape[0]] = block.T
        col += block.shape[0]
    to_sph = to_sph / np.sqrt(np.diag(to_sph.T @ overlap @ to_sph))[None, :]
    overlap_sph = to_sph.T @ overlap @ to_sph
    core_sph = to_sph.T @ core @ to_sph
    eri_sph = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, to_sph, to_sph, to_sph, to_sph, optimize=True)
    return overlap_sph, core_sph, eri_sph, float(e_nuc)

import numpy as np
from scipy.linalg import eigh


def _fock_matrix(core_hamiltonian: "np.ndarray", eri: "np.ndarray", density: "np.ndarray") -> "np.ndarray":
    """Closed-shell Fock matrix h + J[P] - K[P]/2 for the total density matrix P."""
    coulomb = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    exchange = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    return core_hamiltonian + coulomb - 0.5 * exchange


def _diis_extrapolate(history: list) -> "np.ndarray":
    """Pulay DIIS combination of stored (Fock, error) pairs."""
    m = len(history)
    b = -np.ones((m + 1, m + 1))
    b[m, m] = 0.0
    for i in range(m):
        for j in range(m):
            b[i, j] = np.sum(history[i][1] * history[j][1])
    rhs = np.zeros(m + 1)
    rhs[m] = -1.0
    weights = np.linalg.lstsq(b, rhs, rcond=None)[0][:m]
    return sum(w * f for w, (f, _) in zip(weights, history))


def run_rhf(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", e_nuc: float, n_occ: int) -> tuple:
    overlap = np.asarray(overlap, dtype=float)
    core_hamiltonian = np.asarray(core_hamiltonian, dtype=float)
    eri = np.asarray(eri, dtype=float)
    n_bf = overlap.shape[0]
    if overlap.shape != (n_bf, n_bf) or core_hamiltonian.shape != (n_bf, n_bf) or eri.shape != (n_bf,) * 4:
        raise ValueError("inconsistent integral shapes")
    if int(n_occ) != n_occ or not 1 <= n_occ <= n_bf:
        raise ValueError("n_occ must be an integer between 1 and n_bf")
    n_occ = int(n_occ)
    energies, coefficients = eigh(core_hamiltonian, overlap)
    history = []
    e_old = None
    for _ in range(200):
        density = 2.0 * coefficients[:, :n_occ] @ coefficients[:, :n_occ].T
        fock = _fock_matrix(core_hamiltonian, eri, density)
        e_elec = 0.5 * np.sum(density * (core_hamiltonian + fock))
        error = fock @ density @ overlap - overlap @ density @ fock
        if np.abs(error).max() < 1e-10 and e_old is not None and abs(e_elec - e_old) < 1e-12:
            return float(e_elec + e_nuc), energies, coefficients
        e_old = e_elec
        history = (history + [(fock, error)])[-8:]
        energies, coefficients = eigh(_diis_extrapolate(history), overlap)
    raise ValueError("Hartree-Fock iterations did not converge")

import numpy as np


def srg_regulator(delta_p: "np.ndarray", delta_q: "np.ndarray", s: float) -> "np.ndarray":
    s = float(s)
    if not np.isfinite(s) or s < 0.0:
        raise ValueError("the flow parameter s must be finite and non-negative")
    try:
        dp, dq = np.broadcast_arrays(np.asarray(delta_p, dtype=float), np.asarray(delta_q, dtype=float))
    except ValueError as exc:
        raise ValueError("delta_p and delta_q cannot be broadcast together") from exc
    denom = dp * dp + dq * dq
    safe = np.where(denom > 0.0, denom, 1.0)
    factor = (dp + dq) / safe * -np.expm1(-safe * s)
    return np.where(denom > 0.0, factor, 0.0)

import numpy as np


def srg_self_energy(eri_mo: "np.ndarray", orbital_energies: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    eri_mo = np.asarray(eri_mo, dtype=float)
    eps = np.asarray(orbital_energies, dtype=float).ravel()
    n = eps.size
    if eri_mo.shape != (n, n, n, n):
        raise ValueError("eri_mo must have shape (n, n, n, n) matching orbital_energies")
    if int(n_occ) != n_occ or not 1 <= n_occ <= n - 1:
        raise ValueError("n_occ must be an integer between 1 and n - 1")
    if not (np.isfinite(c_ss) and np.isfinite(c_os)):
        raise ValueError("scaling factors must be finite")
    n_occ = int(n_occ)
    occ, vir = slice(0, n_occ), slice(n_occ, n)
    e_occ, e_vir = eps[occ], eps[vir]
    # two-hole-one-particle part: (pi|aj) with D^{pa}_{ij}
    hole = eri_mo[:, occ, vir, occ]
    hole_scaled = (c_ss + c_os) * hole - c_ss * hole.transpose(0, 3, 2, 1)
    d_hole = eps[:, None, None, None] + e_vir[None, None, :, None] - e_occ[None, :, None, None] - e_occ[None, None, None, :]
    f_hole = srg_regulator(d_hole[:, None], d_hole[None, :], s)
    sigma = np.einsum("piaj,pqiaj,qiaj->pq", hole, f_hole, hole_scaled, optimize=True)
    # two-particle-one-hole part: (pa|ib) with D^{pi}_{ab}
    part = eri_mo[:, vir, occ, vir]
    part_scaled = (c_ss + c_os) * part - c_ss * part.transpose(0, 3, 2, 1)
    d_part = eps[:, None, None, None] + e_occ[None, None, :, None] - e_vir[None, :, None, None] - e_vir[None, None, None, :]
    f_part = srg_regulator(d_part[:, None], d_part[None, :], s)
    sigma = sigma + np.einsum("paib,pqaib,qaib->pq", part, f_part, part_scaled, optimize=True)
    return 0.5 * (sigma + sigma.T)

import numpy as np


def qs_fock_matrix(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", coefficients: "np.ndarray", orbital_energies: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    overlap = np.asarray(overlap, dtype=float)
    core_hamiltonian = np.asarray(core_hamiltonian, dtype=float)
    eri = np.asarray(eri, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    n_bf = overlap.shape[0]
    if (overlap.shape != (n_bf, n_bf) or core_hamiltonian.shape != (n_bf, n_bf)
            or eri.shape != (n_bf,) * 4 or coefficients.shape != (n_bf, n_bf)):
        raise ValueError("inconsistent matrix shapes")
    if int(n_occ) != n_occ or not 1 <= n_occ <= n_bf - 1:
        raise ValueError("n_occ must be an integer between 1 and n_bf - 1")
    n_occ = int(n_occ)
    density = 2.0 * coefficients[:, :n_occ] @ coefficients[:, :n_occ].T
    coulomb = np.einsum("pqrs,rs->pq", eri, density, optimize=True)
    exchange = np.einsum("prqs,rs->pq", eri, density, optimize=True)
    eri_mo = np.einsum("pqrs,pi,qj,rk,sl->ijkl", eri, coefficients, coefficients, coefficients, coefficients, optimize=True)
    sigma = srg_self_energy(eri_mo, orbital_energies, n_occ, s, c_ss, c_os)
    sc = overlap @ coefficients
    fock = core_hamiltonian + coulomb - 0.5 * exchange + sc @ sigma @ sc.T
    return 0.5 * (fock + fock.T)

import numpy as np
from scipy.linalg import eigh


def run_srg_qsgf2(overlap: "np.ndarray", core_hamiltonian: "np.ndarray", eri: "np.ndarray", n_occ: int, s: float, c_ss: float, c_os: float) -> "np.ndarray":
    overlap = np.asarray(overlap, dtype=float)
    n_bf = overlap.shape[0]
    if int(n_occ) != n_occ or not 1 <= n_occ <= n_bf - 1:
        raise ValueError("n_occ must be an integer between 1 and n_bf - 1")
    n_occ = int(n_occ)
    _, energies, coefficients = run_rhf(overlap, core_hamiltonian, eri, 0.0, n_occ)
    history = []
    for _ in range(300):
        fock = qs_fock_matrix(overlap, core_hamiltonian, eri, coefficients, energies, n_occ, s, c_ss, c_os)
        density = 2.0 * coefficients[:, :n_occ] @ coefficients[:, :n_occ].T
        error = fock @ density @ overlap - overlap @ density @ fock
        history = (history + [(fock, error)])[-8:]
        new_energies, new_coefficients = eigh(_diis_extrapolate(history), overlap)
        change = np.abs(new_energies - energies).max()
        energies, coefficients = new_energies, new_coefficients
        if change < 1e-10 and np.abs(error).max() < 1e-9:
            return energies
    raise ValueError("quasiparticle self-consistency did not converge")

import numpy as np


def quasiparticle_gap(charges: "np.ndarray", coords: "np.ndarray", shells: list, n_electrons: int, s: float, c_ss: float, c_os: float) -> float:
    if int(n_electrons) != n_electrons or n_electrons <= 0 or int(n_electrons) % 2 != 0:
        raise ValueError("n_electrons must be a positive even integer")
    n_occ = int(n_electrons) // 2
    overlap, core_hamiltonian, eri, _ = compute_ao_integrals(charges, coords, shells)
    if n_occ >= overlap.shape[0]:
        raise ValueError("the basis has no virtual orbital for this electron count")
    qp = run_srg_qsgf2(overlap, core_hamiltonian, eri, n_occ, s, c_ss, c_os)
    hartree_to_ev = 27.211386245988
    return float((qp[n_occ] - qp[n_occ - 1]) * hartree_to_ev)
SCICODE_GOLD_EOF
