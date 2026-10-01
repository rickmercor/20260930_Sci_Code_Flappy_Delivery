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

def apply_fermion_string(
    determinant: 'np.ndarray',
    annihilators: 'np.ndarray',
    creators: 'np.ndarray',
    n_orb: int,
) -> 'np.ndarray':

    if not isinstance(n_orb, (int, np.integer)) or not 1 <= int(n_orb) <= 31:
        raise ValueError("n_orb must be an integer in [1,31]")

    n_orb = int(n_orb)
    det = np.asarray(determinant)
    ann = np.asarray(annihilators)
    cre = np.asarray(creators)

    if det.shape != (2,) or ann.ndim != 1 or cre.ndim != 1:
        raise ValueError("invalid determinant or operator shape")
    if not np.issubdtype(det.dtype, np.integer):
        raise ValueError("determinant masks must be integers")
    if not np.issubdtype(ann.dtype, np.integer) or not np.issubdtype(
        cre.dtype, np.integer
    ):
        raise ValueError("operator indices must be integers")

    alpha, beta = map(int, det)
    if (
        alpha < 0
        or beta < 0
        or alpha >= (1 << n_orb)
        or beta >= (1 << n_orb)
    ):
        raise ValueError("determinant mask is outside n_orb")

    if (
        np.any(ann < 0)
        or np.any(ann >= 2 * n_orb)
        or np.any(cre < 0)
        or np.any(cre >= 2 * n_orb)
    ):
        raise ValueError("spin-orbital index outside range")

    phase = 1

    for q in ann:
        q = int(q)
        orbital, spin = q // 2, q & 1
        mask = alpha if spin == 0 else beta
        bit = 1 << orbital

        if (mask & bit) == 0:
            return np.array([-1, -1, 0], dtype=np.int64)

        if (mask & (bit - 1)).bit_count() & 1:
            phase = -phase

        mask ^= bit
        if spin == 0:
            alpha = mask
        else:
            beta = mask

    for p in cre:
        p = int(p)
        orbital, spin = p // 2, p & 1
        mask = alpha if spin == 0 else beta
        bit = 1 << orbital

        if mask & bit:
            return np.array([-1, -1, 0], dtype=np.int64)

        if (mask & (bit - 1)).bit_count() & 1:
            phase = -phase

        mask |= bit
        if spin == 0:
            alpha = mask
        else:
            beta = mask

    return np.array([alpha, beta, phase], dtype=np.int64)

import numpy as np

def build_selected_hamiltonian(
    determinants: 'np.ndarray',
    h1: 'np.ndarray',
    g2: 'np.ndarray',
    n_orb: int,
) -> 'np.ndarray':

    if not isinstance(n_orb, (int, np.integer)) or not 1 <= int(n_orb) <= 31:
        raise ValueError("invalid n_orb")

    n_orb = int(n_orb)
    n_so = 2 * n_orb
    dets = np.asarray(determinants)
    one = np.asarray(h1, dtype=float)
    two = np.asarray(g2, dtype=float)

    if dets.ndim != 2 or dets.shape[1] != 2 or dets.shape[0] == 0:
        raise ValueError("determinants must have shape (n,2)")
    if not np.issubdtype(dets.dtype, np.integer):
        raise ValueError("determinants must be integer masks")
    if one.shape != (n_so, n_so) or two.shape != (
        n_so,
        n_so,
        n_so,
        n_so,
    ):
        raise ValueError("integral shape mismatch")
    if np.any(~np.isfinite(one)) or np.any(~np.isfinite(two)):
        raise ValueError("integrals must be finite")
    if not np.allclose(one, one.T, atol=1e-12, rtol=0.0):
        raise ValueError("h1 must be symmetric")
    if not np.allclose(
        two,
        -np.swapaxes(two, 0, 1),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("g2 must be antisymmetric in its first pair")
    if not np.allclose(
        two,
        -np.swapaxes(two, 2, 3),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("g2 must be antisymmetric in its second pair")
    if not np.allclose(
        two,
        np.transpose(two, (2, 3, 0, 1)),
        atol=1e-12,
        rtol=0.0,
    ):
        raise ValueError("g2 must be Hermitian under pair exchange")

    keys = [tuple(map(int, d)) for d in dets]
    if len(set(keys)) != len(keys):
        raise ValueError("determinants must be unique")
    if any(
        a < 0
        or b < 0
        or a >= (1 << n_orb)
        or b >= (1 << n_orb)
        for a, b in keys
    ):
        raise ValueError("determinant mask outside n_orb")

    out = np.zeros((len(keys), len(keys)), dtype=float)

    def _occupied_spin_orbitals(det):
        alpha, beta = map(int, det)
        occ = [
            2 * p
            for p in range(n_orb)
            if (alpha >> p) & 1
        ]
        occ += [
            2 * p + 1
            for p in range(n_orb)
            if (beta >> p) & 1
        ]
        return occ

    def _spin_difference(bra_mask, ket_mask):
        holes_mask = ket_mask & ~bra_mask
        particles_mask = bra_mask & ~ket_mask
        holes = [
            p
            for p in range(n_orb)
            if (holes_mask >> p) & 1
        ]
        particles = [
            p
            for p in range(n_orb)
            if (particles_mask >> p) & 1
        ]
        return holes, particles

    for i, bra in enumerate(dets):
        bra_alpha, bra_beta = map(int, bra)

        for j, ket in enumerate(dets):
            ket_alpha, ket_beta = map(int, ket)

            holes_a, parts_a = _spin_difference(
                bra_alpha,
                ket_alpha,
            )
            holes_b, parts_b = _spin_difference(
                bra_beta,
                ket_beta,
            )

            if (
                len(holes_a) != len(parts_a)
                or len(holes_b) != len(parts_b)
            ):
                continue

            degree = len(holes_a) + len(holes_b)
            if degree > 2:
                continue

            if degree == 0:
                occ = _occupied_spin_orbitals(ket)
                value = sum(one[p, p] for p in occ)
                value += sum(
                    two[p, q, p, q]
                    for pos, p in enumerate(occ)
                    for q in occ[:pos]
                )
                out[i, j] = value
                continue

            annihilators = np.array(
                [2 * p for p in holes_a]
                + [2 * p + 1 for p in holes_b],
                dtype=np.int64,
            )
            creators = np.array(
                [2 * p + 1 for p in reversed(parts_b)]
                + [2 * p for p in reversed(parts_a)],
                dtype=np.int64,
            )

            acted = apply_fermion_string(
                ket,
                annihilators,
                creators,
                n_orb,
            )
            if (
                acted[2] == 0
                or tuple(map(int, acted[:2]))
                != tuple(map(int, bra))
            ):
                raise ValueError(
                    "excitation analysis did not reproduce the bra"
                )

            phase = float(acted[2])

            if degree == 1:
                if holes_a:
                    hole = 2 * holes_a[0]
                    particle = 2 * parts_a[0]
                else:
                    hole = 2 * holes_b[0] + 1
                    particle = 2 * parts_b[0] + 1

                value = one[particle, hole]
                for occupied in _occupied_spin_orbitals(ket):
                    if occupied != hole:
                        value += two[
                            particle,
                            occupied,
                            hole,
                            occupied,
                        ]

                out[i, j] = phase * value
                continue

            if len(holes_a) == 2:
                h0, h1_index = (
                    2 * p for p in holes_a
                )
                p0, p1 = (
                    2 * p for p in parts_a
                )
            elif len(holes_b) == 2:
                h0, h1_index = (
                    2 * p + 1 for p in holes_b
                )
                p0, p1 = (
                    2 * p + 1 for p in parts_b
                )
            else:
                h0 = 2 * holes_a[0]
                p0 = 2 * parts_a[0]
                h1_index = 2 * holes_b[0] + 1
                p1 = 2 * parts_b[0] + 1

            out[i, j] = (
                phase
                * two[p0, p1, h0, h1_index]
            )

    if not np.allclose(
        out,
        out.T,
        atol=2e-11,
        rtol=0.0,
    ):
        raise ValueError(
            "constructed Hamiltonian is not Hermitian"
        )

    return 0.5 * (out + out.T)


def _small_integrals(n_orb, seed):
    rng = np.random.default_rng(seed)
    n_so = 2 * n_orb

    h = rng.normal(
        scale=0.2,
        size=(n_so, n_so),
    )
    h = 0.5 * (h + h.T)

    raw = rng.normal(
        scale=0.04,
        size=(n_so, n_so, n_so, n_so),
    )
    g = raw - np.swapaxes(raw, 0, 1)
    g = g - np.swapaxes(g, 2, 3)
    g = 0.5 * (
        g + np.transpose(g, (2, 3, 0, 1))
    )

    return h, g

import numpy as np

def normalized_configuration_probabilities(
    signs: 'np.ndarray',
    logabs: 'np.ndarray',
) -> 'np.ndarray':

    s = np.asarray(signs, dtype=float)
    x = np.asarray(logabs, dtype=float)

    if (
        s.ndim != 1
        or x.ndim != 1
        or s.size == 0
        or s.shape != x.shape
    ):
        raise ValueError(
            "signs and logabs must be matching nonempty vectors"
        )
    if np.any(~np.isfinite(s)) or np.any(~np.isfinite(x)):
        raise ValueError("inputs must be finite")
    if np.any(np.abs(s) != 1.0):
        raise ValueError("signs must be exactly -1 or +1")

    log_prob = 2.0 * x
    shifted = np.exp(log_prob - np.max(log_prob))

    return shifted / np.sum(shifted)

import numpy as np

def select_cumulative_support(
    probabilities: 'np.ndarray',
    fraction: float,
    min_k: int,
    max_k: int,
) -> 'np.ndarray':

    p = np.asarray(probabilities, dtype=float)

    if (
        p.ndim != 1
        or p.size == 0
        or np.any(~np.isfinite(p))
        or np.any(p < 0.0)
    ):
        raise ValueError(
            "probabilities must be a finite nonnegative vector"
        )

    total = float(np.sum(p))
    if total <= 0.0:
        raise ValueError("probability mass must be positive")

    if (
        not np.isfinite(fraction)
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError("fraction must be in (0,1]")

    if not isinstance(
        min_k,
        (int, np.integer),
    ) or not isinstance(
        max_k,
        (int, np.integer),
    ):
        raise ValueError("bounds must be integers")

    min_k = int(min_k)
    max_k = int(max_k)

    if not 1 <= min_k <= max_k:
        raise ValueError("invalid size bounds")

    p = p / total
    index = np.arange(p.size)
    order = np.lexsort((index, -p))

    cumulative = np.cumsum(p[order])
    k = (
        int(
            np.searchsorted(
                cumulative,
                float(fraction),
                side="left",
            )
        )
        + 1
    )
    k = min(
        max(k, min_k),
        max_k,
        p.size,
    )

    return order[:k].astype(np.int64)

import numpy as np

def screen_dynamic_perturbative_space(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    determinants: 'np.ndarray',
    variational_indices: 'np.ndarray',
    eps1: float,
) -> 'np.ndarray':

    h = np.asarray(hamiltonian, dtype=float)
    psi = np.asarray(amplitudes, dtype=float)
    dets = np.asarray(determinants)
    v = np.asarray(variational_indices)

    if (
        h.ndim != 2
        or h.shape[0] == 0
        or h.shape[0] != h.shape[1]
    ):
        raise ValueError(
            "hamiltonian must be a nonempty square matrix"
        )

    n = h.shape[0]

    if (
        psi.shape != (n,)
        or dets.shape != (n, 2)
        or v.ndim != 1
        or v.size == 0
    ):
        raise ValueError("incompatible input shapes")

    if not np.issubdtype(
        dets.dtype,
        np.integer,
    ) or not np.issubdtype(
        v.dtype,
        np.integer,
    ):
        raise ValueError(
            "determinants and indices must be integral"
        )

    if np.any(dets < 0):
        raise ValueError(
            "determinant masks must be nonnegative"
        )

    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(psi)):
        raise ValueError(
            "hamiltonian and amplitudes must be finite"
        )

    if not np.allclose(
        h,
        h.T,
        atol=2e-11,
        rtol=0.0,
    ):
        raise ValueError(
            "hamiltonian must be symmetric"
        )

    if len({tuple(map(int, d)) for d in dets}) != n:
        raise ValueError(
            "determinants must be unique"
        )

    if (
        np.any(v < 0)
        or np.any(v >= n)
        or np.unique(v).size != v.size
    ):
        raise ValueError(
            "variational indices must be unique and in range"
        )

    if not np.isfinite(eps1) or float(eps1) < 0.0:
        raise ValueError(
            "eps1 must be finite and nonnegative"
        )

    norm = float(np.linalg.norm(psi[v]))
    if norm == 0.0:
        raise ValueError(
            "selected amplitudes must have nonzero norm"
        )

    psi_v = psi[v] / norm

    is_external = np.ones(n, dtype=bool)
    is_external[v] = False
    external = np.flatnonzero(is_external)

    if external.size == 0:
        return np.empty(0, dtype=np.int64)

    score = np.max(
        np.abs(
            h[np.ix_(external, v)]
            * psi_v[None, :]
        ),
        axis=1,
    )

    keep = external[score >= float(eps1)]

    if keep.size == 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort(
        (
            dets[keep, 1],
            dets[keep, 0],
        )
    )

    return keep[order].astype(np.int64)

import numpy as np

def deterministic_energy_diagnostics(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    variational_indices: 'np.ndarray',
    perturbative_indices: 'np.ndarray',
) -> 'np.ndarray':

    h = np.asarray(hamiltonian, dtype=float)
    psi = np.asarray(amplitudes, dtype=float)
    v = np.asarray(variational_indices)
    p = np.asarray(perturbative_indices)

    if h.ndim != 2 or h.shape[0] == 0 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian must be a nonempty square matrix")

    n = h.shape[0]

    if psi.shape != (n,) or v.ndim != 1 or p.ndim != 1 or v.size == 0:
        raise ValueError("incompatible input shapes")

    if not np.issubdtype(v.dtype, np.integer) or not np.issubdtype(
        p.dtype, np.integer
    ):
        raise ValueError("indices must be integral")

    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(psi)):
        raise ValueError("hamiltonian and amplitudes must be finite")

    if not np.allclose(h, h.T, atol=2e-11, rtol=0.0):
        raise ValueError("hamiltonian must be symmetric")

    if np.any(v < 0) or np.any(v >= n) or np.unique(v).size != v.size:
        raise ValueError("invalid variational indices")

    if np.any(p < 0) or np.any(p >= n) or np.unique(p).size != p.size:
        raise ValueError("invalid perturbative indices")

    if np.intersect1d(v, p).size:
        raise ValueError(
            "variational and perturbative sets must be disjoint"
        )

    norm_v = float(np.linalg.norm(psi[v]))
    if norm_v == 0.0:
        raise ValueError(
            "variational amplitudes must have nonzero norm"
        )

    psi_v = psi[v] / norm_v
    hvv = h[np.ix_(v, v)]
    e_var = float(psi_v @ hvv @ psi_v)

    target = np.concatenate((v, p)).astype(np.int64)
    psi_target = psi[target]

    e_asym = float(
        psi_v @ h[np.ix_(v, target)] @ psi_target / norm_v
    )

    norm_target = float(np.linalg.norm(psi_target))
    if norm_target == 0.0:
        raise ValueError("target amplitudes must have nonzero norm")

    proxy = h[np.ix_(target, target)].copy()

    if p.size:
        k = v.size
        proxy[k:, k:] = np.diag(np.diag(proxy[k:, k:]))

    psi_t = psi_target / norm_target
    e_proxy = float(psi_t @ proxy @ psi_t)
    e_diag = float(np.linalg.eigvalsh(hvv)[0])

    return np.array(
        [e_var, e_asym, e_proxy, e_diag, e_var - e_diag],
        dtype=float,
    )

import numpy as np

def decomposed_epstein_nesbet_pt2(
    hamiltonian: 'np.ndarray',
    amplitudes: 'np.ndarray',
    variational_indices: 'np.ndarray',
    perturbative_indices: 'np.ndarray',
    reference_total_energy: float,
    nuclear_repulsion: float,
) -> 'np.ndarray':

    h = np.asarray(hamiltonian, dtype=float)
    psi = np.asarray(amplitudes, dtype=float)
    v = np.asarray(variational_indices)
    p = np.asarray(perturbative_indices)

    if h.ndim != 2 or h.shape[0] == 0 or h.shape[0] != h.shape[1]:
        raise ValueError("hamiltonian must be a nonempty square matrix")

    n = h.shape[0]

    if psi.shape != (n,) or v.ndim != 1 or p.ndim != 1 or v.size == 0:
        raise ValueError("incompatible input shapes")

    if not np.issubdtype(v.dtype, np.integer) or not np.issubdtype(
        p.dtype, np.integer
    ):
        raise ValueError("indices must be integral")

    if np.any(~np.isfinite(h)) or np.any(~np.isfinite(psi)):
        raise ValueError("hamiltonian and amplitudes must be finite")

    if not np.isfinite(reference_total_energy) or not np.isfinite(
        nuclear_repulsion
    ):
        raise ValueError("energies must be finite")

    if not np.allclose(h, h.T, atol=2e-11, rtol=0.0):
        raise ValueError("hamiltonian must be symmetric")

    if np.any(v < 0) or np.any(v >= n) or np.unique(v).size != v.size:
        raise ValueError("invalid variational indices")

    if np.any(p < 0) or np.any(p >= n) or np.unique(p).size != p.size:
        raise ValueError("invalid perturbative indices")

    if np.intersect1d(v, p).size:
        raise ValueError(
            "variational and perturbative sets must be disjoint"
        )

    norm = float(np.linalg.norm(psi[v]))
    if norm == 0.0:
        raise ValueError(
            "variational amplitudes must have nonzero norm"
        )

    psi_v = psi[v] / norm
    e_elec = float(reference_total_energy) - float(nuclear_repulsion)

    hvv = h[np.ix_(v, v)]
    residual_v = hvv @ psi_v - e_elec * psi_v
    denom_v = e_elec - np.diag(hvv)

    if np.any(np.abs(denom_v) <= 1e-12):
        raise ValueError(
            "singular internal Epstein-Nesbet denominator"
        )

    delta_internal = float(
        np.sum(residual_v * residual_v / denom_v)
    )

    if p.size:
        residual_p = h[np.ix_(p, v)] @ psi_v
        denom_p = e_elec - np.diag(h)[p]

        if np.any(np.abs(denom_p) <= 1e-12):
            raise ValueError(
                "singular external Epstein-Nesbet denominator"
            )

        delta_external = float(
            np.sum(residual_p * residual_p / denom_p)
        )
    else:
        delta_external = 0.0

    return np.array(
        [
            delta_internal,
            delta_external,
            delta_internal + delta_external,
            float(p.size),
        ],
        dtype=float,
    )

import itertools
import numpy as np

def nqs_internal_pt2_share(
    determinants: 'np.ndarray' = None,
    h1: 'np.ndarray' = None,
    g2: 'np.ndarray' = None,
    signs: 'np.ndarray' = None,
    logabs: 'np.ndarray' = None,
    mass_fraction: float = None,
    eps1: float = None,
    min_k: int = None,
    max_k: int = None,
    nuclear_repulsion: float = None,
) -> float:

    supplied = (
        determinants,
        h1,
        g2,
        signs,
        logabs,
        mass_fraction,
        eps1,
        min_k,
        max_k,
        nuclear_repulsion,
    )

    if all(value is None for value in supplied):
        (
            determinants,
            h1,
            g2,
            signs,
            logabs,
            mass_fraction,
            eps1,
            min_k,
            max_k,
            nuclear_repulsion,
        ) = _make_nqs_fixture(
            26090410,
            5,
            28,
            0.86,
            0.012,
            6,
            11,
            1.3275,
        )
    elif any(value is None for value in supplied):
        raise ValueError(
            "either supply every argument or use the benchmark defaults"
        )

    dets = np.asarray(determinants)
    one = np.asarray(h1, dtype=float)
    n_orb = one.shape[0] // 2 if one.ndim == 2 else 0

    h = build_selected_hamiltonian(
        dets,
        one,
        g2,
        n_orb,
    )

    probability = normalized_configuration_probabilities(
        signs,
        logabs,
    )

    v = select_cumulative_support(
        probability,
        mass_fraction,
        min_k,
        max_k,
    )

    psi = np.asarray(signs, dtype=float) * np.sqrt(probability)

    p = screen_dynamic_perturbative_space(
        h,
        psi,
        dets,
        v,
        eps1,
    )

    ledger = deterministic_energy_diagnostics(
        h,
        psi,
        v,
        p,
    )

    pt2 = decomposed_epstein_nesbet_pt2(
        h,
        psi,
        v,
        p,
        ledger[0] + float(nuclear_repulsion),
        nuclear_repulsion,
    )

    if not np.isfinite(pt2[2]) or abs(float(pt2[2])) <= 1e-15:
        raise ValueError(
            "total PT2 correction must be finite and nonzero"
        )

    return float(100.0 * pt2[0] / pt2[2])


def _make_nqs_fixture(
    seed,
    n_orb,
    n_target,
    mass_fraction,
    eps1,
    min_k,
    max_k,
    nuclear_repulsion,
):
    rng = np.random.default_rng(seed)
    n_so = 2 * n_orb

    h_spatial = rng.normal(
        scale=0.18,
        size=(n_orb, n_orb),
    )
    h_spatial = 0.5 * (h_spatial + h_spatial.T)
    h_spatial[np.diag_indices(n_orb)] -= np.linspace(
        1.55,
        0.25,
        n_orb,
    )

    h1 = np.zeros((n_so, n_so), dtype=float)

    for p in range(n_so):
        for q in range(n_so):
            if p % 2 == q % 2:
                h1[p, q] = h_spatial[p // 2, q // 2]

    pairs = [
        (p, q)
        for p in range(n_so)
        for q in range(p + 1, n_so)
    ]

    pair_block = rng.normal(
        scale=0.055,
        size=(len(pairs), len(pairs)),
    )
    pair_block = 0.5 * (pair_block + pair_block.T)

    spin_count = np.array(
        [(p & 1) + (q & 1) for p, q in pairs]
    )
    pair_block *= spin_count[:, None] == spin_count[None, :]

    g2 = np.zeros(
        (n_so, n_so, n_so, n_so),
        dtype=float,
    )

    for a, (p, q) in enumerate(pairs):
        for b, (r, s) in enumerate(pairs):
            value = pair_block[a, b]
            g2[p, q, r, s] = value
            g2[q, p, r, s] = -value
            g2[p, q, s, r] = -value
            g2[q, p, s, r] = value

    alpha_masks = [
        sum(1 << i for i in occ)
        for occ in itertools.combinations(range(n_orb), 2)
    ]

    beta_masks = [
        sum(1 << i for i in occ)
        for occ in itertools.combinations(range(n_orb), 2)
    ]

    archive = np.array(
        list(itertools.product(alpha_masks, beta_masks)),
        dtype=np.int64,
    )

    if not 1 <= int(n_target) <= archive.shape[0]:
        raise ValueError(
            "n_target exceeds the determinant archive"
        )

    determinants = archive[
        rng.permutation(archive.shape[0])[: int(n_target)]
    ]

    orbital_diag = np.diag(h_spatial)
    one_body_diag = np.empty(int(n_target), dtype=float)

    for k, (alpha, beta) in enumerate(determinants):
        occupied = [
            i
            for i in range(n_orb)
            if (int(alpha) >> i) & 1
        ]
        occupied += [
            i
            for i in range(n_orb)
            if (int(beta) >> i) & 1
        ]
        one_body_diag[k] = np.sum(orbital_diag[occupied])

    logabs = -1.15 * (
        one_body_diag - np.min(one_body_diag)
    )
    logabs += rng.normal(
        scale=0.22,
        size=int(n_target),
    )

    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=int(n_target),
    )
    signs[np.argmax(logabs)] = 1.0

    return (
        determinants,
        h1,
        g2,
        signs,
        logabs,
        float(mass_fraction),
        float(eps1),
        int(min_k),
        int(max_k),
        float(nuclear_repulsion),
    )
SCICODE_GOLD_EOF
