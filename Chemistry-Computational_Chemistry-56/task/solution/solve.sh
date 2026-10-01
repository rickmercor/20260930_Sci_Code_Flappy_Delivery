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


def drude_lorentz_exponents(reorg_cm: float, cutoff_cm: float, temperature_K: float,
                                    n_matsubara: int) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if cutoff_cm <= 0 or temperature_K <= 0 or reorg_cm < 0 or n_matsubara < 0:
        raise ValueError("cutoff and temperature must be positive, reorganization energy and n_matsubara non-negative")
    to_radps = 2.0 * np.pi * 2.99792458e-2
    lam = reorg_cm * to_radps
    gam = cutoff_cm * to_radps
    beta = 1.0 / (0.6950348 * temperature_K * to_radps)
    rows = [[lam * gam / np.tan(beta * gam / 2.0), -lam * gam, gam]]
    for k in range(1, int(n_matsubara) + 1):
        nu = 2.0 * np.pi * k / beta
        if abs(nu - gam) <= 1e-9 * gam:
            raise ValueError("a Matsubara frequency coincides with the cutoff")
        rows.append([4.0 * lam * gam / beta * nu / (nu * nu - gam * gam), 0.0, nu])
    return np.array(rows, dtype=float)

import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _hierarchy_labels(n_modes: int, depth: int) -> list:
    """All index tuples of length n_modes with sum <= depth, ordered by total tier."""
    import itertools
    labels = []
    for tier in range(depth + 1):
        for bars in itertools.combinations(range(tier + n_modes - 1), n_modes - 1):
            prev = -1
            index = []
            for b in bars:
                index.append(b - prev - 1)
                prev = b
            index.append(tier + n_modes - 1 - prev - 1)
            labels.append(tuple(index))
    return labels


def _hierarchy_generator(dimer, bath, n_matsubara, depth):
    """Sparse generator of the truncated hierarchy on stacked row-major 2x2 blocks; block 0 is the system."""
    import numpy as np
    import scipy.sparse as sp
    to_radps = 2.0 * np.pi * 2.99792458e-2
    J, gap, g_rec, kappa = [float(v) for v in dimer]
    H = np.array([[gap / 2.0, J], [J, -gap / 2.0]]) * to_radps
    loss = np.diag([g_rec, g_rec + kappa])
    I2 = np.eye(2)
    # row-major vec: vec(A X B) = kron(A, B.T) vec(X)
    system = (-1j * (np.kron(H, I2) - np.kron(I2, H.T))
              - 0.5 * (np.kron(loss, I2) + np.kron(I2, loss.T)))
    exps = drude_lorentz_exponents(float(bath[0]), float(bath[1]), float(bath[2]), int(n_matsubara))
    amps = exps[:, 0] + 1j * exps[:, 1]
    rates = exps[:, 2]
    modes = [(site, amps[k], rates[k]) for site in range(2) for k in range(len(rates))]
    labels = _hierarchy_labels(len(modes), depth)
    where = {lab: i for i, lab in enumerate(labels)}
    n_ado = len(labels)
    damping = np.einsum("ij,j->i", np.array(labels, dtype=float), np.array([m[2] for m in modes]))
    gen = sp.kron(sp.identity(n_ado, format="csr"), sp.csr_matrix(system), format="csr") \
        - sp.kron(sp.diags(damping), sp.identity(4), format="csr")
    rows, cols, vals = [], [], []
    for m, (site, amp, _) in enumerate(modes):
        left = np.zeros(4)
        right = np.zeros(4)
        for a in range(2):
            left[2 * site + a] = 1.0      # Q_j X
            right[2 * a + site] = 1.0     # X Q_j
        comm = np.nonzero(left - right)[0]
        for i, lab in enumerate(labels):
            if sum(lab) < depth:
                up = list(lab)
                up[m] += 1
                j = where[tuple(up)]
                rows.extend(4 * i + comm)
                cols.extend(4 * j + comm)
                vals.extend(-1j * (left - right)[comm])
            if lab[m] > 0:
                down = list(lab)
                down[m] -= 1
                j = where[tuple(down)]
                coup = -1j * lab[m] * (amp * left - np.conj(amp) * right)
                nz = np.nonzero(coup)[0]
                rows.extend(4 * i + nz)
                cols.extend(4 * j + nz)
                vals.extend(coup[nz])
    gen = gen + sp.csr_matrix((vals, (rows, cols)), shape=(4 * n_ado, 4 * n_ado))
    return gen.tocsr()


def heom_readout_operator(dimer: "np.ndarray", bath: "np.ndarray", n_matsubara: int, depth: int,
                                  readout_site: int, times_ps: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    import scipy.sparse.linalg as spla
    dimer = np.asarray(dimer, dtype=float)
    times = np.asarray(times_ps, dtype=float).reshape(-1)
    if depth < 0 or n_matsubara < 0:
        raise ValueError("depth and n_matsubara must be non-negative")
    if readout_site not in (0, 1):
        raise ValueError("readout_site must be 0 or 1")
    if dimer[2] < 0 or dimer[3] < 0 or np.any(times < 0):
        raise ValueError("rates and delays must be non-negative")
    gen_h = _hierarchy_generator(dimer, bath, n_matsubara, int(depth)).conj().T.tocsr()
    # adjoint propagation: Tr[M rho(t)] = y(t)^H x0 with y(t) = exp(t G^H) vec(|r><r|) in block 0
    y0 = np.zeros(gen_h.shape[0], dtype=complex)
    y0[3 * readout_site] = 1.0
    grid, inverse = np.unique(times, return_inverse=True)
    states = np.zeros((len(grid), gen_h.shape[0]), dtype=complex)
    if len(grid) > 2 and np.allclose(np.diff(grid), grid[1] - grid[0], rtol=1e-9, atol=0.0):
        first = spla.expm_multiply(gen_h * grid[0], y0) if grid[0] > 0 else y0
        states[:] = spla.expm_multiply(gen_h, first, start=0.0, stop=grid[-1] - grid[0],
                                       num=len(grid), endpoint=True)
    else:
        y = y0
        t_prev = 0.0
        for i, t in enumerate(grid):
            if t > t_prev:
                y = spla.expm_multiply(gen_h * (t - t_prev), y)
            states[i] = y
            t_prev = t
    rates = (gen_h @ states.T).T
    # M_lk = conj(y_{2k+l}); M_DA = M_01 = conj(y_2)
    blocks = np.conj(states[:, :4])
    dblocks = np.conj(rates[:, :4])
    table = np.column_stack([blocks[:, 0].real, blocks[:, 3].real, blocks[:, 2].real, blocks[:, 2].imag,
                             dblocks[:, 0].real, dblocks[:, 3].real, dblocks[:, 2].real, dblocks[:, 2].imag])
    return table[inverse]

import numpy as np


def ensemble_readout_operator(dimer: "np.ndarray", site_sigma_cm: float, bath: "np.ndarray",
                                      n_matsubara: int, depth: int, n_nodes: int, readout_site: int,
                                      times_ps: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if site_sigma_cm < 0 or n_nodes < 1:
        raise ValueError("site_sigma_cm must be non-negative and n_nodes at least 1")
    dimer = np.asarray(dimer, dtype=float)
    nodes, weights = np.polynomial.hermite_e.hermegauss(int(n_nodes))
    weights = weights / weights.sum()
    gap_sigma = np.sqrt(2.0) * site_sigma_cm      # difference of two independent site energies
    total = None
    for x, w in zip(nodes, weights):
        member = dimer.copy()
        member[1] = dimer[1] + gap_sigma * x
        rows = heom_readout_operator(member, bath, n_matsubara, depth, readout_site, times_ps)
        total = w * rows if total is None else total + w * rows
    return total

import numpy as np


def coherence_diagnostics(readout: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    rows = np.asarray(readout, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError("readout must have shape (n, 8)")
    m_da = rows[:, 2] + 1j * rows[:, 3]
    coherence = np.abs(m_da)
    # eigenvalues of [[a, m], [m*, b]] are (a + b)/2 +- sqrt(((a - b)/2)^2 + |m|^2)
    mean = 0.5 * (rows[:, 0] + rows[:, 1])
    radius = np.sqrt((0.5 * (rows[:, 0] - rows[:, 1])) ** 2 + coherence ** 2)
    best_any = np.maximum(np.abs(mean + radius), np.abs(mean - radius))
    best_free = np.maximum(np.abs(rows[:, 0]), np.abs(rows[:, 1]))
    unpaired = best_any - best_free
    # Tr[M (rho - G(rho))] = Re(M_DA exp(i phi)) for the equal-weight superposition
    phase = np.where(coherence > 0.0, -np.angle(m_da), 0.0)
    phase = np.where(phase <= -np.pi, phase + 2.0 * np.pi, phase)
    rate = np.hypot(rows[:, 6], rows[:, 7])
    return np.column_stack([coherence, unpaired, phase, rate])

import numpy as np
from scipy.optimize import minimize_scalar


def _hermite_abs_maximum(t0, t1, row0, row1):
    """Maximum of |m(t)| on [t0, t1] for the cubic Hermite interpolant of Re m and Im m from values and derivatives."""
    import numpy as np
    from scipy.optimize import minimize_scalar
    h = t1 - t0

    def _value(t):
        s = (t - t0) / h
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        re = h00 * row0[2] + h10 * h * row0[6] + h01 * row1[2] + h11 * h * row1[6]
        im = h00 * row0[3] + h10 * h * row0[7] + h01 * row1[3] + h11 * h * row1[7]
        return np.hypot(re, im)

    res = minimize_scalar(lambda t: -_value(t), bounds=(t0, t1), method="bounded", options={"xatol": 1e-13})
    candidates = [(_value(t0), t0), (-res.fun, res.x), (_value(t1), t1)]
    best = max(c[0] for c in candidates)
    return min((c for c in candidates if c[0] >= best - 1e-15), key=lambda c: c[1])


def post_overlap_supremum(dimer: "np.ndarray", site_sigma_cm: float, bath: "np.ndarray", n_matsubara: int,
                                  depth: int, n_nodes: int, window_ps: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    t_a, t_b = [float(v) for v in np.asarray(window_ps, dtype=float).reshape(2)]
    if not 0.0 <= t_a < t_b:
        raise ValueError("the window must satisfy 0 <= t_a < t_b")
    # coarse scan on a grid of at most 2 fs, then a 0.1 fs grid around the best coarse point
    n_coarse = int(np.ceil((t_b - t_a) / 0.002 - 1e-9))
    coarse = np.linspace(t_a, t_b, n_coarse + 1)
    rows = ensemble_readout_operator(dimer, site_sigma_cm, bath, n_matsubara, depth, n_nodes, 1, coarse)
    impact = coherence_diagnostics(rows)[:, 0]
    j = int(np.argmax(impact))
    lo, hi = coarse[max(j - 1, 0)], coarse[min(j + 1, n_coarse)]
    fine = np.linspace(lo, hi, int(np.ceil((hi - lo) / 1e-4 - 1e-9)) + 1)
    frows = ensemble_readout_operator(dimer, site_sigma_cm, bath, n_matsubara, depth, n_nodes, 1, fine)
    fimpact = coherence_diagnostics(frows)[:, 0]
    k = int(np.argmax(fimpact))
    best = (fimpact[k], fine[k])
    for a, b in ((k - 1, k), (k, k + 1)):
        if 0 <= a and b < len(fine):
            cand = _hermite_abs_maximum(fine[a], fine[b], frows[a], frows[b])
            if cand[0] > best[0] + 1e-15 or (abs(cand[0] - best[0]) <= 1e-15 and cand[1] < best[1]):
                best = cand
    return np.array([best[0], best[1]])

import numpy as np
from scipy.optimize import brentq


def critical_reorganization_energy(dimer: "np.ndarray", site_sigma_cm: float, cutoff_cm: float,
                                           temperature_K: float, n_matsubara: int, depth: int, n_nodes: int,
                                           window_ps: "np.ndarray", target: float,
                                           reorg_bracket: "np.ndarray") -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    e_lo, e_hi = [float(v) for v in np.asarray(reorg_bracket, dtype=float).reshape(2)]
    if not 0.0 <= e_lo < e_hi:
        raise ValueError("the bracket must satisfy 0 <= E_lo < E_hi")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between 0 and 1")

    def _excess(reorg):
        bath = np.array([reorg, cutoff_cm, temperature_K])
        return post_overlap_supremum(dimer, site_sigma_cm, bath, n_matsubara, depth, n_nodes,
                                             window_ps)[0] - target

    if _excess(e_lo) * _excess(e_hi) > 0.0:
        raise ValueError("Q - target does not change sign inside the bracket")
    return float(brentq(_excess, e_lo, e_hi, xtol=1e-8, rtol=1e-14))

import numpy as np


def coherence_relevance_report(dimer: "np.ndarray", site_sigma_cm: float, cutoff_cm: float,
                                       temperature_K: float, n_matsubara: int, depth: int, n_nodes: int,
                                       window_ps: "np.ndarray", target: float, reorg_bracket: "np.ndarray",
                                       reorg_reference: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if reorg_reference < 0:
        raise ValueError("reorg_reference must be non-negative")
    t_a = float(np.asarray(window_ps, dtype=float).reshape(2)[0])
    if t_a <= 0:
        raise ValueError("the window must start after zero delay")
    reorg_star = critical_reorganization_energy(dimer, site_sigma_cm, cutoff_cm, temperature_K, n_matsubara,
                                                        depth, n_nodes, window_ps, target, reorg_bracket)
    bath_star = np.array([reorg_star, cutoff_cm, temperature_K])
    t_star = post_overlap_supremum(dimer, site_sigma_cm, bath_star, n_matsubara, depth, n_nodes,
                                           window_ps)[1]
    rows = ensemble_readout_operator(dimer, site_sigma_cm, bath_star, n_matsubara, depth, n_nodes, 1,
                                             np.array([t_star]))
    diag = coherence_diagnostics(rows)[0]
    q_reference = post_overlap_supremum(dimer, site_sigma_cm, np.array([reorg_reference, cutoff_cm,
                                                temperature_K]), n_matsubara, depth, n_nodes, window_ps)[0]
    q_homogeneous = post_overlap_supremum(dimer, 0.0, bath_star, n_matsubara, depth, 1, window_ps)[0]
    early = post_overlap_supremum(dimer, site_sigma_cm, bath_star, n_matsubara, depth, n_nodes,
                                          np.array([0.0, t_a]))
    n_int = int(np.ceil((t_star - early[1]) / 2e-4 - 1e-9))
    n_int += n_int % 2
    grid = np.linspace(early[1], t_star, n_int + 1)
    rate = coherence_diagnostics(ensemble_readout_operator(
        dimer, site_sigma_cm, bath_star, n_matsubara, depth, n_nodes, 1, grid))[:, 3]
    h = (t_star - early[1]) / n_int
    variation = h / 3.0 * (rate[0] + rate[-1] + 4.0 * rate[1:-1:2].sum() + 2.0 * rate[2:-1:2].sum())
    rate_early = rate[0]
    return np.array([reorg_star, t_star, diag[2], rows[0, 0], rows[0, 1], q_reference, q_homogeneous,
                     early[0], early[1], diag[1], rate_early, variation])
SCICODE_GOLD_EOF
