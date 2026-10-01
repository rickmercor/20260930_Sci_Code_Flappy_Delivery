#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def build_lattice_adjacency(n_per_class=4):
    """Reference implementation for build_lattice_adjacency."""
    import numpy as np

    if isinstance(n_per_class, bool) or int(n_per_class) != n_per_class:
        raise ValueError("n_per_class must be an integer")
    n_per_class = int(n_per_class)
    if n_per_class < 2:
        raise ValueError("n_per_class must be at least 2")
    ncol = n_per_class
    n = 2 * ncol
    is_v = np.zeros(n, dtype=float)
    adj = np.zeros((n, n), dtype=float)
    for r in range(2):
        for c in range(ncol):
            i = r * ncol + c
            is_v[i] = 1.0 if (r + c) % 2 == 0 else 0.0
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 2 and 0 <= cc < ncol:
                    adj[i, rr * ncol + cc] = 1.0
    return is_v, adj

def contact_energy_changes(adjacency, open_mask):
    """Reference implementation for contact_energy_changes."""
    import numpy as np

    EPS_OO, EPS_CC = -5.7, -0.7  # kT, the treatment's contact energies (Table 1)
    adjacency = np.asarray(adjacency, dtype=float)
    open_mask = np.asarray(open_mask, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.all(np.isfinite(adjacency)):
        raise ValueError("adjacency must be finite")
    if not np.allclose(adjacency, adjacency.T, rtol=0.0, atol=1e-12):
        raise ValueError("adjacency must be symmetric")
    if open_mask.ndim != 1 or open_mask.shape[0] != adjacency.shape[0]:
        raise ValueError("open_mask must align with adjacency")
    if not np.all(np.isfinite(open_mask)) or not np.all((open_mask == 0.0) | (open_mask == 1.0)):
        raise ValueError("open_mask must contain only 0.0/1.0")

    # Each open neighbour contributes EPS_OO to the opening energy change; each closed
    # neighbour's contact (energy EPS_CC) ceases to exist on opening and therefore enters
    # with a minus sign. Inactivated channels enter as closed because open_mask marks them 0.
    degree = adjacency.sum(axis=1)
    n_open = adjacency @ open_mask
    return n_open * EPS_OO - (degree - n_open) * EPS_CC

def channel_transition_rates(state, is_v, delta_energy, eps_v, nu=1.0, nu_v=0.8):
    """Reference implementation for channel_transition_rates."""
    import numpy as np

    # Table 1 constants of the matched treatment (ms^-1 and kT).
    K_PLUS, K_MINUS = 0.001, 2.0
    I1_HT, I2_HT, R1_HT, R2_HT = 3.5, 50.0, 20.0, 50.0
    N_STATES = 4
    if state not in (0, 1, 2, 3):
        raise ValueError("state must be one of 0, 1, 2, 3")
    is_v = bool(is_v)
    for name, value in (("delta_energy", delta_energy), ("eps_v", eps_v),
                        ("nu", nu), ("nu_v", nu_v)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if is_v and state in (2, 3):
        raise ValueError("sensor-coupled channels have no inactivated states")

    # Electrical energy acts on the sensor-coupled class only.
    e_v = float(eps_v) if is_v else 0.0
    rates = np.zeros(N_STATES, dtype=float)

    if state == 0:
        # Opening: both energies enter the forward barrier scaled by their coefficients.
        rates[1] = (K_PLUS * np.exp(-nu * delta_energy) * np.exp(-nu_v * e_v))
    elif state == 1:
        # Closing carries the complementary fractions with the sign of the reverse
        # direction, so that the ratio of the two rates reproduces the treatment's
        # equilibrium constant independently of either coefficient.
        rates[0] = (K_MINUS * np.exp((1.0 - nu) * delta_energy)
                    * np.exp((1.0 - nu_v) * e_v))
        if not is_v:
            rates[2] = np.log(2.0) / I1_HT
    elif state == 2:
        # Two exits compete from the first inactivated state; both are armed on entry and
        # the earlier one fires, which is a race of two exponentials.
        rates[0] = np.log(2.0) / R1_HT
        rates[3] = np.log(2.0) / I2_HT
    else:
        rates[2] = np.log(2.0) / R2_HT

    return rates

def assemble_generator(is_v, adjacency, eps_v, nu=1.0, nu_v=0.8):
    """Reference implementation for assemble_generator."""
    import numpy as np

    def _global_state_table(is_v):
        is_v = np.asarray(is_v, dtype=bool)
        n = is_v.size
        dims = tuple(2 if v else 4 for v in is_v)
        n_states = int(np.prod(dims))
        digits = np.stack(np.unravel_index(np.arange(n_states), dims), axis=1)
        strides = np.array([int(np.prod(dims[c + 1:])) for c in range(n)])
        return digits, strides, n_states

    is_v = np.asarray(is_v, dtype=float)
    adjacency = np.asarray(adjacency, dtype=float)
    for name, value in (("eps_v", eps_v), ("nu", nu), ("nu_v", nu_v)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    n = is_v.size
    if adjacency.shape != (n, n):
        raise ValueError("shape mismatch between is_v and adjacency")

    digits, strides, n_states = _global_state_table(is_v != 0.0)
    rows, cols, data = [], [], []

    for src in range(n_states):
        local = digits[src]
        open_mask = (local == 1).astype(float)
        D = contact_energy_changes(adjacency, open_mask)
        for c in range(n):
            c_local = int(local[c])
            rates = channel_transition_rates(
                c_local, bool(is_v[c]), float(D[c]), float(eps_v),
                nu=float(nu), nu_v=float(nu_v))
            for dst_local in range(4):
                if dst_local == c_local or rates[dst_local] <= 0.0:
                    continue
                dst = int(src + (dst_local - c_local) * strides[c])
                rows.append(dst)
                cols.append(src)
                data.append(float(rates[dst_local]))

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    data = np.array(data, dtype=float)

    # Conservative diagonal: Q[src, src] = -(total exit rate of src). Column sums of the
    # off-diagonal entries give the exit rates directly.
    exit_rate = np.zeros(n_states, dtype=float)
    np.add.at(exit_rate, cols, data)
    diag_rows = np.arange(n_states, dtype=np.int64)
    rows = np.concatenate([rows, diag_rows])
    cols = np.concatenate([cols, diag_rows])
    data = np.concatenate([data, -exit_rate])
    return rows, cols, data

def solve_master_equation(rows, cols, values, flux_per_state):
    """Reference implementation for solve_master_equation."""
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.sparse import csc_matrix

    DT, TPULSE = 0.02, 100.0  # ms, pinned grid (treatment's own resolution bookkeeping)
    RTOL, ATOL = 1e-8, 1e-11  # solver tolerances behind the measured graded numbers
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    values = np.asarray(values, dtype=float).ravel()
    flux_per_state = np.asarray(flux_per_state, dtype=float).ravel()
    if not (rows.size == cols.size == values.size):
        raise ValueError("coordinate arrays must be the same length")
    if not np.all(np.isfinite(values)):
        raise ValueError("generator entries must be finite")
    n_states = int(flux_per_state.size)
    if n_states == 0 or rows.max(initial=-1) >= n_states or cols.max(initial=-1) >= n_states:
        raise ValueError("flux vector does not cover the state space")
    if not np.all(np.isfinite(flux_per_state)):
        raise ValueError("flux observable must be finite")

    Q = csc_matrix((values, (rows, cols)), shape=(n_states, n_states))
    p0 = np.zeros(n_states)
    p0[0] = 1.0  # all-closed resting configuration is global state index 0
    ngrid = int(round(TPULSE / DT)) + 1
    tgrid = np.linspace(0.0, TPULSE, ngrid)
    sol = solve_ivp(lambda t, p: np.asarray(Q @ p), (0.0, TPULSE), p0, method="BDF",
                    t_eval=tgrid, jac=lambda t, p: Q, rtol=RTOL, atol=ATOL)
    if not sol.success:
        raise RuntimeError(f"BDF integration failed: {sol.message}")
    return flux_per_state @ sol.y

def extract_flux_features(flux_time_course):
    """Reference implementation for extract_flux_features."""
    import numpy as np

    DT = 0.02
    STEADY_TAIL_MS = 5.0
    f = np.asarray(flux_time_course, dtype=float)
    if f.ndim != 1:
        raise ValueError("flux_time_course must be one-dimensional")
    if not np.all(np.isfinite(f)):
        raise ValueError("flux_time_course must be finite")
    tail = int(round(STEADY_TAIL_MS / DT))
    if f.size < tail:
        raise ValueError("trace must contain at least one steady window")
    return np.array([float(f.max()), float(f[-tail:].mean())])

import numpy as np
from scipy.optimize import curve_fit


def fit_boltzmann(eps_family, peak_fluxes):
    """Reference implementation for fit_boltzmann."""
    import numpy as np
    from scipy.optimize import curve_fit

    def _fit_model(x, fm, xb, k):
        return fm / (1.0 + np.exp(-(x - xb) / k))

    eps_family = np.asarray(eps_family, dtype=float).ravel()
    peak_fluxes = np.asarray(peak_fluxes, dtype=float).ravel()
    if eps_family.size != peak_fluxes.size:
        raise ValueError("eps_family and peak_fluxes must be the same length")
    if eps_family.size < 3:
        raise ValueError("at least three points are needed for a three-parameter fit")
    if not (np.all(np.isfinite(eps_family)) and np.all(np.isfinite(peak_fluxes))):
        raise ValueError("inputs must be finite")
    p, _ = curve_fit(_fit_model, np.abs(eps_family), peak_fluxes,
                     p0=[float(peak_fluxes.max()), 7.0, 1.2], maxfev=200000)
    return np.array([float(p[0]), float(-p[1]), float(p[2])])

def couplon_kappa_mv(n_per_class=4):
    """Reference implementation for couplon_kappa_mv."""
    import numpy as np

    MV_PER_KT = 7.14
    EPS_FAMILY = (0., -2., -4., -6., -7., -8., -9., -10., -12., -14.)

    if isinstance(n_per_class, bool) or int(n_per_class) != n_per_class or int(n_per_class) < 2:
        raise ValueError("n_per_class must be an integer >= 2")
    n_per_class = int(n_per_class)

    is_v, adjacency = build_lattice_adjacency(n_per_class)

    # Flux observable per global state: open V channels count 1, open C channels count
    # R_CV = 5; inactivated channels count as closed. Rebuilt here with the same
    # mixed-radix encoding the generator uses, so the orchestrator is self-contained.
    is_v_bool = np.asarray(is_v, dtype=bool)
    dims = tuple(2 if v else 4 for v in is_v_bool)
    n_states = int(np.prod(dims))
    digits = np.stack(np.unravel_index(np.arange(n_states), dims), axis=1)
    occ = (digits == 1)
    flux_per_state = (occ & is_v_bool[None, :]).sum(axis=1) \
        + 5.0 * (occ & ~is_v_bool[None, :]).sum(axis=1)

    peaks = np.empty(len(EPS_FAMILY), dtype=float)
    for i, eps_v in enumerate(EPS_FAMILY):
        rows, cols, values = assemble_generator(is_v, adjacency, eps_v)
        trace = solve_master_equation(rows, cols, values, flux_per_state)
        peaks[i] = extract_flux_features(trace)[0]

    _fmax, _eps_bar, kappa = fit_boltzmann(np.array(EPS_FAMILY), peaks)
    return float(kappa * MV_PER_KT)
SCICODE_GOLD_EOF
