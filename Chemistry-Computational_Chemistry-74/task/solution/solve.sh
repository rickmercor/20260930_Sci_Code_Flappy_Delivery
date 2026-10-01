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


def _angular_per_wavenumber() -> float:
    """Angular frequency in rad/ps carried by one cm^-1 (2 pi c with c in cm/ps)."""
    import numpy as np
    return 2.0 * np.pi * 2.99792458e-2


def drude_bath_exponents(reorg_cm: float, tau_bath_fs: float, temperature_k: float,
                                 n_matsubara: int) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(reorg_cm) or reorg_cm < 0.0:
        raise ValueError("reorganization energy must be finite and non-negative")
    if not np.isfinite(tau_bath_fs) or tau_bath_fs <= 0.0:
        raise ValueError("bath relaxation time must be positive")
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if int(n_matsubara) != n_matsubara or n_matsubara < 0:
        raise ValueError("number of Matsubara terms must be a non-negative integer")
    w2c = _angular_per_wavenumber()
    lam = reorg_cm * w2c
    gam = 1000.0 / tau_bath_fs
    beta = 1.0 / (0.6950348 * temperature_k * w2c)
    out = np.zeros((int(n_matsubara) + 1, 4))
    out[0] = [lam * gam / np.tan(0.5 * beta * gam), -lam * gam, gam, 0.0]
    for k in range(1, int(n_matsubara) + 1):
        nu = 2.0 * np.pi * k / beta
        if abs(nu - gam) <= 1e-12 * gam:
            raise ValueError("a Matsubara frequency coincides with the Drude cutoff")
        out[k] = [4.0 * lam * gam / beta * nu / (nu * nu - gam * gam), 0.0, nu, 0.0]
    return out

import numpy as np


def _angular_per_wavenumber() -> float:
    """Angular frequency in rad/ps carried by one cm^-1 (2 pi c with c in cm/ps)."""
    import numpy as np
    return 2.0 * np.pi * 2.99792458e-2


def underdamped_mode_exponents(reorg_cm: float, mode_cm: float, tau_damp_fs: float,
                                       temperature_k: float, n_matsubara: int) -> "np.ndarray":
    import numpy as np
    if not np.isfinite(reorg_cm) or reorg_cm < 0.0:
        raise ValueError("reorganization energy must be finite and non-negative")
    if not np.isfinite(mode_cm) or mode_cm <= 0.0:
        raise ValueError("mode frequency must be positive")
    if not np.isfinite(tau_damp_fs) or tau_damp_fs <= 0.0:
        raise ValueError("damping time must be positive")
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature must be positive")
    if int(n_matsubara) != n_matsubara or n_matsubara < 0:
        raise ValueError("number of Matsubara terms must be a non-negative integer")
    w2c = _angular_per_wavenumber()
    lam = reorg_cm * w2c
    w0 = mode_cm * w2c
    gam = 1000.0 / tau_damp_fs
    if w0 <= 0.5 * gam:
        raise ValueError("the mode must be underdamped: omega_0 > gamma / 2")
    beta = 1.0 / (0.6950348 * temperature_k * w2c)
    om = np.sqrt(w0 * w0 - 0.25 * gam * gam)
    out = np.zeros((int(n_matsubara) + 2, 4))
    for row, s in ((0, -1.0), (1, 1.0)):
        z = s * om - 0.5j * gam
        amp = lam * w0 * w0 / (2.0 * s * om) * (1.0 / np.tanh(0.5 * beta * z) + 1.0)
        rate = 1j * z
        out[row] = [amp.real, amp.imag, rate.real, rate.imag]
    for k in range(1, int(n_matsubara) + 1):
        nu = 2.0 * np.pi * k / beta
        amp = -4.0 * lam * gam * w0 * w0 / beta * nu / ((w0 * w0 + nu * nu) ** 2 - gam * gam * nu * nu)
        out[k + 1] = [amp, 0.0, nu, 0.0]
    return out

import numpy as np


def _ladder(n_photon: int) -> np.ndarray:
    """Truncated photon annihilation operator on Fock states 0..n_photon-1."""
    import numpy as np
    return np.diag(np.sqrt(np.arange(1, n_photon, dtype=float)), 1)


def _electronic(m: np.ndarray, n_photon: int) -> np.ndarray:
    """Embed a 2x2 electronic operator (order D, A) in the electron-photon product space."""
    import numpy as np
    return np.kron(m, np.eye(n_photon))


def dipole_gauge_hamiltonian(driving_force_cm: float, electronic_coupling_cm: float,
                                     cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float,
                                     n_photon: int) -> "np.ndarray":
    import numpy as np
    if int(n_photon) != n_photon or n_photon < 1:
        raise ValueError("n_photon must be a positive integer")
    if not np.isfinite(cavity_energy_cm) or cavity_energy_cm <= 0.0:
        raise ValueError("cavity photon energy must be positive")
    for v in (driving_force_cm, electronic_coupling_cm, x_dd, x_aa, x_da):
        if not np.isfinite(v):
            raise ValueError("parameters must be finite")
    n = int(n_photon)
    a = _ladder(n)
    ph = np.kron(np.eye(2), a)
    p_d = _electronic(np.diag([1.0, 0.0]), n)
    p_a = _electronic(np.diag([0.0, 1.0]), n)
    sx = _electronic(np.array([[0.0, 1.0], [1.0, 0.0]]), n)
    mu = 1j * (x_dd * p_d + x_aa * p_a + x_da * sx)
    h = (-driving_force_cm * p_a + electronic_coupling_cm * sx
         + cavity_energy_cm * (ph.T @ ph)
         + cavity_energy_cm * (ph - ph.T) @ mu
         - cavity_energy_cm * (mu @ mu))
    return 0.5 * (h + h.conj().T)

import numpy as np


def _ladder(n_photon: int) -> np.ndarray:
    """Truncated photon annihilation operator on Fock states 0..n_photon-1."""
    import numpy as np
    return np.diag(np.sqrt(np.arange(1, n_photon, dtype=float)), 1)


def _electronic(m: np.ndarray, n_photon: int) -> np.ndarray:
    """Embed a 2x2 electronic operator (order D, A) in the electron-photon product space."""
    import numpy as np
    return np.kron(m, np.eye(n_photon))


def bath_coupling_operators(n_photon: int) -> "np.ndarray":
    import numpy as np
    if int(n_photon) != n_photon or n_photon < 1:
        raise ValueError("n_photon must be a positive integer")
    n = int(n_photon)
    ph = np.kron(np.eye(2), _ladder(n))
    field = 1j * (ph - ph.T)
    p_a = _electronic(np.diag([0.0, 1.0]), n).astype(complex)
    q_el = _electronic(np.array([[1.0, 1.0], [1.0, 1.0]]), n)
    return np.array([p_a, field @ q_el])

import numpy as np


def _angular_per_wavenumber() -> float:
    """Angular frequency in rad/ps carried by one cm^-1 (2 pi c with c in cm/ps)."""
    import numpy as np
    return 2.0 * np.pi * 2.99792458e-2


def _ladder(n_photon: int) -> np.ndarray:
    """Truncated photon annihilation operator on Fock states 0..n_photon-1."""
    import numpy as np
    return np.diag(np.sqrt(np.arange(1, n_photon, dtype=float)), 1)


def system_liouvillian(hamiltonian_cm: "np.ndarray", kappa_cm: float, n_photon: int) -> "np.ndarray":
    import numpy as np
    h = np.asarray(hamiltonian_cm, dtype=complex)
    if int(n_photon) != n_photon or n_photon < 1:
        raise ValueError("n_photon must be a positive integer")
    if h.ndim != 2 or h.shape != (2 * int(n_photon), 2 * int(n_photon)):
        raise ValueError("hamiltonian must be square with dimension 2 * n_photon")
    if not np.allclose(h, h.conj().T, rtol=0.0, atol=1e-10):
        raise ValueError("hamiltonian must be Hermitian")
    if not np.isfinite(kappa_cm) or kappa_cm < 0.0:
        raise ValueError("cavity loss rate must be non-negative")
    w2c = _angular_per_wavenumber()
    d = h.shape[0]
    eye = np.eye(d)
    hw = h * w2c
    liou = -1j * (np.kron(hw, eye) - np.kron(eye, hw.T))
    a = np.kron(np.eye(2), _ladder(int(n_photon)))
    num = a.T @ a
    k = kappa_cm * w2c
    liou = liou + k * (np.kron(a, a) - 0.5 * np.kron(num, eye) - 0.5 * np.kron(eye, num.T))
    return liou

import numpy as np


def heom_propagate(liouvillian: "np.ndarray", coupling_ops: "np.ndarray", exponents: "np.ndarray",
                           bath_index: "np.ndarray", depth: int, rho0: "np.ndarray", dt_ps: float,
                           n_steps: int, stride: int) -> "np.ndarray":
    import numpy as np
    lv = np.asarray(liouvillian, dtype=complex)
    ops = np.asarray(coupling_ops, dtype=complex)
    ex = np.asarray(exponents, dtype=float)
    bi = np.asarray(bath_index)
    r0 = np.asarray(rho0, dtype=complex)
    if ops.ndim != 3 or ops.shape[1] != ops.shape[2] or ops.shape[0] < 1:
        raise ValueError("coupling_ops must have shape (B, d, d)")
    n_bath, d = ops.shape[0], ops.shape[1]
    if lv.shape != (d * d, d * d) or r0.shape != (d, d):
        raise ValueError("liouvillian, coupling operators and rho0 dimensions do not match")
    for v in ops:
        if not np.allclose(v, v.conj().T, rtol=0.0, atol=1e-10):
            raise ValueError("coupling operators must be Hermitian")
    if ex.ndim != 2 or ex.shape[1] != 4 or ex.shape[0] < 1 or not np.all(np.isfinite(ex)):
        raise ValueError("exponents must be a finite array of shape (K, 4)")
    if bi.shape != (ex.shape[0],) or not np.all(np.equal(np.mod(bi, 1), 0)) or bi.min() < 0 or bi.max() >= n_bath:
        raise ValueError("bath_index must hold one valid bath number per exponential term")
    bi = bi.astype(int)
    amp = ex[:, 0] + 1j * ex[:, 1]
    rate = ex[:, 2] + 1j * ex[:, 3]
    conj_amp = np.zeros_like(amp)
    for k in range(ex.shape[0]):
        same = np.where(bi == bi[k])[0]
        gap = np.abs(rate[same] - np.conj(rate[k]))
        if gap.min() > 1e-9 * max(1.0, abs(rate[k])):
            raise ValueError("the rates of each bath must be closed under complex conjugation")
        conj_amp[k] = np.conj(amp[same[int(np.argmin(gap))]])
    if int(depth) != depth or depth < 0:
        raise ValueError("depth must be a non-negative integer")
    if not np.isfinite(dt_ps) or dt_ps <= 0.0:
        raise ValueError("time step must be positive")
    if int(n_steps) != n_steps or n_steps < 0 or int(stride) != stride or stride < 1 or n_steps % stride:
        raise ValueError("n_steps must be a non-negative multiple of the positive integer stride")
    depth, n_steps, stride = int(depth), int(n_steps), int(stride)
    import scipy.sparse as sp
    n_mode = ex.shape[0]
    index = [()]
    for _ in range(n_mode):
        index = [t + (j,) for t in index for j in range(depth + 1)]
    index = sorted([t for t in index if sum(t) <= depth], key=lambda t: (sum(t), t))
    where = {t: i for i, t in enumerate(index)}
    n_ado = len(index)
    dd = d * d
    eye = np.eye(d)
    left = [sp.csr_matrix(np.kron(ops[bi[k]], eye)) for k in range(n_mode)]
    right = [sp.csr_matrix(np.kron(eye, ops[bi[k]].T)) for k in range(n_mode)]
    ident = sp.identity(dd, format='csr', dtype=complex)
    l_sys = sp.csr_matrix(lv)
    blocks = {}
    for i, t in enumerate(index):
        blocks[(i, i)] = l_sys - complex(sum(t[k] * rate[k] for k in range(n_mode))) * ident
        for k in range(n_mode):
            j = where.get(t[:k] + (t[k] + 1,) + t[k + 1:])
            if j is not None:
                blocks[(i, j)] = blocks.get((i, j), 0) - 1j * (left[k] - right[k])
            if t[k] > 0:
                j = where[t[:k] + (t[k] - 1,) + t[k + 1:]]
                blocks[(i, j)] = blocks.get((i, j), 0) - 1j * t[k] * (amp[k] * left[k] - conj_amp[k] * right[k])
    gen = sp.bmat([[blocks.get((i, j)) for j in range(n_ado)] for i in range(n_ado)], format='csr')
    vec = np.zeros(n_ado * dd, dtype=complex)
    vec[:dd] = r0.reshape(-1)
    frames = [r0.copy()]
    for s in range(1, n_steps + 1):
        k1 = gen @ vec
        k2 = gen @ (vec + 0.5 * dt_ps * k1)
        k3 = gen @ (vec + 0.5 * dt_ps * k2)
        k4 = gen @ (vec + dt_ps * k3)
        vec = vec + dt_ps / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if s % stride == 0:
            frames.append(vec[:dd].reshape(d, d).copy())
    return np.array(frames)

import numpy as np


def donor_population_trace(driving_force_cm: float, electronic_coupling_cm: float,
                                   cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float,
                                   kappa_cm: float, reorg_cm: float, tau_bath_fs: float,
                                   mode_reorg_cm: float, mode_cm: float, tau_mode_fs: float,
                                   temperature_k: float, n_matsubara_bath: int, n_matsubara_mode: int,
                                   depth: int, n_photon: int, t_max_ps: float, dt_ps: float,
                                   sample_ps: float) -> "np.ndarray":
    import numpy as np
    if not (np.isfinite(t_max_ps) and np.isfinite(dt_ps) and np.isfinite(sample_ps)) or min(t_max_ps, dt_ps, sample_ps) <= 0.0:
        raise ValueError("t_max, dt and the sampling interval must be positive")
    stride = int(round(sample_ps / dt_ps))
    n_steps = int(round(t_max_ps / dt_ps))
    if abs(stride * dt_ps - sample_ps) > 1e-9 * sample_ps or abs(n_steps * dt_ps - t_max_ps) > 1e-9 * t_max_ps or n_steps % stride:
        raise ValueError("t_max must be a multiple of the sampling interval, which must be a multiple of dt")
    if not np.isfinite(mode_reorg_cm) or mode_reorg_cm < 0.0:
        raise ValueError("mode reorganization energy must be finite and non-negative")
    n = int(n_photon)
    ham = dipole_gauge_hamiltonian(driving_force_cm, electronic_coupling_cm, cavity_energy_cm,
                                           x_dd, x_aa, x_da, n)
    ops = bath_coupling_operators(n)
    liou = system_liouvillian(ham, kappa_cm, n)
    ex_b = drude_bath_exponents(reorg_cm, tau_bath_fs, temperature_k, n_matsubara_bath)
    ex_m = underdamped_mode_exponents(mode_reorg_cm, mode_cm, tau_mode_fs, temperature_k,
                                              n_matsubara_mode)
    if mode_reorg_cm > 0.0:
        ex = np.vstack([ex_b, ex_m])
        index = np.array([0] * ex_b.shape[0] + [1] * ex_m.shape[0])
    else:
        ex = ex_b
        index = np.zeros(ex_b.shape[0], dtype=int)
    rho0 = np.zeros((2 * n, 2 * n), dtype=complex)
    rho0[0, 0] = 1.0
    frames = heom_propagate(liou, ops, ex, index, depth, rho0, dt_ps, n_steps, stride)
    return np.real(np.einsum('tii->t', frames[:, :n, :n]))

import numpy as np


def fit_transfer_rates(times_ps: "np.ndarray", p_donor: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy.optimize import minimize_scalar
    t = np.asarray(times_ps, dtype=float)
    p = np.asarray(p_donor, dtype=float)
    if t.ndim != 1 or t.shape != p.shape or t.size < 3:
        raise ValueError("times and populations must be 1-D arrays of equal length >= 3")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(p))):
        raise ValueError("inputs must be finite")
    if abs(t[0]) > 1e-12 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at zero and increase strictly")

    def _best(logk):
        e = np.exp(-np.exp(logk) * t)
        u = 1.0 - e
        den = u @ u
        pinf = 0.0 if den == 0.0 else float(np.clip(((p - e) @ u) / den, 0.0, 1.0))
        r = e + pinf * u - p
        return r @ r, pinf

    grid = np.linspace(np.log(1e-3 / t[-1]), np.log(1e3 / t[1]), 2001)
    vals = np.array([_best(g)[0] for g in grid])
    j = int(np.argmin(vals))
    lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, grid.size - 1)]
    res = minimize_scalar(lambda g: _best(g)[0], bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-13, 'maxiter': 500})
    logk = res.x if res.fun <= vals[j] else grid[j]
    k = float(np.exp(logk))
    pinf = _best(logk)[1]
    return np.array([k * (1.0 - pinf), k * pinf])

import numpy as np


def cavity_rate_enhancement(driving_force_cm: float, electronic_coupling_cm: float,
                                    cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float,
                                    kappa_cm: float, reorg_cm: float, tau_bath_fs: float,
                                    mode_reorg_cm: float, mode_cm: float, tau_mode_fs: float,
                                    temperature_k: float, n_matsubara_bath: int, n_matsubara_mode: int,
                                    depth: int, n_photon: int, t_max_ps: float, dt_ps: float,
                                    sample_ps: float) -> float:
    import numpy as np
    cav = donor_population_trace(driving_force_cm, electronic_coupling_cm, cavity_energy_cm,
                                         x_dd, x_aa, x_da, kappa_cm, reorg_cm, tau_bath_fs,
                                         mode_reorg_cm, mode_cm, tau_mode_fs, temperature_k,
                                         n_matsubara_bath, n_matsubara_mode, depth, n_photon,
                                         t_max_ps, dt_ps, sample_ps)
    bare = donor_population_trace(driving_force_cm, electronic_coupling_cm, cavity_energy_cm,
                                          0.0, 0.0, 0.0, 0.0, reorg_cm, tau_bath_fs,
                                          0.0, mode_cm, tau_mode_fs, temperature_k,
                                          n_matsubara_bath, n_matsubara_mode, depth, 1,
                                          t_max_ps, dt_ps, sample_ps)
    n_steps = int(round(t_max_ps / dt_ps))
    stride = int(round(sample_ps / dt_ps))
    times = dt_ps * np.arange(0, n_steps + 1, stride)
    k_cav = fit_transfer_rates(times, cav)
    k_bare = fit_transfer_rates(times, bare)
    if k_bare[0] <= 0.0:
        raise ValueError("the cavity-free forward rate vanished; the ratio is undefined")
    return float(k_cav[0] / k_bare[0])
SCICODE_GOLD_EOF
