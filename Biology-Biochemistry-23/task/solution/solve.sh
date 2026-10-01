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
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def helical_end_to_end_distance(n_bp: int) -> float:
    """End-to-end distance (nm) of an n_bp duplex between its two force-bearing termini, SI eqs (15)-(17).

    Backbones r1(phi) = (R cos phi, R sin phi, p phi / 2 pi), r2(phi) = (-R cos phi, -R sin phi, p phi / 2 pi)
    with R = 1.0 nm, pitch p = 0.34 nm * 10.5, phi(n) = 2 pi n / 10.5. The termini are base 1 on strand 1
    and base n_bp on strand 2: axial separation 0.34 (n_bp - 1) nm and transverse separation
    2 R |cos(pi (n_bp - 1) / 10.5)| nm (2 nm for a single base pair).
    """
    if isinstance(n_bp, bool) or int(n_bp) != n_bp or int(n_bp) < 1:
        raise ValueError("n_bp must be an integer >= 1")
    n = int(n_bp)
    rise, radius, bp_per_turn = 0.34, 1.0, 10.5
    axial = rise * (n - 1)
    transverse = 2.0 * radius * abs(np.cos(np.pi * (n - 1) / bp_per_turn))
    return float(np.sqrt(axial * axial + transverse * transverse))

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def rod_extension(length: float, force: float, k_bt: float) -> float:
    """Mean extension (nm) of a rigid rod of the given length under force: L [coth(fL/kT) - kT/(fL)], eq. (5)."""
    if isinstance(length, bool) or not np.isfinite(length) or float(length) < 0.0:
        raise ValueError("length must be finite and nonnegative")
    f = _check_pos(force, "force")
    kt = _check_pos(k_bt, "k_bt")
    L = float(length)
    if L == 0.0:
        return 0.0
    x = f * L / kt
    return float(L * (1.0 / np.tanh(x) - 1.0 / x))

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def ssdna_extension(n_nt: int, force: float, k_bt: float, persistence_length: float,
                            interphosphate_distance: float) -> float:
    """Marko-Siggia worm-like-chain extension (nm) of an n_nt-nucleotide strand of contour length l_ss n_nt.

    Solves f lambda / kT = 1/(4 (1 - u)^2) - 1/4 + u for u = <z>/L on (0, 1) and returns L u; zero nucleotides
    give zero extension.
    """
    if isinstance(n_nt, bool) or int(n_nt) != n_nt or int(n_nt) < 0:
        raise ValueError("n_nt must be a nonnegative integer")
    m = int(n_nt)
    f = _check_pos(force, "force")
    kt = _check_pos(k_bt, "k_bt")
    lam = _check_pos(persistence_length, "persistence_length")
    l_ss = _check_pos(interphosphate_distance, "interphosphate_distance")
    if m == 0:
        return 0.0
    L = l_ss * m
    g = f * lam / kt
    u = brentq(lambda v: 1.0 / (4.0 * (1.0 - v) ** 2) - 0.25 + v - g, 0.0, 1.0 - 1e-12, xtol=1e-14, rtol=1e-14)
    return float(L * u)

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def _check_int(n, name, lo):
    if isinstance(n, bool) or int(n) != n or int(n) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(n)


def _thermal_energy_pn_nm(temperature):
    """k_B T in pN nm."""
    return 1.380649e-23 * temperature * 1e21


def _length_change(f, len_n, len_n1, n_released, kt, lam, l_ss):
    """dx_n(f): duplex rod gain minus single-strand loss for the (n-1) -> n base-pair formation."""
    ds = rod_extension(len_n, f, kt) - rod_extension(len_n1, f, kt)
    ss = (ssdna_extension(n_released, f, kt, lam, l_ss)
          - ssdna_extension(n_released + 1, f, kt, lam, l_ss))
    return ds + ss


def mechanical_work(n_closed: int, force: float, n_bp: int, temperature: float, persistence_length: float,
                            interphosphate_distance: float) -> float:
    """W_n(F) = int_0^F dx_n(f) df (pN nm), eqs (3)-(5): the work of the n-1 -> n base-pair formation step.

    dx_n(f) = [rod extension of the helical n-bp duplex - rod extension of the (n-1)-bp duplex]
              + [<z>_{N-n} - <z>_{N-n+1}] with the Marko-Siggia strand extensions; a 0-bp duplex has zero length.
    """
    n = _check_int(n_closed, "n_closed", 2)
    N = _check_int(n_bp, "n_bp", 2)
    if n > N:
        raise ValueError("n_closed must not exceed n_bp")
    F = _check_pos(force, "force")
    T = _check_pos(temperature, "temperature")
    lam = _check_pos(persistence_length, "persistence_length")
    l_ss = _check_pos(interphosphate_distance, "interphosphate_distance")
    kt = _thermal_energy_pn_nm(T)
    Ln = helical_end_to_end_distance(n)
    Ln1 = helical_end_to_end_distance(n - 1) if n - 1 >= 1 else 0.0
    val, _ = quad(_length_change, 1e-9, F, args=(Ln, Ln1, N - n, kt, lam, l_ss), limit=200, epsabs=1e-12, epsrel=1e-12)
    return float(val)

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def _check_int(n, name, lo):
    if isinstance(n, bool) or int(n) != n or int(n) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(n)


def _thermal_energy_pn_nm(temperature):
    """k_B T in pN nm."""
    return 1.380649e-23 * temperature * 1e21


def _thermal_energy_kj_mol(temperature):
    """R T in kJ/mol."""
    return 1.380649e-23 * 6.02214076e23 * temperature / 1000.0


def transition_rates(force: float, n_bp: int, temperature: float, dh_stack: float, ds_stack: float,
                             dh_non: float, ds_non: float, persistence_length: float,
                             interphosphate_distance: float) -> "np.ndarray":
    """[k_ot, k_o(2), ..., k_o(N), k_c(2), ..., k_c(N)] in units of k_a, eqs (6)-(8) with alpha = 1.

    Source convention (alpha = 1, transition state at the closed configuration): the whole mechanical work
    W_n(F) sits on the closing transition, k_c^{n-1->n} = exp(W_n / k_B T), and every opening rate stays
    force independent, k_o(n) = exp(dG_s / RT); k_ot = exp(dG_non / RT); dG = dH - T dS.
    """
    F = _check_pos(force, "force")
    N = _check_int(n_bp, "n_bp", 2)
    T = _check_pos(temperature, "temperature")
    for v, nm in ((dh_stack, "dh_stack"), (ds_stack, "ds_stack"), (dh_non, "dh_non"), (ds_non, "ds_non")):
        if isinstance(v, bool) or not np.isfinite(v):
            raise ValueError(f"{nm} must be finite")
    lam = _check_pos(persistence_length, "persistence_length")
    l_ss = _check_pos(interphosphate_distance, "interphosphate_distance")
    rt = _thermal_energy_kj_mol(T)
    kt = _thermal_energy_pn_nm(T)
    dg_s = float(dh_stack) - T * float(ds_stack)
    dg_non = float(dh_non) - T * float(ds_non)
    out = np.zeros(2 * N - 1, dtype=np.float64)
    out[0] = np.exp(dg_non / rt)
    for n in range(2, N + 1):
        w_n = mechanical_work(n, F, N, T, lam, l_ss)
        out[n - 1] = np.exp(dg_s / rt)                 # opening n -> n-1: alpha = 1, no force factor
        out[N + n - 2] = np.exp(w_n / kt)              # closing n-1 -> n carries exp(W_n / k_B T)
    return out

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_int(n, name, lo):
    if isinstance(n, bool) or int(n) != n or int(n) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(n)


def master_equation_generator(n_bp: int, rates: "np.ndarray") -> "np.ndarray":
    """Generator T of dP/dt = T P over the states (i, j), i + j <= n_bp - 1, plus the ruptured state last.

    States are ordered lexicographically by (i, j) (i = ruptured pairs from the left, j from the right, n = n_bp
    - i - j closed pairs) with the ruptured single-strand state as the final index;
    rates = [k_ot, k_o(2..N), k_c(2..N)]. From a state with n > 1 closed pairs each end opens at k_o(n); with
    n = 1 the last pair opens at k_ot into the ruptured state; an end with i > 0 (or j > 0) closes at
    k_c(n + 1); the ruptured state is absorbing.
    """
    N = _check_int(n_bp, "n_bp", 2)
    r = np.asarray(rates, dtype=float)
    if r.ndim != 1 or r.size != 2 * N - 1 or not np.all(np.isfinite(r)) or np.any(r < 0.0):
        raise ValueError("rates must be a finite nonnegative array of length 2 n_bp - 1")
    k_ot = r[0]
    k_o = {n: r[n - 1] for n in range(2, N + 1)}
    k_c = {n: r[N + n - 2] for n in range(2, N + 1)}
    states = [(i, j) for i in range(N) for j in range(N) if i + j <= N - 1]
    idx = {s: k for k, s in enumerate(states)}
    size = len(states) + 1
    last = size - 1
    Tm = np.zeros((size, size), dtype=np.float64)
    for (i, j), k in idx.items():
        n = N - i - j
        if n == 1:
            Tm[last, k] += k_ot
            Tm[k, k] -= k_ot
        else:
            for tgt in ((i + 1, j), (i, j + 1)):
                Tm[idx[tgt], k] += k_o[n]
                Tm[k, k] -= k_o[n]
        if i > 0:
            Tm[idx[(i - 1, j)], k] += k_c[n + 1]
            Tm[k, k] -= k_c[n + 1]
        if j > 0:
            Tm[idx[(i, j - 1)], k] += k_c[n + 1]
            Tm[k, k] -= k_c[n + 1]
    return Tm

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def _check_int(n, name, lo):
    if isinstance(n, bool) or int(n) != n or int(n) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(n)


def rupture_off_rate(generator: "np.ndarray", t_end: float, n_points: int) -> float:
    """Off-rate k (units of k_a) from a single-exponential fit P_ss(t) ~ 1 - exp(-k t), SI eq. (30) protocol.

    P(t) = expm(T t) P0 with P0 = full duplex (index 0); P_ss is the last component; sampled at n_points equally
    spaced times in [0, t_end] and fitted by unweighted nonlinear least squares with the single parameter k
    (initial guess 1 / t_end), iterated to convergence (parameter, residual and gradient tolerances 1e-14).
    """
    Tm = np.asarray(generator, dtype=float)
    if Tm.ndim != 2 or Tm.shape[0] != Tm.shape[1] or Tm.shape[0] < 2 or not np.all(np.isfinite(Tm)):
        raise ValueError("generator must be a finite square matrix of size >= 2")
    te = _check_pos(t_end, "t_end")
    npts = _check_int(n_points, "n_points", 3)
    w, V = np.linalg.eig(Tm)
    coef = np.linalg.solve(V, np.eye(Tm.shape[0])[:, 0])
    ts = np.linspace(0.0, te, npts)
    p_ss = np.real(V[-1, :] @ (np.exp(np.outer(w, ts)) * coef[:, None]))
    k, _ = curve_fit(lambda t, kk: 1.0 - np.exp(-kk * t), ts, p_ss, p0=[1.0 / te], xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return float(k[0])

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _check_pos(x, name):
    if isinstance(x, bool) or not np.isfinite(x) or float(x) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return float(x)


def _check_int(n, name, lo):
    if isinstance(n, bool) or int(n) != n or int(n) < lo:
        raise ValueError(f"{name} must be an integer >= {lo}")
    return int(n)


def _thermal_energy_pn_nm(temperature):
    """k_B T in pN nm."""
    return 1.380649e-23 * temperature * 1e21


def _thermal_energy_kj_mol(temperature):
    """R T in kJ/mol."""
    return 1.380649e-23 * 6.02214076e23 * temperature / 1000.0


def apparent_transition_state_distance(forces: "np.ndarray", n_bp: int, temperature: float, dh_stack: float,
                                               ds_stack: float, dh_non: float, ds_non: float, persistence_length: float,
                                               interphosphate_distance: float, n_points: int) -> float:
    """Apparent transition-state distance d (nm) = k_B T * slope of ln k(F) versus F over the given forces (Bell)."""
    Fs = np.asarray(forces, dtype=float)
    if Fs.ndim != 1 or Fs.size < 2 or not np.all(np.isfinite(Fs)) or np.any(Fs <= 0.0) or np.any(np.diff(Fs) <= 0.0):
        raise ValueError("forces must be an increasing 1-D array of at least two positive values")
    N = _check_int(n_bp, "n_bp", 2)
    T = _check_pos(temperature, "temperature")
    npts = _check_int(n_points, "n_points", 3)
    rt = _thermal_energy_kj_mol(T)
    kt = _thermal_energy_pn_nm(T)
    dg_s = float(dh_stack) - T * float(ds_stack)
    dg_non = float(dh_non) - T * float(ds_non)
    t_end = float(np.exp(-(dg_non + (N - 1) * dg_s) / rt))       # thermal dissociation time in 1/k_a
    # geometric and mechanical ingredients, evaluated here so that the chain is complete (reported, not returned)
    rod_lengths = [helical_end_to_end_distance(n) for n in range(1, N + 1)]
    rod_extension(rod_lengths[-1], float(Fs[0]), kt)
    ssdna_extension(1, float(Fs[0]), kt, persistence_length, interphosphate_distance)
    mechanical_work(2, float(Fs[0]), N, T, persistence_length, interphosphate_distance)
    ks = []
    for F in Fs:
        r = transition_rates(F, N, T, dh_stack, ds_stack, dh_non, ds_non, persistence_length, interphosphate_distance)
        Tm = master_equation_generator(N, r)
        ks.append(rupture_off_rate(Tm, t_end, npts))
    slope = np.polyfit(Fs, np.log(np.asarray(ks)), 1)[0]
    return float(kt * slope)
SCICODE_GOLD_EOF
