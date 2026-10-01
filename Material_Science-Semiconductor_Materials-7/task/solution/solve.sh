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

def charge_state_formation_energies(e_neutral: float, charges: "np.ndarray", levels: "np.ndarray") -> "np.ndarray":
    charges = np.asarray(charges, dtype=int).ravel()
    levels = np.asarray(levels, dtype=float).ravel()
    if charges.size == 0 or not np.any(charges == 0):
        raise ValueError("charges must contain the neutral state 0")
    if charges.size > 1 and not np.all(np.diff(charges) == -1):
        raise ValueError("charges must decrease in steps of one")
    if levels.size != charges.size - 1:
        raise ValueError("levels must have length len(charges) - 1")

    energies = np.empty(charges.size, dtype=float)
    i0 = int(np.flatnonzero(charges == 0)[0])
    energies[i0] = float(e_neutral)
    # Walk outward from the neutral state; states i and i+1 cross at levels[i].
    for i in range(i0 - 1, -1, -1):
        energies[i] = energies[i + 1] + (charges[i + 1] - charges[i]) * levels[i]
    for i in range(i0 + 1, charges.size):
        energies[i] = energies[i - 1] + (charges[i - 1] - charges[i]) * levels[i - 1]
    return energies

import numpy as np

def band_edges_and_densities(temperature: float, host: dict) -> "np.ndarray":
    t = float(temperature)
    if not t > 0.0:
        raise ValueError("temperature must be positive")
    f_cb = float(host["f_cb"])
    if not 0.0 <= f_cb <= 1.0:
        raise ValueError("f_cb must lie in [0, 1]")
    d_gap = -float(host["varshni_alpha"]) * t * t / (t + float(host["varshni_beta"]))
    ec = float(host["eg0"]) + f_cb * d_gap
    ev = -(1.0 - f_cb) * d_gap
    scale = (t / float(host["t_ref"])) ** 1.5
    return np.array([ec, ev, float(host["nc_ref"]) * scale, float(host["nv_ref"]) * scale])

import numpy as np

def vibrational_free_energy_shift(temperature: float, hw0: float, n_added: "np.ndarray") -> "np.ndarray":
    kb_ev = 8.617333262e-5  # Boltzmann constant (eV/K)
    t = float(temperature)
    if not t > 0.0:
        raise ValueError("temperature must be positive")
    if not float(hw0) > 0.0:
        raise ValueError("hw0 must be positive")
    x = float(hw0) / (kb_ev * t)
    # Entropy (in kB) of one quantum harmonic oscillator, written in an overflow-safe form.
    s_mode = x / np.expm1(x) - np.log(-np.expm1(-x))
    n = np.asarray(n_added, dtype=float).ravel()
    return -3.0 * n * kb_ev * t * s_mode

import numpy as np

def open_charge_state_concentrations(temperature: float, fermi_level: float, mu_dopant: float, host: dict, defects: dict) -> "np.ndarray":
    kb_ev = 8.617333262e-5  # Boltzmann constant (eV/K)
    t = float(temperature)
    if not t > 0.0:
        raise ValueError("temperature must be positive")
    cs_def = np.asarray(defects["cs_def"], dtype=int).ravel()
    cs_q = np.asarray(defects["cs_q"], dtype=float).ravel()
    cs_e = np.asarray(defects["cs_e"], dtype=float).ravel()
    if not (cs_def.size == cs_q.size == cs_e.size):
        raise ValueError("per-charge-state arrays must have equal lengths")
    site = np.asarray(defects["site"], dtype=float).ravel()
    n_dop = np.asarray(defects["n_dopant"], dtype=float).ravel()
    g_vib = vibrational_free_energy_shift(t, host["hw0"], defects["n_added"])
    g = cs_e + cs_q * float(fermi_level) - n_dop[cs_def] * float(mu_dopant) + g_vib[cs_def]
    return site[cs_def] * np.exp(-g / (kb_ev * t))

import numpy as np
from scipy.special import expi

def remaining_diffusion_length(temperature: float, em: float, d0: float, gamma: float, t_min: float) -> float:
    kb_ev = 8.617333262e-5  # Boltzmann constant (eV/K)
    t = float(temperature)
    t_lo = float(t_min)
    if not (float(em) > 0.0 and float(d0) > 0.0 and float(gamma) > 0.0 and t_lo > 0.0):
        raise ValueError("em, d0, gamma and t_min must be positive")
    if t < t_lo:
        raise ValueError("temperature must not be below t_min")
    a = float(em) / kb_ev

    def _antiderivative(tt):
        return tt * np.exp(-a / tt) + a * expi(-a / tt)

    x2 = float(d0) / float(gamma) * (_antiderivative(t) - _antiderivative(t_lo))
    return float(np.sqrt(max(x2, 0.0)))

import numpy as np
from scipy.optimize import brentq

def freeze_in_temperature(em: float, d0: float, gamma: float, t_max: float, t_min: float, grain_size: float) -> float:
    if not (float(em) > 0.0 and float(d0) > 0.0 and float(gamma) > 0.0 and float(grain_size) > 0.0):
        raise ValueError("em, d0, gamma and grain_size must be positive")
    t_hi = float(t_max)
    t_lo = float(t_min)
    if not 0.0 < t_lo < t_hi:
        raise ValueError("temperatures must satisfy 0 < t_min < t_max")
    sink_distance = 0.5 * float(grain_size)

    def _excess(t):
        return remaining_diffusion_length(t, em, d0, gamma, t_lo) - sink_distance

    if _excess(t_hi) <= 0.0:
        return t_hi
    return float(brentq(_excess, t_lo, t_hi, xtol=1e-10, rtol=1e-15, maxiter=500))

import numpy as np
from scipy.optimize import brentq

def _frozen_fractions(g, cs_def, idx, kt):
    """Charge-state fractions of defect idx from relative formation energies g."""
    sel = cs_def == idx
    gs = g[sel]
    w = np.exp(-(gs - gs.min()) / kt)
    return sel, w / w.sum()

def solve_partial_equilibrium(temperature: float, frozen_totals: "np.ndarray", dopant_total: float, host: dict, defects: dict) -> "np.ndarray":
    kb_ev = 8.617333262e-5  # Boltzmann constant (eV/K)
    t = float(temperature)
    if not t > 0.0:
        raise ValueError("temperature must be positive")
    site = np.asarray(defects["site"], dtype=float).ravel()
    n_def = site.size
    frozen = np.array(frozen_totals, dtype=float).ravel()
    if frozen.size != n_def:
        raise ValueError("frozen_totals must have one entry per defect")
    is_frozen = np.isfinite(frozen)
    if np.any(frozen[is_frozen] < 0.0):
        raise ValueError("frozen totals must be non-negative")
    cs_def = np.asarray(defects["cs_def"], dtype=int).ravel()
    cs_q = np.asarray(defects["cs_q"], dtype=float).ravel()
    cs_e = np.asarray(defects["cs_e"], dtype=float).ravel()
    n_dop = np.asarray(defects["n_dopant"], dtype=int).ravel()

    ec, ev, nc, nv = band_edges_and_densities(t, host)
    kt = kb_ev * t
    open_dopant = (~is_frozen) & (n_dop == 1)
    dopant_left = float(dopant_total) - float(np.sum(frozen[is_frozen] * n_dop[is_frozen]))
    if np.any(open_dopant) and not dopant_left > 0.0:
        raise ValueError("no dopant left for the open dopant-containing defects")
    open_dopant_cs = open_dopant[cs_def]

    def _populations(ef):
        conc = open_charge_state_concentrations(t, ef, 0.0, host, defects)
        if np.any(open_dopant_cs):
            conc[open_dopant_cs] *= dopant_left / conc[open_dopant_cs].sum()
        g_rel = cs_e + cs_q * ef
        for idx in np.flatnonzero(is_frozen):
            sel, frac = _frozen_fractions(g_rel, cs_def, idx, kt)
            conc[sel] = frozen[idx] * frac
        n = nc * np.exp((ef - ec) / kt)
        p = nv * np.exp((ev - ef) / kt)
        return conc, n, p

    def _net_charge(ef):
        conc, n, p = _populations(ef)
        return float(np.sum(cs_q * conc) + p - n)

    ef = brentq(_net_charge, ev - 1.0, ec + 1.0, xtol=1e-14, rtol=1e-15, maxiter=1000)
    conc, n, p = _populations(ef)
    totals = np.bincount(cs_def, weights=conc, minlength=n_def)
    return np.concatenate(([ef, n, p], totals))

import numpy as np

def sequential_freeze_in_totals(process: dict, host: dict, defects: dict) -> "np.ndarray":
    em = np.asarray(defects["em"], dtype=float).ravel()
    t_freeze = np.array([
        freeze_in_temperature(e, process["d0"], process["gamma"], process["t_max"], process["t_min"], process["grain_size"])
        for e in em
    ])
    frozen = np.full(em.size, np.nan)
    # Highest freeze-in temperature first; each species keeps the total it has when it freezes.
    for idx in np.argsort(-t_freeze, kind="stable"):
        state = solve_partial_equilibrium(t_freeze[idx], frozen, process["dopant_total"], host, defects)
        frozen[idx] = state[3 + idx]
    return frozen

import numpy as np

def dopant_activation_ratio(process: dict, host: dict, defects: dict) -> float:
    cs_def, cs_q, cs_e = [], [], []
    for j, (ch, e0, lv) in enumerate(zip(defects["charges"], defects["e_neutral"], defects["levels"])):
        e = charge_state_formation_energies(e0, np.asarray(ch), np.asarray(lv, dtype=float))
        cs_def += [j] * len(ch)
        cs_q += list(ch)
        cs_e += list(e)
    full = {k: np.asarray(defects[k]) for k in ("site", "n_added", "n_dopant", "em")}
    full.update(cs_def=np.array(cs_def), cs_q=np.array(cs_q), cs_e=np.array(cs_e))
    totals = sequential_freeze_in_totals(process, host, full)
    state = solve_partial_equilibrium(process["t_min"], totals, process["dopant_total"], host, full)
    return float(state[2] / float(process["dopant_total"]))
SCICODE_GOLD_EOF
