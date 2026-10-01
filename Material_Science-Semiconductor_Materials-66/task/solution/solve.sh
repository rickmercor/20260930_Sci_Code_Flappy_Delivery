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


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def log_intrinsic_carrier_density(T: float, NC300: float, NV300: float) -> float:
    for name, v in (("T", T), ("NC300", NC300), ("NV300", NV300)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    T = float(T)
    s = (T / 300.0) ** 1.5
    return float(0.5 * (np.log(float(NC300) * s) + np.log(float(NV300) * s))
                 - _bandgap(T) / (2.0 * _thermal_voltage(T)))

import numpy as np


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def _psi_bulk(T, NA, gA, EA):
    """Bulk potential from charge neutrality, evaluated stably in log space."""
    ut = _thermal_voltage(T)
    lni = log_intrinsic_carrier_density(T, 2.86e19, 2.66e19)
    psiA = EA - _bandgap(T) / 2.0
    S = np.log(4.0 * gA * NA) + psiA / ut - lni
    corr = np.logaddexp(0.0, 0.5 * np.logaddexp(0.0, S)) - np.log(2.0)
    return float(ut * ((lni - np.log(NA)) + corr))


def bulk_potential(T: float, NA: float, gA: float, EA_above_EV: float) -> float:
    for name, v in (("T", T), ("NA", NA), ("gA", gA), ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    return _psi_bulk(float(T), float(NA), float(gA), float(EA_above_EV))

import numpy as np


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def work_function_difference(T: float, NA: float, phi_m: float, chi_si: float,
                             gA: float, EA_above_EV: float) -> float:
    for name, v in (("T", T), ("NA", NA), ("phi_m", phi_m), ("chi_si", chi_si),
                    ("gA", gA), ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    T = float(T)
    psib = bulk_potential(T, float(NA), float(gA), float(EA_above_EV))
    return float(float(phi_m) - (float(chi_si) + _bandgap(T) / 2.0 - psib))

import numpy as np


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def _occupation(z, g):
    """Overflow-free 1 / (1 + g * exp(z))."""
    u = np.clip(np.asarray(z, dtype=float) + np.log(g), -700.0, 700.0)
    e = np.exp(-np.abs(u))
    return np.where(u > 0.0, e / (1.0 + e), 1.0 / (1.0 + e))


def _gauss_legendre(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def interface_trap_charge(psi_s: float, Vch: float, T: float, DitC: float,
                          Edecay: float, gt: float, n_nodes: int) -> float:
    for name, v in (("psi_s", psi_s), ("Vch", Vch)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("DitC", DitC), ("Edecay", Edecay), ("gt", gt)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")
    Q = 1.602176634e-19
    T = float(T)
    eg = _bandgap(T)
    ut = _thermal_voltage(T)
    dE, w = _gauss_legendre(0.0, eg / 2.0, int(n_nodes))
    psi_t = eg / 2.0 - dE
    f = _occupation((psi_t - (float(psi_s) - float(Vch))) / ut, float(gt))
    return float(-Q * np.sum(w * float(DitC) * np.exp(-dE / float(Edecay)) * f))

import numpy as np
from typing import Union


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def band_tail_carrier_density(psi: Union[float, np.ndarray], Vch: float, T: float,
                                      UBT: float, Usp: float, Voffset: float,
                                      n_nodes: int) -> Union[float, np.ndarray]:
    arr = np.asarray(psi, dtype=float)
    if arr.ndim > 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("psi must be a finite scalar or a finite 1D array")
    for name, v in (("Vch", Vch), ("Voffset", Voffset)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("UBT", UBT), ("Usp", Usp)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    T = float(T); Vch = float(Vch)
    ut = _thermal_voltage(T)
    psi_X = Vch + _bandgap(T) / 2.0 + float(Voffset)
    log_nX = log_intrinsic_carrier_density(T, 2.86e19, 2.66e19) + (psi_X - Vch) / ut

    p = np.atleast_1d(arr)
    x, w = np.polynomial.legendre.leggauss(int(n_nodes))
    t = 0.5 * (p[:, None] - psi_X) * x[None, :] + 0.5 * (p[:, None] + psi_X)
    ww = 0.5 * (p[:, None] - psi_X) * w[None, :]
    W = 1.0 / (1.0 + np.exp(np.clip((t - psi_X) / float(Usp), -700.0, 700.0)))
    U_stat = W * float(UBT) + (1.0 - W) * ut
    out = np.exp(np.clip(log_nX + np.sum(ww / U_stat, axis=1), -700.0, 700.0))
    return float(out[0]) if arr.ndim == 0 else out

import numpy as np


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def _occupation(z, g):
    """Overflow-free 1 / (1 + g * exp(z))."""
    u = np.clip(np.asarray(z, dtype=float) + np.log(g), -700.0, 700.0)
    e = np.exp(-np.abs(u))
    return np.where(u > 0.0, e / (1.0 + e), 1.0 / (1.0 + e))


def depletion_charge_density(psi_s: float, Vch: float, T: float, NA: float,
                             gA: float, EA_above_EV: float) -> float:
    for name, v in (("psi_s", psi_s), ("Vch", Vch)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("NA", NA), ("gA", gA), ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    Q = 1.602176634e-19
    EPS_SI = 11.7 * 8.8541878128e-14
    T = float(T); NA = float(NA); gA = float(gA); EA = float(EA_above_EV)
    psi_s = float(psi_s); Vch = float(Vch)
    ut = _thermal_voltage(T)
    psib = bulk_potential(T, NA, gA, EA)
    if psi_s < psib:
        raise ValueError("psi_s must not lie below the bulk potential")
    psiA = EA - _bandgap(T) / 2.0
    fs = _occupation((psiA - (psi_s - Vch)) / ut, gA)
    fb = _occupation((psiA - (psib - Vch)) / ut, gA)
    G = (psi_s - psib) - ut * np.log(fs / fb)
    return float(-np.sqrt(2.0 * Q * NA * EPS_SI * max(0.0, G)))

import numpy as np


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def _occupation(z, g):
    """Overflow-free 1 / (1 + g * exp(z))."""
    u = np.clip(np.asarray(z, dtype=float) + np.log(g), -700.0, 700.0)
    e = np.exp(-np.abs(u))
    return np.where(u > 0.0, e / (1.0 + e), 1.0 / (1.0 + e))


def _gauss_legendre(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def mobile_charge_density(psi_s: float, Vch: float, T: float, NA: float, UBT: float,
                          Usp: float, Voffset: float, gA: float, EA_above_EV: float,
                          n_nodes: int) -> float:
    for name, v in (("psi_s", psi_s), ("Vch", Vch), ("Voffset", Voffset)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("NA", NA), ("UBT", UBT), ("Usp", Usp), ("gA", gA),
                    ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    Q = 1.602176634e-19
    EPS_SI = 11.7 * 8.8541878128e-14
    T = float(T); NA = float(NA); gA = float(gA); EA = float(EA_above_EV)
    psi_s = float(psi_s); Vch = float(Vch); n_nodes = int(n_nodes)
    ut = _thermal_voltage(T)
    psib = bulk_potential(T, NA, gA, EA)
    if psi_s < psib:
        raise ValueError("psi_s must not lie below the bulk potential")

    t, w = _gauss_legendre(psib, psi_s, n_nodes)
    n = band_tail_carrier_density(t, Vch, T, UBT, Usp, Voffset, n_nodes)
    NAm = NA * _occupation(((EA - _bandgap(T) / 2.0) - (t - Vch)) / ut, gA)
    Es2 = max(0.0, (2.0 * Q / EPS_SI) * float(np.sum(w * (n + NAm))))
    Qsc = -EPS_SI * np.sqrt(Es2)
    return float(Qsc - depletion_charge_density(psi_s, Vch, T, NA, gA, EA))

import numpy as np


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def surface_potential(VGB: float, Vch: float, T: float, NA: float, Cox: float,
                      phi_ms: float, DitC: float, Edecay: float, UBT: float,
                      Usp: float, Voffset: float, gt: float, gA: float,
                      EA_above_EV: float, n_nodes: int, tol: float) -> float:
    for name, v in (("VGB", VGB), ("Vch", Vch), ("phi_ms", phi_ms), ("Voffset", Voffset)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("NA", NA), ("Cox", Cox), ("DitC", DitC), ("Edecay", Edecay),
                    ("UBT", UBT), ("Usp", Usp), ("gt", gt), ("gA", gA),
                    ("EA_above_EV", EA_above_EV), ("tol", tol)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    T = float(T); NA = float(NA); Cox = float(Cox); Vch = float(Vch)
    gA = float(gA); EA = float(EA_above_EV); n_nodes = int(n_nodes)
    psib = bulk_potential(T, NA, gA, EA)

    def _residual(ps):
        Qm = mobile_charge_density(ps, Vch, T, NA, UBT, Usp, Voffset, gA, EA, n_nodes)
        Qd = depletion_charge_density(ps, Vch, T, NA, gA, EA)
        Qit = interface_trap_charge(ps, Vch, T, DitC, Edecay, gt, n_nodes)
        return (float(phi_ms) - Qit / Cox) + (ps - psib) - (Qm + Qd) / Cox - float(VGB)

    lo = psib + 1e-9
    hi = _bandgap(T) / 2.0 + Vch + 0.20
    flo, fhi = _residual(lo), _residual(hi)
    if flo * fhi > 0.0:
        raise ValueError("surface potential is not bracketed on the search interval")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = _residual(mid)
        if flo * fm <= 0.0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if hi - lo < float(tol):
            break
    return float(0.5 * (lo + hi))

import numpy as np


def effective_mobility(Qdep: float, Qm: float, mu0: float, theta1: float,
                       theta2: float, eta: float) -> float:
    for name, v in (("Qdep", Qdep), ("Qm", Qm)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    if not np.isscalar(mu0) or not np.isfinite(float(mu0)) or float(mu0) <= 0.0:
        raise ValueError("mu0 must be a finite strictly positive scalar")
    for name, v in (("theta1", theta1), ("theta2", theta2), ("eta", eta)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) < 0.0:
            raise ValueError("%s must be a finite non-negative scalar" % name)
    EPS_SI = 11.7 * 8.8541878128e-14
    E0 = 1.0e6
    Eeff = (abs(float(Qdep)) + float(eta) * abs(float(Qm))) / EPS_SI
    F = Eeff / E0
    return float(float(mu0) / (1.0 + float(theta1) * F + float(theta2) * F * F))

import numpy as np


def _bandgap(T):
    """Silicon band gap in eV at temperature T in kelvin."""
    return 1.170 - 4.730e-4 * T * T / (T + 636.0)


def _thermal_voltage(T):
    """Thermal voltage kT/q in volts."""
    return 1.380649e-23 * T / 1.602176634e-19


def _occupation(z, g):
    """Overflow-free 1 / (1 + g * exp(z))."""
    u = np.clip(np.asarray(z, dtype=float) + np.log(g), -700.0, 700.0)
    e = np.exp(-np.abs(u))
    return np.where(u > 0.0, e / (1.0 + e), 1.0 / (1.0 + e))


def _gauss_legendre(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def coupling_factor(psi_s: float, Vch: float, T: float, NA: float, Cox: float,
                    DitC: float, Edecay: float, gt: float, gA: float,
                    EA_above_EV: float, n_nodes: int) -> float:
    for name, v in (("psi_s", psi_s), ("Vch", Vch)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    for name, v in (("T", T), ("NA", NA), ("Cox", Cox), ("DitC", DitC),
                    ("Edecay", Edecay), ("gt", gt), ("gA", gA), ("EA_above_EV", EA_above_EV)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    Q = 1.602176634e-19
    EPS_SI = 11.7 * 8.8541878128e-14
    T = float(T); NA = float(NA); gA = float(gA); EA = float(EA_above_EV)
    psi_s = float(psi_s); Vch = float(Vch); n_nodes = int(n_nodes)
    ut = _thermal_voltage(T)
    eg = _bandgap(T)
    psib = bulk_potential(T, NA, gA, EA)
    if psi_s <= psib:
        raise ValueError("psi_s must lie strictly above the bulk potential")

    psiA = EA - eg / 2.0
    fs = _occupation((psiA - (psi_s - Vch)) / ut, gA)
    fb = _occupation((psiA - (psib - Vch)) / ut, gA)
    G = (psi_s - psib) - ut * np.log(fs / fb)
    Cdep = Q * NA * EPS_SI * fs / np.sqrt(2.0 * Q * NA * EPS_SI * G)

    dE, w = _gauss_legendre(0.0, eg / 2.0, n_nodes)
    ft = _occupation(((eg / 2.0 - dE) - (psi_s - Vch)) / ut, float(gt))
    Cit = (Q / ut) * np.sum(w * float(DitC) * np.exp(-dE / float(Edecay)) * ft * (1.0 - ft))

    return float(1.0 + (float(Cdep) + float(Cit)) / float(Cox))

import numpy as np


def drain_current_microamp(T: float, NA: float, tox: float, Wch: float, Lch: float,
                           VGB: float, VDS: float, phi_m: float, chi_si: float,
                           DitC: float, Edecay: float, UBT: float, Usp: float,
                           Voffset: float, mu0: float, theta1: float, theta2: float,
                           eta: float, gt: float, gA: float, EA_above_EV: float,
                           n_nodes: int, tol: float) -> float:
    for name, v in (("VGB", VGB), ("Voffset", Voffset)):
        if not np.isscalar(v) or not np.isfinite(float(v)):
            raise ValueError("%s must be a finite scalar" % name)
    if not np.isscalar(VDS) or not np.isfinite(float(VDS)) or float(VDS) < 0.0:
        raise ValueError("VDS must be a finite non-negative scalar")
    for name, v in (("T", T), ("NA", NA), ("tox", tox), ("Wch", Wch), ("Lch", Lch),
                    ("phi_m", phi_m), ("chi_si", chi_si), ("DitC", DitC),
                    ("Edecay", Edecay), ("UBT", UBT), ("Usp", Usp), ("mu0", mu0),
                    ("gt", gt), ("gA", gA), ("EA_above_EV", EA_above_EV), ("tol", tol)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) <= 0.0:
            raise ValueError("%s must be a finite strictly positive scalar" % name)
    for name, v in (("theta1", theta1), ("theta2", theta2), ("eta", eta)):
        if not np.isscalar(v) or not np.isfinite(float(v)) or float(v) < 0.0:
            raise ValueError("%s must be a finite non-negative scalar" % name)
    if not isinstance(n_nodes, (int, np.integer)) or int(n_nodes) < 2:
        raise ValueError("n_nodes must be an integer greater than or equal to 2")

    KB = 1.380649e-23
    Q = 1.602176634e-19
    EPS_OX = 3.9 * 8.8541878128e-14
    T = float(T); VDS = float(VDS); n_nodes = int(n_nodes)
    ut = KB * T / Q
    Cox = EPS_OX / float(tox)

    phi_ms = work_function_difference(T, NA, phi_m, chi_si, gA, EA_above_EV)

    psi_S = surface_potential(VGB, 0.0, T, NA, Cox, phi_ms, DitC, Edecay,
                                      UBT, Usp, Voffset, gt, gA, EA_above_EV, n_nodes, tol)
    psi_D = surface_potential(VGB, VDS, T, NA, Cox, phi_ms, DitC, Edecay,
                                      UBT, Usp, Voffset, gt, gA, EA_above_EV, n_nodes, tol)

    Qm_S = mobile_charge_density(psi_S, 0.0, T, NA, UBT, Usp, Voffset,
                                         gA, EA_above_EV, n_nodes)
    Qm_D = mobile_charge_density(psi_D, VDS, T, NA, UBT, Usp, Voffset,
                                         gA, EA_above_EV, n_nodes)

    Qdep_S = depletion_charge_density(psi_S, 0.0, T, NA, gA, EA_above_EV)
    mu_n = effective_mobility(Qdep_S, Qm_S, mu0, theta1, theta2, eta)
    m = coupling_factor(psi_S, 0.0, T, NA, Cox, DitC, Edecay, gt, gA,
                                EA_above_EV, n_nodes)

    I = (float(Wch) / float(Lch)) * mu_n * (
        -(Qm_S ** 2 - Qm_D ** 2) / (2.0 * m * Cox) + ut * (Qm_S - Qm_D))
    return float(abs(I) * 1.0e6)
SCICODE_GOLD_EOF
