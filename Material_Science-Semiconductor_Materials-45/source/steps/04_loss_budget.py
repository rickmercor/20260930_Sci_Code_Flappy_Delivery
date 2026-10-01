"""
From the converged light state (generation s G_ex) and dark state (s = 0) of the second step at the same voltage V, evaluate the currents and the exact budget of the photogenerated carriers. The current density of a state is the total Scharfetter-Gummel current J_n + J_p on the middle interval (it is the same on every interval). The photocurrent is J_light - J_dark, the collected fraction its magnitude over q G d. With the excess densities Delta n = n_light - n_dark and Delta p = p_light - p_dark at the nodes, the excess recombination gamma_L (n_light p_light - n_dark p_dark) splits exactly into a first-order part gamma_L (n_dark Delta p + p_dark Delta n), recombination of photogenerated carriers with the dark (injected) populations, and a second-order part gamma_L Delta n Delta p; integrate each over the interior nodes with weight h and divide by G d. The remaining loss is extraction of photogenerated minority carriers at the wrong contact: the excess electron current on the first interval (light minus dark) plus the generation q G h/2 of the anode half-cell, over q G d, and likewise for the excess hole current on the last interval plus q G h/2 at the cathode. Return [J_light, J_dark (both in mA/cm^2), collected fraction, first-order bulk loss, second-order bulk loss, anode extraction loss, cathode extraction loss]; the five fractions sum to one. Raise ValueError if the two states are not (N + 1, 3) arrays of the same shape, if par does not have 14 entries, if the generation scale is not positive, if either state does not satisfy the contact potentials at V, or if either is not a converged steady state at its generation (any scaled residual above 1e-9).

A photogenerated carrier that is not collected is lost either to bimolecular recombination with another photogenerated carrier, a second-order process that weakens at low intensity, or to a first-order process that does not: recombination with the equilibrium charge the contacts inject, or diffusion into the wrong contact. Separating the two in the exact model tells which loss a fill factor or a saturating photocurrent is reporting.

Returns
-------
A (7,) float64 array [J_light, J_dark, collected, first-order, second-order, anode extraction, cathode extraction].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def loss_budget(state_light: "np.ndarray", state_dark: "np.ndarray", voltage: float, generation_scale: float, par: "np.ndarray") -> "np.ndarray":
    """From the converged light state (generation s G_ex) and dark state (s = 0) of the second
        step at the same voltage V, evaluate the currents and the exact budget of the
        photogenerated carriers. A (7,) float64 array [J_light, J_dark, collected, first-order,
        second-order, anode extraction, cathode extraction].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19


def _bernoulli(x):
    """B(x) = x / (exp(x) - 1) with a series branch near zero."""
    x = np.asarray(x, dtype=float); out = np.empty_like(x); s = np.abs(x) < 1e-8
    out[s] = 1.0 - x[s] / 2.0 + x[s] ** 2 / 12.0
    xb = x[~s]; out[~s] = xb / np.expm1(xb)
    return out

def _dd_params(par):
    """Dictionary of SI quantities from the parameter vector of the first step, with the generation scaled by s."""
    VT, Vbi, d, ee, p_an, n_an, n_cat, p_cat, ni2, gb, Dn, Dp, G, qGd = [float(v) for v in par]
    return dict(VT=VT, Vbi=Vbi, d=d, ee=ee, p_an=p_an, n_an=n_an, n_cat=n_cat, p_cat=p_cat, ni2=ni2, gamma=gb, Dn=Dn, Dp=Dp, G=G, qGd=qGd)

def _dd_unpack(state, prm, V):
    """Full node arrays (psi in V, n and p in 1/m^3) from the interior state (M, 3) = [psi/VT, ln n, ln p]."""
    VT = prm["VT"]
    psi = np.concatenate([[0.0], state[:, 0] * VT, [prm["Vbi"] - V]])
    n = np.concatenate([[prm["n_an"]], np.exp(state[:, 1]), [prm["n_cat"]]])
    p = np.concatenate([[prm["p_an"]], np.exp(state[:, 2]), [prm["p_cat"]]])
    return psi, n, p

def _dd_fluxes(psi, n, p, prm, h):
    """Scharfetter-Gummel electron and hole current densities (A/m^2) on the N cell midpoints."""
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q() * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q() * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _dd_residual(state, prm, V, h, G):
    """Scaled residuals (M, 3): Poisson in units of VT, continuity per carrier at each interior node; G in 1/(m^3 s)."""
    psi, n, p = _dd_unpack(state, prm, V); VT = prm["VT"]; ee = prm["ee"]
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    U = prm["gamma"] * (n * p - prm["ni2"]) - G
    Fpsi = ((psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / h ** 2 + _Q() / ee * (p[1:-1] - n[1:-1])) * h ** 2 / VT
    Fn = ((Jn[1:] - Jn[:-1]) / h - _Q() * U[1:-1]) * h ** 2 / (_Q() * prm["Dn"] * n[1:-1])
    Fp = ((Jp[1:] - Jp[:-1]) / h + _Q() * U[1:-1]) * h ** 2 / (_Q() * prm["Dp"] * p[1:-1])
    return np.stack([Fpsi, Fn, Fp], axis=1)

def _dd_current(state, prm, V, N):
    """Total current density (A/m^2) at the mid-cell, with the SG fluxes of the full node arrays."""
    h = prm["d"] / N; psi, n, p = _dd_unpack(state, prm, V)
    Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
    return float((Jn + Jp)[N // 2]), Jn, Jp, psi, n, p

def _oracle_loss_budget(state_light: "np.ndarray", state_dark: "np.ndarray", voltage: float, generation_scale: float, par: "np.ndarray") -> "np.ndarray":
    """Currents and the exact budget of the photogenerated carriers at one bias: collected, lost to first-order and
    second-order bulk recombination, and extracted at the wrong contacts; all in units of the generation current."""
    sl = np.asarray(state_light, dtype=float); sd = np.asarray(state_dark, dtype=float); par = np.asarray(par, dtype=float).ravel()
    V = float(voltage); s = float(generation_scale)
    if sl.ndim != 2 or sl.shape[1] != 3 or sl.shape[0] < 5 or sl.shape != sd.shape:
        raise ValueError("the two states must be (N + 1, 3) arrays of the same shape")
    if par.size != 14 or s <= 0.0:
        raise ValueError("par must be the 14-entry parameter vector and the generation scale positive")
    N = sl.shape[0] - 1; prm = _dd_params(par); h = prm["d"] / N; G = prm["G"] * s
    for st in (sl, sd):
        if abs(st[0, 0]) > 1e-9 or abs(st[-1, 0] * prm["VT"] - (prm["Vbi"] - V)) > 1e-9:
            raise ValueError("a state does not satisfy the contact potentials at this bias")
    Ul = sl[1:-1].copy(); Ud = sd[1:-1].copy()
    if np.max(np.abs(_dd_residual(Ul, prm, V, h, G))) > 1e-9 or np.max(np.abs(_dd_residual(Ud, prm, V, h, 0.0))) > 1e-9:
        raise ValueError("a state is not a converged steady state at this bias and generation")
    Jl, Jnl, Jpl, psil, nl, pl = _dd_current(Ul, prm, V, N)
    Jd, Jnd, Jpd, psid, nd, pd = _dd_current(Ud, prm, V, N)
    qGd = _Q() * G * prm["d"]
    dn = nl - nd; dp = pl - pd
    R1 = prm["gamma"] * (nd * dp + pd * dn); R2 = prm["gamma"] * dn * dp
    first = _Q() * h * R1[1:N].sum() / qGd; second = _Q() * h * R2[1:N].sum() / qGd
    # contact currents of the excess carriers: the first-interval flux plus the generation of the boundary half-cell
    anode = ((Jnl[0] - Jnd[0]) + _Q() * G * h / 2.0) / qGd        # electrons extracted by the anode
    cathode = ((Jpl[-1] - Jpd[-1]) + _Q() * G * h / 2.0) / qGd    # holes extracted by the cathode
    collected = -(Jl - Jd) / qGd
    return np.array([Jl / 10.0, Jd / 10.0, collected, first, second, anode, cathode], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nvoltage = 0.0\ngeneration_scale = 1.0\nintervals = 200\nstate_light = _oracle_steady_state(voltage, generation_scale, intervals, par)\nstate_dark = _oracle_steady_state(voltage, 0.0, intervals, par)\n',
         'call': 'loss_budget(state_light, state_dark, voltage, generation_scale, par)',
         'gold_call': '_oracle_loss_budget(state_light, state_dark, voltage, generation_scale, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nvoltage = -1.0\ngeneration_scale = 1.0\nintervals = 400\nstate_light = _oracle_steady_state(voltage, generation_scale, intervals, par)\nstate_dark = _oracle_steady_state(voltage, 0.0, intervals, par)\n',
         'call': 'loss_budget(state_light, state_dark, voltage, generation_scale, par)',
         'gold_call': '_oracle_loss_budget(state_light, state_dark, voltage, generation_scale, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 0.30, 0.15, 5.0e-4, 1.0e-4, 0.2, 5.0e21)\nvoltage = 0.3\ngeneration_scale = 0.5\nintervals = 200\nstate_light = _oracle_steady_state(voltage, generation_scale, intervals, par)\nstate_dark = _oracle_steady_state(voltage, 0.0, intervals, par)\n',
         'call': 'loss_budget(state_light, state_dark, voltage, generation_scale, par)',
         'gold_call': '_oracle_loss_budget(state_light, state_dark, voltage, generation_scale, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.10, 0.10, 1.0e-2, 1.0e-2, 0.01, 1.0e22)\nvoltage = 0.0\ngeneration_scale = 0.1\nintervals = 200\nstate_light = _oracle_steady_state(voltage, generation_scale, intervals, par)\nstate_dark = _oracle_steady_state(voltage, 0.0, intervals, par)\n',
         'call': 'loss_budget(state_light, state_dark, voltage, generation_scale, par)',
         'gold_call': '_oracle_loss_budget(state_light, state_dark, voltage, generation_scale, par)'},
    ]
