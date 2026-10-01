"""
Current density (mA/cm^2, forward bias positive) of the cell at each applied voltage under the generation s G_ex, from the converged steady states of the third step (continuation from zero bias outward, so that the states at neighbouring voltages start each other). Return the currents in the order of the voltages given. Raise ValueError if voltages is empty, par does not have 14 entries, N is not an even integer of at least 4, the generation scale is negative, or any voltage is not below V_bi + 0.5 V.

Light and dark current-voltage characteristics are the only steady-state observables of a solar cell; the difference between them, the photocurrent, is what the procedure under examination interprets.

Returns
-------
A (len(voltages),) float64 array of current densities in mA/cm^2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def jv_characteristics(voltages: "np.ndarray", generation_scale: float, intervals: int, par: "np.ndarray") -> "np.ndarray":
    """Current density (mA/cm^2, forward bias positive) of the cell at each applied voltage
        under the generation s G_ex, from the converged steady states of the third step
        (continuation from zero bias outward, so that the states at neighbouring voltages start
        each other). A (len(voltages),) float64 array of current densities in mA/cm^2.
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

def _dd_fluxes(psi, n, p, prm, h):
    """Scharfetter-Gummel electron and hole current densities (A/m^2) on the N cell midpoints."""
    dl = np.diff(psi) / prm["VT"]; Bp = _bernoulli(dl); Bm = _bernoulli(-dl)
    Jn = _Q() * prm["Dn"] / h * (n[1:] * Bp - n[:-1] * Bm)
    Jp = _Q() * prm["Dp"] / h * (p[:-1] * Bp - p[1:] * Bm)
    return Jn, Jp

def _Sweep(par, N):
    class _SweepState:
        """Cache of converged states of one cell keyed by (bias, generation scale); every new state is continued from
        the nearest cached one through the third step's oracle."""
        def __init__(self, par, N):
            self.par = np.asarray(par, dtype=float).ravel(); self.prm = _dd_params(self.par); self.N = N; self.cache = {}
    
        def _state(self, V, s):
            key = (round(float(V), 12), round(float(s), 12))
            if key in self.cache: return self.cache[key]
            same = [k for k in self.cache if k[1] == key[1]]
            lit = [k for k in self.cache if k[1] > 0.0]
            if same:
                k0 = min(same, key=lambda k: abs(k[0] - key[0]))
            elif s > 0.0 and lit:
                k0 = min(lit, key=lambda k: abs(k[0] - key[0]))
            else:
                k0 = None
            st = _oracle_steady_state(V, s, self.N, self.par, None if k0 is None else (self.cache[k0], k0[0]))
            self.cache[key] = st
            return st
    
        def _current(self, V, s):
            st = self._state(V, s); prm = self.prm; h = prm["d"] / self.N
            psi = st[:, 0] * prm["VT"]; n = np.exp(st[:, 1]); p = np.exp(st[:, 2])
            Jn, Jp = _dd_fluxes(psi, n, p, prm, h)
            return float((Jn + Jp)[self.N // 2])
    return _SweepState(par, N)

def _oracle_jv_characteristics(voltages: "np.ndarray", generation_scale: float, intervals: int, par: "np.ndarray") -> "np.ndarray":
    """Current density (mA/cm^2) at each bias, by continuation outward from zero bias."""
    Vs = np.asarray(voltages, dtype=float).ravel(); par = np.asarray(par, dtype=float).ravel(); N = int(intervals); s = float(generation_scale)
    if Vs.size == 0 or par.size != 14 or N < 4 or N % 2 != 0 or s < 0.0:
        raise ValueError("voltages must be non-empty, par the 14-entry vector, intervals an even integer of at least 4 and the generation scale non-negative")
    prm = _dd_params(par)
    if np.any(Vs >= prm["Vbi"] + 0.5):
        raise ValueError("every bias must lie below the built-in voltage plus 0.5 V")
    sw = _Sweep(par, N)
    out = np.empty(Vs.size)
    order = np.argsort(np.abs(Vs))
    for k in order:
        out[k] = sw._current(Vs[k], s) / 10.0
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nvoltages = np.array([-1.0, -0.5, 0.0, 0.4, 0.7, 0.85])\ngeneration_scale = 1.0\nintervals = 200\n',
         'call': 'jv_characteristics(voltages, generation_scale, intervals, par)',
         'gold_call': '_oracle_jv_characteristics(voltages, generation_scale, intervals, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nvoltages = np.array([0.85, 0.0, -1.0, 0.5])\ngeneration_scale = 0.0\nintervals = 200\n',
         'call': 'jv_characteristics(voltages, generation_scale, intervals, par)',
         'gold_call': '_oracle_jv_characteristics(voltages, generation_scale, intervals, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 0.30, 0.15, 5.0e-4, 1.0e-4, 0.2, 5.0e21)\nvoltages = np.array([-2.0, 0.0, 0.6, 1.0])\ngeneration_scale = 0.3\nintervals = 200\n',
         'call': 'jv_characteristics(voltages, generation_scale, intervals, par)',
         'gold_call': '_oracle_jv_characteristics(voltages, generation_scale, intervals, par)'},
    ]
