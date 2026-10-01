"""
Apply the photocurrent-versus-effective-voltage procedure to the exact model of one cell whose free-carrier generation carries the Onsager-Braun yield of the second step evaluated at the nominal field |F| = (V_bi - V)/d of each voltage (G = G_ex P_gen(F); a zero-field yield of 1 means unity, field-independent generation). With J_light(V) the current under that generation and J_dark(V) the dark current, the photocurrent is J_ph(V) = J_light(V) - J_dark(V); locate the compensation voltage V_0 at which J_ph = 0 by bisection on (0, V_bi + 0.4 V) to an interval width of 1e-10 V, the open-circuit voltage V_oc at which J_light = 0 the same way, and the maximum-power point by golden-section search of -V J_light(V) on (0, V_oc) to an interval width of 1e-9 V; the fill factor is P_max / (V_oc J_sc) with J_sc = -J_light(0). For each reverse bias V_rev the saturation current read by the procedure is J_sat,exp = J_ph(V_rev) and the apparent dissociation probability is P_diss = J_ph(0) / J_sat,exp. Return [V_0, J_sc / (q G_ex d), V_oc, V_mpp, FF, P_gen(F_sc) at short circuit, the short-circuit collection efficiency J_sc / (q G_ex d P_gen(F_sc))] followed by [-J_sat,exp / (q G_ex d), P_diss] for each reverse bias in order. Raise ValueError if the zero-field yield is not in (0, 1], if any reverse bias is not negative, if par does not have 14 entries or N is not an even integer of at least 4.

The procedure reads a single ratio of photocurrents, at short circuit and at a reverse bias, and reports it as an exciton dissociation probability; applied to a model in which the generation yield is prescribed, the ratio can be compared with what it is supposed to measure.

Returns
-------
A (7 + 2 len(reverse_biases),) float64 array [V_0, J_sc/(q G_ex d), V_oc, V_mpp, FF, P_gen(F_sc), collection efficiency, then -J_sat,exp/(q G_ex d) and P_diss per reverse bias].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def procedure_metrics(zero_field_yield: float, reverse_biases: "np.ndarray", intervals: int, par: "np.ndarray") -> "np.ndarray":
    """Apply the photocurrent-versus-effective-voltage procedure to the exact model of one cell
        whose free-carrier generation carries the Onsager-Braun yield of the second step
        evaluated at the nominal field |F| = (V_bi - V)/d of each voltage (G = G_ex P_gen(F); a
        zero-field yield of 1 means unity, field-independent generation). A (7 + 2
        len(reverse_biases),) float64 array [V_0, J_sc/(q G_ex d), V_oc, V_mpp, FF, P_gen(F_sc),
        collection efficiency, then -J_sat,exp/(q G_ex d) and P_diss per reverse bias].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded
def _Q():
    return 1.602176634e-19
def _KB():
    return 1.380649e-23
def _E0():
    return 8.8541878128e-12


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

def _bisect(f, a, b, tol):
    """Root of f on [a, b] (sign change required) to an interval width below tol."""
    fa = f(a); fb = f(b)
    if not (fa * fb < 0):
        raise ValueError("no sign change on the bracket")
    while b - a > tol:
        m = 0.5 * (a + b); fm = f(m)
        if fa * fm <= 0: b, fb = m, fm
        else: a, fa = m, fm
    return 0.5 * (a + b)

def _golden_max(f, a, b, tol):
    """Maximiser of f on [a, b] by golden-section search to an interval width below tol; returns (x, f(x))."""
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - g * (b - a); d = a + g * (b - a); fc = f(c); fd = f(d)
    while b - a > tol:
        if fc > fd:
            b, d, fd = d, c, fc; c = b - g * (b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd; d = a + g * (b - a); fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)

def _oracle_procedure_metrics(zero_field_yield: float, reverse_biases: "np.ndarray", intervals: int, par: "np.ndarray") -> "np.ndarray":
    """The Jph-Veff procedure and the light metrics of one device whose generation carries the Onsager-Braun yield
    evaluated at the nominal field (Vbi - V)/d."""
    P0 = float(zero_field_yield); Vrs = np.asarray(reverse_biases, dtype=float).ravel(); N = int(intervals); par = np.asarray(par, dtype=float).ravel()
    if not (0.0 < P0 <= 1.0) or Vrs.size == 0 or np.any(Vrs >= 0.0) or par.size != 14 or N < 4 or N % 2 != 0:
        raise ValueError("the zero-field yield must lie in (0, 1], the reverse biases must be negative, par the 14-entry vector and intervals an even integer of at least 4")
    prm = _dd_params(par); VT = prm["VT"]; Vbi = prm["Vbi"]; qGd = prm["qGd"]
    eps = prm["ee"] / _E0(); T = VT * _Q() / _KB()
    sw = _Sweep(par, N)
    def _yld(V):
        return 1.0 if P0 >= 1.0 else float(_oracle_onsager_braun_yield((Vbi - V) / prm["d"], P0, eps, T)[0])
    def _jlight(V): return sw._current(V, _yld(V))
    def _jdark(V): return sw._current(V, 0.0)
    def _jph(V): return _jlight(V) - _jdark(V)
    Jsc = -_jlight(0.0); Jph0 = _jph(0.0)
    V0 = _bisect(_jph, 0.0, Vbi + 0.4, 1e-10)
    Voc = _bisect(_jlight, 0.0, Vbi + 0.4, 1e-10)
    Vm, Pm = _golden_max(lambda V: -V * _jlight(V), 0.0, Voc, 1e-9)
    FF = Pm / (Voc * Jsc)
    Pg_sc = _yld(0.0)
    out = [V0, Jsc / qGd, Voc, Vm, FF, Pg_sc, Jsc / (qGd * Pg_sc)]
    for Vr in Vrs:
        Jsat = _jph(Vr); out += [-Jsat / qGd, Jph0 / Jsat]
    return np.array(out, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nzero_field_yield = 1.0\nreverse_biases = np.array([-1.0, -3.0])\nintervals = 200\n',
         'call': 'procedure_metrics(zero_field_yield, reverse_biases, intervals, par)',
         'gold_call': '_oracle_procedure_metrics(zero_field_yield, reverse_biases, intervals, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 0.25, 0.25, 2.0e-4, 2.0e-4, 1.0, 1.0e22)\nzero_field_yield = 0.8\nreverse_biases = np.array([-1.0])\nintervals = 400\n',
         'call': 'procedure_metrics(zero_field_yield, reverse_biases, intervals, par)',
         'gold_call': '_oracle_procedure_metrics(zero_field_yield, reverse_biases, intervals, par)'},
        {'setup': 'import numpy as np\npar = _oracle_device_parameters(150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 0.30, 0.15, 5.0e-4, 1.0e-4, 0.2, 5.0e21)\nzero_field_yield = 0.6\nreverse_biases = np.array([-2.0])\nintervals = 200\n',
         'call': 'procedure_metrics(zero_field_yield, reverse_biases, intervals, par)',
         'gold_call': '_oracle_procedure_metrics(zero_field_yield, reverse_biases, intervals, par)'},
    ]
