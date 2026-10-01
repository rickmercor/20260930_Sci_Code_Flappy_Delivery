"""
Short-circuit collection efficiency -(J_light(0) - J_dark(0)) / (s q G_ex d) of a cell with symmetric barriers phi and reduction factor zeta (material as in the seventh step) under the generation s G_ex for each intensity s (suns) and each balanced mobility (rows). Return the (len(mobilities), len(intensities)) array. Raise ValueError if intensities or mobilities are empty or not positive, the barrier is negative or twice the barrier reaches the gap, the reduction factor is negative, material does not have 7 entries or N is not an even integer of at least 4.

A collection loss that is independent of the illumination intensity is a first-order process; one that shrinks as the intensity is lowered is a second-order, bimolecular process. Comparing a transport-limited cell with an ideal-transport cell across intensities exposes which loss the ceiling on the collection efficiency belongs to.

Returns
-------
A (len(mobilities), len(intensities)) float64 array of short-circuit collection efficiencies.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def intensity_dependence(intensities: "np.ndarray", mobilities: "np.ndarray", barrier: float, reduction_factor: float, intervals: int, material: "np.ndarray") -> "np.ndarray":
    """Short-circuit collection efficiency -(J_light(0) - J_dark(0)) / (s q G_ex d) of a cell
        with symmetric barriers phi and reduction factor zeta (material as in the seventh step)
        under the generation s G_ex for each intensity s (suns) and each balanced mobility
        (rows). A (len(mobilities), len(intensities)) float64 array of short-circuit collection
        efficiencies.
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

def _oracle_intensity_dependence(intensities: "np.ndarray", mobilities: "np.ndarray", barrier: float, reduction_factor: float, intervals: int, material: "np.ndarray") -> "np.ndarray":
    """Short-circuit collection efficiency against the illumination intensity for each balanced mobility (rows)."""
    Is = np.asarray(intensities, dtype=float).ravel(); mus = np.asarray(mobilities, dtype=float).ravel()
    phi = float(barrier); g = float(reduction_factor); N = int(intervals); mat = [float(v) for v in np.asarray(material, dtype=float).ravel()]
    if Is.size == 0 or np.any(Is <= 0) or mus.size == 0 or np.any(mus <= 0) or phi < 0 or g < 0 or len(mat) != 7 or N < 4 or N % 2 != 0:
        raise ValueError("positive intensities and mobilities, non-negative barrier and reduction factor, material the 7-entry tuple, intervals an even integer of at least 4")
    d_nm, eps, T, Nc, Nv, Eg, Gex = mat
    if 2.0 * phi >= Eg:
        raise ValueError("twice the barrier must lie below the gap")
    out = np.empty((mus.size, Is.size))
    for i, mu in enumerate(mus):
        pv = _oracle_device_parameters(d_nm, eps, T, Nc, Nv, Eg, phi, phi, mu, mu, g, Gex)
        prm = _dd_params(pv); sw = _Sweep(pv, N); Jd = sw._current(0.0, 0.0)
        for j, s in enumerate(Is):
            out[i, j] = -(sw._current(0.0, s) - Jd) / (prm["qGd"] * s)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nmaterial = (100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 1.0e22)\nintensities = np.array([0.01, 1.0])\nmobilities = np.array([2.0e-4, 1.0])\nbarrier = 0.25\nreduction_factor = 1.0\nintervals = 200\n',
         'call': 'intensity_dependence(intensities, mobilities, barrier, reduction_factor, intervals, material)',
         'gold_call': '_oracle_intensity_dependence(intensities, mobilities, barrier, reduction_factor, intervals, material)'},
        {'setup': 'import numpy as np\nmaterial = (100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 1.0e22)\nintensities = np.array([0.1, 0.5, 2.0])\nmobilities = np.array([1.0e-3])\nbarrier = 0.20\nreduction_factor = 0.1\nintervals = 200\n',
         'call': 'intensity_dependence(intensities, mobilities, barrier, reduction_factor, intervals, material)',
         'gold_call': '_oracle_intensity_dependence(intensities, mobilities, barrier, reduction_factor, intervals, material)'},
        {'setup': 'import numpy as np\nmaterial = (150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 5.0e21)\nintensities = np.array([0.01, 1.0])\nmobilities = np.array([5.0e-4, 1.0])\nbarrier = 0.30\nreduction_factor = 0.2\nintervals = 200\n',
         'call': 'intensity_dependence(intensities, mobilities, barrier, reduction_factor, intervals, material)',
         'gold_call': '_oracle_intensity_dependence(intensities, mobilities, barrier, reduction_factor, intervals, material)'},
    ]
