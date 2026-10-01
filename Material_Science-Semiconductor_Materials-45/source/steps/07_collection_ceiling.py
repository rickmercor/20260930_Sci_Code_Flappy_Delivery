"""
Short-circuit collection efficiency -(J_light(0) - J_dark(0)) / (q G_ex d) of cells with symmetric injection barriers phi (so V_bi = (E_g - 2 phi)/q) for one balanced mobility mu_n = mu_p = mu and each Langevin reduction factor, with unity field-independent generation, from the converged states of the third step; material = (d in nm, eps, T in K, N_c, N_v in 1/cm^3, E_g in eV, G_ex in 1/(cm^3 s)) supplies the other inputs of the first step. Return the (len(barriers), len(reduction_factors) + 1) array whose column j is the efficiency for reduction factor j and whose last column is the Sokel-Hughes diffusion-only value coth(u/2) - 2/u with u = q V_bi / kT. Raise ValueError if barriers or reduction factors are empty or negative, twice a barrier reaches the gap, the mobility is not positive, material does not have 7 entries or N is not an even integer of at least 4.

With the bulk recombination outrun by extraction, what remains of the collection loss is set by the contacts: the injected equilibrium charge that photogenerated carriers can recombine with, and the diffusion of photogenerated carriers into the wrong contact, whose classic estimate for a uniformly generated insulator with a built-in field is the Sokel-Hughes form.

Returns
-------
A (len(barriers), len(reduction_factors) + 1) float64 array of short-circuit collection efficiencies, last column the Sokel-Hughes form.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def collection_ceiling(barriers: "np.ndarray", reduction_factors: "np.ndarray", mobility: float, intervals: int, material: "np.ndarray") -> "np.ndarray":
    """Short-circuit collection efficiency -(J_light(0) - J_dark(0)) / (q G_ex d) of cells with
        symmetric injection barriers phi (so V_bi = (E_g - 2 phi)/q) for one balanced mobility
        mu_n = mu_p = mu and each Langevin reduction factor, with unity field-independent
        generation, from the converged states of the third step; material = (d in nm, eps, T in
        K, N_c, N_v in 1/cm^3, E_g in eV, G_ex in 1/(cm^3 s)) supplies the other inputs of the
        first step. A (len(barriers), len(reduction_factors) + 1) float64 array of short-circuit
        collection efficiencies, last column the Sokel-Hughes form.
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

def _oracle_collection_ceiling(barriers: "np.ndarray", reduction_factors: "np.ndarray", mobility: float, intervals: int, material: "np.ndarray") -> "np.ndarray":
    """Short-circuit collection efficiency against the symmetric injection barrier for a balanced mobility and each
    Langevin reduction factor (columns), with the Sokel-Hughes diffusion-loss form as the last column."""
    Ph = np.asarray(barriers, dtype=float).ravel(); gs = np.asarray(reduction_factors, dtype=float).ravel(); mu = float(mobility)
    N = int(intervals); mat = [float(v) for v in np.asarray(material, dtype=float).ravel()]
    if Ph.size == 0 or gs.size == 0 or mu <= 0 or len(mat) != 7 or N < 4 or N % 2 != 0 or np.any(gs < 0) or np.any(Ph < 0):
        raise ValueError("barriers (non-negative) and reduction factors (non-negative) must be non-empty, the mobility positive, material the 7-entry tuple and intervals an even integer of at least 4")
    d_nm, eps, T, Nc, Nv, Eg, Gex = mat
    if np.any(2.0 * Ph >= Eg):
        raise ValueError("twice the barrier must lie below the gap")
    VT = _KB() * T / _Q()
    out = np.empty((Ph.size, gs.size + 1))
    for i, phi in enumerate(Ph):
        for j, g in enumerate(gs):
            pv = _oracle_device_parameters(d_nm, eps, T, Nc, Nv, Eg, phi, phi, mu, mu, g, Gex)
            prm = _dd_params(pv); sw = _Sweep(pv, N)
            out[i, j] = -(sw._current(0.0, 1.0) - sw._current(0.0, 0.0)) / prm["qGd"]
        u = (Eg - 2.0 * phi) / VT
        out[i, gs.size] = 1.0 / np.tanh(u / 2.0) - 2.0 / u
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\nmaterial = (100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 1.0e22)\nbarriers = np.array([0.10, 0.25])\nreduction_factors = np.array([0.0, 1.0])\nmobility = 1.0\nintervals = 200\n',
         'call': 'collection_ceiling(barriers, reduction_factors, mobility, intervals, material)',
         'gold_call': '_oracle_collection_ceiling(barriers, reduction_factors, mobility, intervals, material)'},
        {'setup': 'import numpy as np\nmaterial = (100.0, 3.5, 300.0, 1.0e20, 1.0e20, 1.4, 1.0e22)\nbarriers = np.array([0.05, 0.20, 0.35])\nreduction_factors = np.array([0.1])\nmobility = 1.0e-2\nintervals = 200\n',
         'call': 'collection_ceiling(barriers, reduction_factors, mobility, intervals, material)',
         'gold_call': '_oracle_collection_ceiling(barriers, reduction_factors, mobility, intervals, material)'},
        {'setup': 'import numpy as np\nmaterial = (150.0, 3.0, 290.0, 1.0e21, 5.0e20, 1.5, 5.0e21)\nbarriers = np.array([0.15, 0.30])\nreduction_factors = np.array([0.0, 0.5])\nmobility = 2.0e-4\nintervals = 200\n',
         'call': 'collection_ceiling(barriers, reduction_factors, mobility, intervals, material)',
         'gold_call': '_oracle_collection_ceiling(barriers, reduction_factors, mobility, intervals, material)'},
    ]
