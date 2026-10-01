"""
Fit the sheet separation of the displaced-sheet model to a set of as-grown structures.

Given several heterostructures of known Al fraction and barrier thickness with measured zero-bias sheet
densities, and a common metal barrier height, find the single sheet separation delta that minimises the
sum over structures of the squared relative residual

    sum_i ( n_model,i(delta) / n_meas,i - 1 )^2,

where n_model,i is the self-consistent sheet density of structure i. The search runs over
0 <= delta <= 1.5 nm, where the cost has one minimum; return that minimiser converged to at least
1e-15 m.

Inputs: al_fractions, barrier_thicknesses, sheet_densities: 1-D arrays of equal length >= 1; thicknesses in m,
  densities in m^-2 and strictly positive barrier_height: float, common phi_B in V conduction_band_offset_eV, effective_mass_ratio: as in the earlier steps gan, aln: binary parameter dictionaries keyed 'a' (a-axis lattice constant, m), 'psp' (spontaneous polarization, C/m^2, negative), 'e31' and 'e33' (piezoelectric constants, C/m^2), 'c13' and 'c33' (elastic constants, Pa) and 'eps_r' (relative permittivity)

 Returns: float, fitted sheet separation delta in m

 Raises: ValueError if the three arrays differ in length, are empty, or any measured density is not positive.

Returns
-------
float, fitted sheet separation delta in m
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_sheet_separation(al_fractions: "np.ndarray", barrier_thicknesses: "np.ndarray", sheet_densities: "np.ndarray",
                         barrier_height: float, conduction_band_offset_eV: float, effective_mass_ratio: float,
                         gan: dict, aln: dict) -> float:
    """Least-squares (relative residual) sheet separation in m over 0 <= delta <= 1.5 nm.
 
    Raises ValueError if the arrays differ in length, are empty, or a measured density is not positive.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq, minimize_scalar
 
def _oracle_fit_sheet_separation(al_fractions: "np.ndarray", barrier_thicknesses: "np.ndarray", sheet_densities: "np.ndarray",
                                 barrier_height: float, conduction_band_offset_eV: float, effective_mass_ratio: float,
                                 gan: dict, aln: dict) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    x = np.asarray(al_fractions, dtype=float).ravel()
    d = np.asarray(barrier_thicknesses, dtype=float).ravel()
    n = np.asarray(sheet_densities, dtype=float).ravel()
    if not (x.size == d.size == n.size) or x.size == 0:
        raise ValueError('the three arrays must be non-empty and of equal length')
    if not np.all(n > 0.0):
        raise ValueError('measured sheet densities must be positive')
 
    def _cost(delta):
        total = 0.0
        for xi, di, ni in zip(x, d, n):
            nm = _oracle_self_consistent_sheet_density(xi, di, delta, barrier_height, conduction_band_offset_eV,
                                                       effective_mass_ratio, gan, aln)
            total += (nm / ni - 1.0) ** 2
        return total
 
    res = minimize_scalar(_cost, bounds=(0.0, 1.5e-9), method='bounded', options={'xatol': 1e-16, 'maxiter': 500})
    return float(res.x)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    s = ("import numpy as np\nfrom scipy.optimize import brentq, minimize_scalar\nfrom scipy.linalg import eigh_tridiagonal\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\n"
         "X = np.array([0.22, 0.26, 0.30, 0.33, 0.36])\nD = np.array([26.0, 21.0, 17.0, 14.0, 12.0]) * 1e-9\n"
         "N = np.array([8.28, 9.55, 10.6, 11.1, 11.5]) * 1e16\n")
    return [
        {"setup": s,
         "call": "round(fit_sheet_separation(X, D, N, 1.32, 0.301, 0.22, GAN, ALN) * 1e10, 4)",
         "gold_call": "round(_oracle_fit_sheet_separation(X, D, N, 1.32, 0.301, 0.22, GAN, ALN) * 1e10, 4)"},
        {"setup": s + "N2 = np.array([_oracle_self_consistent_sheet_density(xi, di, 4.0e-10, 1.32, 0.301, 0.22, GAN, ALN) for xi, di in zip(X, D)])\n",
         "call": "round(fit_sheet_separation(X, D, N2, 1.32, 0.301, 0.22, GAN, ALN) * 1e10, 4)",
         "gold_call": "round(_oracle_fit_sheet_separation(X, D, N2, 1.32, 0.301, 0.22, GAN, ALN) * 1e10, 4)"},
        {"setup": s + "N3 = np.array([_oracle_self_consistent_sheet_density(0.31, 9.5e-9, 0.0, 1.32, 0.301, 0.22, GAN, ALN)])\n",
         "call": "round(fit_sheet_separation(np.array([0.31]), np.array([9.5e-9]), N3, 1.32, 0.301, 0.22, GAN, ALN) * 1e10, 4)",
         "gold_call": "round(_oracle_fit_sheet_separation(np.array([0.31]), np.array([9.5e-9]), N3, 1.32, 0.301, 0.22, GAN, ALN) * 1e10, 4)"},
        {"setup": s + "def run_model():\n    try:\n        fit_sheet_separation(X, D[:3], N, 1.32, 0.301, 0.22, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle():\n    try:\n        _oracle_fit_sheet_separation(X, D[:3], N, 1.32, 0.301, 0.22, GAN, ALN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
