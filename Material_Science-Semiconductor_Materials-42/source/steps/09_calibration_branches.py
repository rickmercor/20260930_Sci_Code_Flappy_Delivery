"""
Resolve the inverse calibration’s isolated branches rather than depending on one optimizer starting point; this is a constructed inverse use of the paper’s forward model.

Resolve the inverse calibration’s isolated branches rather than depending on one optimizer starting point; this is a constructed inverse use of the paper’s forward model.

Calibration is a coupled inverse use of the microscopic screening forward model. The supplied function class has isolated branches; the public API specifies the domain and rounding.

Returns
-------
return result  # real ndarray, shape (R,2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def calibration_branches(forward, observed: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Parameters
    ----------
    forward : callable
        Maps a finite real [kappa,d] array of shape (2,) to two finite ground-state
        energies (eV) in a real array of shape (2,). On the supplied rectangle,
        the first energy is strictly increasing in kappa. Its feasible first-
        energy contour has simple isolated second-energy crossings separated
        by at least 0.15 angstrom; each crossing has a feasible neighborhood of
        radius (d_max-d_min)/32 relative to the rectangle. The supplied experiment and analytical unit
        fixtures satisfy this class. Interior tangent roots are outside this class.
    observed : finite real ndarray, shape (2,)
        Exact target energies in the same order, in eV.
    bounds : finite real ndarray, shape (2,2)
        Rows [kappa_min,kappa_max] and [d_min,d_max], each strictly increasing.
    Returns
    -------
    real ndarray, shape (R,2)
        All admissible root pairs [kappa,d], with absolute coordinate
        convergence 1e-8 and with both energy residuals below 1e-9 eV before rounding. Coordinates are rounded to 6 decimal places, and
        rows sorted by increasing d then kappa. Empty shape (0,2) means no fit.
    Raises
    ------
    ValueError : malformed/nonfinite observations or bounds."""
    return np.zeros((0,2))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_calibration_branches(forward, observed, bounds):
    observed = np.asarray(observed, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if observed.shape != (2,) or bounds.shape != (2,2) or not np.isfinite(observed).all() or not np.isfinite(bounds).all() or np.any(bounds[:,0] >= bounds[:,1]):
        raise ValueError('two observations and nondegenerate rectangular bounds required')
    roots = []
    cache = {}
    def reduced(d):
        if d in cache:
            return cache[d]
        def first(k):
            return float(forward(np.array([k, d]))[0]-observed[0])
        lo, hi = bounds[0]
        flo, fhi = first(lo), first(hi)
        if flo*fhi > 0:
            cache[d] = (np.nan, np.nan)
        else:
            k = brentq(first, lo, hi, xtol=1e-10)
            cache[d] = (float(forward(np.array([k, d]))[1]-observed[1]), k)
        return cache[d]
    ds = np.linspace(*bounds[1], 65)
    values = [reduced(d)[0] for d in ds]
    for i in range(len(ds)-1):
        if not np.isfinite(values[i]) or not np.isfinite(values[i+1]):
            continue
        if values[i] == 0:
            d = ds[i]
        elif values[i]*values[i+1] < 0:
            d = brentq(lambda d: reduced(d)[0], ds[i], ds[i+1], xtol=2e-9)
        else:
            continue
        candidate = np.array([reduced(d)[1], d])
        if not any(np.max(np.abs(candidate-x)) < 2e-4 for x in roots):
            roots.append(candidate)
    if values[-1] == 0:
        roots.append(np.array([reduced(ds[-1])[1], ds[-1]]))
    if not roots:
        return np.empty((0, 2), dtype=float)
    result = np.round(np.asarray(roots), 6)
    return result[np.lexsort((result[:, 0], result[:, 1]))]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ndef f(p):\n k,d=p\n return np.array([k+d,(d-1.)*(d-2.)+.1*(k+d)])\nobs=np.array([8.,.8]); bounds=np.array([[4.,9.],[.6,2.7]])\n', 'call': 'calibration_branches(f,obs,bounds)', 'gold_call': '_oracle_calibration_branches(f,obs,bounds)'}, {'setup': 'import numpy as np\ndef f(p):\n k,d=p\n return np.array([k+d,d+.1*k])\nobs=np.array([8.,2.6]); bounds=np.array([[4.,9.],[.6,2.7]])\n', 'call': 'calibration_branches(f,obs,bounds)', 'gold_call': '_oracle_calibration_branches(f,obs,bounds)'}, {'setup': 'import numpy as np\ndef f(p):\n k,d=p\n return np.array([k+d,d])\nobs=np.array([8.,.6]); bounds=np.array([[4.,9.],[.6,2.7]])\n', 'call': 'calibration_branches(f,obs,bounds)', 'gold_call': '_oracle_calibration_branches(f,obs,bounds)'}, {'setup': 'import numpy as np\ndef f(p):\n k,d=p\n return np.array([k+d,d])\nobs=np.array([8.,4.]); bounds=np.array([[4.,9.],[.6,2.7]])\n', 'call': 'calibration_branches(f,obs,bounds)', 'gold_call': '_oracle_calibration_branches(f,obs,bounds)'}, {'setup': 'import numpy as np\ndef f(p):\n k,d=p\n return np.array([k+d,d])\nobs=np.array([20.,1.]); bounds=np.array([[4.,9.],[.6,2.7]])\n', 'call': 'calibration_branches(f,obs,bounds)', 'gold_call': '_oracle_calibration_branches(f,obs,bounds)'}, {'setup': 'import numpy as np\nf=lambda p:np.asarray(p)\ndef _exception_code(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': '_exception_code(lambda: calibration_branches(f,np.ones(2),np.array([[2.,1.],[0.,1.]])))', 'gold_call': '_exception_code(lambda: _oracle_calibration_branches(f,np.ones(2),np.array([[2.,1.],[0.,1.]])))'}]
