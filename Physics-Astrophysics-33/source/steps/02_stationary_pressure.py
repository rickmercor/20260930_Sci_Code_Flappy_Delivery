"""
Normalize the thermal zero-flux state on the density equilibrium of step 1.

Thermal energy is p/(gamma-1). Its turbulent flux divided by diffusivity is

-gradient(E_th) + 2\*gamma\*E_th\*(gradient(ln b) + M_A\*gradient(ln u)).

The corresponding mass flux is the one in step 1 and vanishes at every point.

Density and pressure have prescribed, separately conserved nodal means.

Normalize the thermal zero-flux state on the density equilibrium of step 1.

Thermal energy is p/(gamma-1). Its turbulent flux divided by diffusivity is

-gradient(E_th) + 2\*gamma\*E_th\*(gradient(ln b) + M_A\*gradient(ln u)).

The corresponding mass flux is the one in step 1 and vanishes at every point.

Density and pressure have prescribed, separately conserved nodal means.



Inputs: rho is a nonempty one-dimensional positive density array; mean_pressure

is positive; gamma>1 is the constant ratio of specific heats.

Returns: out, the same-shaped positive pressure array with the requested

arithmetic mean and zero thermal flux. All quantities are dimensionless.

Returns
-------
out, the same-shaped positive pressure array with the requested arithmetic mean and zero thermal flux. All quantities are dimensionless.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def stationary_pressure(rho: np.ndarray, mean_pressure: float, gamma: float) -> np.ndarray:
    """Return the pressure of the simultaneous zero-flux equilibrium.

    rho is the positive nodal density vector, mean_pressure is the prescribed
    arithmetic pressure mean, and gamma is the ratio of specific heats.
    Return a same-shaped float pressure vector in the stated normalized units.

    Raise ValueError for invalid shapes, nonfinite values, nonpositive density
    or mean pressure, gamma<=1, or nonfinite derived output.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_stationary_pressure(rho: np.ndarray, mean_pressure: float, gamma: float) -> np.ndarray:
    r,m,g=np.asarray(rho,dtype=float),np.asarray(mean_pressure,dtype=float),np.asarray(gamma,dtype=float)
    if r.ndim!=1 or not r.size or m.ndim or g.ndim:
        raise ValueError('invalid shapes')
    if not np.all(np.isfinite(r)) or not np.isfinite(m) or not np.isfinite(g):
        raise ValueError('nonfinite input')
    if np.any(r<=0) or m<=0 or g<=1: raise ValueError('invalid thermodynamic state')
    logs=g*np.log(r)
    weights=np.exp(logs-np.max(logs))
    out=float(m)*weights/np.mean(weights)
    if not np.all(np.isfinite(out)) or np.any(out<=0): raise ValueError('invalid pressure')
    return out

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    return [
        {'setup': 'import numpy as np\nr=np.array([1.0, 1.0, 1.0]); m=0.8; g=1.6666666666666667\n', 'call': 'stationary_pressure(r,m,g)', 'gold_call': '_oracle_stationary_pressure(r,m,g)'},
        {'setup': 'import numpy as np\nr=np.array([0.2, 1.0, 2.8]); m=1.0; g=1.6666666666666667\n', 'call': 'stationary_pressure(r,m,g)', 'gold_call': '_oracle_stationary_pressure(r,m,g)'},
        {'setup': 'import numpy as np\nr=np.array([3.0, 0.7, 1.8, 0.1]); m=2.0; g=1.4\n', 'call': 'stationary_pressure(r,m,g)', 'gold_call': '_oracle_stationary_pressure(r,m,g)'},
        {'setup': 'import numpy as np\nr=np.array([0.01, 3.0, 0.5]); m=0.3; g=1.01\n', 'call': 'stationary_pressure(r,m,g)', 'gold_call': '_oracle_stationary_pressure(r,m,g)'},
        {'setup': 'import numpy as np\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_status(stationary_pressure, np.ones(3), 1.0, 1.0)', 'gold_call': '_status(_oracle_stationary_pressure, np.ones(3), 1.0, 1.0)'},
    ]
