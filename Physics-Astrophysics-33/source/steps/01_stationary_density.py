"""
Determine the positive zero-mass-flux state of a transverse, periodic coronal

transport layer. Field strength b and parallel velocity u are externally fixed.

Units absorb the reference density and magnetic permeability, so v_A=b/sqrt(r).

The slaved turbulent mass flux divided by its positive diffusivity is

-gradient(r) + 2\*r\*(gradient(ln b) + M_A\*gradient(ln u)), with M_A=u/v_A.

The spatial mean of r is prescribed; M_A is a local state variable.

Determine the positive zero-mass-flux state of a transverse, periodic coronal

transport layer. Field strength b and parallel velocity u are externally fixed.

Units absorb the reference density and magnetic permeability, so v_A=b/sqrt(r).

The slaved turbulent mass flux divided by its positive diffusivity is

-gradient(r) + 2\*r\*(gradient(ln b) + M_A\*gradient(ln u)), with M_A=u/v_A.

The spatial mean of r is prescribed; M_A is a local state variable.



Inputs: b,u are same-length flattened nodal arrays, b>0 and u>=0;

mean_density is positive. The smooth profiles pass through these samples.

Use the arithmetic nodal mean to fix the normalization. Select the positive

branch whose Alfvén speed remains positive everywhere.



Returns: out, a float array of length N+1: N equilibrium density samples

followed by the spatially constant total outward Alfvén characteristic speed.

Returns
-------
out, a float array of length N+1: N equilibrium density samples followed by the spatially constant total outward Alfvén characteristic speed.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def stationary_density(b: np.ndarray, u: np.ndarray, mean_density: float) -> np.ndarray:
    """Return the normalized zero-flux density and its characteristic speed.

    b and u are same-length nodal vectors of normalized field strength and
    parallel velocity. mean_density fixes the arithmetic mean of the density.
    Return an (N+1,) float array: density samples followed by the constant
    characteristic speed, in the units and branch specified above.

    Raise ValueError for mismatched, empty, non-vector or nonfinite inputs,
    nonpositive b/mean_density, negative u, or a nonfinite derived state.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq

def _oracle_stationary_density(b: np.ndarray, u: np.ndarray, mean_density: float) -> np.ndarray:
    b,u,m=np.asarray(b,dtype=float),np.asarray(u,dtype=float),np.asarray(mean_density,dtype=float)
    if b.ndim!=1 or not b.size or u.shape!=b.shape or m.ndim:
        raise ValueError('invalid shapes')
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(u)) or not np.isfinite(m):
        raise ValueError('nonfinite inputs')
    if np.any(b<=0) or np.any(u<0) or m<=0:
        raise ValueError('invalid positive state')
    # Integrating the zero flux gives b/sqrt(r)+u=C. The mean is strictly
    # decreasing in C on C>max(u), hence this root is unique.
    umax=float(np.max(u))
    lo=np.nextafter(umax,np.inf)
    hi=umax+float(np.max(b)/np.sqrt(m))+1.
    with np.errstate(over='ignore', divide='ignore'):
        c=brentq(lambda s: np.mean((b/(s-u))**2)-m,lo,hi,xtol=2e-14,rtol=1e-14)
        out=np.r_[(b/(c-u))**2,c]
    if not np.all(np.isfinite(out)) or np.any(out<=0):
        raise ValueError('invalid equilibrium')
    return out

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    return [
        {'setup': 'import numpy as np\nb=np.array([1.0, 1.0, 1.0]); u=np.array([0.2, 0.2, 0.2]); m=1.0\n', 'call': 'stationary_density(b,u,m)', 'gold_call': '_oracle_stationary_density(b,u,m)'},
        {'setup': 'import numpy as np\nb=np.array([1.7, 0.8, 0.4, 1.2]); u=np.array([0.1, 0.8, 0.3, 0.5]); m=1.3\n', 'call': 'stationary_density(b,u,m)', 'gold_call': '_oracle_stationary_density(b,u,m)'},
        {'setup': 'import numpy as np\nb=np.array([0.5, 1.0, 2.0]); u=np.array([0.0, 0.0, 0.0]); m=0.7\n', 'call': 'stationary_density(b,u,m)', 'gold_call': '_oracle_stationary_density(b,u,m)'},
        {'setup': 'import numpy as np\nb=np.array([1.0, 0.6, 1.4, 2.0]); u=np.array([1.8, 0.1, 2.5, 0.6]); m=2.0\n', 'call': 'stationary_density(b,u,m)', 'gold_call': '_oracle_stationary_density(b,u,m)'},
        {'setup': 'import numpy as np\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_status(stationary_density, np.array([0.0, 1.0]), np.zeros(2), 1.0)', 'gold_call': '_status(_oracle_stationary_density, np.array([0.0, 1.0]), np.zeros(2), 1.0)'},
    ]
