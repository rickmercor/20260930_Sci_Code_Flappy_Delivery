"""
Build the radial field metric and the fixed-action stiffness data.

The 4D radial measure is 2*pi**2*r**3 dr. The finite-volume convention in the background defines a variable origin cell and a fixed zero field at r_N.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def radial_geometry(n: int, radius: float, power: float) -> "np.ndarray":
    """Build the radial field metric and the fixed-action stiffness data.

    n : int, n >= 2
        Number of variable radial fields, excluding the fixed outer endpoint.
    radius : float > 0
        Dimensionless outer radius.
    power : float >= 1
        Power in r_j=radius*(j/n)**power.
    Returns
    -------
    ndarray, shape (n,4)
        Columns [r_j,w_j,c_j,L_jj] in outward order. L_jj=c_j+c_(j-1), with c_(-1)=0. The definitions of a,w,c are those in the background.

    Raises
    ------
    ValueError
        If numerical input domains or array shapes are invalid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def _oracle_radial_geometry(n: int, radius: float, power: float) -> "np.ndarray":
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<2 or not np.isfinite(radius) or radius<=0 or not np.isfinite(power) or power<1:
        raise ValueError("Invalid radial geometry domain")
    r=radius*(np.arange(n+1,dtype=float)/n)**power
    mid=(r[:-1]+r[1:])/2
    w=np.pi**2/2*np.diff(np.r_[0.,mid]**4)
    c=2*np.pi**2*mid**3/np.diff(r)
    return np.column_stack((r[:-1],w,c,c+np.r_[0.,c[:-1]]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Constructed scientific cases, not published numerical examples."""
    return [{'setup': '',
      'call': 'radial_geometry(2,2.0,1.0)',
      'gold_call': '_oracle_radial_geometry(2,2.0,1.0)',
      'tol': 2e-06},
     {'setup': '',
      'call': 'radial_geometry(9,3.0,1.0)',
      'gold_call': '_oracle_radial_geometry(9,3.0,1.0)',
      'tol': 2e-06},
     {'setup': '',
      'call': 'radial_geometry(15,3.0,2.0)',
      'gold_call': '_oracle_radial_geometry(15,3.0,2.0)',
      'tol': 2e-06},
     {'setup': '',
      'call': 'radial_geometry(15,6.0,2.0)',
      'gold_call': '_oracle_radial_geometry(15,6.0,2.0)',
      'tol': 2e-06},
     {'setup': '',
      'call': 'radial_geometry(27,4.0,1.5)',
      'gold_call': '_oracle_radial_geometry(27,4.0,1.5)',
      'tol': 2e-06},
     {'setup': '',
      'call': 'radial_geometry(64,9.0,2.5)',
      'gold_call': '_oracle_radial_geometry(64,9.0,2.5)',
      'tol': 2e-06}]
