"""
Select the energy-feasible protocol from its robust finite-block certificates.

The finite menu and mean launch-photon constraint are task-specific protocol design choices. Certificate rates retain their signs, including when all rates are negative.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_protocol(menu: np.ndarray, certificates: np.ndarray, photon_budget: float) -> np.ndarray:
    """Select the energy-feasible protocol from its robust finite-block certificates.

    menu : ndarray, shape (C,2), float
        Rows [N,amplitude], N in {2,4}, amplitude >= 0, in candidate order.
    certificates : ndarray, shape (C,8), float
        Corresponding robust_rate outputs, in candidate order.
    photon_budget : float
        Nonnegative maximum amplitude**2 in mean photons per transmitted signal.
    Returns
    -------
    ndarray, shape (9,), float
        [R_star, menu_index, active_scenario, t_star, h_star, R_down, R_B, R_AEP,
        R_inf] for the feasible candidate with largest signed R_star.
        Index tie-break: first candidate within 1e-9 of the best feasible rate.
        Budget equality is feasible (1e-12 absolute arithmetic tolerance).
        Units are rates in bits/signal, h_star in bits, t dimensionless.
    Raises
    ------
    ValueError
        If arrays or budget are invalid, or no candidate is energy feasible.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _oracle_select_protocol(menu: np.ndarray, certificates: np.ndarray, photon_budget: float) -> np.ndarray:
    m=np.asarray(menu,float);c=np.asarray(certificates,float)
    if m.ndim!=2 or m.shape[1]!=2 or c.shape!=(len(m),8) or not np.isfinite(m).all() or not np.isfinite(c).all() or not np.isfinite(photon_budget) or photon_budget<0 or np.any(m[:,1]<0) or not np.isin(m[:,0],[2,4]).all():
        raise ValueError('invalid menu, certificate shape or photon budget')
    feasible=np.flatnonzero(m[:,1]**2<=photon_budget+1e-12)
    if not len(feasible):raise ValueError('no protocol satisfies photon budget')
    best=np.max(c[feasible,0]);i=int(feasible[np.flatnonzero(c[feasible,0]>=best-1e-9)[0]])
    z=c[i];return np.r_[z[0],i,z[2],z[1],z[3:]]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight distinct deterministic scientific regimes."""
    return [{'setup': 'import numpy as np\n'
               'm=np.array([[2, 1], [4, '
               '1.5]],float);c=np.tile(np.array([0.,.4,0.,1.1,-.2,-.4,-.7,-.1]),(len(m),1));c[:,0]=np.array([0.1, '
               '0.2])',
      'call': 'select_protocol(m.copy(),c.copy(),2.25)',
      'gold_call': '_oracle_select_protocol(m.copy(),c.copy(),2.25)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'm=np.array([[2, 0.8], [4, '
               '1.7]],float);c=np.tile(np.array([0.,.4,0.,1.1,-.2,-.4,-.7,-.1]),(len(m),1));c[:,0]=np.array([0.11, '
               '0.23])',
      'call': 'select_protocol(m.copy(),c.copy(),1.0)',
      'gold_call': '_oracle_select_protocol(m.copy(),c.copy(),1.0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'm=np.array([[2, 0.7], [4, '
               '1.1]],float);c=np.tile(np.array([0.,.4,0.,1.1,-.2,-.4,-.7,-.1]),(len(m),1));c[:,0]=np.array([-0.2, '
               '-0.1])',
      'call': 'select_protocol(m.copy(),c.copy(),2.0)',
      'gold_call': '_oracle_select_protocol(m.copy(),c.copy(),2.0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'm=np.array([[2, 0.7], [4, '
               '0.9]],float);c=np.tile(np.array([0.,.4,0.,1.1,-.2,-.4,-.7,-.1]),(len(m),1));c[:,0]=np.array([0.2, '
               '0.2000000003])',
      'call': 'select_protocol(m.copy(),c.copy(),2.0)',
      'gold_call': '_oracle_select_protocol(m.copy(),c.copy(),2.0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'm=np.array([[4, 0.9], [2, '
               '0.7]],float);c=np.tile(np.array([0.,.4,0.,1.1,-.2,-.4,-.7,-.1]),(len(m),1));c[:,0]=np.array([0.2, '
               '0.2])',
      'call': 'select_protocol(m.copy(),c.copy(),2.0)',
      'gold_call': '_oracle_select_protocol(m.copy(),c.copy(),2.0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'm=np.array([[4, 1.4], [2, 0.2], [4, '
               '1.8]],float);c=np.tile(np.array([0.,.4,0.,1.1,-.2,-.4,-.7,-.1]),(len(m),1));c[:,0]=np.array([0.5, '
               '0.1, 0.6])',
      'call': 'select_protocol(m.copy(),c.copy(),0.04)',
      'gold_call': '_oracle_select_protocol(m.copy(),c.copy(),0.04)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'm=np.array([[2, 0], [4, '
               '0.1]],float);c=np.tile(np.array([0.,.4,0.,1.1,-.2,-.4,-.7,-.1]),(len(m),1));c[:,0]=np.array([-0.13, '
               '0.2])',
      'call': 'select_protocol(m.copy(),c.copy(),0.0)',
      'gold_call': '_oracle_select_protocol(m.copy(),c.copy(),0.0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'm=np.array([[2, 0.9], [4, '
               '1.2]],float);c=np.tile(np.array([0.,.4,0.,1.1,-.2,-.4,-.7,-.1]),(len(m),1));c[:,0]=np.array([0.31, '
               '0.3100001])',
      'call': 'select_protocol(m.copy(),c.copy(),2.0)',
      'gold_call': '_oracle_select_protocol(m.copy(),c.copy(),2.0)',
      'tol': 1e-08}]
