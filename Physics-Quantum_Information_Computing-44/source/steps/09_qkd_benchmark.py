"""
Compose the preceding scientific functions into the QKD design benchmark.

Final orchestrator: the submitted implementation must call and combine every preceding public subproblem function, directly or transitively through robust_rate, and use their results. All setup, basis, rate and precision conventions are in the global task and public interfaces.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qkd_benchmark(menu: np.ndarray, scenarios: np.ndarray, n: int, epsilon: float, epsilon_prime: float, photon_budget: float) -> np.ndarray:
    """Compose the preceding scientific functions into the QKD design benchmark.

    menu : ndarray, shape (C,2), float
        Ordered rows [N,amplitude], N=2 or 4 and amplitude >= 0.
    scenarios : ndarray, shape (S,2), float
        Ordered rows [eta,phase], eta in [0,1] and finite phase in radians.
        Require (1-eta)*amplitude**2 <= 16 for every evaluated pair.
    n : int
        Positive block size in transmitted signals.
    epsilon, epsilon_prime : float
        Smoothing and extraction parameters in (0,1).
    photon_budget : float
        Nonnegative maximum mean launch photon number per signal.
    Returns
    -------
    ndarray, shape (9,), float
        [R_star, menu_index, active_scenario, t_star, h_star, R_down, R_B, R_AEP,
        R_inf]. These are exactly the selected certificate from select_protocol.
        Rates are signed bits/signal, h_star is bits, t dimensionless, and the
        two indices are zero-based integers stored as floats. Compose all eight
        preceding public functions, with transitive calls to the two quantum
        optimizers through robust_rate. Preserve all caller arrays.
    Raises
    ------
    ValueError
        If any input is outside the preceding functions' domains, arrays are empty
        or malformed, or no menu entry is energy feasible.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import erf, logsumexp
from scipy.optimize import minimize, minimize_scalar, brentq

def _oracle_qkd_benchmark(menu: np.ndarray, scenarios: np.ndarray, n: int, epsilon: float, epsilon_prime: float, photon_budget: float) -> np.ndarray:
    menu=np.asarray(menu,float);scenarios=np.asarray(scenarios,float)
    if menu.ndim!=2 or menu.shape[1]!=2 or not len(menu) or scenarios.ndim!=2 or scenarios.shape[1]!=2 or not len(scenarios):raise ValueError('menu and scenarios must be nonempty two-column arrays')
    certificates=[]
    for N,amplitude in menu:
        if N not in (2,4):raise ValueError('N must be 2 or 4')
        ps=[];rs=[];stats=[]
        for eta,phase in scenarios:
            P=_oracle_psk_channel(int(N),amplitude,eta,phase)
            w=_oracle_cyclic_weights(int(N),(1-eta)*amplitude**2)
            rho=_oracle_reverse_states(P,w)[0]
            ps.append(P);rs.append(rho);stats.append(_oracle_entropy_statistics(rho))
        certificates.append(_oracle_robust_rate(np.array(rs),np.array(ps),np.array(stats),n,epsilon,epsilon_prime))
    return _oracle_select_protocol(menu,np.array(certificates),photon_budget)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Eight distinct deterministic scientific regimes."""
    return [{'setup': 'import numpy as np\nm=np.array([[2, 0.71], [2, 1.09]],float);sc=np.array([[0.82, 0.05]],float)',
      'call': '(qkd_benchmark(m.copy(),sc.copy(),190,1e-08,1e-08,2.0)) * '
              'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'gold_call': '(_oracle_qkd_benchmark(m.copy(),sc.copy(),190,1e-08,1e-08,2.0)) * '
                   'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'tol': 5e-07},
     {'setup': 'import numpy as np\n'
               'm=np.array([[4, 1.12], [4, 1.42]],float);sc=np.array([[0.89, 0.28], [0.92, 0.63]],float)',
      'call': '(qkd_benchmark(m.copy(),sc.copy(),590,1e-08,1e-08,3.0)) * '
              'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'gold_call': '(_oracle_qkd_benchmark(m.copy(),sc.copy(),590,1e-08,1e-08,3.0)) * '
                   'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'tol': 5e-07},
     {'setup': 'import numpy as np\n'
               'm=np.array([[4, 1.24], [4, 1.44]],float);sc=np.array([[0.879, 0.01], [0.896, 0.71]],float)',
      'call': '(qkd_benchmark(m.copy(),sc.copy(),410,1e-08,1e-08,3.0)) * '
              'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'gold_call': '(_oracle_qkd_benchmark(m.copy(),sc.copy(),410,1e-08,1e-08,3.0)) * '
                   'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'tol': 5e-07},
     {'setup': 'import numpy as np\n'
               'm=np.array([[2, 0.83], [4, 1.19], [4, 1.49]],float);sc=np.array([[0.9, 0.04], [0.92, '
               '0.62]],float)',
      'call': '(qkd_benchmark(m.copy(),sc.copy(),730,1e-08,1e-08,1.6)) * '
              'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'gold_call': '(_oracle_qkd_benchmark(m.copy(),sc.copy(),730,1e-08,1e-08,1.6)) * '
                   'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'tol': 5e-07},
     {'setup': 'import numpy as np\n'
               'm=np.array([[4, 1.16], [4, 1.36]],float);sc=np.array([[0.92, 0.66], [0.87, 0.02]],float)',
      'call': '(qkd_benchmark(m.copy(),sc.copy(),460,1e-07,1e-09,2.0)) * '
              'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'gold_call': '(_oracle_qkd_benchmark(m.copy(),sc.copy(),460,1e-07,1e-09,2.0)) * '
                   'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nm=np.array([[2, 0.78], [2, 0.78]],float);sc=np.array([[0.91, 0.02]],float)',
      'call': '(qkd_benchmark(m.copy(),sc.copy(),930,1e-08,1e-08,1.0)) * '
              'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'gold_call': '(_oracle_qkd_benchmark(m.copy(),sc.copy(),930,1e-08,1e-08,1.0)) * '
                   'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nm=np.array([[2, 0.68], [4, 0.72]],float);sc=np.array([[0.37, 0.4]],float)',
      'call': '(qkd_benchmark(m.copy(),sc.copy(),260,1e-08,1e-08,1.0)) * '
              'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'gold_call': '(_oracle_qkd_benchmark(m.copy(),sc.copy(),260,1e-08,1e-08,1.0)) * '
                   'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'tol': 5e-07},
     {'setup': 'import numpy as np\nm=np.array([[2, 0.64], [4, 1.23]],float);sc=np.array([[1.0, 0.12]],float)',
      'call': '(qkd_benchmark(m.copy(),sc.copy(),620,1e-08,1e-08,2.0)) * '
              'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'gold_call': '(_oracle_qkd_benchmark(m.copy(),sc.copy(),620,1e-08,1e-08,2.0)) * '
                   'np.array([1.,1.,1.,.025,.01,.01,.01,.01,.01])',
      'tol': 5e-07}]
