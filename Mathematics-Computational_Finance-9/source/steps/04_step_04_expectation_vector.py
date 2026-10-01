"""
The physical moment family consumed by the truncated coefficient matrix.

Normalized moments are converted to physical units and organized into the four families needed by the matrix rows.

Returns
-------
dict base_even and base_odd are float arrays of shape (N_max+1,). high_even is a list of M_max//2 arrays of that shape; high_odd is a list of (M_max-1)//2 arrays of that shape. Keys are exactly 'base_even', 'base_odd', 'high_even', 'high_odd'.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def expectation_vector(M_max: int, N_max: int, sigma0: float, nu: float, T: float) -> dict:
    """Collect physical moments E(beta,alpha)=E[xi_T**beta*v_T**(-2*alpha)],
    with xi_T=(sigma_T-sigma0)/nu, v_T**2=(1/T)integral_0^T sigma_s**2 ds,
    sigma_t=sigma0*exp(nu*W_t-nu**2*t/2), tau=nu**2*T.
    In terms of normalized_moment, E(beta,alpha) equals
    sigma0**(beta-2*alpha)*nu**(-beta)*normalized_moment(beta,alpha,tau).
    For n=0,...,N_max return the following named families:
    base_even[n]=E(0,n-.5); base_odd[n]=E(1,n+.5);
    high_even[p][n]=E(2*p+2,n+p+.5), p=0,...,M_max//2-1;
    high_odd[p][n]=E(2*p+3,n+p+1.5), p=0,...,(M_max-1)//2-1.
    This step reuses normalized_moment and never depends on the strike grid.

    Parameters
    ----------
    M_max : int
        Integer from 1 through 4, excluding booleans.
    N_max : int
        Integer from 0 through 12, excluding booleans.
    sigma0, nu, T : float
        Strictly positive finite model constants; .04 <= nu**2*T <= .20.

    Returns
    -------
    dict
        base_even and base_odd are float arrays of shape (N_max+1,).
        high_even is a list of M_max//2 arrays of that shape; high_odd is
        a list of (M_max-1)//2 arrays of that shape. Keys are exactly
        'base_even', 'base_odd', 'high_even', 'high_odd'.

    Raises
    ------
    ValueError
        If an order or model parameter is outside the stated domain.
    """
    return {}

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_expectation_vector(M_max: int, N_max: int, sigma0: float, nu: float, T: float) -> dict:
    np = __import__('numpy')
    if isinstance(M_max,bool) or not isinstance(M_max,(int,np.integer)) or not 1 <= M_max <= 4:
        raise ValueError('M_max outside [1,4]')
    if isinstance(N_max,bool) or not isinstance(N_max,(int,np.integer)) or not 0 <= N_max <= 12:
        raise ValueError('N_max outside [0,12]')
    if any(not np.isfinite(v) or v <= 0 for v in [sigma0,nu,T]) or not .04 <= nu*nu*T <= .20:
        raise ValueError('invalid model parameters')
    def E(beta,alpha):
        return sigma0**(beta-2*alpha)*nu**(-beta)*_oracle_normalized_moment(beta,alpha,nu*nu*T)
    n=range(N_max+1)
    return dict(base_even=np.array([E(0,j-.5) for j in n]),base_odd=np.array([E(1,j+.5) for j in n]),
                high_even=[np.array([E(2*p+2,j+p+.5) for j in n]) for p in range(M_max//2)],
                high_odd=[np.array([E(2*p+3,j+p+1.5) for j in n]) for p in range((M_max-1)//2)])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Normal, boundary, edge and invalid-input cases."""
    return [{'setup': "import numpy as np\ndef packed(e):\n    return np.concatenate([e['base_even'],e['base_odd']]+e['high_even']+e['high_odd'])\n", 'call': 'packed(expectation_vector(4,12,1.0,0.7,0.125))', 'gold_call': 'packed(_oracle_expectation_vector(4,12,1.0,0.7,0.125))'}, {'setup': "import numpy as np\ndef packed(e):\n    return np.concatenate([e['base_even'],e['base_odd']]+e['high_even']+e['high_odd'])\n", 'call': 'packed(expectation_vector(1,0,35.0,0.7,0.125))', 'gold_call': 'packed(_oracle_expectation_vector(1,0,35.0,0.7,0.125))'}, {'setup': "import numpy as np\ndef packed(e):\n    return np.concatenate([e['base_even'],e['base_odd']]+e['high_even']+e['high_odd'])\n", 'call': 'packed(expectation_vector(3,5,2.0,1.0,0.1))', 'gold_call': 'packed(_oracle_expectation_vector(3,5,2.0,1.0,0.1))'}, {'setup': 'def run_model_invalid():\n    try:\n        expectation_vector(5,2,35.,.7,.125)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_expectation_vector(5,2,35.,.7,.125)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'}]
