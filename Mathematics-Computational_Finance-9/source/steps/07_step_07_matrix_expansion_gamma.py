"""
Short-maturity matrix valuation: final orchestrator.

The final calculation combines the scales, normalized moments, physical moment family, coefficient matrix and analytic strike contractions to obtain the selected Gamma.

Returns
-------
float Gamma at report_strike, in the input asset units.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def matrix_expansion_gamma(X0: float=100., sigma0: float=35., nu: float=.7, rho: float=-.6,
                           T: float=.125, M_max: int=4, N_max: int=12,
                           strikes: tuple=(95.,97.5,100.,102.5,105.), report_strike: float=95.) -> float:
    """Compute the selected Gamma from the finite coefficient-matrix method.
    Reuse scale_constants, normal_call_terms, normalized_moment,
    expectation_vector, coefficient_matrix and matrix_values. The base
    normal-call volatility is sqrt(1-rho**2)*v, with v the RMS level from
    scale_constants. Evaluate the signed physical moment family, assemble
    the common coefficient matrix and compute all prices and both spot
    derivatives; return the Gamma at report_strike. This is the finite
    M_max,N_max expansion, not the converged stochastic-volatility value.

    Parameters
    ----------
    X0, report_strike : float
        Finite asset level and a strike occurring exactly once in strikes.
    sigma0, nu, T : float
        Positive finite parameters with .04<=nu**2*T<=.20.
    rho : float
        Finite and abs(rho)<1/sqrt(2).
    M_max, N_max : int
        Integers excluding booleans, 1<=M_max<=4 and 0<=N_max<=12.
    strikes : tuple
        Nonempty finite distinct strikes, in arbitrary order. All satisfy
        abs(X0-k)<sigma0*sqrt(1-rho**2)/nu.

    Returns
    -------
    float
        Gamma at report_strike, in the input asset units.

    Raises
    ------
    ValueError
        If any parameter violates its stated domain, report_strike is not
        present exactly once.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_matrix_expansion_gamma(X0: float=100., sigma0: float=35., nu: float=.7, rho: float=-.6,
                                  T: float=.125, M_max: int=4, N_max: int=12,
                                  strikes: tuple=(95.,97.5,100.,102.5,105.), report_strike: float=95.) -> float:
    np = __import__('numpy')
    k=np.asarray(strikes,dtype=float)
    if not np.isfinite(X0) or not np.isfinite(report_strike) or k.ndim!=1 or not k.size or not np.all(np.isfinite(k)) or len(np.unique(k))!=k.size or np.count_nonzero(k==report_strike)!=1:
        raise ValueError('invalid strike inputs')
    if any(not np.isfinite(v) or v<=0 for v in [sigma0,nu,T]) or not .04<=nu*nu*T<=.20 or not np.isfinite(rho) or abs(rho)>=1/np.sqrt(2):
        raise ValueError('invalid model parameters')
    if isinstance(M_max,bool) or not isinstance(M_max,(int,np.integer)) or not 1<=M_max<=4 or isinstance(N_max,bool) or not isinstance(N_max,(int,np.integer)) or not 0<=N_max<=12:
        raise ValueError('invalid expansion orders')
    tau,v,radius=_oracle_scale_constants(sigma0,nu,rho,T)
    if np.any(np.abs(X0-k)>=radius):
        raise ValueError('strike outside convergence radius')
    e=_oracle_expectation_vector(M_max,N_max,sigma0,nu,T)
    A=_oracle_coefficient_matrix(e,M_max,N_max,rho,T,v)
    base=_oracle_normal_call_terms(T,X0,k,np.sqrt(1-rho*rho)*v)
    values=_oracle_matrix_values(A,X0,k,rho,base)
    return float(values[2][np.flatnonzero(k==report_strike)[0]])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Normal, boundary, edge and invalid-input cases."""
    return [{'setup': '', 'call': 'matrix_expansion_gamma()', 'gold_call': '_oracle_matrix_expansion_gamma()'}, {'setup': '', 'call': 'matrix_expansion_gamma(sigma0=20.,nu=1.,T=.1,rho=.4,M_max=3,N_max=8,strikes=(97.,100.,103.),report_strike=103.)', 'gold_call': '_oracle_matrix_expansion_gamma(sigma0=20.,nu=1.,T=.1,rho=.4,M_max=3,N_max=8,strikes=(97.,100.,103.),report_strike=103.)'}, {'setup': '', 'call': 'matrix_expansion_gamma(T=.2,nu=.5,rho=0.,M_max=1,N_max=6,report_strike=100.)', 'gold_call': '_oracle_matrix_expansion_gamma(T=.2,nu=.5,rho=0.,M_max=1,N_max=6,report_strike=100.)'}, {'setup': 'def run_model_invalid():\n    try:\n        matrix_expansion_gamma(rho=.8)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_matrix_expansion_gamma(rho=.8)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'}, {'setup': 'def run_model_invalid():\n    try:\n        matrix_expansion_gamma(report_strike=80.)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_matrix_expansion_gamma(report_strike=80.)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'}]
