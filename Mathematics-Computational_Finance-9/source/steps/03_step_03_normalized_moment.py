"""
Signed normalized Brownian exponential-functional moments at short maturity.

The Brownian exponential-functional law determines signed moments of terminal volatility and integrated variance. Short maturities make the real oscillatory representation strongly cancelling, so stable deterministic evaluation is needed.

Returns
-------
float The signed normalized moment R.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def normalized_moment(beta: int, alpha: float, tau: float) -> float:
    """Compute R = tau**alpha E[(exp(W_tau-tau/2)-1)**beta A_tau**(-alpha)],
    where W is standard Brownian motion and
    A_tau = integral_0^tau exp(2*W_s-s) ds.

    A deterministic integral representation is R = tau**alpha Gamma(alpha+1)
    exp(-tau/8+pi**2/(2*tau))/sqrt(2*pi**3*tau) times

    integral over x in R, eta in (0,infinity) of
    (exp(x)-1)**beta exp(-(alpha+1/2)*x-eta**2/(2*tau))
    sinh(eta) sin(pi*eta/tau) / (cosh(x)+cosh(eta))**(alpha+1).

    For beta=0, the power is one. For odd beta, preserve its sign. This is
    the integral over the infinite domain, not a specified finite grid sum.
    At these small tau values the oscillatory real integral has severe
    cancellation. Use a deterministic, numerically stable evaluation;
    mathematically equivalent integral transformations are allowed. If a
    complex contour is used, continue the denominator power analytically
    from the real contour without crossing a zero or branch cut. Simulation
    and a small-tau truncation are not the target. Aim for relative error
    at most 1e-10, with absolute error 1e-12 allowed for near-zero values:
    abs(computed-R) <= 1e-12 + 1e-10*abs(R).
    Cost contract: one call must return in well under a second on a single
    core, and later steps of this task evaluate R at more than a hundred
    parameter combinations inside one shared process wall-clock limit.
    Reaching the stated accuracy by raising the working precision instead
    of by choosing a better representation does not meet that contract.

    Parameters
    ----------
    beta : int
        Integer from 0 through 4, excluding booleans.
    alpha : float
        Finite, -0.5 <= alpha <= 13.5 and 2*alpha+1.5 > beta.
        The last inequality makes the displayed double integral absolutely
        integrable before the oscillatory cancellations are taken.
    tau : float
        Finite dimensionless maturity in [0.04, 0.20].

    Returns
    -------
    float
        The signed normalized moment R.

    Raises
    ------
    ValueError
        If beta, alpha or tau is outside its stated domain.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _contour_grid(tau):
    np = __import__('numpy')
    roots = __import__('scipy.special',fromlist=['roots_legendre']).roots_legendre
    cache = getattr(_contour_grid, 'cache', {})
    if tau in cache:
        return cache[tau]
    z, w = roots(24)
    def panels(edges):
        a, b = edges[:-1], edges[1:]
        return ((a[:,None]+b[:,None])/2+(b-a)[:,None]*z/2).ravel(), ((b-a)[:,None]*w/2).ravel()
    pos = np.r_[0., np.geomspace(.005,80.,40)]
    x, wx = panels(np.r_[-pos[:0:-1],pos])
    u, wu = panels(np.linspace(-12*np.sqrt(tau),12*np.sqrt(tau),25))
    c = np.pi-3*np.sqrt(tau)
    v = u+1j*c
    denominator = np.log(np.cosh(x[:,None])+np.cosh(v[None,:]))
    kernel = wu*np.exp(-u*u/(2*tau)+1j*(np.pi-c)*u/tau)*np.sinh(v)
    const = -tau/8+(np.pi-c)**2/(2*tau)-.5*np.log(2*np.pi**3*tau)-np.log(2)
    data = x,wx,denominator,kernel,const
    cache[tau] = data
    _contour_grid.cache = cache
    return data

def _oracle_normalized_moment(beta: int, alpha: float, tau: float) -> float:
    """Reference contour integral, shifted inside the analytic strip."""
    np = __import__('numpy')
    gammaln = __import__('scipy.special',fromlist=['gammaln']).gammaln
    if isinstance(beta,bool) or not isinstance(beta,(int,np.integer)) or not 0 <= beta <= 4:
        raise ValueError('beta must be an integer from zero through four')
    alpha,tau = float(alpha),float(tau)
    if not np.isfinite(alpha) or not -.5 <= alpha <= 13.5 or 2*alpha+1.5 <= beta:
        raise ValueError('alpha outside integrable moment domain')
    if not np.isfinite(tau) or not .04 <= tau <= .20:
        raise ValueError('tau outside short-maturity domain')
    cache = getattr(_oracle_normalized_moment,'cache',{})
    key = int(beta),alpha,tau
    if key in cache:
        return cache[key]
    x,wx,den,kernel,const = _contour_grid(tau)
    logs = -(alpha+.5)*x[:,None]-(alpha+1)*den+const+gammaln(alpha+1)+alpha*np.log(tau)
    if beta:
        logs += beta*np.log(np.abs(np.expm1(x[:,None])))
    value = float(np.sum(wx[:,None]*np.sign(x[:,None])**beta*np.exp(logs)*kernel).imag)
    cache[key] = value
    _oracle_normalized_moment.cache = cache
    return value

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Normal, boundary, edge and invalid-input cases."""
    return [{'setup': 'pairs=[(0, -0.5), (1, 0.5), (2, 0.5), (4, 1.5)]', 'call': '[normalized_moment(b,a,0.06125) for b,a in pairs]', 'gold_call': '[_oracle_normalized_moment(b,a,0.06125) for b,a in pairs]'}, {'setup': 'pairs=[(0, 0), (1, 0), (0, 12.5), (4, 13.5)]', 'call': '[normalized_moment(b,a,0.04) for b,a in pairs]', 'gold_call': '[_oracle_normalized_moment(b,a,0.04) for b,a in pairs]'}, {'setup': 'pairs=[(0, -0.5), (1, 2.5), (3, 3.5)]', 'call': '[normalized_moment(b,a,0.2) for b,a in pairs]', 'gold_call': '[_oracle_normalized_moment(b,a,0.2) for b,a in pairs]'}, {'setup': 'pairs=[(0, 6.5), (2, 7.5)]', 'call': '[normalized_moment(b,a,0.1) for b,a in pairs]', 'gold_call': '[_oracle_normalized_moment(b,a,0.1) for b,a in pairs]'}, {'setup': 'def run_model_invalid():\n    try:\n        normalized_moment(2,0.,.1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_normalized_moment(2,0.,.1)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'}, {'setup': 'def run_model_invalid():\n    try:\n        normalized_moment(0,0.,.01)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle_invalid():\n    try:\n        _oracle_normalized_moment(0,0.,.01)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n', 'call': 'run_model_invalid()', 'gold_call': 'run_oracle_invalid()'}]
