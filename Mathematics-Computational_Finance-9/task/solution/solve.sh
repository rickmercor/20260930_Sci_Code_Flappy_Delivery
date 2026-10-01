#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def scale_constants(sigma0: float, nu: float, rho: float, T: float) -> tuple:
    """Reference implementation of the model scale constants."""
    np = __import__("numpy")

    for name, value in (("sigma0", sigma0), ("nu", nu), ("T", T)):
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    rho_val = float(rho)
    if not np.isfinite(rho_val) or not -1.0 < rho_val < 1.0:
        raise ValueError("rho must be finite with -1 < rho < 1")

    sigma0 = float(sigma0)
    nu = float(nu)
    T = float(T)
    tau = nu * nu * T
    v = sigma0 * np.sqrt(np.expm1(tau) / tau)
    radius = sigma0 * np.sqrt(1.0 - rho_val * rho_val) / nu
    return float(tau), float(v), float(radius)

def normal_call_terms(T: float, X0: float, strikes: np.ndarray, sigma: float) -> tuple:
    """Reference implementation of the normal-model call terms."""
    np = __import__("numpy")

    for name, value in (("T", T), ("sigma", sigma)):
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if not np.isfinite(float(X0)):
        raise ValueError("X0 must be finite")
    k = np.asarray(strikes, dtype=float)
    if k.ndim != 1 or k.size == 0:
        raise ValueError("strikes must be a one-dimensional non-empty array")
    if not np.all(np.isfinite(k)):
        raise ValueError("strikes must be finite")

    s = float(sigma) * np.sqrt(float(T))
    d = (float(X0) - k) / s
    dens = np.exp(-0.5 * d * d) / np.sqrt(2.0 * np.pi)
    ndtr = __import__("scipy.special", fromlist=["ndtr"]).ndtr
    cdf = np.asarray(ndtr(d), dtype=float)
    price = (float(X0) - k) * cdf + s * dens
    return price, cdf, dens / s

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

def normalized_moment(beta: int, alpha: float, tau: float) -> float:
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
    cache = getattr(normalized_moment,'cache',{})
    key = int(beta),alpha,tau
    if key in cache:
        return cache[key]
    x,wx,den,kernel,const = _contour_grid(tau)
    logs = -(alpha+.5)*x[:,None]-(alpha+1)*den+const+gammaln(alpha+1)+alpha*np.log(tau)
    if beta:
        logs += beta*np.log(np.abs(np.expm1(x[:,None])))
    value = float(np.sum(wx[:,None]*np.sign(x[:,None])**beta*np.exp(logs)*kernel).imag)
    cache[key] = value
    normalized_moment.cache = cache
    return value

def expectation_vector(M_max: int, N_max: int, sigma0: float, nu: float, T: float) -> dict:
    np = __import__('numpy')
    if isinstance(M_max,bool) or not isinstance(M_max,(int,np.integer)) or not 1 <= M_max <= 4:
        raise ValueError('M_max outside [1,4]')
    if isinstance(N_max,bool) or not isinstance(N_max,(int,np.integer)) or not 0 <= N_max <= 12:
        raise ValueError('N_max outside [0,12]')
    if any(not np.isfinite(v) or v <= 0 for v in [sigma0,nu,T]) or not .04 <= nu*nu*T <= .20:
        raise ValueError('invalid model parameters')
    def E(beta,alpha):
        return sigma0**(beta-2*alpha)*nu**(-beta)*normalized_moment(beta,alpha,nu*nu*T)
    n=range(N_max+1)
    return dict(base_even=np.array([E(0,j-.5) for j in n]),base_odd=np.array([E(1,j+.5) for j in n]),
                high_even=[np.array([E(2*p+2,j+p+.5) for j in n]) for p in range(M_max//2)],
                high_odd=[np.array([E(2*p+3,j+p+1.5) for j in n]) for p in range((M_max-1)//2)])

def coefficient_matrix(exps: dict, M_max: int, N_max: int, rho: float,
                               T: float, v: float) -> np.ndarray:
    """Reference implementation of the coefficient matrix."""
    np = __import__("numpy")
    gammaln = __import__("scipy.special", fromlist=["gammaln"]).gammaln

    if isinstance(M_max, bool) or not isinstance(M_max, (int, np.integer)) or int(M_max) < 1:
        raise ValueError("M_max must be an integer >= 1")
    if isinstance(N_max, bool) or not isinstance(N_max, (int, np.integer)) or int(N_max) < 0:
        raise ValueError("N_max must be an integer >= 0")
    M_max = int(M_max)
    N_max = int(N_max)
    if not isinstance(exps, dict) or any(
            key not in exps for key in ("base_even", "base_odd", "high_even", "high_odd")):
        raise ValueError("exps must carry base_even, base_odd, high_even and high_odd")
    base_even = np.asarray(exps["base_even"], dtype=float)
    base_odd = np.asarray(exps["base_odd"], dtype=float)
    if base_even.shape != (N_max + 1,) or base_odd.shape != (N_max + 1,):
        raise ValueError("base_even and base_odd must both have length N_max + 1")
    high_even = [np.asarray(a, dtype=float) for a in exps["high_even"]]
    high_odd = [np.asarray(a, dtype=float) for a in exps["high_odd"]]
    if len(high_even) != M_max // 2 or len(high_odd) != (M_max - 1) // 2:
        raise ValueError("high_even and high_odd hold the wrong number of arrays")
    if any(a.shape != (N_max + 1,) for a in high_even + high_odd):
        raise ValueError("every high-order array must have length N_max + 1")
    rho = float(rho)
    if not np.isfinite(rho) or not -1.0 < rho < 1.0:
        raise ValueError("rho must be finite with -1 < rho < 1")
    for name, value in (("T", T), ("v", v)):
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    T = float(T)
    v = float(v)

    q = 1.0 - rho * rho
    root_two_pi = np.sqrt(2.0 * np.pi)
    n = np.arange(N_max + 1)
    g = gammaln(np.arange(max(4 * N_max + 12, 2 * N_max + M_max + 3)) + 1.0)
    A = np.zeros((M_max + 1, 2 * N_max + 2))

    delta = base_even - v ** (1.0 - 2.0 * n)
    log_den = g[n] + n * np.log(2.0 * T) + (n - 0.5) * np.log(q)
    A[0, 0::2] = (-np.sqrt(T) / root_two_pi * ((-1.0) ** n) * delta
                  / ((2.0 * n - 1.0) * np.exp(log_den)))

    log_den = g[n] + n * np.log(2.0) + (n + 0.5) * np.log(T) + (n + 0.5) * np.log(q)
    A[1, 1::2] = ((1.0 / (2.0 * root_two_pi)) * ((-1.0) ** n) * base_odd
                  / np.exp(log_den) * (4.0 / (4.0 * n + 2.0)))

    for p, moments in enumerate(high_even):
        log_ratio = (g[2 * n + 2 * p] - g[2 * n] - g[n + p] - (n + p) * np.log(2.0)
                     - (n + p + 0.5) * np.log(q * T))
        A[2 * p + 2, 0::2] = ((1.0 / root_two_pi) * ((-1.0) ** (n + p))
                              * np.exp(log_ratio) * moments / np.exp(g[2 * p + 2]))

    for p, moments in enumerate(high_odd):
        log_ratio = (g[2 * n + 2 * p + 2] - g[2 * n + 1] - g[n + p + 1]
                     - (n + p + 1) * np.log(2.0) - (n + p + 1.5) * np.log(q * T))
        A[2 * p + 3, 1::2] = (-(1.0 / root_two_pi) * ((-1.0) ** (n + p))
                              * np.exp(log_ratio) * moments / np.exp(g[2 * p + 3]))

    return A

def matrix_values(A: np.ndarray, X0: float, strikes: np.ndarray, rho: float, base: tuple) -> tuple:
    np = __import__('numpy')
    A,strikes=np.asarray(A,dtype=float),np.asarray(strikes,dtype=float)
    if A.ndim!=2 or not all(A.shape) or strikes.ndim!=1 or not strikes.size or not np.all(np.isfinite(A)) or not np.all(np.isfinite(strikes)):
        raise ValueError('invalid array dimensions or values')
    if not np.isfinite(X0) or not np.isfinite(rho) or abs(rho)>=1 or not isinstance(base,(tuple,list)) or len(base)!=3:
        raise ValueError('invalid scalars or base')
    b=[np.asarray(v,dtype=float) for v in base]
    if any(v.shape!=strikes.shape or not np.all(np.isfinite(v)) for v in b):
        raise ValueError('invalid base arrays')
    c=np.power(rho,np.arange(A.shape[0]))@A
    d=X0-strikes
    return tuple(b[j]+np.polynomial.polynomial.polyval(d,np.polynomial.polynomial.polyder(c,j)) for j in range(3))

def matrix_expansion_gamma(X0: float=100., sigma0: float=35., nu: float=.7, rho: float=-.6,
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
    tau,v,radius=scale_constants(sigma0,nu,rho,T)
    if np.any(np.abs(X0-k)>=radius):
        raise ValueError('strike outside convergence radius')
    e=expectation_vector(M_max,N_max,sigma0,nu,T)
    A=coefficient_matrix(e,M_max,N_max,rho,T,v)
    base=normal_call_terms(T,X0,k,np.sqrt(1-rho*rho)*v)
    values=matrix_values(A,X0,k,rho,base)
    return float(values[2][np.flatnonzero(k==report_strike)[0]])
SCICODE_GOLD_EOF
