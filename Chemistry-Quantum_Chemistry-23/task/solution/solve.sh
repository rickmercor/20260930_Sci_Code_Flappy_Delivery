#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def _p_radial_polynomial(n_principal):
    if n_principal == 2:
        return 1.0/(2.0*np.sqrt(6.0)), [0.0,1.0]
    if n_principal == 3:
        return 8.0/(27.0*np.sqrt(6.0)), [0.0,1.0,-1.0/6.0]
    return np.sqrt(5.0)/(16.0*np.sqrt(3.0)), [0.0,1.0,-0.25,1.0/80.0]


def p_shell_semilocal_ingredients(n_principal: int, Z: float, r: np.ndarray) -> np.ndarray:
    if n_principal not in (2,3,4):
        raise ValueError("n_principal must be 2, 3 or 4.")
    if Z <= 0.0:
        raise ValueError("Z must be positive.")
    r=np.asarray(r,dtype=float)
    if np.any(r<=0.0):
        raise ValueError("Radii must be positive.")
    c,coeffs=_p_radial_polynomial(n_principal)
    poly=np.polynomial.Polynomial(coeffs)
    x=Z*r; decay=np.exp(-x/n_principal)
    R=c*Z**1.5*poly(x)*decay
    dR=c*Z**2.5*(poly.deriv()(x)-poly(x)/n_principal)*decay
    n=3.0/(2.0*np.pi)*R**2
    dn=3.0/np.pi*R*dR
    tau=3.0/(4.0*np.pi)*(dR**2+2.0*R**2/r**2)
    keep=n>=1e-30
    tau_w=np.zeros_like(n); grad=np.zeros_like(n); z=np.zeros_like(n)
    nk,dnk,tauk=n[keep],dn[keep],tau[keep]
    tau_w[keep]=dnk**2/(8.0*nk)
    grad[keep]=np.abs(dnk)/(2.0*(3.0*np.pi**2)**(1.0/3.0)*nk**(4.0/3.0))
    z[keep]=np.minimum(tau_w[keep]/tauk,1.0)
    return np.array([R,dR,n,dn,tau,tau_w,grad,z])

import numpy as np





def _radial_trapezoid(r, f):

    return float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(r)))





def pc_strong_interaction(r: np.ndarray, n: np.ndarray, dn: np.ndarray) -> tuple[float, float]:

    r = np.asarray(r, dtype=float)

    n = np.asarray(n, dtype=float)

    dn = np.asarray(dn, dtype=float)

    if not (len(r) == len(n) == len(dn)):

        raise ValueError("Arrays must have the same length.")

    if np.any(np.diff(r) <= 0.0):

        raise ValueError("The grid must be strictly increasing.")

    A, B, C, D = -1.451, 5.317e-3, 1.535, -0.02558

    keep = n >= 1e-30

    f_w = np.zeros_like(r)

    f_wp = np.zeros_like(r)

    nk, g2 = n[keep], dn[keep] ** 2

    w4 = 4.0 * np.pi * r[keep] ** 2

    f_w[keep] = w4 * (A * nk ** (4.0 / 3.0) + B * g2 / nk ** (4.0 / 3.0))

    f_wp[keep] = w4 * (C * nk**1.5 + D * g2 / nk ** (7.0 / 6.0))

    return (_radial_trapezoid(r, f_w), _radial_trapezoid(r, f_wp))

import numpy as np


def epc_w_inf_branches(s: np.ndarray) -> np.ndarray:
    s=np.asarray(s,dtype=float)
    if np.any(~np.isfinite(s)) or np.any(s<0.0):
        raise ValueError("s must be finite and nonnegative.")
    A=-1.451
    a_x=-0.75*(3.0/np.pi)**(1.0/3.0)
    kappa=1.0-a_x/A
    mu=0.14
    a1,a2,a3=0.1,0.9342,0.22447
    s2=s**2
    x=mu*s2/kappa
    f0=1.0-kappa+kappa/(1.0+x+x*x)
    f1=a1+a2/(1.0+a3*s2**4)
    return np.array([f0,f1])

import numpy as np


def _radial_trapezoid(r, f):
    return float(np.sum(0.5*(f[1:]+f[:-1])*np.diff(r)))


def epc_w_inf(r: np.ndarray, n: np.ndarray, s: np.ndarray, z: np.ndarray) -> float:
    r=np.asarray(r,dtype=float); n=np.asarray(n,dtype=float); s=np.asarray(s,dtype=float); z=np.asarray(z,dtype=float)
    if not (len(r)==len(n)==len(s)==len(z)):
        raise ValueError("Arrays must have the same length.")
    if np.any(np.diff(r)<=0.0) or np.any(n<0.0) or np.any((z<0.0)|(z>1.0)):
        raise ValueError("Invalid radial grid, density or z.")
    f0,f1=epc_w_inf_branches(s)
    f=f0+(z*f1-f0)*z**6.65
    keep=n>=1e-30
    integrand=np.zeros_like(r)
    integrand[keep]=4.0*np.pi*r[keep]**2*(-1.451)*n[keep]**(4.0/3.0)*f[keep]
    return _radial_trapezoid(r,integrand)

import numpy as np


def epc_wprime_branches(s: np.ndarray, zeta: float = 0.0) -> np.ndarray:
    s=np.asarray(s,dtype=float); zeta=float(zeta)
    if np.any(~np.isfinite(s)) or np.any(s<0.0) or not np.isfinite(zeta) or abs(zeta)>1.0:
        raise ValueError("Invalid s or zeta.")
    mu_p=0.491
    b1,b2,b3=0.04865,4.3217,16.581
    s2=s**2
    f0=(1.0+(mu_p+1.0)*s2)/(1.0+s2)
    f1=(b1+(b1+b2*s2)*np.exp(-b3*s2**3))*(1.0-zeta**10)
    return np.array([f0,f1])

import numpy as np


def _radial_trapezoid(r, f):
    return float(np.sum(0.5*(f[1:]+f[:-1])*np.diff(r)))


def epc_wprime_inf(r: np.ndarray, n: np.ndarray, s: np.ndarray, z: np.ndarray) -> float:
    r=np.asarray(r,dtype=float); n=np.asarray(n,dtype=float); s=np.asarray(s,dtype=float); z=np.asarray(z,dtype=float)
    if not (len(r)==len(n)==len(s)==len(z)):
        raise ValueError("Arrays must have the same length.")
    if np.any(np.diff(r)<=0.0) or np.any(n<0.0) or np.any((z<0.0)|(z>1.0)):
        raise ValueError("Invalid radial grid, density or z.")
    f0,f1=epc_wprime_branches(s,0.0)
    f=f0+(z**11*f1-f0)*z**2
    keep=n>=1e-30
    integrand=np.zeros_like(r)
    integrand[keep]=4.0*np.pi*r[keep]**2*1.535*n[keep]**1.5*f[keep]
    return _radial_trapezoid(r,integrand)

import numpy as np


def p_shell_epc_wprime(n_principal: int, Z: float) -> float:
    if n_principal not in (2,3,4):
        raise ValueError("n_principal must be 2, 3 or 4.")
    if Z<=0.0:
        raise ValueError("Z must be positive.")
    r=np.geomspace(1e-6,60.0*n_principal**2,20001)/Z
    ing=p_shell_semilocal_ingredients(n_principal,Z,r)
    pc_reference = pc_strong_interaction(r,ing[2],ing[3])
    if not np.all(np.isfinite(pc_reference)):
        raise ValueError("The PC reference audit requires two finite values.")
    w_inf=epc_w_inf(r,ing[2],ing[6],ing[7])
    w_prime=epc_wprime_inf(r,ing[2],ing[6],ing[7])
    if w_inf>=0.0 or w_prime<0.0:
        raise ValueError("The ePC model requires W_inf < 0 and W'_inf >= 0.")
    return w_prime
SCICODE_GOLD_EOF
