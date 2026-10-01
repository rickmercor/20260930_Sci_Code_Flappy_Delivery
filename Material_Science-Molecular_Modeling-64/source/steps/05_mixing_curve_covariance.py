"""
Construct excess-mixing responses and shared-endmember sampling covariance.

The paper forms G_mix from a common pure-solute endpoint. Reusing its noisy estimate correlates every interior composition after subtracting x Delta G(1).

Returns
-------
return np.column_stack((interior, response, covariance)).astype(float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def mixing_curve_covariance(compositions, gibbs_certificates, temperature_k):
    """Build excess-mixing responses with shared-endmember covariance.

    `compositions` is strictly increasing and ends at x=1. Each row of
    `gibbs_certificates` is ordered [Delta_G, standard_error, F_TI, F_mass,
    F_PV]. For C compositions, returns a float64 ndarray of shape
    (C-1,C+1). Column 0 is the increasing interior composition x, column 1
    is the excess response in eV/atom, and columns 2..C are its (C-1)x(C-1)
    covariance matrix in the same x row/column order, in (eV/atom)^2.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_mixing_curve_covariance(compositions, gibbs_certificates, temperature_k):
    """Construct excess-mixing responses and shared-endmember covariance."""
    import math
    import numpy as np
    x = np.asarray(compositions, dtype=float)
    certificates = np.asarray(gibbs_certificates, dtype=float)
    temperature = float(temperature_k)
    if x.ndim != 1 or x.size < 5 or certificates.shape != (x.size, 5):
        raise ValueError("need C>=5 compositions and Gibbs certificates of shape (C,5)")
    if not math.isfinite(temperature) or temperature <= 0.0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(certificates)):
        raise ValueError("inputs must be finite and temperature positive")
    if np.any(x <= 0.0) or np.any(x > 1.0) or np.any(np.diff(x) <= 0.0) or x[-1] != 1.0:
        raise ValueError("compositions must increase strictly in (0,1] and end at 1")
    if np.any(certificates[:, 1] <= 0.0):
        raise ValueError("all Gibbs standard errors must be positive")
    k_b_ev_per_k = 8.617333262145e-5
    interior = x[:-1]
    ideal = k_b_ev_per_k * temperature * (
        interior * np.log(interior) + (1.0 - interior) * np.log(1.0 - interior)
    )
    response = certificates[:-1, 0] - interior * certificates[-1, 0] - ideal
    variances = certificates[:, 1] ** 2
    covariance = np.diag(variances[:-1]) + variances[-1] * np.outer(interior, interior)
    return np.column_stack((interior, response, covariance)).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np, math\nx=np.array([0.2, 0.4, 0.6, 0.8, 1],float);L=np.array([0.1, 0.02, -0.01],float);T=500;d1=-0.4;err=np.array([0.001, 0.001, 0.001, 0.001, 0.001],float);kb=8.617333262145e-5\ng=[]\nfor i,q in enumerate(x):\n ideal=0.0 if q==1 else kb*T*(q*math.log(q)+(1-q)*math.log(1-q));z=2*q-1;dg=q*d1+ideal+q*(1-q)*sum(L[k]*z**k for k in range(3));g.append([dg,err[i],0,0,0])\ng=np.array(g)', 'call': 'mixing_curve_covariance(x,g,T)', 'gold_call': '_oracle_mixing_curve_covariance(x,g,T)'}, {'setup': 'import numpy as np, math\nx=np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1],float);L=np.array([0.08, -0.01, 0.025],float);T=650;d1=0.2;err=np.array([0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011],float);kb=8.617333262145e-5\ng=[]\nfor i,q in enumerate(x):\n ideal=0.0 if q==1 else kb*T*(q*math.log(q)+(1-q)*math.log(1-q));z=2*q-1;dg=q*d1+ideal+q*(1-q)*sum(L[k]*z**k for k in range(3));g.append([dg,err[i],0,0,0])\ng=np.array(g)', 'call': 'mixing_curve_covariance(x,g,T)', 'gold_call': '_oracle_mixing_curve_covariance(x,g,T)'}, {'setup': 'import numpy as np, math\nx=np.array([0.15, 0.35, 0.55, 0.75, 0.95, 1],float);L=np.array([0.14, 0, 0],float);T=450;d1=-0.1;err=np.array([0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0015],float);kb=8.617333262145e-5\ng=[]\nfor i,q in enumerate(x):\n ideal=0.0 if q==1 else kb*T*(q*math.log(q)+(1-q)*math.log(1-q));z=2*q-1;dg=q*d1+ideal+q*(1-q)*sum(L[k]*z**k for k in range(3));g.append([dg,err[i],0,0,0])\ng=np.array(g)', 'call': 'mixing_curve_covariance(x,g,T)', 'gold_call': '_oracle_mixing_curve_covariance(x,g,T)'}, {'setup': 'import numpy as np, math\nx=np.array([0.05, 0.2, 0.45, 0.7, 0.92, 1],float);L=np.array([0.11, 0.03, 0.02],float);T=300;d1=-0.8;err=np.array([0.002, 0.001, 0.0007, 0.001, 0.0015, 0.003],float);kb=8.617333262145e-5\ng=[]\nfor i,q in enumerate(x):\n ideal=0.0 if q==1 else kb*T*(q*math.log(q)+(1-q)*math.log(1-q));z=2*q-1;dg=q*d1+ideal+q*(1-q)*sum(L[k]*z**k for k in range(3));g.append([dg,err[i],0,0,0])\ng=np.array(g)', 'call': 'mixing_curve_covariance(x,g,T)', 'gold_call': '_oracle_mixing_curve_covariance(x,g,T)'}, {'setup': 'import numpy as np, math\nx=np.array([0.25, 0.5, 0.75, 0.9, 1],float);L=np.array([0.09, -0.04, 0.03],float);T=800;d1=0.5;err=np.array([0.001, 0.0012, 0.0014, 0.0016, 0.002],float);kb=8.617333262145e-5\ng=[]\nfor i,q in enumerate(x):\n ideal=0.0 if q==1 else kb*T*(q*math.log(q)+(1-q)*math.log(1-q));z=2*q-1;dg=q*d1+ideal+q*(1-q)*sum(L[k]*z**k for k in range(3));g.append([dg,err[i],0,0,0])\ng=np.array(g)', 'call': 'mixing_curve_covariance(x,g,T)', 'gold_call': '_oracle_mixing_curve_covariance(x,g,T)'}, {'setup': 'import numpy as np, math\nx=np.array([0.12, 0.28, 0.44, 0.61, 0.79, 0.93, 1],float);L=np.array([0.13, 0.01, -0.02],float);T=520;d1=-0.2;err=np.array([0.0004, 0.0005, 0.0007, 0.0006, 0.0009, 0.001, 0.0018],float);kb=8.617333262145e-5\ng=[]\nfor i,q in enumerate(x):\n ideal=0.0 if q==1 else kb*T*(q*math.log(q)+(1-q)*math.log(1-q));z=2*q-1;dg=q*d1+ideal+q*(1-q)*sum(L[k]*z**k for k in range(3));g.append([dg,err[i],0,0,0])\ng=np.array(g)', 'call': 'mixing_curve_covariance(x,g,T)', 'gold_call': '_oracle_mixing_curve_covariance(x,g,T)'}, {'setup': 'import numpy as np, math\nx=np.array([0.18, 0.36, 0.54, 0.72, 0.9, 1],float);L=np.array([0.07, 0.05, 0.01],float);T=1000;d1=0.1;err=np.array([0.003, 0.0025, 0.002, 0.0015, 0.001, 0.004],float);kb=8.617333262145e-5\ng=[]\nfor i,q in enumerate(x):\n ideal=0.0 if q==1 else kb*T*(q*math.log(q)+(1-q)*math.log(1-q));z=2*q-1;dg=q*d1+ideal+q*(1-q)*sum(L[k]*z**k for k in range(3));g.append([dg,err[i],0,0,0])\ng=np.array(g)', 'call': 'mixing_curve_covariance(x,g,T)', 'gold_call': '_oracle_mixing_curve_covariance(x,g,T)'}, {'setup': 'import numpy as np, math\nx=np.array([0.08, 0.17, 0.29, 0.46, 0.68, 0.87, 1],float);L=np.array([0.16, -0.025, 0.012],float);T=575;d1=-0.35;err=np.array([0.0005, 0.0008, 0.0006, 0.0011, 0.0009, 0.0013, 0.0021],float);kb=8.617333262145e-5\ng=[]\nfor i,q in enumerate(x):\n ideal=0.0 if q==1 else kb*T*(q*math.log(q)+(1-q)*math.log(1-q));z=2*q-1;dg=q*d1+ideal+q*(1-q)*sum(L[k]*z**k for k in range(3));g.append([dg,err[i],0,0,0])\ng=np.array(g)', 'call': 'mixing_curve_covariance(x,g,T)', 'gold_call': '_oracle_mixing_curve_covariance(x,g,T)'}]
