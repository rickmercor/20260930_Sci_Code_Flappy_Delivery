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
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def radial_geometry(n: int, radius: float, power: float) -> "np.ndarray":
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<2 or not np.isfinite(radius) or radius<=0 or not np.isfinite(power) or power<1:
        raise ValueError("Invalid radial geometry domain")
    r=radius*(np.arange(n+1,dtype=float)/n)**power
    mid=(r[:-1]+r[1:])/2
    w=np.pi**2/2*np.diff(np.r_[0.,mid]**4)
    c=2*np.pi**2*mid**3/np.diff(r)
    return np.column_stack((r[:-1],w,c,c+np.r_[0.,c[:-1]]))

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def action_jet(phi: "np.ndarray", kappa: float, geometry: "np.ndarray") -> "np.ndarray":
    phi=np.asarray(phi,dtype=float);geometry=np.asarray(geometry,dtype=float)
    if phi.ndim!=1 or len(phi)<2 or geometry.shape!=(len(phi),4) or not np.all(np.isfinite(phi)) or not np.all(np.isfinite(geometry)) or np.any(geometry[:,1]<=0) or not np.isfinite(kappa):
        raise ValueError("Invalid action-jet inputs")
    x=np.asarray(phi,dtype=float)
    r,w,c,ld=np.asarray(geometry).T
    lx=ld*x
    lx[:-1]-=c[:-1]*x[1:]
    lx[1:]-=c[:-1]*x[:-1]
    action=.5*np.sum(c*np.diff(np.r_[x,0.])**2)+w@(x*x/2-x**4/24)
    constraint=w@x**3
    gradient=(lx+w*(x-x**3/6+3*kappa*x*x))/np.sqrt(w)
    hdiag=ld/w+1-x*x/2+6*kappa*x
    return np.r_[action,constraint,action+kappa*constraint,gradient,hdiag]

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp


def stationary_profile(
    kappa: float,
    geometry: "np.ndarray",
) -> "np.ndarray":
    geometry = np.asarray(geometry, dtype=float)

    if (
        not np.isfinite(kappa)
        or not -0.65 <= kappa <= -0.15
        or geometry.ndim != 2
        or geometry.shape[1] != 4
        or len(geometry) < 2
        or not np.all(np.isfinite(geometry))
        or np.any(geometry[:, 1] <= 0)
    ):
        raise ValueError("Invalid stationary-family inputs")

    n = len(geometry)
    r, w, c, ld = np.asarray(geometry).T

    def _residual(x, k):
        return action_jet(x, k, geometry)[3 : 3 + n] / np.sqrt(w)

    def _jac(x, k):
        diagonal = action_jet(x, k, geometry)[3 + n :]
        return (
            np.diag(diagonal)
            + np.diag(-c[:-1] / w[:-1], 1)
            + np.diag(-c[:-1] / w[1:], -1)
        )

    def _attempt(k, guess):
        fit = root(
            _residual,
            guess,
            args=(k,),
            jac=_jac,
            tol=2e-11,
        )
        x = fit.x
        good = (
            np.max(abs(_residual(x, k)) / (1 + abs(x) ** 3)) < 3e-9
            and x[0] > 0.2
            and x.min() > -1e-8
            and np.max(np.diff(x)) < 1e-6
        )
        return x, good

    # Tridiagonal Newton continuation stays on the positive radial branch.
    rho = 1.0
    x = (
        4
        * np.sqrt(3)
        * rho
        / (rho * rho + r * r)
        * np.exp(-0.3 * r)
    )

    good = True

    continuation = np.r_[
        -0.5,
        np.linspace(
            -0.5,
            kappa,
            max(2, int(abs(kappa + 0.5) / 0.012) + 2),
        )[1:],
    ]

    for k in continuation:
        for iteration in range(45):
            jet = action_jet(x, k, geometry)
            gradient = jet[3 : 3 + n]

            if np.max(abs(gradient)) < 2e-8:
                break

            off = -c[:-1] / np.sqrt(w[:-1] * w[1:])
            band = np.zeros((3, n))
            band[1] = jet[3 + n :]
            band[0, 1:] = off
            band[2, :-1] = off

            x += solve_banded(
                (1, 1),
                band,
                -gradient,
            ) / np.sqrt(w)
        else:
            good = False
            break

    good = (
        good
        and x[0] > 0.2
        and x.min() > -1e-8
        and np.max(np.diff(x)) < 1e-6
    )

    if not good:
        rho = -2 * kappa
        guess = (
            4
            * np.sqrt(3)
            * rho
            / (rho * rho + r * r)
            * np.exp(-0.3 * r)
        )
        x, good = _attempt(kappa, guess)

    if not good:
        raise RuntimeError(
            "Positive decreasing stationary profile failed to converge"
        )

    return x

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def normal_response(jet: "np.ndarray", phi: "np.ndarray", geometry: "np.ndarray") -> "np.ndarray":
    jet=np.asarray(jet,dtype=float);phi=np.asarray(phi,dtype=float);geometry=np.asarray(geometry,dtype=float)
    if phi.ndim!=1 or len(phi)<2 or geometry.shape!=(len(phi),4) or jet.shape!=(3+2*len(phi),) or not np.all(np.isfinite(jet)) or not np.all(np.isfinite(phi)) or not np.all(np.isfinite(geometry)) or np.any(geometry[:,1]<=0) or not np.any(phi):
        raise ValueError("Invalid normal-response inputs")
    n=len(phi);w=geometry[:,1];c=geometry[:,2]
    diagonal=jet[3+n:]
    off=-c[:-1]/np.sqrt(w[:-1]*w[1:])
    band=np.zeros((3,n));band[1]=diagonal;band[0,1:]=off;band[2,:-1]=off
    zeta=3*np.sqrt(w)*np.asarray(phi)**2
    psi=solve_banded((1,1),band,zeta)
    norm=zeta@zeta;numerator=zeta@psi
    values=eigvalsh_tridiagonal(diagonal,off,lapack_driver='stebz')
    if np.min(abs(values))<1e-8:raise ValueError('Full modified Hessian must be nonsingular')
    return np.array([np.sum(values<0),numerator/norm,-numerator,
                     np.sum(np.log(abs(values))),norm])

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def constraint_fold(geometry: "np.ndarray") -> "np.ndarray":
    def _normal(k):
        p=stationary_profile(k,geometry)
        j=action_jet(p,k,geometry)
        return normal_response(j,p,geometry)[1]
    k=brentq(_normal,-.32,-.21,xtol=2e-12)
    p=stationary_profile(k,geometry);j=action_jet(p,k,geometry)
    return np.array([k,j[1],j[0]])

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def matched_branches(fraction: float, fold: "np.ndarray", geometry: "np.ndarray") -> "np.ndarray":
    fold=np.asarray(fold,dtype=float)
    if not np.isfinite(fraction) or not 0<fraction<1 or fold.shape!=(3,) or not np.all(np.isfinite(fold)) or fold[1]<=0:
        raise ValueError("Invalid matched-branch inputs")
    target=fraction*fold[1];kc=fold[0]
    def _difference(k):
        p=stationary_profile(k,geometry)
        return action_jet(p,k,geometry)[1]-target
    km=brentq(_difference,-.65,kc,xtol=2e-12)
    ks=brentq(_difference,kc,-.15,xtol=2e-12)
    # Row labels are determined by projected inertia, not action ordering.
    rows=[]
    for k in (km,ks):
        p=stationary_profile(k,geometry);j=action_jet(p,k,geometry)
        response=normal_response(j,p,geometry)
        index=int(round(response[0]))-int(response[1]<0)
        rows.append((index,np.r_[k,p]))
    rows.sort(key=lambda item:item[0])
    if [item[0] for item in rows]!=[0,1]:raise RuntimeError('Expected one minimum and one constrained saddle')
    return np.stack([item[1] for item in rows])

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def branch_logweights(branches: "np.ndarray", coupling: float, geometry: "np.ndarray") -> "np.ndarray":
    branches=np.asarray(branches,dtype=float);geometry=np.asarray(geometry,dtype=float)
    if not np.isfinite(coupling) or coupling<=0 or geometry.ndim!=2 or geometry.shape[1]!=4 or branches.shape!=(2,len(geometry)+1) or not np.all(np.isfinite(branches)):
        raise ValueError("Invalid Gaussian-weight inputs")
    rows=[]
    for branch in branches:
        k=branch[0];p=branch[1:]
        jet=action_jet(p,k,geometry)
        response=normal_response(jet,p,geometry)
        logdet=response[3]+np.log(abs(response[1]))
        # The action is s/lambda, so the tangent Hessian has N-1 factors of 1/lambda.
        logweight=-jet[0]/coupling-.5*(logdet-(len(p)-1)*np.log(coupling))
        rows.append([jet[0],response[1],logdet,logweight])
    return np.array(rows)

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def integrated_suppression(fractions: "np.ndarray", weights: "np.ndarray", logweights: "np.ndarray", xi_critical: float) -> "np.ndarray":
    fractions=np.asarray(fractions,dtype=float);weights=np.asarray(weights,dtype=float);logweights=np.asarray(logweights,dtype=float)
    if fractions.ndim!=1 or not len(fractions) or weights.shape!=fractions.shape or logweights.shape!=(len(fractions),2) or not np.all(np.isfinite(fractions)) or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(logweights)) or np.any(weights<=0) or not np.isfinite(xi_critical) or xi_critical<=0:
        raise ValueError("Invalid quadrature inputs")
    logs=logsumexp(np.log(np.asarray(weights))[:,None]+logweights,axis=0)+np.log(xi_critical)
    posterior=np.exp(np.log(weights)+logweights[:,1]-logs[1]+np.log(xi_critical))
    mean_fraction=posterior@np.asarray(fractions)
    return np.array([logs[0]-logs[1],logs[0],logs[1],mean_fraction])

import numpy as np
from scipy.linalg import solve_banded, eigvalsh_tridiagonal
from scipy.optimize import root, brentq
from scipy.special import logsumexp

def radial_competition(coupling: float = 0.7, lower: float = 0.84, upper: float = 0.97, n: int = 160, radius: float = 16.0, power: float = 2.0, order: int = 24) -> "np.ndarray":
    if not np.isfinite(coupling) or coupling<=0 or not 0<lower<upper<1 or not isinstance(order,(int,np.integer)) or isinstance(order,bool) or order<2:
        raise ValueError("Invalid competition inputs")
    geometry=radial_geometry(n,radius,power)
    fold=constraint_fold(geometry)
    # An anchor supplies a compact, independently scored spectral certificate.
    anchor=stationary_profile(-.22,geometry)
    jet=action_jet(anchor,-.22,geometry)
    response=normal_response(jet,anchor,geometry)
    nodes,weights=np.polynomial.legendre.leggauss(order)
    fractions=(upper+lower)/2+(upper-lower)/2*nodes
    weights=weights*(upper-lower)/2
    logweights=[]
    for fraction in fractions:
        pair=matched_branches(fraction,fold,geometry)
        values=branch_logweights(pair,coupling,geometry)
        logweights.append(values[:,3])
    integrated=integrated_suppression(fractions,weights,np.asarray(logweights),fold[1])
    middle=matched_branches((lower+upper)/2,fold,geometry)
    values=branch_logweights(middle,coupling,geometry)
    return np.r_[integrated[0],fold[0],fold[1],middle[:,0],
                 values[1,0]-values[0,0],values[1,2]-values[0,2],integrated[3],response[1]]
SCICODE_GOLD_EOF
