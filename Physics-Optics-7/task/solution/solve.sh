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
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def fold_frame(precision: "np.ndarray", point: "np.ndarray") -> "np.ndarray":
    A = np.asarray(precision, dtype=float)
    x = np.asarray(point, dtype=float)
    v = A @ x
    eig, vec = np.linalg.eigh(np.outer(v, v) - A)
    g = -1.0 / eig[0]
    alpha = g * np.exp(x @ A @ x / 2)
    H = np.eye(2) + g * (np.outer(v, v) - A)
    e = vec[:, 0]
    cubic = g * (3 * (e @ v) * (e @ A @ e) - (e @ v)**3)
    if cubic < 0:
        e = -e
        cubic = -cubic
    V = np.column_stack((e, [-e[1], e[0]]))
    lam = V[:, 1] @ H @ V[:, 1]
    return np.r_[alpha, x - g * v, V.ravel(), lam, (2 / cubic)**(1 / 3)]

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def gaussian_jet(precision: "np.ndarray", alpha: float, point: "np.ndarray", basis: "np.ndarray", degree: int) -> "np.ndarray":
    A = np.asarray(precision, dtype=float)
    x = np.asarray(point, dtype=float)
    V = np.asarray(basis, dtype=float)
    linear = -V.T @ A @ x
    quadratic = -0.5 * V.T @ A @ V
    C = np.zeros((degree + 1, degree + 1))
    C[0, 0] = alpha * np.exp(-x @ A @ x / 2)
    def _at(a, b):
        return C[a, b] if a >= 0 and b >= 0 else 0.0
    for b in range(1, degree + 1):
        C[0, b] = (linear[1] * _at(0, b-1) + 2 * quadratic[1, 1] * _at(0, b-2)) / b
    for a in range(1, degree + 1):
        for b in range(degree - a + 1):
            C[a, b] = (linear[0] * _at(a-1, b) + 2 * quadratic[0, 0] * _at(a-2, b)
                       + 2 * quadratic[0, 1] * _at(a-1, b-1)) / a
    return C

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def fold_powers(jet: "np.ndarray", scale: float, transverse: float, order: int) -> "np.ndarray":
    C = np.asarray(jet, dtype=float)
    R = [{} for _ in range(order + 1)]
    for a in range(C.shape[0]):
        for b in range(C.shape[1]):
            k = 2*a + 3*b - 6
            if a+b >= 3 and 1 <= k <= order and C[a, b] != 0:
                R[k][a, b] = C[a, b] * scale**a * abs(transverse)**(-b/2)
    P = [{(0, 0): 1.0}] + [{} for _ in range(order)]
    for k in range(1, order + 1):
        for j in range(1, k + 1):
            for (a, b), v in R[j].items():
                for (aa, bb), w in P[k-j].items():
                    key = a+aa, b+bb
                    P[k][key] = P[k].get(key, 0) + 1j*j/k * v*w
    out = np.zeros((order+1, 2*order+1, order+1), dtype=complex)
    for k, p in enumerate(P):
        for (a, b), v in p.items():
            out[k, a, b] = v
    return out

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def fold_contractions(powers: "np.ndarray", unfolding: float, transverse: float) -> "np.ndarray":
    P = np.asarray(powers, dtype=complex)
    D = np.zeros(P.shape[1] + 1)
    D[:2] = airy(unfolding)[:2]
    for n in range(P.shape[1]-1):
        D[n+2] = unfolding * D[n] + (n * D[n-1] if n else 0)
    out = np.zeros(P.shape[0], dtype=complex)
    for a in range(P.shape[1]):
        for b in range(0, P.shape[2], 2):
            factor = 1j**(-a) * D[a] * (1j*np.sign(transverse))**(b//2) * prod(range(1,b,2))
            out += P[:, a, b] * factor
    return out

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def saddle_series(jet: "np.ndarray", eigenvalues: "np.ndarray", order: int) -> "np.ndarray":
    C = np.asarray(jet, dtype=float)
    lam = np.asarray(eigenvalues, dtype=float)
    R = {(a,b): C[a,b] for a in range(len(C)) for b in range(len(C))
         if 3 <= a+b <= 2*order+2 and C[a,b] != 0}
    P = {(0,0): 1.0}
    out = np.zeros(order+1, dtype=complex)
    for r in range(2*order+1):
        for (a,b), v in P.items():
            if a % 2 or b % 2:
                continue
            m = (a+b)//2 - r
            if 0 <= m <= order:
                out[m] += (1j)**(r+(a+b)//2) * v / factorial(r) * prod(range(1,a,2)) * prod(range(1,b,2)) / lam[0]**(a//2) / lam[1]**(b//2)
        nxt = {}
        for (a,b), v in P.items():
            for (aa,bb), w in R.items():
                if a+b+aa+bb <= 2*(order+r+1):
                    key = a+aa,b+bb
                    nxt[key] = nxt.get(key,0) + v*w
        P = nxt
    return out

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def gaussian_diffraction(precision: "np.ndarray", alpha: float, position: "np.ndarray", frequency: float) -> "np.ndarray":
    A = np.asarray(precision, dtype=float)
    vals, V = np.linalg.eigh(A)
    y = V.T @ np.asarray(position, dtype=float)
    # The absolute coefficients peak at frequency*alpha. Guard the ensuing cancellation.
    dps = max(60, int(abs(frequency*alpha)/np.log(10)) + 45)
    if alpha == 0:
        return np.array([1., 0.])
    with localcontext() as ctx:
        ctx.prec = dps
        one, zero = Decimal(1), Decimal(0)
        w, al = Decimal(float(frequency)), Decimal(float(alpha))
        vals = [Decimal(float(v)) for v in vals]
        ys = [Decimal(float(v)) for v in y]
        tolerance = one.scaleb(-dps + 5)

        def _atan_reciprocal(q):
            x = one / q
            term, answer, k = x, x, 1
            while abs(term) > tolerance:
                term *= -x*x
                answer += term / (2*k+1)
                k += 1
            return answer

        pi = 16*_atan_reciprocal(Decimal(5))-4*_atan_reciprocal(Decimal(239))

        def _sincos(angle):
            x = angle % (2*pi)
            if x > pi:
                x -= 2*pi
            sine, cosine, st, ct, k = x, one, x, one, 1
            while max(abs(st), abs(ct)) > tolerance:
                st *= -x*x / ((2*k)*(2*k+1))
                ct *= -x*x / ((2*k-1)*(2*k))
                sine += st
                cosine += ct
                k += 1
            return sine, cosine

        factor = one
        total_re, total_im = one, zero
        for n in range(1, int(4*abs(frequency*alpha)) + 201):
            factor *= w*al/n
            gr, gi, er, ei = one, zero, zero, zero
            for aj, yj in zip(vals, ys):
                a = n*aj
                den = a*a+w*w
                norm = w/den.sqrt()
                sr = ((norm+w*w/den)/2).sqrt()
                si = -w*a/(2*den*sr)
                gr, gi = gr*sr-gi*si, gr*si+gi*sr
                er -= w*w*yj*yj*a/(2*den)
                ei += w*yj*yj*a*a/(2*den)
            sine, cosine = _sincos(ei)
            magnitude = factor*er.exp()
            tr, ti = magnitude*(gr*cosine-gi*sine), magnitude*(gr*sine+gi*cosine)
            if n % 4 == 1:
                tr, ti = -ti, tr
            elif n % 4 == 2:
                tr, ti = -tr, -ti
            elif n % 4 == 3:
                tr, ti = ti, -tr
            total_re += tr
            total_im += ti
        return np.array([float(total_re), float(total_im)])

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def coherent_hybrid(fold_coefficients: "np.ndarray", regular_coefficients: "np.ndarray", fold_delay: float, regular_delay: float, scale: float, transverse: float, regular_eigenvalues: "np.ndarray", frequency: float) -> "np.ndarray":
    u = np.asarray(fold_coefficients, dtype=complex)
    t = np.asarray(regular_coefficients, dtype=complex)
    ev = np.asarray(regular_eigenvalues, dtype=float)
    fold_prefactor = scale * frequency**(1/6) * np.sqrt(2*np.pi) / np.sqrt(abs(transverse))
    fold_prefactor *= np.exp(1j*frequency*fold_delay + 1j*np.pi*np.sign(transverse)/4) / 1j
    fold = fold_prefactor * np.cumsum(u[::2] / frequency**(np.arange(len(u[::2]))/3))
    regular = np.exp(1j*frequency*regular_delay - 1j*np.pi*np.sum(ev < 0)/2) / np.sqrt(abs(np.prod(ev)))
    regular *= np.sum(t / frequency**np.arange(len(t)))
    return fold + regular

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def error_profile(approximations: "np.ndarray", references: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    fields = np.asarray(approximations, dtype=complex)
    exact = np.asarray(references, dtype=complex)
    w = np.asarray(weights, dtype=float)
    denominator = np.sum(w[None,:] * abs(exact)**2, axis=1)
    errors = np.sqrt(np.sum(w[None,:,None] * abs(fields-exact[:,:,None])**2,axis=1)/denominator[:,None])
    return errors

import numpy as np
from decimal import Decimal, localcontext
from math import factorial, prod
from scipy.optimize import brentq
from scipy.special import airy

def plasma_fold_benchmark(precision: "np.ndarray", point: "np.ndarray", frequencies: "np.ndarray", unfoldings: "np.ndarray", weights: "np.ndarray", max_order: int = 6) -> "np.ndarray":
    A = np.asarray(precision, dtype=float)
    xc = np.asarray(point, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    unfoldings = np.asarray(unfoldings, dtype=float)
    frame = fold_frame(A, xc)
    alpha, yc, V = frame[0], frame[1:3], frame[3:7].reshape(2,2)
    lam, scale = frame[7:9]
    C = gaussian_jet(A, alpha, xc, V, max_order+3)
    P = fold_powers(C, scale, lam, 2*max_order)
    eigA, U = np.linalg.eigh(A)
    approx = np.zeros((len(frequencies),len(unfoldings),max_order+1),complex)
    exact = np.zeros((len(frequencies),len(unfoldings)),complex)
    for i, w in enumerate(frequencies):
        for j, z in enumerate(unfoldings):
            y = yc - z/(scale*w**(2/3))*V[:,0]
            u = fold_contractions(P,z,lam)
            yy = U.T@y
            def _ray_equation(q):
                return np.log(q/alpha)+0.5*np.sum(eigA*yy**2/(1-q*eigA)**2)
            upper = min(alpha, (1-1e-13)/eigA[-1])
            q = brentq(_ray_equation, np.finfo(float).tiny, upper, xtol=5e-15, rtol=1e-14)
            xr = np.linalg.solve(np.eye(2)-q*A,y)
            H = np.eye(2)+q*(np.outer(A@xr,A@xr)-A)
            ev, Vr = np.linalg.eigh(H)
            regular_jet = gaussian_jet(A,alpha,xr,Vr,6)
            regular_coeff = saddle_series(regular_jet,ev,2)
            Tc = 0.5*np.sum((xc-y)**2)+C[0,0]
            Tr = 0.5*np.sum((xr-y)**2)+q
            approx[i,j] = coherent_hybrid(u,regular_coeff,Tc,Tr,scale,lam,ev,w)
            xy = gaussian_diffraction(A,alpha,y,w)
            exact[i,j] = xy[0]+1j*xy[1]
    profile = error_profile(approx,exact,weights)
    worst = np.max(profile,axis=0)
    ranking = np.argsort(worst,kind='stable')
    winner = int(ranking[0])
    second = int(ranking[1])
    frequency_index = int(np.argmax(profile[:,winner]))
    # [best percent error, retained correction order, worst frequency, leading percent,
    # maximum-order percent, runner-up percent gap]
    return np.array([100*worst[winner],winner,frequencies[frequency_index],100*worst[0],
                     100*worst[-1],100*(worst[second]-worst[winner])])

def _integration_expected():
    """Independent real-plane quadrature certificate; evaluator-only literal."""
    return np.array([12.545986320161948, 3.0, 50.0, 27.65538538343503, 15.143256888428361, 1.56621074465505],dtype=float)
SCICODE_GOLD_EOF
