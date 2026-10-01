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
from scipy import special
from scipy.linalg import eigh
from scipy.special import gammaln

A0_NM = 0.052917721
EH_MEV = 27211.386
B0_T = 235051.757

def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError("bad " + name)
    return v
def _posint(x, name):
    v = _fin(x, name)
    if v <= 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _nnint(x, name):
    v = _fin(x, name)
    if v < 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _vec(x, name, positive=True):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)): raise ValueError("bad " + name)
    if positive and np.any(arr <= 0.0): raise ValueError("bad " + name)
    return arr
def _params(mex, mey, mhx, mhy):
    mux = mex*mhx/(mex+mhx); muy = mey*mhy/(mey+mhy); Mx = mex+mhx; My = mey+mhy
    rx = mex/mhx; ry = mey/mhy
    alpha = (1.0-rx*ry)/((1.0+rx)*(1.0+ry)); bx = 4.0*rx/(1.0+rx)**2; by = 4.0*ry/(1.0+ry)**2
    return mux, muy, Mx, My, alpha, (by+alpha*alpha)/muy, (bx+alpha*alpha)/mux
def _keldysh(r, r0, kappa):
    z = kappa*np.asarray(r, dtype=float)/r0
    return -(np.pi/(2.0*r0))*(special.struve(0, z) - special.y0(z))
def _lagfun(nmax, al, t):
    t = np.asarray(t, dtype=float)
    psi = np.zeros((nmax, t.size)); dpsi = np.zeros_like(psi); d2psi = np.zeros_like(psi)
    psi[0] = np.exp(0.5*al*np.log(t) - 0.5*t - 0.5*gammaln(al+1.0))
    if nmax > 1: psi[1] = (1.0+al-t)*psi[0]/np.sqrt(1.0+al)
    for n in range(1, nmax-1):
        psi[n+1] = ((2*n+1+al-t)*psi[n] - np.sqrt(n*(n+al))*psi[n-1])/np.sqrt((n+1)*(n+1+al))
    g = al/(2.0*t) - 0.5
    for n in range(nmax):
        prev = psi[n-1] if n >= 1 else 0.0
        dpsi[n] = g*psi[n] + (n*psi[n] - np.sqrt(n*(n+al))*prev)/t
        Q = dpsi[n] - g*psi[n]
        d2psi[n] = -al/(2.0*t*t)*psi[n] + g*dpsi[n] + g*Q + (-(al+1.0-t)*Q - n*psi[n])/t
    return psi, dpsi, d2psi
def _basis(a, N, m, r):
    t = np.asarray(r, dtype=float)/a; am = abs(int(m))
    psi, dpsi, d2psi = _lagfun(N, 2*am+1, t)
    s = t**-0.5
    return psi*s/a, (dpsi*s - 0.5*psi*t**-1.5)/a**2, (d2psi*s - dpsi*t**-1.5 + 0.75*psi*t**-2.5)/a**3
def _quad(a, Nq):
    n = np.arange(Nq)
    J = np.diag(2.0*n+1.0) + np.diag(np.arange(1, Nq, dtype=float), 1) + np.diag(np.arange(1, Nq, dtype=float), -1)
    t = np.linalg.eigvalsh(J)
    ssum = np.sum(_lagfun(Nq, 0.0, t)[0]**2, axis=0)
    wt = np.where(ssum > 0.0, 1.0/np.where(ssum > 0.0, ssum, 1.0), 0.0)
    return a*t, a*a*t*wt
def _hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, Kx, Ky, a, N, M, Nq, factorized=False):
    mux, muy, Mx, My, alpha, cX, cY = _params(mex, mey, mhx, mhy)
    if factorized:
        alpha = 0.0; cX = 1.0/muy; cY = 1.0/mux
    r, W = _quad(a, Nq); V = _keldysh(r, r0, kappa)
    ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r) for m in ms}
    kiso = 0.25*(1.0/mux + 1.0/muy); kan = 0.25*(1.0/mux - 1.0/muy)
    zi = 0.5*(1.0/mux + 1.0/muy); za = 0.5*(1.0/mux - 1.0/muy)
    cd = 0.5*(cX + cY)*B*B/8.0; cc = 0.5*(cX - cY)*B*B/8.0
    H = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f, df, d2f = F[m]; sl = slice(i*N, (i+1)*N)
        lap = d2f + df/r - (m*m)*f/r**2
        H[sl, sl] += -kiso*(f*W) @ lap.T + (f*W) @ (V*f).T + (f*W) @ ((cd*r*r)*f).T + (0.5*alpha*B*zi*m)*(f*W) @ f.T
        if i-2 >= 0:
            g = F[m-2][0]; sg = slice((i-2)*N, (i-1)*N)
            opz = 0.25*(d2f + (2*m-1)*df/r + m*(m-2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opz.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(r*df + m*f)).T
        if i+2 < nm:
            g = F[m+2][0]; sg = slice((i+2)*N, (i+3)*N)
            opzb = 0.25*(d2f - (2*m+1)*df/r + m*(m+2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opzb.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(m*f - r*df)).T
        if Kx != 0.0 or Ky != 0.0:
            for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
                if 0 <= i+dm < nm:
                    g = F[m+dm][0]; sg = slice((i+dm)*N, (i+dm+1)*N)
                    H[sg, sl] += (B*(Ky/My*cxk - Kx/Mx*cyk))*(g*W) @ (r*f).T
    return H
def _dipoles(a, N, M, Nq):
    r, W = _quad(a, Nq); ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r)[0] for m in ms}
    X = np.zeros((D, D), dtype=complex); Y = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f = F[m]; sl = slice(i*N, (i+1)*N)
        for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
            if 0 <= i+dm < nm:
                g = F[m+dm]; sg = slice((i+dm)*N, (i+dm+1)*N); blk = (g*W) @ (r*f).T
                X[sg, sl] += cxk*blk; Y[sg, sl] += cyk*blk
    return X, Y
def _spectrum(H, k):
    w = eigh(H, eigvals_only=True)
    return w[:k]
def _polarizability(H, X, Y):
    w, v = eigh(H)
    dx = v.conj().T @ (X @ v[:, 0]); dy = v.conj().T @ (Y @ v[:, 0])
    axx = 2.0*np.sum(np.abs(dx[1:])**2/(w[1:]-w[0])); ayy = 2.0*np.sum(np.abs(dy[1:])**2/(w[1:]-w[0]))
    return float(w[0]), float(axx), float(ayy)

def mx_parameters(mex, mey, mhx, mhy, r0_nm, kappa, B_T):
    if not np.isfinite(float(mex)) or float(mex) <= 0.0: raise ValueError("mex must be positive")
    mex = _pos(mex, "mex"); mey = _pos(mey, "mey"); mhx = _pos(mhx, "mhx"); mhy = _pos(mhy, "mhy")
    r0 = _pos(r0_nm, "r0_nm"); kappa = _pos(kappa, "kappa"); BT = _nonneg(B_T, "B_T")
    mux, muy, Mx, My, alpha, cX, cY = _params(mex, mey, mhx, mhy)
    return np.array([mux, muy, Mx, My, alpha, cX, cY, r0/0.052917721, BT/235051.757], dtype=np.float64)

import numpy as np
from scipy import special
from scipy.linalg import eigh
from scipy.special import gammaln

A0_NM = 0.052917721
EH_MEV = 27211.386
B0_T = 235051.757

def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError("bad " + name)
    return v
def _posint(x, name):
    v = _fin(x, name)
    if v <= 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _nnint(x, name):
    v = _fin(x, name)
    if v < 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _vec(x, name, positive=True):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)): raise ValueError("bad " + name)
    if positive and np.any(arr <= 0.0): raise ValueError("bad " + name)
    return arr
def _params(mex, mey, mhx, mhy):
    mux = mex*mhx/(mex+mhx); muy = mey*mhy/(mey+mhy); Mx = mex+mhx; My = mey+mhy
    rx = mex/mhx; ry = mey/mhy
    alpha = (1.0-rx*ry)/((1.0+rx)*(1.0+ry)); bx = 4.0*rx/(1.0+rx)**2; by = 4.0*ry/(1.0+ry)**2
    return mux, muy, Mx, My, alpha, (by+alpha*alpha)/muy, (bx+alpha*alpha)/mux
def _keldysh(r, r0, kappa):
    z = kappa*np.asarray(r, dtype=float)/r0
    return -(np.pi/(2.0*r0))*(special.struve(0, z) - special.y0(z))
def _lagfun(nmax, al, t):
    t = np.asarray(t, dtype=float)
    psi = np.zeros((nmax, t.size)); dpsi = np.zeros_like(psi); d2psi = np.zeros_like(psi)
    psi[0] = np.exp(0.5*al*np.log(t) - 0.5*t - 0.5*gammaln(al+1.0))
    if nmax > 1: psi[1] = (1.0+al-t)*psi[0]/np.sqrt(1.0+al)
    for n in range(1, nmax-1):
        psi[n+1] = ((2*n+1+al-t)*psi[n] - np.sqrt(n*(n+al))*psi[n-1])/np.sqrt((n+1)*(n+1+al))
    g = al/(2.0*t) - 0.5
    for n in range(nmax):
        prev = psi[n-1] if n >= 1 else 0.0
        dpsi[n] = g*psi[n] + (n*psi[n] - np.sqrt(n*(n+al))*prev)/t
        Q = dpsi[n] - g*psi[n]
        d2psi[n] = -al/(2.0*t*t)*psi[n] + g*dpsi[n] + g*Q + (-(al+1.0-t)*Q - n*psi[n])/t
    return psi, dpsi, d2psi
def _basis(a, N, m, r):
    t = np.asarray(r, dtype=float)/a; am = abs(int(m))
    psi, dpsi, d2psi = _lagfun(N, 2*am+1, t)
    s = t**-0.5
    return psi*s/a, (dpsi*s - 0.5*psi*t**-1.5)/a**2, (d2psi*s - dpsi*t**-1.5 + 0.75*psi*t**-2.5)/a**3
def _quad(a, Nq):
    n = np.arange(Nq)
    J = np.diag(2.0*n+1.0) + np.diag(np.arange(1, Nq, dtype=float), 1) + np.diag(np.arange(1, Nq, dtype=float), -1)
    t = np.linalg.eigvalsh(J)
    ssum = np.sum(_lagfun(Nq, 0.0, t)[0]**2, axis=0)
    wt = np.where(ssum > 0.0, 1.0/np.where(ssum > 0.0, ssum, 1.0), 0.0)
    return a*t, a*a*t*wt
def _hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, Kx, Ky, a, N, M, Nq, factorized=False):
    mux, muy, Mx, My, alpha, cX, cY = _params(mex, mey, mhx, mhy)
    if factorized:
        alpha = 0.0; cX = 1.0/muy; cY = 1.0/mux
    r, W = _quad(a, Nq); V = _keldysh(r, r0, kappa)
    ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r) for m in ms}
    kiso = 0.25*(1.0/mux + 1.0/muy); kan = 0.25*(1.0/mux - 1.0/muy)
    zi = 0.5*(1.0/mux + 1.0/muy); za = 0.5*(1.0/mux - 1.0/muy)
    cd = 0.5*(cX + cY)*B*B/8.0; cc = 0.5*(cX - cY)*B*B/8.0
    H = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f, df, d2f = F[m]; sl = slice(i*N, (i+1)*N)
        lap = d2f + df/r - (m*m)*f/r**2
        H[sl, sl] += -kiso*(f*W) @ lap.T + (f*W) @ (V*f).T + (f*W) @ ((cd*r*r)*f).T + (0.5*alpha*B*zi*m)*(f*W) @ f.T
        if i-2 >= 0:
            g = F[m-2][0]; sg = slice((i-2)*N, (i-1)*N)
            opz = 0.25*(d2f + (2*m-1)*df/r + m*(m-2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opz.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(r*df + m*f)).T
        if i+2 < nm:
            g = F[m+2][0]; sg = slice((i+2)*N, (i+3)*N)
            opzb = 0.25*(d2f - (2*m+1)*df/r + m*(m+2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opzb.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(m*f - r*df)).T
        if Kx != 0.0 or Ky != 0.0:
            for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
                if 0 <= i+dm < nm:
                    g = F[m+dm][0]; sg = slice((i+dm)*N, (i+dm+1)*N)
                    H[sg, sl] += (B*(Ky/My*cxk - Kx/Mx*cyk))*(g*W) @ (r*f).T
    return H
def _dipoles(a, N, M, Nq):
    r, W = _quad(a, Nq); ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r)[0] for m in ms}
    X = np.zeros((D, D), dtype=complex); Y = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f = F[m]; sl = slice(i*N, (i+1)*N)
        for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
            if 0 <= i+dm < nm:
                g = F[m+dm]; sg = slice((i+dm)*N, (i+dm+1)*N); blk = (g*W) @ (r*f).T
                X[sg, sl] += cxk*blk; Y[sg, sl] += cyk*blk
    return X, Y
def _spectrum(H, k):
    w = eigh(H, eigvals_only=True)
    return w[:k]
def _polarizability(H, X, Y):
    w, v = eigh(H)
    dx = v.conj().T @ (X @ v[:, 0]); dy = v.conj().T @ (Y @ v[:, 0])
    axx = 2.0*np.sum(np.abs(dx[1:])**2/(w[1:]-w[0])); ayy = 2.0*np.sum(np.abs(dy[1:])**2/(w[1:]-w[0]))
    return float(w[0]), float(axx), float(ayy)

def mx_keldysh(r, r0_au, kappa):
    if not np.isfinite(float(r0_au)) or float(r0_au) <= 0.0: raise ValueError("r0_au must be positive")
    r = _vec(r, "r"); r0 = _pos(r0_au, "r0_au"); kappa = _pos(kappa, "kappa")
    return _keldysh(r, r0, kappa).astype(np.float64)

import numpy as np
from scipy import special
from scipy.linalg import eigh
from scipy.special import gammaln

A0_NM = 0.052917721
EH_MEV = 27211.386
B0_T = 235051.757

def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError("bad " + name)
    return v
def _posint(x, name):
    v = _fin(x, name)
    if v <= 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _nnint(x, name):
    v = _fin(x, name)
    if v < 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _vec(x, name, positive=True):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)): raise ValueError("bad " + name)
    if positive and np.any(arr <= 0.0): raise ValueError("bad " + name)
    return arr
def _params(mex, mey, mhx, mhy):
    mux = mex*mhx/(mex+mhx); muy = mey*mhy/(mey+mhy); Mx = mex+mhx; My = mey+mhy
    rx = mex/mhx; ry = mey/mhy
    alpha = (1.0-rx*ry)/((1.0+rx)*(1.0+ry)); bx = 4.0*rx/(1.0+rx)**2; by = 4.0*ry/(1.0+ry)**2
    return mux, muy, Mx, My, alpha, (by+alpha*alpha)/muy, (bx+alpha*alpha)/mux
def _keldysh(r, r0, kappa):
    z = kappa*np.asarray(r, dtype=float)/r0
    return -(np.pi/(2.0*r0))*(special.struve(0, z) - special.y0(z))
def _lagfun(nmax, al, t):
    t = np.asarray(t, dtype=float)
    psi = np.zeros((nmax, t.size)); dpsi = np.zeros_like(psi); d2psi = np.zeros_like(psi)
    psi[0] = np.exp(0.5*al*np.log(t) - 0.5*t - 0.5*gammaln(al+1.0))
    if nmax > 1: psi[1] = (1.0+al-t)*psi[0]/np.sqrt(1.0+al)
    for n in range(1, nmax-1):
        psi[n+1] = ((2*n+1+al-t)*psi[n] - np.sqrt(n*(n+al))*psi[n-1])/np.sqrt((n+1)*(n+1+al))
    g = al/(2.0*t) - 0.5
    for n in range(nmax):
        prev = psi[n-1] if n >= 1 else 0.0
        dpsi[n] = g*psi[n] + (n*psi[n] - np.sqrt(n*(n+al))*prev)/t
        Q = dpsi[n] - g*psi[n]
        d2psi[n] = -al/(2.0*t*t)*psi[n] + g*dpsi[n] + g*Q + (-(al+1.0-t)*Q - n*psi[n])/t
    return psi, dpsi, d2psi
def _basis(a, N, m, r):
    t = np.asarray(r, dtype=float)/a; am = abs(int(m))
    psi, dpsi, d2psi = _lagfun(N, 2*am+1, t)
    s = t**-0.5
    return psi*s/a, (dpsi*s - 0.5*psi*t**-1.5)/a**2, (d2psi*s - dpsi*t**-1.5 + 0.75*psi*t**-2.5)/a**3
def _quad(a, Nq):
    n = np.arange(Nq)
    J = np.diag(2.0*n+1.0) + np.diag(np.arange(1, Nq, dtype=float), 1) + np.diag(np.arange(1, Nq, dtype=float), -1)
    t = np.linalg.eigvalsh(J)
    ssum = np.sum(_lagfun(Nq, 0.0, t)[0]**2, axis=0)
    wt = np.where(ssum > 0.0, 1.0/np.where(ssum > 0.0, ssum, 1.0), 0.0)
    return a*t, a*a*t*wt
def _hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, Kx, Ky, a, N, M, Nq, factorized=False):
    mux, muy, Mx, My, alpha, cX, cY = _params(mex, mey, mhx, mhy)
    if factorized:
        alpha = 0.0; cX = 1.0/muy; cY = 1.0/mux
    r, W = _quad(a, Nq); V = _keldysh(r, r0, kappa)
    ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r) for m in ms}
    kiso = 0.25*(1.0/mux + 1.0/muy); kan = 0.25*(1.0/mux - 1.0/muy)
    zi = 0.5*(1.0/mux + 1.0/muy); za = 0.5*(1.0/mux - 1.0/muy)
    cd = 0.5*(cX + cY)*B*B/8.0; cc = 0.5*(cX - cY)*B*B/8.0
    H = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f, df, d2f = F[m]; sl = slice(i*N, (i+1)*N)
        lap = d2f + df/r - (m*m)*f/r**2
        H[sl, sl] += -kiso*(f*W) @ lap.T + (f*W) @ (V*f).T + (f*W) @ ((cd*r*r)*f).T + (0.5*alpha*B*zi*m)*(f*W) @ f.T
        if i-2 >= 0:
            g = F[m-2][0]; sg = slice((i-2)*N, (i-1)*N)
            opz = 0.25*(d2f + (2*m-1)*df/r + m*(m-2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opz.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(r*df + m*f)).T
        if i+2 < nm:
            g = F[m+2][0]; sg = slice((i+2)*N, (i+3)*N)
            opzb = 0.25*(d2f - (2*m+1)*df/r + m*(m+2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opzb.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(m*f - r*df)).T
        if Kx != 0.0 or Ky != 0.0:
            for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
                if 0 <= i+dm < nm:
                    g = F[m+dm][0]; sg = slice((i+dm)*N, (i+dm+1)*N)
                    H[sg, sl] += (B*(Ky/My*cxk - Kx/Mx*cyk))*(g*W) @ (r*f).T
    return H
def _dipoles(a, N, M, Nq):
    r, W = _quad(a, Nq); ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r)[0] for m in ms}
    X = np.zeros((D, D), dtype=complex); Y = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f = F[m]; sl = slice(i*N, (i+1)*N)
        for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
            if 0 <= i+dm < nm:
                g = F[m+dm]; sg = slice((i+dm)*N, (i+dm+1)*N); blk = (g*W) @ (r*f).T
                X[sg, sl] += cxk*blk; Y[sg, sl] += cyk*blk
    return X, Y
def _spectrum(H, k):
    w = eigh(H, eigvals_only=True)
    return w[:k]
def _polarizability(H, X, Y):
    w, v = eigh(H)
    dx = v.conj().T @ (X @ v[:, 0]); dy = v.conj().T @ (Y @ v[:, 0])
    axx = 2.0*np.sum(np.abs(dx[1:])**2/(w[1:]-w[0])); ayy = 2.0*np.sum(np.abs(dy[1:])**2/(w[1:]-w[0]))
    return float(w[0]), float(axx), float(ayy)

def mx_radial_basis(a, N, m, r):
    if not np.isfinite(float(a)) or float(a) <= 0.0: raise ValueError("a must be positive")
    a = _pos(a, "a"); N = _posint(N, "N"); mf = _fin(m, "m")
    if abs(mf - round(mf)) > 0: raise ValueError("m must be an integer")
    m = int(round(mf)); r = _vec(r, "r")
    f, df, d2f = _basis(a, N, m, r)
    return np.stack([f, df, d2f]).astype(np.float64)

import numpy as np
from scipy import special
from scipy.linalg import eigh
from scipy.special import gammaln

A0_NM = 0.052917721
EH_MEV = 27211.386
B0_T = 235051.757

def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError("bad " + name)
    return v
def _posint(x, name):
    v = _fin(x, name)
    if v <= 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _nnint(x, name):
    v = _fin(x, name)
    if v < 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _vec(x, name, positive=True):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)): raise ValueError("bad " + name)
    if positive and np.any(arr <= 0.0): raise ValueError("bad " + name)
    return arr
def _params(mex, mey, mhx, mhy):
    mux = mex*mhx/(mex+mhx); muy = mey*mhy/(mey+mhy); Mx = mex+mhx; My = mey+mhy
    rx = mex/mhx; ry = mey/mhy
    alpha = (1.0-rx*ry)/((1.0+rx)*(1.0+ry)); bx = 4.0*rx/(1.0+rx)**2; by = 4.0*ry/(1.0+ry)**2
    return mux, muy, Mx, My, alpha, (by+alpha*alpha)/muy, (bx+alpha*alpha)/mux
def _keldysh(r, r0, kappa):
    z = kappa*np.asarray(r, dtype=float)/r0
    return -(np.pi/(2.0*r0))*(special.struve(0, z) - special.y0(z))
def _lagfun(nmax, al, t):
    t = np.asarray(t, dtype=float)
    psi = np.zeros((nmax, t.size)); dpsi = np.zeros_like(psi); d2psi = np.zeros_like(psi)
    psi[0] = np.exp(0.5*al*np.log(t) - 0.5*t - 0.5*gammaln(al+1.0))
    if nmax > 1: psi[1] = (1.0+al-t)*psi[0]/np.sqrt(1.0+al)
    for n in range(1, nmax-1):
        psi[n+1] = ((2*n+1+al-t)*psi[n] - np.sqrt(n*(n+al))*psi[n-1])/np.sqrt((n+1)*(n+1+al))
    g = al/(2.0*t) - 0.5
    for n in range(nmax):
        prev = psi[n-1] if n >= 1 else 0.0
        dpsi[n] = g*psi[n] + (n*psi[n] - np.sqrt(n*(n+al))*prev)/t
        Q = dpsi[n] - g*psi[n]
        d2psi[n] = -al/(2.0*t*t)*psi[n] + g*dpsi[n] + g*Q + (-(al+1.0-t)*Q - n*psi[n])/t
    return psi, dpsi, d2psi
def _basis(a, N, m, r):
    t = np.asarray(r, dtype=float)/a; am = abs(int(m))
    psi, dpsi, d2psi = _lagfun(N, 2*am+1, t)
    s = t**-0.5
    return psi*s/a, (dpsi*s - 0.5*psi*t**-1.5)/a**2, (d2psi*s - dpsi*t**-1.5 + 0.75*psi*t**-2.5)/a**3
def _quad(a, Nq):
    n = np.arange(Nq)
    J = np.diag(2.0*n+1.0) + np.diag(np.arange(1, Nq, dtype=float), 1) + np.diag(np.arange(1, Nq, dtype=float), -1)
    t = np.linalg.eigvalsh(J)
    ssum = np.sum(_lagfun(Nq, 0.0, t)[0]**2, axis=0)
    wt = np.where(ssum > 0.0, 1.0/np.where(ssum > 0.0, ssum, 1.0), 0.0)
    return a*t, a*a*t*wt
def _hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, Kx, Ky, a, N, M, Nq, factorized=False):
    mux, muy, Mx, My, alpha, cX, cY = _params(mex, mey, mhx, mhy)
    if factorized:
        alpha = 0.0; cX = 1.0/muy; cY = 1.0/mux
    r, W = _quad(a, Nq); V = _keldysh(r, r0, kappa)
    ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r) for m in ms}
    kiso = 0.25*(1.0/mux + 1.0/muy); kan = 0.25*(1.0/mux - 1.0/muy)
    zi = 0.5*(1.0/mux + 1.0/muy); za = 0.5*(1.0/mux - 1.0/muy)
    cd = 0.5*(cX + cY)*B*B/8.0; cc = 0.5*(cX - cY)*B*B/8.0
    H = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f, df, d2f = F[m]; sl = slice(i*N, (i+1)*N)
        lap = d2f + df/r - (m*m)*f/r**2
        H[sl, sl] += -kiso*(f*W) @ lap.T + (f*W) @ (V*f).T + (f*W) @ ((cd*r*r)*f).T + (0.5*alpha*B*zi*m)*(f*W) @ f.T
        if i-2 >= 0:
            g = F[m-2][0]; sg = slice((i-2)*N, (i-1)*N)
            opz = 0.25*(d2f + (2*m-1)*df/r + m*(m-2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opz.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(r*df + m*f)).T
        if i+2 < nm:
            g = F[m+2][0]; sg = slice((i+2)*N, (i+3)*N)
            opzb = 0.25*(d2f - (2*m+1)*df/r + m*(m+2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opzb.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(m*f - r*df)).T
        if Kx != 0.0 or Ky != 0.0:
            for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
                if 0 <= i+dm < nm:
                    g = F[m+dm][0]; sg = slice((i+dm)*N, (i+dm+1)*N)
                    H[sg, sl] += (B*(Ky/My*cxk - Kx/Mx*cyk))*(g*W) @ (r*f).T
    return H
def _dipoles(a, N, M, Nq):
    r, W = _quad(a, Nq); ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r)[0] for m in ms}
    X = np.zeros((D, D), dtype=complex); Y = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f = F[m]; sl = slice(i*N, (i+1)*N)
        for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
            if 0 <= i+dm < nm:
                g = F[m+dm]; sg = slice((i+dm)*N, (i+dm+1)*N); blk = (g*W) @ (r*f).T
                X[sg, sl] += cxk*blk; Y[sg, sl] += cyk*blk
    return X, Y
def _spectrum(H, k):
    w = eigh(H, eigvals_only=True)
    return w[:k]
def _polarizability(H, X, Y):
    w, v = eigh(H)
    dx = v.conj().T @ (X @ v[:, 0]); dy = v.conj().T @ (Y @ v[:, 0])
    axx = 2.0*np.sum(np.abs(dx[1:])**2/(w[1:]-w[0])); ayy = 2.0*np.sum(np.abs(dy[1:])**2/(w[1:]-w[0]))
    return float(w[0]), float(axx), float(ayy)

def mx_quadrature(a, Nq):
    if not np.isfinite(float(a)) or float(a) <= 0.0: raise ValueError("a must be positive")
    a = _pos(a, "a"); Nq = _posint(Nq, "Nq")
    r, W = _quad(a, Nq)
    return np.stack([r, W]).astype(np.float64)

import numpy as np
from scipy import special
from scipy.linalg import eigh
from scipy.special import gammaln

A0_NM = 0.052917721
EH_MEV = 27211.386
B0_T = 235051.757

def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError("bad " + name)
    return v
def _posint(x, name):
    v = _fin(x, name)
    if v <= 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _nnint(x, name):
    v = _fin(x, name)
    if v < 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _vec(x, name, positive=True):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)): raise ValueError("bad " + name)
    if positive and np.any(arr <= 0.0): raise ValueError("bad " + name)
    return arr
def _params(mex, mey, mhx, mhy):
    mux = mex*mhx/(mex+mhx); muy = mey*mhy/(mey+mhy); Mx = mex+mhx; My = mey+mhy
    rx = mex/mhx; ry = mey/mhy
    alpha = (1.0-rx*ry)/((1.0+rx)*(1.0+ry)); bx = 4.0*rx/(1.0+rx)**2; by = 4.0*ry/(1.0+ry)**2
    return mux, muy, Mx, My, alpha, (by+alpha*alpha)/muy, (bx+alpha*alpha)/mux
def _keldysh(r, r0, kappa):
    z = kappa*np.asarray(r, dtype=float)/r0
    return -(np.pi/(2.0*r0))*(special.struve(0, z) - special.y0(z))
def _lagfun(nmax, al, t):
    t = np.asarray(t, dtype=float)
    psi = np.zeros((nmax, t.size)); dpsi = np.zeros_like(psi); d2psi = np.zeros_like(psi)
    psi[0] = np.exp(0.5*al*np.log(t) - 0.5*t - 0.5*gammaln(al+1.0))
    if nmax > 1: psi[1] = (1.0+al-t)*psi[0]/np.sqrt(1.0+al)
    for n in range(1, nmax-1):
        psi[n+1] = ((2*n+1+al-t)*psi[n] - np.sqrt(n*(n+al))*psi[n-1])/np.sqrt((n+1)*(n+1+al))
    g = al/(2.0*t) - 0.5
    for n in range(nmax):
        prev = psi[n-1] if n >= 1 else 0.0
        dpsi[n] = g*psi[n] + (n*psi[n] - np.sqrt(n*(n+al))*prev)/t
        Q = dpsi[n] - g*psi[n]
        d2psi[n] = -al/(2.0*t*t)*psi[n] + g*dpsi[n] + g*Q + (-(al+1.0-t)*Q - n*psi[n])/t
    return psi, dpsi, d2psi
def _basis(a, N, m, r):
    t = np.asarray(r, dtype=float)/a; am = abs(int(m))
    psi, dpsi, d2psi = _lagfun(N, 2*am+1, t)
    s = t**-0.5
    return psi*s/a, (dpsi*s - 0.5*psi*t**-1.5)/a**2, (d2psi*s - dpsi*t**-1.5 + 0.75*psi*t**-2.5)/a**3
def _quad(a, Nq):
    n = np.arange(Nq)
    J = np.diag(2.0*n+1.0) + np.diag(np.arange(1, Nq, dtype=float), 1) + np.diag(np.arange(1, Nq, dtype=float), -1)
    t = np.linalg.eigvalsh(J)
    ssum = np.sum(_lagfun(Nq, 0.0, t)[0]**2, axis=0)
    wt = np.where(ssum > 0.0, 1.0/np.where(ssum > 0.0, ssum, 1.0), 0.0)
    return a*t, a*a*t*wt
def _hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, Kx, Ky, a, N, M, Nq, factorized=False):
    mux, muy, Mx, My, alpha, cX, cY = _params(mex, mey, mhx, mhy)
    if factorized:
        alpha = 0.0; cX = 1.0/muy; cY = 1.0/mux
    r, W = _quad(a, Nq); V = _keldysh(r, r0, kappa)
    ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r) for m in ms}
    kiso = 0.25*(1.0/mux + 1.0/muy); kan = 0.25*(1.0/mux - 1.0/muy)
    zi = 0.5*(1.0/mux + 1.0/muy); za = 0.5*(1.0/mux - 1.0/muy)
    cd = 0.5*(cX + cY)*B*B/8.0; cc = 0.5*(cX - cY)*B*B/8.0
    H = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f, df, d2f = F[m]; sl = slice(i*N, (i+1)*N)
        lap = d2f + df/r - (m*m)*f/r**2
        H[sl, sl] += -kiso*(f*W) @ lap.T + (f*W) @ (V*f).T + (f*W) @ ((cd*r*r)*f).T + (0.5*alpha*B*zi*m)*(f*W) @ f.T
        if i-2 >= 0:
            g = F[m-2][0]; sg = slice((i-2)*N, (i-1)*N)
            opz = 0.25*(d2f + (2*m-1)*df/r + m*(m-2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opz.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(r*df + m*f)).T
        if i+2 < nm:
            g = F[m+2][0]; sg = slice((i+2)*N, (i+3)*N)
            opzb = 0.25*(d2f - (2*m+1)*df/r + m*(m+2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opzb.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(m*f - r*df)).T
        if Kx != 0.0 or Ky != 0.0:
            for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
                if 0 <= i+dm < nm:
                    g = F[m+dm][0]; sg = slice((i+dm)*N, (i+dm+1)*N)
                    H[sg, sl] += (B*(Ky/My*cxk - Kx/Mx*cyk))*(g*W) @ (r*f).T
    return H
def _dipoles(a, N, M, Nq):
    r, W = _quad(a, Nq); ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r)[0] for m in ms}
    X = np.zeros((D, D), dtype=complex); Y = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f = F[m]; sl = slice(i*N, (i+1)*N)
        for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
            if 0 <= i+dm < nm:
                g = F[m+dm]; sg = slice((i+dm)*N, (i+dm+1)*N); blk = (g*W) @ (r*f).T
                X[sg, sl] += cxk*blk; Y[sg, sl] += cyk*blk
    return X, Y
def _spectrum(H, k):
    w = eigh(H, eigvals_only=True)
    return w[:k]
def _polarizability(H, X, Y):
    w, v = eigh(H)
    dx = v.conj().T @ (X @ v[:, 0]); dy = v.conj().T @ (Y @ v[:, 0])
    axx = 2.0*np.sum(np.abs(dx[1:])**2/(w[1:]-w[0])); ayy = 2.0*np.sum(np.abs(dy[1:])**2/(w[1:]-w[0]))
    return float(w[0]), float(axx), float(ayy)

def mx_hamiltonian(mex, mey, mhx, mhy, r0_au, kappa, B_au, Kx, Ky, a, N, M, Nq, factorized):
    if not np.isfinite(float(B_au)) or float(B_au) < 0.0: raise ValueError("B_au must be non-negative")
    mex = _pos(mex, "mex"); mey = _pos(mey, "mey"); mhx = _pos(mhx, "mhx"); mhy = _pos(mhy, "mhy")
    r0 = _pos(r0_au, "r0_au"); kappa = _pos(kappa, "kappa"); B = _nonneg(B_au, "B_au"); Kx = _fin(Kx, "Kx"); Ky = _fin(Ky, "Ky")
    a = _pos(a, "a"); N = _posint(N, "N"); M = _nnint(M, "M"); Nq = _posint(Nq, "Nq"); fz = _fin(factorized, "factorized")
    if fz not in (0.0, 1.0): raise ValueError("factorized must be 0 or 1")
    H = _hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, Kx, Ky, a, N, M, Nq, factorized=(fz == 1.0))
    return np.stack([H.real, H.imag]).astype(np.float64)

import numpy as np
from scipy import special
from scipy.linalg import eigh
from scipy.special import gammaln

A0_NM = 0.052917721
EH_MEV = 27211.386
B0_T = 235051.757

def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError("bad " + name)
    return v
def _posint(x, name):
    v = _fin(x, name)
    if v <= 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _nnint(x, name):
    v = _fin(x, name)
    if v < 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _vec(x, name, positive=True):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)): raise ValueError("bad " + name)
    if positive and np.any(arr <= 0.0): raise ValueError("bad " + name)
    return arr
def _params(mex, mey, mhx, mhy):
    mux = mex*mhx/(mex+mhx); muy = mey*mhy/(mey+mhy); Mx = mex+mhx; My = mey+mhy
    rx = mex/mhx; ry = mey/mhy
    alpha = (1.0-rx*ry)/((1.0+rx)*(1.0+ry)); bx = 4.0*rx/(1.0+rx)**2; by = 4.0*ry/(1.0+ry)**2
    return mux, muy, Mx, My, alpha, (by+alpha*alpha)/muy, (bx+alpha*alpha)/mux
def _keldysh(r, r0, kappa):
    z = kappa*np.asarray(r, dtype=float)/r0
    return -(np.pi/(2.0*r0))*(special.struve(0, z) - special.y0(z))
def _lagfun(nmax, al, t):
    t = np.asarray(t, dtype=float)
    psi = np.zeros((nmax, t.size)); dpsi = np.zeros_like(psi); d2psi = np.zeros_like(psi)
    psi[0] = np.exp(0.5*al*np.log(t) - 0.5*t - 0.5*gammaln(al+1.0))
    if nmax > 1: psi[1] = (1.0+al-t)*psi[0]/np.sqrt(1.0+al)
    for n in range(1, nmax-1):
        psi[n+1] = ((2*n+1+al-t)*psi[n] - np.sqrt(n*(n+al))*psi[n-1])/np.sqrt((n+1)*(n+1+al))
    g = al/(2.0*t) - 0.5
    for n in range(nmax):
        prev = psi[n-1] if n >= 1 else 0.0
        dpsi[n] = g*psi[n] + (n*psi[n] - np.sqrt(n*(n+al))*prev)/t
        Q = dpsi[n] - g*psi[n]
        d2psi[n] = -al/(2.0*t*t)*psi[n] + g*dpsi[n] + g*Q + (-(al+1.0-t)*Q - n*psi[n])/t
    return psi, dpsi, d2psi
def _basis(a, N, m, r):
    t = np.asarray(r, dtype=float)/a; am = abs(int(m))
    psi, dpsi, d2psi = _lagfun(N, 2*am+1, t)
    s = t**-0.5
    return psi*s/a, (dpsi*s - 0.5*psi*t**-1.5)/a**2, (d2psi*s - dpsi*t**-1.5 + 0.75*psi*t**-2.5)/a**3
def _quad(a, Nq):
    n = np.arange(Nq)
    J = np.diag(2.0*n+1.0) + np.diag(np.arange(1, Nq, dtype=float), 1) + np.diag(np.arange(1, Nq, dtype=float), -1)
    t = np.linalg.eigvalsh(J)
    ssum = np.sum(_lagfun(Nq, 0.0, t)[0]**2, axis=0)
    wt = np.where(ssum > 0.0, 1.0/np.where(ssum > 0.0, ssum, 1.0), 0.0)
    return a*t, a*a*t*wt
def _hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, Kx, Ky, a, N, M, Nq, factorized=False):
    mux, muy, Mx, My, alpha, cX, cY = _params(mex, mey, mhx, mhy)
    if factorized:
        alpha = 0.0; cX = 1.0/muy; cY = 1.0/mux
    r, W = _quad(a, Nq); V = _keldysh(r, r0, kappa)
    ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r) for m in ms}
    kiso = 0.25*(1.0/mux + 1.0/muy); kan = 0.25*(1.0/mux - 1.0/muy)
    zi = 0.5*(1.0/mux + 1.0/muy); za = 0.5*(1.0/mux - 1.0/muy)
    cd = 0.5*(cX + cY)*B*B/8.0; cc = 0.5*(cX - cY)*B*B/8.0
    H = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f, df, d2f = F[m]; sl = slice(i*N, (i+1)*N)
        lap = d2f + df/r - (m*m)*f/r**2
        H[sl, sl] += -kiso*(f*W) @ lap.T + (f*W) @ (V*f).T + (f*W) @ ((cd*r*r)*f).T + (0.5*alpha*B*zi*m)*(f*W) @ f.T
        if i-2 >= 0:
            g = F[m-2][0]; sg = slice((i-2)*N, (i-1)*N)
            opz = 0.25*(d2f + (2*m-1)*df/r + m*(m-2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opz.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(r*df + m*f)).T
        if i+2 < nm:
            g = F[m+2][0]; sg = slice((i+2)*N, (i+3)*N)
            opzb = 0.25*(d2f - (2*m+1)*df/r + m*(m+2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opzb.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(m*f - r*df)).T
        if Kx != 0.0 or Ky != 0.0:
            for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
                if 0 <= i+dm < nm:
                    g = F[m+dm][0]; sg = slice((i+dm)*N, (i+dm+1)*N)
                    H[sg, sl] += (B*(Ky/My*cxk - Kx/Mx*cyk))*(g*W) @ (r*f).T
    return H
def _dipoles(a, N, M, Nq):
    r, W = _quad(a, Nq); ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r)[0] for m in ms}
    X = np.zeros((D, D), dtype=complex); Y = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f = F[m]; sl = slice(i*N, (i+1)*N)
        for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
            if 0 <= i+dm < nm:
                g = F[m+dm]; sg = slice((i+dm)*N, (i+dm+1)*N); blk = (g*W) @ (r*f).T
                X[sg, sl] += cxk*blk; Y[sg, sl] += cyk*blk
    return X, Y
def _spectrum(H, k):
    w = eigh(H, eigvals_only=True)
    return w[:k]
def _polarizability(H, X, Y):
    w, v = eigh(H)
    dx = v.conj().T @ (X @ v[:, 0]); dy = v.conj().T @ (Y @ v[:, 0])
    axx = 2.0*np.sum(np.abs(dx[1:])**2/(w[1:]-w[0])); ayy = 2.0*np.sum(np.abs(dy[1:])**2/(w[1:]-w[0]))
    return float(w[0]), float(axx), float(ayy)

def mx_spectrum(mex, mey, mhx, mhy, r0_au, kappa, B_au, Kx, Ky, a, N, M, Nq, factorized, k):
    if not np.isfinite(float(k)) or float(k) <= 0 or abs(float(k) - round(float(k))) > 0: raise ValueError("k must be a positive integer")
    HH = mx_hamiltonian(mex, mey, mhx, mhy, r0_au, kappa, B_au, Kx, Ky, a, N, M, Nq, factorized)
    k = _posint(k, "k")
    if k > HH.shape[1]: raise ValueError("k exceeds the basis size")
    H = HH[0] + 1j*HH[1]
    return (_spectrum(H, k)*27211.386).astype(np.float64)

import numpy as np
from scipy import special
from scipy.linalg import eigh
from scipy.special import gammaln

A0_NM = 0.052917721
EH_MEV = 27211.386
B0_T = 235051.757

def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError("bad " + name)
    return v
def _posint(x, name):
    v = _fin(x, name)
    if v <= 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _nnint(x, name):
    v = _fin(x, name)
    if v < 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _vec(x, name, positive=True):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)): raise ValueError("bad " + name)
    if positive and np.any(arr <= 0.0): raise ValueError("bad " + name)
    return arr
def _params(mex, mey, mhx, mhy):
    mux = mex*mhx/(mex+mhx); muy = mey*mhy/(mey+mhy); Mx = mex+mhx; My = mey+mhy
    rx = mex/mhx; ry = mey/mhy
    alpha = (1.0-rx*ry)/((1.0+rx)*(1.0+ry)); bx = 4.0*rx/(1.0+rx)**2; by = 4.0*ry/(1.0+ry)**2
    return mux, muy, Mx, My, alpha, (by+alpha*alpha)/muy, (bx+alpha*alpha)/mux
def _keldysh(r, r0, kappa):
    z = kappa*np.asarray(r, dtype=float)/r0
    return -(np.pi/(2.0*r0))*(special.struve(0, z) - special.y0(z))
def _lagfun(nmax, al, t):
    t = np.asarray(t, dtype=float)
    psi = np.zeros((nmax, t.size)); dpsi = np.zeros_like(psi); d2psi = np.zeros_like(psi)
    psi[0] = np.exp(0.5*al*np.log(t) - 0.5*t - 0.5*gammaln(al+1.0))
    if nmax > 1: psi[1] = (1.0+al-t)*psi[0]/np.sqrt(1.0+al)
    for n in range(1, nmax-1):
        psi[n+1] = ((2*n+1+al-t)*psi[n] - np.sqrt(n*(n+al))*psi[n-1])/np.sqrt((n+1)*(n+1+al))
    g = al/(2.0*t) - 0.5
    for n in range(nmax):
        prev = psi[n-1] if n >= 1 else 0.0
        dpsi[n] = g*psi[n] + (n*psi[n] - np.sqrt(n*(n+al))*prev)/t
        Q = dpsi[n] - g*psi[n]
        d2psi[n] = -al/(2.0*t*t)*psi[n] + g*dpsi[n] + g*Q + (-(al+1.0-t)*Q - n*psi[n])/t
    return psi, dpsi, d2psi
def _basis(a, N, m, r):
    t = np.asarray(r, dtype=float)/a; am = abs(int(m))
    psi, dpsi, d2psi = _lagfun(N, 2*am+1, t)
    s = t**-0.5
    return psi*s/a, (dpsi*s - 0.5*psi*t**-1.5)/a**2, (d2psi*s - dpsi*t**-1.5 + 0.75*psi*t**-2.5)/a**3
def _quad(a, Nq):
    n = np.arange(Nq)
    J = np.diag(2.0*n+1.0) + np.diag(np.arange(1, Nq, dtype=float), 1) + np.diag(np.arange(1, Nq, dtype=float), -1)
    t = np.linalg.eigvalsh(J)
    ssum = np.sum(_lagfun(Nq, 0.0, t)[0]**2, axis=0)
    wt = np.where(ssum > 0.0, 1.0/np.where(ssum > 0.0, ssum, 1.0), 0.0)
    return a*t, a*a*t*wt
def _hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, Kx, Ky, a, N, M, Nq, factorized=False):
    mux, muy, Mx, My, alpha, cX, cY = _params(mex, mey, mhx, mhy)
    if factorized:
        alpha = 0.0; cX = 1.0/muy; cY = 1.0/mux
    r, W = _quad(a, Nq); V = _keldysh(r, r0, kappa)
    ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r) for m in ms}
    kiso = 0.25*(1.0/mux + 1.0/muy); kan = 0.25*(1.0/mux - 1.0/muy)
    zi = 0.5*(1.0/mux + 1.0/muy); za = 0.5*(1.0/mux - 1.0/muy)
    cd = 0.5*(cX + cY)*B*B/8.0; cc = 0.5*(cX - cY)*B*B/8.0
    H = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f, df, d2f = F[m]; sl = slice(i*N, (i+1)*N)
        lap = d2f + df/r - (m*m)*f/r**2
        H[sl, sl] += -kiso*(f*W) @ lap.T + (f*W) @ (V*f).T + (f*W) @ ((cd*r*r)*f).T + (0.5*alpha*B*zi*m)*(f*W) @ f.T
        if i-2 >= 0:
            g = F[m-2][0]; sg = slice((i-2)*N, (i-1)*N)
            opz = 0.25*(d2f + (2*m-1)*df/r + m*(m-2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opz.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(r*df + m*f)).T
        if i+2 < nm:
            g = F[m+2][0]; sg = slice((i+2)*N, (i+3)*N)
            opzb = 0.25*(d2f - (2*m+1)*df/r + m*(m+2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opzb.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(m*f - r*df)).T
        if Kx != 0.0 or Ky != 0.0:
            for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
                if 0 <= i+dm < nm:
                    g = F[m+dm][0]; sg = slice((i+dm)*N, (i+dm+1)*N)
                    H[sg, sl] += (B*(Ky/My*cxk - Kx/Mx*cyk))*(g*W) @ (r*f).T
    return H
def _dipoles(a, N, M, Nq):
    r, W = _quad(a, Nq); ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r)[0] for m in ms}
    X = np.zeros((D, D), dtype=complex); Y = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f = F[m]; sl = slice(i*N, (i+1)*N)
        for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
            if 0 <= i+dm < nm:
                g = F[m+dm]; sg = slice((i+dm)*N, (i+dm+1)*N); blk = (g*W) @ (r*f).T
                X[sg, sl] += cxk*blk; Y[sg, sl] += cyk*blk
    return X, Y
def _spectrum(H, k):
    w = eigh(H, eigvals_only=True)
    return w[:k]
def _polarizability(H, X, Y):
    w, v = eigh(H)
    dx = v.conj().T @ (X @ v[:, 0]); dy = v.conj().T @ (Y @ v[:, 0])
    axx = 2.0*np.sum(np.abs(dx[1:])**2/(w[1:]-w[0])); ayy = 2.0*np.sum(np.abs(dy[1:])**2/(w[1:]-w[0]))
    return float(w[0]), float(axx), float(ayy)

def mx_magnetopolarizability(mex, mey, mhx, mhy, r0_au, kappa, B_au, a, N, M, Nq, factorized):
    if not np.isfinite(float(kappa)) or float(kappa) <= 0.0: raise ValueError("kappa must be positive")
    HH = mx_hamiltonian(mex, mey, mhx, mhy, r0_au, kappa, B_au, 0.0, 0.0, a, N, M, Nq, factorized)
    if HH.shape[1] < 2: raise ValueError("the basis holds a single state")
    H = HH[0] + 1j*HH[1]
    X, Y = _dipoles(float(a), int(round(float(N))), int(round(float(M))), int(round(float(Nq))))
    E0, axx, ayy = _polarizability(H, X, Y)
    return np.array([E0*27211.386, axx, ayy], dtype=np.float64)

import numpy as np
from scipy import special
from scipy.linalg import eigh
from scipy.special import gammaln

A0_NM = 0.052917721
EH_MEV = 27211.386
B0_T = 235051.757

def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _pos(x, name):
    v = _fin(x, name)
    if v <= 0.0: raise ValueError("bad " + name)
    return v
def _nonneg(x, name):
    v = _fin(x, name)
    if v < 0.0: raise ValueError("bad " + name)
    return v
def _posint(x, name):
    v = _fin(x, name)
    if v <= 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _nnint(x, name):
    v = _fin(x, name)
    if v < 0 or abs(v - round(v)) > 0: raise ValueError("bad " + name)
    return int(round(v))
def _vec(x, name, positive=True):
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)): raise ValueError("bad " + name)
    if positive and np.any(arr <= 0.0): raise ValueError("bad " + name)
    return arr
def _params(mex, mey, mhx, mhy):
    mux = mex*mhx/(mex+mhx); muy = mey*mhy/(mey+mhy); Mx = mex+mhx; My = mey+mhy
    rx = mex/mhx; ry = mey/mhy
    alpha = (1.0-rx*ry)/((1.0+rx)*(1.0+ry)); bx = 4.0*rx/(1.0+rx)**2; by = 4.0*ry/(1.0+ry)**2
    return mux, muy, Mx, My, alpha, (by+alpha*alpha)/muy, (bx+alpha*alpha)/mux
def _keldysh(r, r0, kappa):
    z = kappa*np.asarray(r, dtype=float)/r0
    return -(np.pi/(2.0*r0))*(special.struve(0, z) - special.y0(z))
def _lagfun(nmax, al, t):
    t = np.asarray(t, dtype=float)
    psi = np.zeros((nmax, t.size)); dpsi = np.zeros_like(psi); d2psi = np.zeros_like(psi)
    psi[0] = np.exp(0.5*al*np.log(t) - 0.5*t - 0.5*gammaln(al+1.0))
    if nmax > 1: psi[1] = (1.0+al-t)*psi[0]/np.sqrt(1.0+al)
    for n in range(1, nmax-1):
        psi[n+1] = ((2*n+1+al-t)*psi[n] - np.sqrt(n*(n+al))*psi[n-1])/np.sqrt((n+1)*(n+1+al))
    g = al/(2.0*t) - 0.5
    for n in range(nmax):
        prev = psi[n-1] if n >= 1 else 0.0
        dpsi[n] = g*psi[n] + (n*psi[n] - np.sqrt(n*(n+al))*prev)/t
        Q = dpsi[n] - g*psi[n]
        d2psi[n] = -al/(2.0*t*t)*psi[n] + g*dpsi[n] + g*Q + (-(al+1.0-t)*Q - n*psi[n])/t
    return psi, dpsi, d2psi
def _basis(a, N, m, r):
    t = np.asarray(r, dtype=float)/a; am = abs(int(m))
    psi, dpsi, d2psi = _lagfun(N, 2*am+1, t)
    s = t**-0.5
    return psi*s/a, (dpsi*s - 0.5*psi*t**-1.5)/a**2, (d2psi*s - dpsi*t**-1.5 + 0.75*psi*t**-2.5)/a**3
def _quad(a, Nq):
    n = np.arange(Nq)
    J = np.diag(2.0*n+1.0) + np.diag(np.arange(1, Nq, dtype=float), 1) + np.diag(np.arange(1, Nq, dtype=float), -1)
    t = np.linalg.eigvalsh(J)
    ssum = np.sum(_lagfun(Nq, 0.0, t)[0]**2, axis=0)
    wt = np.where(ssum > 0.0, 1.0/np.where(ssum > 0.0, ssum, 1.0), 0.0)
    return a*t, a*a*t*wt
def _hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, Kx, Ky, a, N, M, Nq, factorized=False):
    mux, muy, Mx, My, alpha, cX, cY = _params(mex, mey, mhx, mhy)
    if factorized:
        alpha = 0.0; cX = 1.0/muy; cY = 1.0/mux
    r, W = _quad(a, Nq); V = _keldysh(r, r0, kappa)
    ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r) for m in ms}
    kiso = 0.25*(1.0/mux + 1.0/muy); kan = 0.25*(1.0/mux - 1.0/muy)
    zi = 0.5*(1.0/mux + 1.0/muy); za = 0.5*(1.0/mux - 1.0/muy)
    cd = 0.5*(cX + cY)*B*B/8.0; cc = 0.5*(cX - cY)*B*B/8.0
    H = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f, df, d2f = F[m]; sl = slice(i*N, (i+1)*N)
        lap = d2f + df/r - (m*m)*f/r**2
        H[sl, sl] += -kiso*(f*W) @ lap.T + (f*W) @ (V*f).T + (f*W) @ ((cd*r*r)*f).T + (0.5*alpha*B*zi*m)*(f*W) @ f.T
        if i-2 >= 0:
            g = F[m-2][0]; sg = slice((i-2)*N, (i-1)*N)
            opz = 0.25*(d2f + (2*m-1)*df/r + m*(m-2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opz.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(r*df + m*f)).T
        if i+2 < nm:
            g = F[m+2][0]; sg = slice((i+2)*N, (i+3)*N)
            opzb = 0.25*(d2f - (2*m+1)*df/r + m*(m+2)*f/r**2)
            H[sg, sl] += -2.0*kan*(g*W) @ opzb.T + (g*W) @ ((0.5*cc*r*r)*f).T + (1j*0.5*alpha*B*za)*(g*W) @ ((0.5j)*(m*f - r*df)).T
        if Kx != 0.0 or Ky != 0.0:
            for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
                if 0 <= i+dm < nm:
                    g = F[m+dm][0]; sg = slice((i+dm)*N, (i+dm+1)*N)
                    H[sg, sl] += (B*(Ky/My*cxk - Kx/Mx*cyk))*(g*W) @ (r*f).T
    return H
def _dipoles(a, N, M, Nq):
    r, W = _quad(a, Nq); ms = np.arange(-M, M+1); nm = len(ms); D = nm*N
    F = {m: _basis(a, N, m, r)[0] for m in ms}
    X = np.zeros((D, D), dtype=complex); Y = np.zeros((D, D), dtype=complex)
    for i, m in enumerate(ms):
        f = F[m]; sl = slice(i*N, (i+1)*N)
        for dm, cxk, cyk in ((+1, 0.5, -0.5j), (-1, 0.5, 0.5j)):
            if 0 <= i+dm < nm:
                g = F[m+dm]; sg = slice((i+dm)*N, (i+dm+1)*N); blk = (g*W) @ (r*f).T
                X[sg, sl] += cxk*blk; Y[sg, sl] += cyk*blk
    return X, Y
def _spectrum(H, k):
    w = eigh(H, eigvals_only=True)
    return w[:k]
def _polarizability(H, X, Y):
    w, v = eigh(H)
    dx = v.conj().T @ (X @ v[:, 0]); dy = v.conj().T @ (Y @ v[:, 0])
    axx = 2.0*np.sum(np.abs(dx[1:])**2/(w[1:]-w[0])); ayy = 2.0*np.sum(np.abs(dy[1:])**2/(w[1:]-w[0]))
    return float(w[0]), float(axx), float(ayy)

def mx_audit(mex, mey, mhx, mhy, r0_nm, kappa, B_T, a, N, M, Nq, K0):
    K0 = _pos(K0, "K0")
    p = mx_parameters(mex, mey, mhx, mhy, r0_nm, kappa, B_T)
    Mx, My, r0, B = float(p[2]), float(p[3]), float(p[7]), float(p[8])
    if B <= 0.0: raise ValueError("a finite field is required")
    HH = mx_hamiltonian(mex, mey, mhx, mhy, r0, kappa, B, 0.0, 0.0, a, N, M, Nq, 0)
    H = HH[0] + 1j*HH[1]
    if np.max(np.abs(H - H.conj().T)) > 1e-10: raise ValueError("the Hamiltonian is not Hermitian")
    rr = mx_quadrature(a, Nq)
    vv = mx_keldysh(rr[0], r0, kappa)
    bb = mx_radial_basis(a, N, 0, rr[0])
    if not np.all(np.isfinite(vv)) or not np.all(np.isfinite(bb)): raise ValueError("non-finite potential or basis values")
    EB = mx_spectrum(mex, mey, mhx, mhy, r0, kappa, B, 0.0, 0.0, a, N, M, Nq, 0, 2)
    E00 = mx_spectrum(mex, mey, mhx, mhy, r0, kappa, 0.0, 0.0, 0.0, a, N, M, Nq, 0, 1)[0]
    EBf = mx_spectrum(mex, mey, mhx, mhy, r0, kappa, B, 0.0, 0.0, a, N, M, Nq, 1, 1)[0]
    E0f = mx_spectrum(mex, mey, mhx, mhy, r0, kappa, 0.0, 0.0, 0.0, a, N, M, Nq, 1, 1)[0]
    pol = mx_magnetopolarizability(mex, mey, mhx, mhy, r0, kappa, B, a, N, M, Nq, 0)
    axx, ayy = float(pol[1]), float(pol[2])
    dx = 1.0/(1.0 - ayy*B*B/Mx) - 1.0; dy = 1.0/(1.0 - axx*B*B/My) - 1.0
    if not (dx > 0.0 and dy > 0.0): raise ValueError("mass enhancement must be positive")
    EK = mx_spectrum(mex, mey, mhx, mhy, r0, kappa, B, K0, 0.0, a, N, M, Nq, 0, 1)[0]
    direct = (EK - EB[0])/27211.386; pred = -0.5*ayy*(B*K0/Mx)**2
    ratio = direct/pred
    if abs(ratio - 1.0) > 0.02: raise ValueError("the direct pseudomomentum shift does not match its second-order prediction")
    return np.array([EB[0], EB[1], EB[0]-E00, EBf-E0f, axx, ayy, dx, dy, ratio], dtype=np.float64)
SCICODE_GOLD_EOF
