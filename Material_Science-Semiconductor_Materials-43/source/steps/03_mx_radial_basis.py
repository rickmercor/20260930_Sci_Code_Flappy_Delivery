"""
Builds the orthonormal generalized-Laguerre radial basis of one angular channel with its first two derivatives.

A complete orthonormal radial basis with the correct short-distance power and a common decay length turns the two-dimensional Schrodinger equation into a finite matrix problem.

Returns
-------
A float64 array of shape (3, N, n).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mx_radial_basis(a, N, m, r):
    r"""a: positive float, the length scale of the basis in Bohr radii. N: positive integer, the number of radial
    functions. m: integer, the angular momentum of the channel. r: one-dimensional array of positive distances
    in Bohr radii.

    Returns a numpy float64 array of shape $(3, N, n)$: the orthonormal radial functions
    $\phi^{(m)}_n(r) = \frac{1}{a}\sqrt{\frac{n!}{(n + 2|m| + 1)!}}\,(r/a)^{|m|}\, e^{-r/(2a)}\, L^{(2|m|+1)}_n(r/a)$,
    $n = 0, \dots, N-1$, with $L^{(\alpha)}_n$ the generalized Laguerre polynomials, which satisfy
    $\int_0^\infty \phi^{(m)}_n \phi^{(m)}_{n'}\, r\, dr = \delta_{nn'}$; their first derivatives $d\phi/dr$; and their
    second derivatives $d^2\phi/dr^2$, each evaluated at the given distances.

    Raises:
        ValueError: on a non-positive or non-finite a, a non-integral or non-positive N, a non-integral m, or an
            invalid r.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

def _oracle_mx_radial_basis(a, N, m, r):
    if not np.isfinite(float(a)) or float(a) <= 0.0: raise ValueError("a must be positive")
    a = _pos(a, "a"); N = _posint(N, "N"); mf = _fin(m, "m")
    if abs(mf - round(mf)) > 0: raise ValueError("m must be an integer")
    m = int(round(mf)); r = _vec(r, "r")
    f, df, d2f = _basis(a, N, m, r)
    return np.stack([f, df, d2f]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nr=np.array([0.1,0.5,1.0,2.0,4.0,8.0,16.0,32.0,64.0])\n', 'call': 'mx_radial_basis(6.0,12,0,r)', 'gold_call': '_oracle_mx_radial_basis(6.0,12,0,r)'},
 {'setup': 'import numpy as np\nr=np.array([0.1,0.5,1.0,2.0,4.0,8.0,16.0,32.0,64.0])\n',
  'call': 'mx_radial_basis(6.0,10,-3,r)',
  'gold_call': '_oracle_mx_radial_basis(6.0,10,-3,r)'},
 {'setup': 'import numpy as np\nr=np.array([0.3,1.5,3.0,6.0,12.0,24.0,48.0])\n', 'call': 'mx_radial_basis(3.0,16,2,r)', 'gold_call': '_oracle_mx_radial_basis(3.0,16,2,r)'},
 {'setup': 'import numpy as np\n# boundary: a single function, the nodeless ground radial function of the channel\nr=np.array([1.0,5.0,20.0])\n',
  'call': 'mx_radial_basis(6.0,1,1,r)',
  'gold_call': '_oracle_mx_radial_basis(6.0,1,1,r)'},
 {'setup': 'import numpy as np\n'
           '# invalid input: a non-integral angular momentum must raise ValueError\n'
           'def _probe():\n'
           '    try:\n'
           '        mx_radial_basis(6.0,12,0.5,np.array([1.0,2.0]))\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '\n'
           'def _expected_probe():\n'
           '    try:\n'
           '        _oracle_mx_radial_basis(6.0,12,0.5,np.array([1.0,2.0]))\n'
           '        return 0\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n',
  'call': '_probe()',
  'gold_call': '_expected_probe()'}]
