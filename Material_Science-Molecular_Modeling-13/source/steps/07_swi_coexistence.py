"""
Locates the liquid-vapor coexistence of the square-well solvent from the equality of chemical potential and pressure.

A liquid-vapor interface exists only between coexisting phases; their densities set the amplitude of the interfacial profile and the reservoir the solute is measured against.

Returns
-------
A float64 array of shape (4,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def swi_coexistence(beps: float, lam: float) -> "np.ndarray":
    r"""beps: positive float, the well depth in units of $k_B T$. lam: float not below one, the well
    range in units of the diameter.

    Returns a numpy float64 array of shape $(4,)$: the coexisting vapor and liquid densities of the
    solvent and the common chemical potential $\beta\mu$ and pressure $\beta P\sigma^3$ at
    liquid-vapor coexistence, the two densities at which the chemical potentials and the pressures of
    the two branches agree, located to a relative precision of 1e-14.

    Raises:
        ValueError: on invalid beps or lam, or when the equation of state has no liquid-vapor
            coexistence at this temperature.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial.legendre import leggauss

def _pos(x, name):
    v = float(x)
    if not np.isfinite(v) or v <= 0.0: raise ValueError("bad " + name)
    return v
def _fin(x, name):
    v = float(x)
    if not np.isfinite(v): raise ValueError("bad " + name)
    return v
def _f3(n3):
    n3 = np.asarray(n3, dtype=float); out = np.empty_like(n3); s = n3 < 1e-4; b = n3[s]
    out[s] = 1.5 + 8.0*b/3.0 + 15.0*b*b/4.0 + 24.0*b**3/5.0
    b = n3[~s]; out[~s] = (b + (1.0-b)**2*np.log(1.0-b))/(b*b*(1.0-b)**2)
    return out
def _f3p(n3):
    n3 = np.asarray(n3, dtype=float); out = np.empty_like(n3); s = n3 < 1e-4; b = n3[s]
    out[s] = 8.0/3.0 + 7.5*b + 72.0*b*b/5.0 + 70.0*b**3/3.0
    b = n3[~s]; L = np.log(1.0-b)
    out[~s] = (b*(b-1.0)*(b+2.0*(b-1.0)*L) - 2.0*b*(b+(b-1.0)**2*L) - 2.0*(b-1.0)*(b+(b-1.0)**2*L))/(b**3*(b-1.0)**3)
    return out
def _phi(n0, n1, n2, n3, v1, v2):
    f = _f3(n3)
    return -n0*np.log(1.0-n3) + (n1*n2 - v1*v2)/(1.0-n3) + (n2**3 - 3.0*n2*v2**2)*f/(36.0*np.pi)
def _dphi(n0, n1, n2, n3, v1, v2):
    f = _f3(n3); fp = _f3p(n3)
    d0 = -np.log(1.0-n3); d1 = n2/(1.0-n3)
    d2 = n1/(1.0-n3) + 3.0*(n2**2 - v2**2)*f/(36.0*np.pi)
    d3 = n0/(1.0-n3) + (n1*n2 - v1*v2)/(1.0-n3)**2 + (n2**3 - 3.0*n2*v2**2)*fp/(36.0*np.pi)
    dv1 = -v2/(1.0-n3); dv2 = -v1/(1.0-n3) - 6.0*n2*v2*f/(36.0*np.pi)
    return d0, d1, d2, d3, dv1, dv2
def _bulk_n(rho, R):
    return rho, rho*R, rho*4.0*np.pi*R*R, rho*4.0*np.pi*R**3/3.0
def _mu_hs(rho, R, Rs):
    n0, n1, n2, n3 = _bulk_n(rho, R); z = np.zeros_like(np.asarray(rho, dtype=float))
    d0, d1, d2, d3, _, _ = _dphi(n0, n1, n2, n3, z, z)
    return d0 + Rs*d1 + 4.0*np.pi*Rs*Rs*d2 + 4.0*np.pi*Rs**3/3.0*d3
def _thermo(rho, beps, lam):
    eta = np.pi*rho/6.0; mu_hs = eta*(8.0 - 9.0*eta + 3.0*eta*eta)/(1.0-eta)**3; P_hs = rho*(1.0 + eta + eta*eta - eta**3)/(1.0-eta)**3
    L = lam; mu_sw = -4.0*np.pi/3.0*L**3*beps*rho; P_sw = -2.0*np.pi/3.0*L**3*beps*rho*rho
    return np.log(rho) + mu_hs + mu_sw, P_hs + P_sw, mu_hs, mu_sw
def _coex(beps, lam):
    rs = np.linspace(1e-4, 1.2, 4000); P = _thermo(rs, beps, lam)[1]; dP = np.diff(P)
    if not np.any(dP < 0.0): raise ValueError("no liquid-vapor coexistence at this temperature")
    imax = int(np.argmax(dP < 0.0)); imin = imax + int(np.argmax(dP[imax:] > 0.0)); rv_sp, rl_sp = rs[imax], rs[imin]
    mu_of = lambda r: float(_thermo(r, beps, lam)[0]); P_of = lambda r: float(_thermo(r, beps, lam)[1])
    def _root(fun, a, b, target):
        fa = fun(a) - target
        for _ in range(200):
            m = 0.5*(a+b); fm = fun(m) - target
            if fa*fm <= 0.0: b = m
            else: a, fa = m, fm
            if b - a < 1e-16*max(1.0, b): break
        return 0.5*(a+b)
    def _dp(mu):
        rv = _root(mu_of, 1e-12, rv_sp, mu); rl = _root(mu_of, rl_sp, 1.2, mu); return P_of(rl) - P_of(rv), rv, rl
    lo, hi = sorted([mu_of(rl_sp) + 1e-9, mu_of(rv_sp) - 1e-9]); fa = _dp(lo)[0]; fb = _dp(hi)[0]
    if fa*fb > 0.0: raise ValueError("no liquid-vapor coexistence at this temperature")
    for _ in range(200):
        m = 0.5*(lo+hi); fm = _dp(m)[0]
        if fa*fm <= 0.0: hi = m
        else: lo, fa = m, fm
        if hi - lo < 1e-16*max(1.0, abs(hi)): break
    mu = 0.5*(lo+hi); _, rv, rl = _dp(mu)
    return rv, rl, mu, P_of(rl)
def _rules(sigma2, lam1):
    lam2 = 1.0 + (lam1-1.0)/sigma2; sigma12 = 0.5*(1.0+sigma2); lam12 = (lam1 + lam2*sigma2)/(1.0+sigma2)
    return lam2, sigma12, lam12
def _mu2(rho1, sigma2, beps12, lam12):
    L12 = lam12*0.5*(1.0+sigma2); hs = float(_mu_hs(rho1, 0.5, 0.5*sigma2)); sw = -4.0*np.pi/3.0*L12**3*beps12*rho1
    return hs + sw, hs, sw
def _prof(z, rv, rl, a):
    return rv + (rl-rv)/(1.0 + np.exp(-a*np.asarray(z, dtype=float)))
def _nodes(lo, hi, panels, nodes):
    x, w = leggauss(nodes); lo = np.atleast_1d(np.asarray(lo, dtype=float)); hi = np.atleast_1d(np.asarray(hi, dtype=float))
    e = np.linspace(0.0, 1.0, panels+1); c = 0.5*(e[:-1]+e[1:]); h = 0.5*(e[1:]-e[:-1])
    t = (c[:, None] + h[:, None]*x[None, :]).ravel(); wt = (h[:, None]*w[None, :]).ravel()
    return lo[:, None] + (hi-lo)[:, None]*t[None, :], (hi-lo)[:, None]*wt[None, :]
def _wd(z, rv, rl, a, R, panels=8, nodes=32):
    z = np.atleast_1d(np.asarray(z, dtype=float)); zp, w = _nodes(z-R, z+R, panels, nodes)
    r = _prof(zp, rv, rl, a); d = z[:, None] - zp
    n3 = np.sum(w*r*np.pi*(R*R-d*d), axis=1); n2 = np.sum(w*r*2.0*np.pi*R, axis=1); v2 = np.sum(w*r*2.0*np.pi*d, axis=1)
    return n2/(4.0*np.pi*R*R), n2/(4.0*np.pi*R), n2, n3, v2/(4.0*np.pi*R), v2
def _mf(z, rv, rl, a, L, panels=8, nodes=32):
    z = np.atleast_1d(np.asarray(z, dtype=float)); zp, w = _nodes(z-L, z+L, panels, nodes)
    r = _prof(zp, rv, rl, a); d = z[:, None] - zp
    return np.sum(w*r*np.pi*(L*L-d*d), axis=1)
def _c1(z, rv, rl, a, sigma2, beps12, lam12, panels=8, nodes=32, chunk=32):
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if len(z) > chunk:
        return np.concatenate([_c1(z[i:i+chunk], rv, rl, a, sigma2, beps12, lam12, panels, nodes, chunk) for i in range(0, len(z), chunk)])
    R1 = 0.5; R2 = 0.5*sigma2; L12 = lam12*0.5*(1.0+sigma2)
    zp, w = _nodes(z-R2, z+R2, panels, nodes)
    n = _wd(zp.ravel(), rv, rl, a, R1, panels, nodes)
    d0, d1, d2, d3, dv1, dv2 = [q.reshape(zp.shape) for q in _dphi(*n)]
    d = zp - z[:, None]
    k3 = np.pi*(R2*R2-d*d); k2 = 2.0*np.pi*R2; k1 = k2/(4.0*np.pi*R2); k0 = k2/(4.0*np.pi*R2*R2); kv2 = 2.0*np.pi*d; kv1 = kv2/(4.0*np.pi*R2)
    hs = np.sum(w*(d0*k0 + d1*k1 + d2*k2 + d3*k3 + dv1*kv1 + dv2*kv2), axis=1)
    return -hs + beps12*_mf(z, rv, rl, a, L12, panels, nodes)
def _veff(z, rv, rl, a, sigma2, beps12, lam12):
    return -_mu2(rl, sigma2, beps12, lam12)[0] - _c1(z, rv, rl, a, sigma2, beps12, lam12)
def _line(fun, W=25.0, panel_width=0.5, nodes=32):
    npan = int(round(2.0*W/panel_width)); zp, w = _nodes(np.array([-W]), np.array([W]), npan, nodes)
    return float(np.sum(w[0]*fun(zp[0])))
def _omega(z, rv, rl, a, beps, lam, mu):
    z = np.atleast_1d(np.asarray(z, dtype=float)); r = _prof(z, rv, rl, a); n = _wd(z, rv, rl, a, 0.5)
    return r*(np.log(r)-1.0) + _phi(*n) - 0.5*beps*r*_mf(z, rv, rl, a, lam) - mu*r
def _minimise(fun, lo, hi):
    gr = (np.sqrt(5.0)-1.0)/2.0; c = hi - gr*(hi-lo); d = lo + gr*(hi-lo); fc = fun(c); fd = fun(d)
    for _ in range(200):
        if fc < fd: hi, d, fd = d, c, fc; c = hi - gr*(hi-lo); fc = fun(c)
        else: lo, c, fc = c, d, fd; d = lo + gr*(hi-lo); fd = fun(d)
        if hi - lo < 1e-12: break
    zm = 0.5*(lo+hi); return zm, fun(zm)

def _oracle_swi_coexistence(beps: float, lam: float) -> "np.ndarray":
    beps = _pos(beps, "beps"); lam = _fin(lam, "lam")
    if lam < 1.0: raise ValueError("lam must not be below one")
    rv, rl, mu, P = _coex(beps, lam)
    return np.array([rv, rl, mu, P], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\n","call":"swi_coexistence(1.0,1.5)","gold_call":"_oracle_swi_coexistence(1.0,1.5)"},
        {"setup":"import numpy as np\n","call":"swi_coexistence(1.1,1.5)","gold_call":"_oracle_swi_coexistence(1.1,1.5)"},
        {"setup":"import numpy as np\n","call":"swi_coexistence(1.2,1.4)","gold_call":"_oracle_swi_coexistence(1.2,1.4)"},
        {"setup":"import numpy as np\n# boundary: a temperature close to the critical point, where the two densities approach each other\n","call":"swi_coexistence(0.8,1.5)","gold_call":"_oracle_swi_coexistence(0.8,1.5)"},
        {"setup":"import numpy as np\n# invalid input: above the critical temperature there is no coexistence and ValueError must be raised\ndef _probe(fn):\n    try:\n        fn(0.5,1.5)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n","call":"_probe(swi_coexistence)","gold_call":"_probe(_oracle_swi_coexistence)"},
    ]
