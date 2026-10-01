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

def _check_phases(phases):
    p=np.asarray(phases,dtype=float)
    if p.ndim!=2 or p.shape[0]!=3 or p.shape[1]<1 or not np.isfinite(p).all():
        raise ValueError('phases must be finite with shape (3,m), m>=1')
    return p


def _check_bounds(bounds):
    b=np.asarray(bounds,dtype=float)
    if b.shape!=(2,2) or not np.isfinite(b).all() or np.any(b[:,1]<=b[:,0]):
        raise ValueError('bounds must contain two finite increasing pairs')
    return b


def _canonical_phases():
    return np.array([[5.93,2.26,4.93,3.72,1.85],
                     [5.80,5.46,2.29,6.11,1.41],
                     [5.06,4.28,2.96,0.19,5.62]])


def _wave(t, phases, derivative=0):
    k=np.arange(1,len(phases)+1)
    return np.sum(k**derivative*np.cos(np.asarray(t)[...,None]*k+phases+derivative*np.pi/2),axis=-1)


def seed_bounds(phases):
    phases=_check_phases(phases);m=phases.shape[1];ranges=[]
    for row in phases:
        c=np.zeros(2*m+1,dtype=complex)
        for k,angle in enumerate(row,1):
            c[m+k]+=1j*k*np.exp(1j*angle)/2
            c[m-k]-=1j*k*np.exp(-1j*angle)/2
        roots=np.polynomial.polynomial.polyroots(c)
        angles=np.angle(roots[np.abs(np.abs(roots)-1)<1e-7])%(2*np.pi)
        if len(angles)<2:
            raise ValueError('stationary roots unresolved')
        values=_wave(np.r_[angles,0.],row)
        ranges.append([values.min(),values.max()])
    pp=np.array([x*y for x in ranges[0] for y in ranges[1]])
    tt=np.array([v*z for v in pp for z in ranges[2]])
    return np.array([[pp.min(),pp.max()],[tt.min(),tt.max()]])

import numpy as np

def prescribed_component(x,y,z,phases,bounds):
    p=_check_phases(phases);bounds=_check_bounds(bounds)
    x,y,z=[np.asarray(v,dtype=float) for v in (x,y,z)]
    if not all(np.isfinite(v).all() for v in (x,y,z)):
        raise ValueError('coordinates must be finite')
    try:
        np.broadcast_shapes(x.shape,y.shape,z.shape)
    except ValueError as exc:
        raise ValueError('coordinates must broadcast') from exc
    sx,sy,sz=_wave(x,p[0]),_wave(y,p[1]),_wave(z,p[2])
    u=(sx*sy*sz-bounds[1,0])/np.diff(bounds[1])[0]
    a=.1+(np.pi-.2)*np.expm1(5*u)/np.expm1(5.)
    factor=-np.sin(a)*(np.pi-.2)*5*np.exp(5*u)/np.expm1(5.)/np.diff(bounds[1])[0]
    values=np.broadcast_arrays(np.cos(a),factor*_wave(x,p[0],1)*sy*sz,
                               factor*sx*_wave(y,p[1],1)*sz,factor*sx*sy*_wave(z,p[2],1))
    if not all(np.isfinite(v).all() for v in values):
        raise ValueError('prescribed field is not finite')
    return np.stack(values,axis=-1)

import numpy as np

def boundary_field(x,y,phases,bounds):
    p=_check_phases(phases);bounds=_check_bounds(bounds)
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    g=prescribed_component(x,y,0.,p,bounds)[...,0]
    v=(_wave(x,p[0])*_wave(y,p[1])-bounds[0,0])/np.diff(bounds[0])[0]
    a2=1-g*g
    normal=np.sqrt(np.maximum(0.,a2))*np.cos(1+(np.pi-2)*v)
    positive2=a2-normal*normal
    if not np.isfinite(positive2).all() or np.any(positive2<=0):
        raise ValueError('boundary is not on the positive branch')
    return np.stack([g,np.sqrt(positive2),normal],axis=-1)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

def volume_field(xs,ys,zs,phases,bounds):
    phases=_check_phases(phases);bounds=_check_bounds(bounds)
    xs,ys,zs=[np.asarray(v,dtype=float) for v in (xs,ys,zs)]
    if any(v.ndim!=1 or not len(v) or not np.isfinite(v).all() for v in (xs,ys,zs)):
        raise ValueError('query axes must be finite nonempty vectors')
    if np.any(xs<0) or np.any(xs>=2*np.pi) or np.any(ys<0) or np.any(ys>=2*np.pi):
        raise ValueError('periodic coordinates must be in [0,2*pi)')
    if zs[0]<0 or np.any(np.diff(zs)<=0):
        raise ValueError('z queries must be nonnegative and strictly increasing')
    n=2048
    x=xs[:,None];y=2*np.pi*np.arange(n)[None,:]/n
    initial=boundary_field(x,y,phases,bounds)[...,2]
    freq=np.fft.fftfreq(n,1/n);freq[n//2]=0
    def derivative(z,flat):
        normal=flat.reshape(len(xs),n)
        data=prescribed_component(x,y,z,phases,bounds)
        g,gx,gy,gz=np.moveaxis(data,-1,0)
        p2=1-g*g-normal*normal
        if np.any(p2<=0):
            raise ValueError('positive branch lost')
        positive=np.sqrt(p2)
        normal_y=np.fft.ifft(1j*freq*np.fft.fft(normal,axis=1),axis=1).real
        positive_y=-(g*gy+normal*normal_y)/positive
        return (-gx-positive_y).ravel()
    if zs[-1]==0:
        states=initial.ravel()[None,:]
    else:
        sol=solve_ivp(derivative,(0.,float(zs[-1])),initial.ravel(),method='DOP853',
                      rtol=2e-11,atol=2e-13,t_eval=zs,max_step=max(.0005,float(zs[-1])/200))
        if not sol.success:
            raise ValueError('smooth continuation could not be resolved')
        states=sol.y.T
    output=[]
    for z,flat in zip(zs,states):
        normal=flat.reshape(len(xs),n)
        indices=ys*n/(2*np.pi)
        if np.all(np.abs(indices-np.rint(indices))<1e-10):
            evaluated=normal[:,np.rint(indices).astype(int)%n]
        else:
            spline=CubicSpline(np.linspace(0,2*np.pi,n+1),np.concatenate([normal,normal[:,:1]],axis=1),axis=1,bc_type='periodic')
            evaluated=spline(ys)
        g=prescribed_component(x,ys[None,:],float(z),phases,bounds)[...,0]
        p2=1-g*g-evaluated*evaluated
        if np.any(p2<=0):
            raise ValueError('query field leaves the positive branch')
        output.append(np.stack([g,np.sqrt(p2),evaluated],axis=-1))
    return np.array(output)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

_TWO_PI = 2 * np.pi


def _two_scalars(first, second):
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.ndim or b.ndim or not np.isfinite(a) or not np.isfinite(b):
        raise ValueError('both arguments must be finite real scalars')
    if b < 0:
        raise ValueError('the height argument must not be negative')
    return float(a), float(b)


def _theta_pack(phases, bounds, xpk, y, z):
    """theta above the plane, with the derivatives the continuation needs."""
    sx, dsx = xpk
    sy = _wave(y, phases[1])
    dsy = _wave(y, phases[1], 1)
    d2sy = _wave(y, phases[1], 2)
    sz = _wave(np.asarray(z, dtype=float), phases[2])
    dsz = _wave(np.asarray(z, dtype=float), phases[2], 1)
    span = np.diff(bounds[1])[0]
    beta = (np.pi - .2) / np.expm1(5.)
    u = (sx * sy * sz - bounds[1, 0]) / span
    theta = .1 + beta * np.expm1(5 * u)
    fp = 5 * beta * np.exp(5 * u)
    fpp = 5 * fp
    ux, uy, uz = dsx * sy * sz / span, sx * dsy * sz / span, sx * sy * dsz / span
    uxy, uyy, uzy = dsx * dsy * sz / span, sx * d2sy * sz / span, sx * dsy * dsz / span
    return (theta, fp * ux, fp * uy, fp * uz,
            fpp * ux * uy + fp * uxy, fpp * uy * uy + fp * uyy, fpp * uz * uy + fp * uzy)


def _azimuth0(phases, bounds, x, y):
    v = (_wave(x, phases[0]) * _wave(y, phases[1]) - bounds[0, 0]) / np.diff(bounds[0])[0]
    return 1 + (np.pi - 2) * v


def _dazimuth0(phases, bounds, x, y):
    return (np.pi - 2) * _wave(x, phases[0]) * _wave(y, phases[1], 1) / np.diff(bounds[0])[0]


def _transport(phases, bounds, xpk, z, y, ph, jac, kap):
    """Right-hand side of the transported state (y, azimuth, and their y0 sensitivities)."""
    th, tx, ty, tz, txy, tyy, tzy = _theta_pack(phases, bounds, xpk, y, z)
    st, ct = np.sin(th), np.cos(th)
    sp, cp = np.sin(ph), np.cos(ph)
    cott, dcott, cotp = ct / st, -1. / st ** 2, cp / sp
    forcing = -tx / sp + cott * ty + cott * cotp * tz
    force_y = -txy / sp + dcott * ty * ty + cott * tyy + (dcott * ty * tz + cott * tzy) * cotp
    force_p = tx * cp / sp ** 2 - cott * tz / sp ** 2
    return -cotp, forcing, kap / sp ** 2, force_y * jac + force_p * kap


def _march(phases, bounds, xs, y0s, dz, zcap, track_margin=False):
    """Vectorised fixed-step march of the transported state over a footpoint family."""
    grid_x = np.repeat(np.asarray(xs, dtype=float)[:, None], len(y0s), 1)
    grid_y = np.repeat(np.asarray(y0s, dtype=float)[None, :], len(xs), 0)
    xpk = (_wave(grid_x, phases[0]), _wave(grid_x, phases[0], 1))
    y, ph = grid_y.copy(), _azimuth0(phases, bounds, grid_x, grid_y)
    jac, kap = np.ones_like(y), _dazimuth0(phases, bounds, grid_x, grid_y)
    stop = np.full(y.shape, np.inf)
    margin = np.sin(_theta_pack(phases, bounds, xpk, y, 0.)[0]) * np.sin(ph) if track_margin else None
    z = 0.
    with np.errstate(all='ignore'):
        while z < zcap - 1e-15:
            step = min(dz, zcap - z)
            a1, b1, c1, d1 = _transport(phases, bounds, xpk, z, y, ph, jac, kap)
            a2, b2, c2, d2 = _transport(phases, bounds, xpk, z + step / 2, y + step / 2 * a1,
                                        ph + step / 2 * b1, jac + step / 2 * c1, kap + step / 2 * d1)
            a3, b3, c3, d3 = _transport(phases, bounds, xpk, z + step / 2, y + step / 2 * a2,
                                        ph + step / 2 * b2, jac + step / 2 * c2, kap + step / 2 * d2)
            a4, b4, c4, d4 = _transport(phases, bounds, xpk, z + step, y + step * a3,
                                        ph + step * b3, jac + step * c3, kap + step * d3)
            yn = y + step / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
            pn = ph + step / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
            jn = jac + step / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
            kn = kap + step / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
            folded = (jac > 0) & (jn <= 0) & np.isinf(stop)
            stop[folded] = z + step * jac[folded] / (jac[folded] - jn[folded])
            sp, spn = np.sin(ph), np.sin(pn)
            lost = (sp > 0) & (spn <= 0) & np.isinf(stop)
            stop[lost] = z + step * sp[lost] / (sp[lost] - spn[lost])
            blown = ~np.isfinite(yn + pn + jn + kn) & np.isinf(stop)
            stop[blown] = z + step
            y, ph, jac, kap = yn, pn, jn, kn
            z += step
            if track_margin:
                margin = np.fmin(margin, np.sin(_theta_pack(phases, bounds, xpk, y, z)[0]) * np.sin(ph))
            if np.isfinite(stop).any() and z > 1.3 * np.nanmin(stop[np.isfinite(stop)]):
                break
    return stop, margin


def _breakdown(phases, bounds, x, y0, ceiling):
    """Height at which the transported state above (x, y0) first stops being admissible."""
    xpk = (_wave(np.float64(x), phases[0]), _wave(np.float64(x), phases[0], 1))

    def rhs(z, s):
        return _transport(phases, bounds, xpk, z, s[0], s[1], s[2], s[3])

    def fold(z, s):
        return s[2]
    fold.terminal, fold.direction = True, -1

    def branch(z, s):
        return np.sin(s[1]) - 1e-6
    branch.terminal, branch.direction = True, -1

    state = [float(y0), float(_azimuth0(phases, bounds, np.float64(x), np.float64(y0))), 1.,
             float(_dazimuth0(phases, bounds, np.float64(x), np.float64(y0)))]
    sol = solve_ivp(rhs, (0., float(ceiling)), state, method='DOP853', rtol=1e-11, atol=1e-13,
                    events=(fold, branch))
    hits = [float(e[0]) for e in sol.t_events if e.size]
    if hits:
        return min(hits)
    # The transport is singular where the azimuth reaches 0 or pi, so an integration that
    # cannot be continued has reached the loss of the positive branch there.
    return float(sol.t[-1]) if sol.status == -1 else np.inf


def _margin_along(phases, bounds, x, y0, zmax):
    """Minimum y component of the field along the characteristic above (x, y0) up to zmax."""
    xpk = (_wave(np.float64(x), phases[0]), _wave(np.float64(x), phases[0], 1))
    if zmax == 0.:
        th = _theta_pack(phases, bounds, xpk, np.float64(y0), 0.)[0]
        return float(np.sin(th) * np.sin(_azimuth0(phases, bounds, np.float64(x), np.float64(y0))))

    def rhs(z, s):
        return _transport(phases, bounds, xpk, z, s[0], s[1], s[2], s[3])

    state = [float(y0), float(_azimuth0(phases, bounds, np.float64(x), np.float64(y0))), 1.,
             float(_dazimuth0(phases, bounds, np.float64(x), np.float64(y0)))]
    sol = solve_ivp(rhs, (0., float(zmax)), state, method='DOP853', rtol=1e-12, atol=1e-14,
                    dense_output=True)
    if not sol.success:
        raise ValueError('the transported state could not be resolved')

    def profile(zs):
        zs = np.atleast_1d(np.asarray(zs, dtype=float))
        s = sol.sol(zs)
        return np.sin(_theta_pack(phases, bounds, xpk, s[0], zs)[0]) * np.sin(s[1])
    grid = np.linspace(0., float(zmax), 201)
    vals = profile(grid)
    k = int(np.argmin(vals))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    if hi > lo:
        res = minimize_scalar(lambda zz: float(profile(zz)[0]), bounds=(lo, hi),
                              method='bounded', options={'xatol': 1e-12})
        return float(min(vals[k], res.fun))
    return float(vals[k])


def branch_margin(x, zmax, phases, bounds):
    phases = _check_phases(phases)
    bounds = _check_bounds(bounds)
    x, zmax = _two_scalars(x, zmax)
    family = _TWO_PI * np.arange(256) / 256
    if zmax > 0:
        stop, margin = _march(phases, bounds, [x], family, zmax / 400., zmax, track_margin=True)
        if np.any(np.isfinite(stop)):
            raise ValueError('the properties cannot be maintained above this plane up to zmax')
        order = np.argsort(margin[0])[:3]
    else:
        th = _theta_pack(phases, bounds,
                         (_wave(np.full(len(family), x), phases[0]),
                          _wave(np.full(len(family), x), phases[0], 1)), family, 0.)[0]
        order = np.argsort(np.sin(th) * np.sin(_azimuth0(phases, bounds, x, family)))[:3]
    best = np.inf
    for j in order:
        spacing = _TWO_PI / len(family)
        fun = lambda t: _margin_along(phases, bounds, x, t, zmax)
        left, right = family[j] - spacing, family[j] + spacing
        res = minimize_scalar(fun, bounds=(left, right), method='bounded',
                              options={'xatol': 1e-9})
        best = min(best, float(res.fun), fun(family[j]))
    if not np.isfinite(best):
        raise ValueError('the field above this plane is not finite')
    return float(best)

import numpy as np
from scipy.optimize import minimize_scalar

_TWO_PI = 2 * np.pi


def _plane_candidates(phases, bounds, x, ceiling, family):
    """Footpoints whose transported state stops earliest on a coarse march."""
    y0s = _TWO_PI * np.arange(family) / family
    stop, _ = _march(phases, bounds, [x], y0s, min(ceiling, 1.0) / 250., ceiling)
    row = stop[0]
    finite = np.isfinite(row)
    if not finite.any():
        return y0s, None
    return y0s, np.argsort(np.where(finite, row, np.inf))[:3]


def plane_height(x, phases, bounds, ceiling=2.0):
    phases = _check_phases(phases)
    bounds = _check_bounds(bounds)
    x, ceiling = _two_scalars(x, ceiling)
    if ceiling <= 0:
        raise ValueError('ceiling must be positive')
    family = 192
    y0s, order = _plane_candidates(phases, bounds, x, ceiling, family)
    if order is None:
        raise ValueError('the field above this plane survives the whole search range')
    spacing = _TWO_PI / family
    best = np.inf
    for j in order:
        fun = lambda t: min(_breakdown(phases, bounds, x, t, ceiling), ceiling)
        node = float(y0s[j])
        res = minimize_scalar(fun, bounds=(node - spacing, node + spacing), method='bounded',
                              options={'xatol': 1e-11})
        best = min(best, float(res.fun), fun(node))
    if not np.isfinite(best) or best >= ceiling * (1 - 1e-12):
        raise ValueError('the field above this plane survives the whole search range')
    return float(best)

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

_TWO_PI = 2 * np.pi


def _global_candidates(phases, bounds, ceiling, nx=24, ny0=48, keep=4, reach=1.3, most=8):
    """Coarse locator: starting cells in every basin a two-dimensional family can see.

    The earliest-stopping cells, plus the earliest cell of every plane that is a local
    minimum along x, limited to cells stopping within `reach` of the earliest one. One
    earliest cell is not enough: separate basins can sit within a few percent of each
    other, and a coarse family can rank them in the wrong order.
    """
    xs = _TWO_PI * np.arange(nx) / nx
    y0s = _TWO_PI * np.arange(ny0) / ny0
    stop, _ = _march(phases, bounds, xs, y0s, min(ceiling, 1.0) / 100., ceiling)
    if not np.isfinite(stop).any():
        raise ValueError('no breakdown of the field was located below the search range')
    flat = np.where(np.isfinite(stop), stop, np.inf)
    picks = [np.unravel_index(int(k), flat.shape)
             for k in np.argsort(flat, axis=None, kind='stable')[:keep]]
    column = flat.min(axis=1)
    for i in np.argsort(column, kind='stable'):
        if np.isfinite(column[i]) and column[i] <= column[i - 1] and column[i] <= column[(i + 1) % nx]:
            picks.append((int(i), int(np.argmin(flat[i]))))
    limit = reach * flat.min()
    starts = []
    for i, j in picks:
        cell = (float(xs[i]), float(y0s[j]))
        if flat[i, j] <= limit and cell not in starts:
            starts.append(cell)
    return starts[:most], _TWO_PI / nx, _TWO_PI / ny0


def _refine_pair(phases, bounds, x0, y0, hx, hy, ceiling, rough=False):
    """Local grid then Nelder-Mead on the two-variable breakdown height.

    A rough pass ranks the basins cheaply; the final pass resolves the winner. The final
    tolerances sit just above the event-location noise of the integrator (about 1e-14 in
    height), because a tighter function tolerance is never met and only exhausts the
    evaluation budget.
    """
    grid_x = x0 + hx * np.linspace(-.5, .5, 5)
    grid_y = y0 + hy * np.linspace(-.5, .5, 5)
    best = (np.inf, x0, y0)
    for xv in grid_x:
        for yv in grid_y:
            value = _breakdown(phases, bounds, xv, yv, ceiling)
            if value < best[0]:
                best = (value, float(xv), float(yv))
    if not np.isfinite(best[0]):
        raise ValueError('no breakdown of the field was located below the search range')
    start = np.array([best[1], best[2]])
    scale = .25 * min(hx, hy)
    simplex = np.array([start, start + [scale, 0.], start + [0., scale]])
    stopping = (dict(xatol=1e-5, fatol=1e-10, maxfev=300) if rough
                else dict(xatol=1e-8, fatol=1e-13, maxfev=3000))
    result = minimize(lambda v: min(_breakdown(phases, bounds, v[0], v[1], ceiling), ceiling),
                      start, method='Nelder-Mead',
                      options=dict(initial_simplex=simplex, **stopping))
    return float(result.fun), float(result.x[0]), float(result.x[1])


def _global_margin(phases, bounds, zmax, nx=24, ny0=48):
    """Smallest y component of the field anywhere in 0 <= z <= zmax."""
    xs = _TWO_PI * np.arange(nx) / nx
    y0s = _TWO_PI * np.arange(ny0) / ny0
    stop, margin = _march(phases, bounds, xs, y0s, zmax / 400., zmax, track_margin=True)
    if np.any(np.isfinite(stop)):
        raise ValueError('the field does not reach the requested height everywhere')
    i, j = np.unravel_index(int(np.argmin(margin)), margin.shape)
    start = np.array([float(xs[i]), float(y0s[j])])
    scale = .25 * min(_TWO_PI / nx, _TWO_PI / ny0)
    simplex = np.array([start, start + [scale, 0.], start + [0., scale]])
    result = minimize(lambda v: _margin_along(phases, bounds, v[0], v[1], zmax), start,
                      method='Nelder-Mead',
                      options=dict(initial_simplex=simplex, xatol=1e-7, fatol=1e-12,
                                   maxfev=600))
    return float(min(float(result.fun), float(margin[i, j])))


def _transported_field(phases, bounds, x, ys, z):
    """The same field at (x, ys, z) from the transported state, for the cross-check."""
    count = 20001
    footpoints = _TWO_PI * np.arange(count) / count
    xpk = (_wave(np.full(count, float(x)), phases[0]), _wave(np.full(count, float(x)), phases[0], 1))
    state = np.stack([footpoints, _azimuth0(phases, bounds, np.full(count, float(x)), footpoints),
                      np.ones(count),
                      _dazimuth0(phases, bounds, np.full(count, float(x)), footpoints)])

    def rhs(zz, flat):
        s = flat.reshape(4, count)
        return np.stack(_transport(phases, bounds, xpk, zz, s[0], s[1], s[2], s[3])).ravel()

    sol = solve_ivp(rhs, (0., float(z)), state.ravel(), method='DOP853', rtol=1e-11, atol=1e-13)
    if not sol.success:
        raise ValueError('the transported family could not be resolved')
    end = sol.y[:, -1].reshape(4, count)
    order = np.argsort(end[0] % _TWO_PI)
    mapped = (end[0] % _TWO_PI)[order]
    azimuth = end[1][order]
    extended_y = np.concatenate([mapped - _TWO_PI, mapped, mapped + _TWO_PI])
    extended_p = np.concatenate([azimuth, azimuth, azimuth])
    queried = np.interp(np.asarray(ys, dtype=float), extended_y, extended_p)
    theta = _theta_pack(phases, bounds,
                        (_wave(np.full(len(ys), float(x)), phases[0]),
                         _wave(np.full(len(ys), float(x)), phases[0], 1)),
                        np.asarray(ys, dtype=float), float(z))[0]
    return np.stack([np.cos(theta), np.sin(theta) * np.sin(queried),
                     np.sin(theta) * np.cos(queried)], axis=-1)


def existence_height(phases=None, ceiling=2.0):
    p = _canonical_phases() if phases is None else _check_phases(phases)
    ceiling = float(np.asarray(ceiling, dtype=float))
    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError('ceiling must be a finite positive scalar')
    bounds = seed_bounds(p)
    starts, hx, hy = _global_candidates(p, bounds, ceiling)
    ranked = min(_refine_pair(p, bounds, x0, y0, hx, hy, ceiling, rough=True) for x0, y0 in starts)
    pair = _refine_pair(p, bounds, ranked[1], ranked[2], hx / 8, hy / 8, ceiling)
    height, plane_x = pair[0], pair[1]
    if height >= ceiling * (1 - 1e-12):
        raise ValueError('the field satisfies every stated property throughout the range')

    # The answer is a minimum of the per-plane heights: certify the critical plane and
    # its immediate neighbours, and keep the smallest value any of them reports.
    offsets = (0., -2e-4, 2e-4)
    plane_values = []
    for shift in offsets:
        plane_values.append(plane_height(plane_x + shift, p, bounds, ceiling))
    if plane_values[0] > min(plane_values[1], plane_values[2]) + 1e-9:
        raise ValueError('the critical plane is not a local minimum of the plane heights')
    if abs(plane_values[0] - height) > 1e-8:
        raise ValueError('the plane height disagrees with the two-variable minimum')
    height = min([height] + plane_values)

    # The positive branch has to survive strictly below the answer over the WHOLE domain,
    # otherwise the height would be set by loss of that branch instead. The plane the
    # answer comes from need not be the plane where the y component is smallest, so the
    # margin is minimised globally and the critical plane's own margin must not undercut it.
    inner = .999 * height
    plane_x = plane_x % _TWO_PI
    margin = branch_margin(plane_x, inner, p, bounds)
    global_margin = min(_global_margin(p, bounds, inner), margin)
    if not np.isfinite(global_margin) or global_margin <= 0:
        raise ValueError('the positive branch does not survive below the returned height')

    # Independent reconstruction of the same field, from the component form rather than
    # the transported state, checked against the prescribed data and the boundary values.
    probe_y = _TWO_PI * np.arange(8) / 8
    inner = .7 * height
    field = volume_field(np.array([plane_x]), probe_y, np.array([inner]), p, bounds)
    prescribed = prescribed_component(np.array([plane_x])[None, :, None],
                                              probe_y[None, None, :], np.array([inner])[:, None, None],
                                              p, bounds)[..., 0]
    if np.max(np.abs(np.linalg.norm(field, axis=-1) - 1)) > 1e-9:
        raise ValueError('the reconstructed field does not have unit magnitude')
    if np.max(np.abs(field[..., 0] - prescribed)) > 1e-9:
        raise ValueError('the reconstructed field does not carry the prescribed component')
    transported = _transported_field(p, bounds, plane_x, probe_y, inner)
    if np.max(np.abs(field[0, 0] - transported)) > 1e-5:
        raise ValueError('the two reconstructions of the field disagree')
    boundary = boundary_field(np.array([plane_x]), probe_y, p, bounds)
    at_zero = volume_field(np.array([plane_x]), probe_y, np.array([0.]), p, bounds)[0]
    if not np.allclose(at_zero, boundary, rtol=0, atol=2e-6):
        raise ValueError('the reconstruction does not reproduce the boundary data')
    return float(height)
SCICODE_GOLD_EOF
