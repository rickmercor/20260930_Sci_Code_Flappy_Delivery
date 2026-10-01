#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def copying_state_posteriors(g: np.ndarray, mismatch: np.ndarray, nref: int, \
    ne: float) -> tuple[np.ndarray, np.ndarray]:
    import math
    import numpy as np
    from scipy.special import logsumexp
    g = _array(g, 1); z = _array(mismatch, 2)
    nref = float(_array(nref, 0)); ne = float(_array(ne, 0))
    if not (len(g) == z.shape[0] and z.shape[1] >= 2):
        raise ValueError('input outside the declared domain')
    if not (np.all(np.diff(g) > 0) and np.all((z == 0) | (z == 1))):
        raise ValueError('input outside the declared domain')
    if not (np.isfinite(nref) and nref >= 2 and int(nref) == nref):
        raise ValueError('input outside the declared domain')
    if not (np.isfinite(ne) and ne > 0):
        raise ValueError('input outside the declared domain')
    nref = int(nref)

    g = np.asarray(g, float)
    z = np.asarray(mismatch, int)
    mcount, hcount = z.shape
    lam = 1.0 / (math.log(nref) + 0.5)
    theta = lam / (2.0 * (lam + nref))
    em = np.where(z, theta, 1.0-theta)
    r = -np.expm1(-0.04*ne/nref*np.r_[0.,np.diff(g)])
    f = np.empty_like(em); b = np.ones_like(em)
    f[0] = em[0]/em[0].sum()
    for m in range(1,mcount):
        f[m] = em[m]*((1-r[m])*f[m-1]+r[m]/hcount)
        f[m] /= f[m].sum()
    for m in range(mcount-2,-1,-1):
        v = b[m+1]*em[m+1]
        b[m] = (1-r[m+1])*v+r[m+1]/hcount*v.sum()
        b[m] /= b[m].sum()
    gamma = f*b
    gamma /= gamma.sum(axis=1)[:,None]
    same = np.empty_like(gamma)
    same[0] = gamma[0]
    for m in range(1,mcount):
        v = b[m]*em[m]
        den = np.dot((1-r[m])*f[m-1]+r[m]/hcount,v)
        same[m] = f[m-1]*((1-r[m])+r[m]/hcount)*v/den
    return gamma, same

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def panel_statistics(gamma: np.ndarray, same: np.ndarray, panels: np.ndarray, \
    npanels: int) -> tuple[np.ndarray, np.ndarray]:
    import math
    import numpy as np
    from scipy.special import logsumexp
    gamma = _array(gamma, 2); same = _array(same, 2)
    j = _array(panels, 2)
    npanels = float(_array(npanels, 0))
    if not (gamma.shape == same.shape == j.shape and gamma.shape[1] >= 2):
        raise ValueError('input outside the declared domain')
    if not (int(npanels) == npanels and npanels >= 1):
        raise ValueError('input outside the declared domain')
    if not (np.all(j == np.floor(j)) and np.all((j >= 0) & (j < npanels))):
        raise ValueError('input outside the declared domain')
    if not (np.all(gamma >= 0) and np.allclose(gamma.sum(1), 1, atol=1e-9)):
        raise ValueError('input outside the declared domain')
    if not (np.all(same >= 0) and np.all(same.sum(1) <= 1+1e-9)):
        raise ValueError('input outside the declared domain')
    npanels = int(npanels)

    panels = np.asarray(panels,int)
    gamma, same = np.asarray(gamma,float), np.asarray(same,float)
    p = np.zeros((len(gamma),npanels))
    for m in range(len(gamma)):
        for h in range(gamma.shape[1]):
            p[m,panels[m,h]] += gamma[m,h]
    tau = gamma.shape[1]/(gamma.shape[1]-1.0)*(1.0-same.sum(axis=1))
    tau[0] = 0.
    return p, tau

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def window_records(g: np.ndarray, post: np.ndarray, tau: np.ndarray, width: \
    float, min_markers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import math
    import numpy as np
    from scipy.special import logsumexp
    g = _array(g, 1); post = _array(post, 3); tau = _array(tau, 2)
    width = float(_array(width, 0)); min_markers = float(_array(min_markers, 0))
    if not (post.shape[:2] == tau.shape and post.shape[1] == len(g)):
        raise ValueError('input outside the declared domain')
    if not (np.all(np.diff(g) > 0) and len(g) >= 2):
        raise ValueError('input outside the declared domain')
    if not (np.all(post >= 0) and np.allclose(post.sum(2), 1, atol=1e-9)):
        raise ValueError('input outside the declared domain')
    if not (np.all(tau >= -1e-9) and np.isfinite(width) and width > 0):
        raise ValueError('input outside the declared domain')
    if not (min_markers >= 2 and int(min_markers) == min_markers):
        raise ValueError('input outside the declared domain')
    min_markers = int(min_markers)

    g = np.asarray(g, float)
    post, tau = np.asarray(post), np.asarray(tau)
    ends = []
    threshold = float(g[0]+width)
    for m in range(len(g)):
        if g[m] > threshold:
            ends.append(m)
            threshold += width
    ends.append(len(g))
    probs, rates, ids, spans = [], [], [], []
    start = 0
    for w, end in enumerate(ends):
        if end-start >= min_markers:
            pp, rr = [], []
            for q in range(post.shape[0]):
                v = [sum(float(x) for x in post[q,start:end,j])/(end-start) for j in \
                    range(post.shape[2])]
                pp.append([float(np.rint(x*1000.))/1000. for x in v])
                rv = sum(float(x) for x in tau[q,start:end])*100./(g[end-1]-g[start])
                # The benchmark keeps the rate in binary64.
                rr.append(float(rv))
            probs.append(pp); rates.append(rr); ids.append(w); spans.append((start,end))
        start = end
    if not (len(probs) > 0):
        raise ValueError('input outside the declared domain')
    return np.array(probs), np.array(rates), np.array(ids,int), np.array(spans,int)

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def _coordinates(profiles):
    a = np.asarray(profiles, float).reshape(-1, np.shape(profiles)[-1])
    drop = int(a.sum(axis=0).argmax())
    return np.delete(a, drop, axis=1), drop

def _fit(x, starts, cycles=60, ridge=1e-6):
    x = np.asarray(x, float)
    starts = np.asarray(starts, int)
    n, d = x.shape; k = starts.shape[1]
    v = x-x.mean(axis=0)
    base = v.T@v/n+ridge*np.eye(d)
    def evaluate(pi, means, covs):
        logp = np.empty((n,k))
        for a in range(k):
            v = x-means[a]
            sign, ld = np.linalg.slogdet(covs[a])
            if sign <= 0: raise ValueError('non-positive covariance')
            quad = np.einsum('ij,ji->i',v,np.linalg.solve(covs[a],v.T))
            logp[:,a] = math.log(pi[a])-.5*(d*math.log(2*math.pi)+ld+quad)
        norm = logsumexp(logp,axis=1)
        return np.exp(logp-norm[:,None]),float(norm.sum())
    fits = []
    for start in starts:
        pi = np.full(k,1./k); means = x[start].copy()
        covs = np.repeat(base[None],k,axis=0)
        for _ in range(cycles):
            resp, _ = evaluate(pi,means,covs)
            nk = resp.sum(axis=0)
            if np.any(nk <= 1e-12): raise ValueError('empty numerical component')
            pi = nk/n; means = resp.T@x/nk[:,None]
            for a in range(k):
                v = x-means[a]
                covs[a] = (v.T*resp[:,a])@v/nk[a]+ridge*np.eye(d)
        resp,ll = evaluate(pi,means,covs)
        fits.append((ll,means.copy(),resp.copy()))
    ll = np.array([v[0] for v in fits])
    # Task tie tolerance makes numerically identical restarts reproducible.
    eligible = np.flatnonzero(ll >= ll.max()-1e-9)
    win = int(eligible[0]); _, means, resp = fits[win]
    return means, resp, ll, win

def fit_profiles(profiles: np.ndarray, starts: np.ndarray, cycles: int = 80, \
    ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    import math
    import numpy as np
    from scipy.special import logsumexp
    profiles = _array(profiles, 3); st = _array(starts, 2)
    cycles = float(_array(cycles, 0)); ridge = float(_array(ridge, 0))
    if not (profiles.shape[2] >= 2 and np.all((profiles >= 0) & (profiles <= 1))):
        raise ValueError('input outside the declared domain')
    if not (np.allclose(profiles.sum(2), 1, atol=.0005*profiles.shape[2]+1e-9)):
        raise ValueError('input outside the declared domain')
    if not (st.shape[1] >= 2 and np.all(st == np.floor(st))):
        raise ValueError('input outside the declared domain')
    if not (np.all((st >= 0) & (st < profiles.shape[0]*profiles.shape[1]))):
        raise ValueError('input outside the declared domain')
    if not (all(len(set(row)) == len(row) for row in st)):
        raise ValueError('input outside the declared domain')
    if not (cycles >= 1 and int(cycles) == cycles and np.isfinite(ridge) and ridge > 0):
        raise ValueError('input outside the declared domain')
    cycles = int(cycles)

    x, drop = _coordinates(profiles)
    means, resp, ll, win = _fit(x,starts,cycles,ridge)
    return means, resp, ll, win, drop

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def ancestry_continuity(labels: np.ndarray, n_windows: int, n_targets: int, k: \
    int) -> np.ndarray:
    import math
    import numpy as np
    from scipy.special import logsumexp
    values = _array(labels)
    n_windows, n_targets, k = [float(_array(v, 0)) for v in (n_windows, n_targets, k)]
    if not (n_windows >= 3 and n_targets >= 1 and k >= 2):
        raise ValueError('input outside the declared domain')
    if not (all(int(v) == v for v in (n_windows, n_targets, k))):
        raise ValueError('input outside the declared domain')
    if not (values.size == n_windows*n_targets and np.all(values == np.floor(values))):
        raise ValueError('input outside the declared domain')
    if not (np.all((values >= 0) & (values < k))):
        raise ValueError('input outside the declared domain')
    n_windows, n_targets, k = map(int, (n_windows, n_targets, k))

    lab = np.asarray(labels,int).reshape(n_windows,n_targets)
    out = []
    for a in range(k):
        x = (lab[:-1].T.ravel()==a).astype(float)
        y = (lab[1:].T.ravel()==a).astype(float)
        xx=x-x.mean(); yy=y-y.mean()
        den=math.sqrt(float(xx@xx)*float(yy@yy))
        if den == 0.: raise ValueError('undefined continuity statistic')
        out.append(float(xx@yy)/den)
    return np.r_[out,min(out)]

def initial_model(
    means: np.ndarray,
    resp: np.ndarray,
    drop: int,
    rates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    import math
    import numpy as np
    from scipy.special import logsumexp

    means = _array(means, 2)
    resp = _array(resp, 2)
    rv = _array(rates)
    drop = float(_array(drop, 0))

    if not (
        means.shape[0] == resp.shape[1]
        and resp.shape[0] == rv.size
    ):
        raise ValueError('input outside the declared domain')

    if not (
        np.all(means >= 0)
        and np.all(means.sum(axis=1) <= 1)
    ):
        raise ValueError('input outside the declared domain')

    if not (
        np.all(resp >= 0)
        and np.allclose(resp.sum(axis=1), 1, atol=1e-9)
    ):
        raise ValueError('input outside the declared domain')

    if not (
        np.all(rv > 0)
        and 0 <= drop <= means.shape[1]
        and int(drop) == drop
    ):
        raise ValueError('input outside the declared domain')

    if len(set(resp.argmax(axis=1))) != len(means):
        raise ValueError('input outside the declared domain')

    drop = int(drop)

    means = np.asarray(means)
    resp = np.asarray(resp)

    p = np.insert(
        means,
        drop,
        1.0 - means.sum(axis=1),
        axis=1
    )

    labels = resp.argmax(axis=1)
    rv = np.asarray(rates).ravel()

    rho = np.array([
        rv[labels == i].mean()
        for i in range(len(p))
    ])

    if (
        not np.isfinite(rho).all()
        or np.any(rho <= 0)
        or np.any(p <= 0)
    ):
        raise ValueError('invalid model parameters')

    # Ancestry IDs are canonicalized by complete P rows.
    order = np.array(
        sorted(
            range(len(p)),
            key=lambda i: tuple(p[i])
        ),
        dtype=int
    )

    return p[order], rho[order]

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def ancestry_posterior(g: np.ndarray, mismatch: np.ndarray, panels: np.ndarray \
    , counts: np.ndarray, p: np.ndarray, rho: np.ndarray, mu: np.ndarray, time: float) \
    -> np.ndarray:
    import math
    import numpy as np
    from scipy.special import logsumexp
    g = _array(g, 1); z0 = _array(mismatch, 2); j0 = _array(panels, 2)
    c0 = _array(counts, 1); p = _array(p, 2); rho = _array(rho, 1); mu = _array(mu, 1)
    time = float(_array(time, 0))
    if not (z0.shape == j0.shape and len(g) == len(z0)):
        raise ValueError('input outside the declared domain')
    if not (np.all(np.diff(g) > 0) and np.all((z0 == 0) | (z0 == 1))):
        raise ValueError('input outside the declared domain')
    if not (np.all(j0 == np.floor(j0)) and np.all((j0 >= 0) & (j0 < len(c0)))):
        raise ValueError('input outside the declared domain')
    if not (np.all(c0 >= 1) and np.all(c0 == np.floor(c0)) and c0.sum() >= 2):
        raise ValueError('input outside the declared domain')
    if not (p.shape == (len(rho), len(c0)) and len(mu) == len(rho)):
        raise ValueError('input outside the declared domain')
    if not (np.all(p >= 0) and np.allclose(p.sum(1), 1, atol=1e-9)):
        raise ValueError('input outside the declared domain')
    if not (np.all(rho > 0) and np.all(mu >= 0) and abs(mu.sum()-1) <= 1e-9):
        raise ValueError('input outside the declared domain')
    if not (np.isfinite(time) and time > 0):
        raise ValueError('input outside the declared domain')

    g = np.asarray(g,float); z = np.asarray(mismatch,int)
    panels = np.asarray(panels,int); p = np.asarray(p,float)
    rho, mu = np.asarray(rho,float), np.asarray(mu,float)
    mcount,hcount = z.shape; k = len(p)
    nref = sum(counts)
    lam = 1./(math.log(nref)+.5); theta = lam/(2*(lam+nref))
    em = np.where(z,theta,1-theta)
    dg = np.r_[0.,np.diff(g)]*.01
    t = -np.expm1(-time*dg)
    r = -np.expm1(-rho[:,None]*dg[None,:])
    stay = (1.-t)[None,:]*(1.-r)
    within = (1.-t)[None,:]*r
    q = p/np.asarray(counts)[None,:]
    f = np.empty((mcount,k,hcount)); b = np.ones_like(f)
    f[0] = mu[:,None]*q[:,panels[0]]*em[0][None,:]
    if not (f[0].sum() > 0):
        raise ValueError('input outside the declared domain')
    f[0] /= f[0].sum()
    for m in range(1,mcount):
        shift = (t[m]*mu + within[:,m]*f[m-1].sum(axis=1))[:,None]*q[:,panels[m]]
        f[m] = (stay[:,m,None]*f[m-1]+shift)*em[m][None,:]
        if not (f[m].sum() > 0):
            raise ValueError('input outside the declared domain')
        f[m] /= f[m].sum()
    for m in range(mcount-2,-1,-1):
        v = b[m+1]*em[m+1][None,:]
        a = (v*q[:,panels[m+1]]).sum(axis=1)
        shift = t[m+1]*np.dot(mu,a)+within[:,m+1]*a
        b[m] = stay[:,m+1,None]*v+shift[:,None]
        b[m] /= b[m].sum()
    gamma = f*b
    gamma /= gamma.sum(axis=(1,2))[:,None,None]
    return gamma

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def update_copying(gammas: list[np.ndarray], panels: list[np.ndarray], old_p: \
    np.ndarray) -> np.ndarray:
    import math
    import numpy as np
    from scipy.special import logsumexp
    old_p = _array(old_p, 2)
    if not (len(gammas) == len(panels) and len(gammas) > 0):
        raise ValueError('input outside the declared domain')
    if not (np.all(old_p >= 0) and np.allclose(old_p.sum(1), 1, atol=1e-9)):
        raise ValueError('input outside the declared domain')
    for a0, j0 in zip(gammas, panels):
        a0 = _array(a0, 3); j0 = _array(j0, 2)
        if not (a0.shape[0] == j0.shape[0] and a0.shape[2] == j0.shape[1]):
            raise ValueError('input outside the declared domain')
        if not (a0.shape[1] == len(old_p) and np.all(a0 >= 0)):
            raise ValueError('input outside the declared domain')
        if not (np.allclose(a0.sum((1,2)), 1, atol=1e-9)):
            raise ValueError('input outside the declared domain')
        if not (np.all(j0 == np.floor(j0)) and np.all((j0 >= 0) & (j0 < old_p.shape[1] \
            ))):
            raise ValueError('input outside the declared domain')

    out = np.zeros_like(np.asarray(old_p,float))
    for gamma, panel in zip(gammas,panels):
        gamma = np.asarray(gamma,float); panel = np.asarray(panel,int)
        for m in range(len(gamma)):
            for h in range(gamma.shape[2]):
                out[:,panel[m,h]] += gamma[m,:,h]
    for i in range(len(out)):
        total = out[i].sum()
        out[i] = old_p[i] if total == 0 else out[i]/total
    return out

import math
import numpy as np
from scipy.special import logsumexp

def _array(value, ndim=None):
    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('numeric input required') from exc
    if not np.isfinite(a).all() or a.size == 0:
        raise ValueError('finite nonempty input required')
    if ndim is not None and a.ndim != ndim:
        raise ValueError('wrong array dimension')
    return a

def _require(condition):
    if not condition:
        raise ValueError('input outside the declared domain')

def _fixture():
    return {'map_cm': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, \
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, \
        1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, \
        1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6, \
        2.65, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95, 3.0, 3.05, 3.1, 3.15, 3.2, 3.25, 3.3, \
        3.35, 3.4, 3.45, 3.5, 3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95, 4.0, \
        4.05, 4.1, 4.15, 4.2, 4.25, 4.3, 4.35, 4.4, 4.45, 4.5, 4.55, 4.6, 4.65, 4.7, \
        4.75, 4.8, 4.85, 4.9, 4.95], 'states': [{'target': 0, 'state': 0, 'panel': (
        '3222222222333333333300000000002222222222111111332122222222'
        '221111333333222221111133333333323333333322'
    ), 'mismatch': (
        '0111100001111111000001110110011000100001011001110001100111'
        '101111111100111000110010111111001010110100'
    )}, {'target': 0, 'state': 1, 'panel': (
        '3333333333223333333322333333333333333333333333333333333333'
        '331111111122333333333323333322223333333333'
    ), 'mismatch': (
        '0111111101111101111110011011100000010101100001011111010111'
        '111011111011111010111110010110011101010010'
    )}, {'target': 0, 'state': 2, 'panel': (
        '2333333311333333333323333222222222221111333333133311222222'
        '222222222233113333222311111111333333333222'
    ), 'mismatch': (
        '0010110011111111001101101111101110110011111111000011111001'
        '011010111110100111101001101111001010011011'
    )}, {'target': 0, 'state': 3, 'panel': (
        '2222222222333333322233333333331111111111111111111133333222'
        '222222222222333331111122222222222222222222'
    ), 'mismatch': (
        '0011100101001011011001101110010000011111011101110010111001'
        '000001101001000110101111111111011110101110'
    )}, {'target': 0, 'state': 4, 'panel': (
        '1111111033222222222233332222222222222222222222233311331111'
        '331113333333333333333323333333331112222222'
    ), 'mismatch': (
        '1001101110111010110000110111101100111011111110001110100010'
        '111101010111111101001110101101011101111110'
    )}, {'target': 0, 'state': 5, 'panel': (
        '2222222231333333333333333311112222111111333333300033333333'
        '333333333333000000000023333333332222222222'
    ), 'mismatch': (
        '1111001011111100011101000001111101101001111110101110011100'
        '010111011100100111110111111000111001111111'
    )}, {'target': 0, 'state': 6, 'panel': (
        '2333333333222222333300000000002222111111111111111111111331'
        '133333311112333331333333333333331113333333'
    ), 'mismatch': (
        '1101111111100000101110110010010101111010100110110111101011'
        '000110011101111101100110000111110111110010'
    )}, {'target': 1, 'state': 0, 'panel': (
        '0000011111000000000022222222220000000000333333333333322222'
        '223331112222333333333322222222222222222233'
    ), 'mismatch': (
        '0011001111101001000111111101010101011111111100010101011110'
        '011010111011101110101000110110010010110111'
    )}, {'target': 1, 'state': 1, 'panel': (
        '1111111111111011100000000000000000000000000000000011113333'
        '331111111111111111111233333333332333333333'
    ), 'mismatch': (
        '0001110110010110011101111100011100110011011100000110010111'
        '011001011000111011010111011101101110110111'
    )}, {'target': 1, 'state': 2, 'panel': (
        '0000000000000000000000000111110000000000000000031133311111'
        '112222222222333333332222222233332222222222'
    ), 'mismatch': (
        '0010001111011111111111001000011001111101111010101011111111'
        '000111101100011111011001001111101111111110'
    )}, {'target': 1, 'state': 3, 'panel': (
        '0000000000333333111100333333331111111000111111111133333333'
        '333333333333333311111133333333332222222222'
    ), 'mismatch': (
        '1101111111001111010010101011101101100111100111110110010111'
        '101111110111100111101101011110001111111100'
    )}, {'target': 1, 'state': 4, 'panel': (
        '0000000000100000000033333333300000000000110000000000000000'
        '022233333333222223333333111111112222222223'
    ), 'mismatch': (
        '0001101111000110111011101110111110100011111011111011101110'
        '110011011110110101100101100001111001101001'
    )}, {'target': 1, 'state': 5, 'panel': (
        '1111222222000000000011111111110000000000000000000033333333'
        '331113333333222222233133333333331333333333'
    ), 'mismatch': (
        '1010111001111100111001110100110001110011010101011111011101'
        '001110000111110101111011011111010111010111'
    )}, {'target': 1, 'state': 6, 'panel': (
        '2222222220000000000011111111110000000000000000000011122222'
        '222222223311220000122233333333332222222222'
    ), 'mismatch': (
        '1001010010101001011000111011110110010010011010011110011100'
        '110100011011011110001010000110010110011000'
    )}, {'target': 1, 'state': 7, 'panel': (
        '1111111111000000000000000000000000001111111111111033333333'
        '332333333333111111122211111111111111111110'
    ), 'mismatch': (
        '1110101011100111100110000010010101010101110111011110111010'
        '001100011011100110110110111111111110011011'
    )}, {'target': 2, 'state': 0, 'panel': (
        '1111111222033311111111111111111000000001111111112222111111'
        '111111000000111111111100000000011111110000'
    ), 'mismatch': (
        '1111101011001110100111010010001001111010010110010011000010'
        '110011011100101101001101011110111110110100'
    )}, {'target': 2, 'state': 1, 'panel': (
        '2233333333000000000022211111112222211100222211111122000000'
        '000000000000000000000000000233300000000002'
    ), 'mismatch': (
        '1101010000101110001011110001010111111111010111011011101011'
        '110011011110111010110011011110111101001000'
    )}, {'target': 2, 'state': 2, 'panel': (
        '0000000001111111002222222222222222222222111111111122211111'
        '111111111111111111111120000000000001111111'
    ), 'mismatch': (
        '0101110111011111100111001100100011110011111111110101111101'
        '011100001011111111111100101100100010111101'
    )}, {'target': 2, 'state': 3, 'panel': (
        '1111111111100000002210000000002222222222111111111100011111'
        '110000000000111111000000000000000001100000'
    ), 'mismatch': (
        '1100011110000111110111101110011110000001111011010100010100'
        '101110111111110111011101011111100101101101'
    )}, {'target': 2, 'state': 4, 'panel': (
        '1111111111222202222211111111110000000000000000000011111111'
        '000000002222111111110003333333332000000000'
    ), 'mismatch': (
        '0110011111011001110111110011100110110110101011111111100110'
        '110111110100011110111101111011001111011000'
    )}, {'target': 2, 'state': 5, 'panel': (
        '1111111111222222222202222222002222210000222222222201111222'
        '230000000000111111111100000000110000000000'
    ), 'mismatch': (
        '1010010001111101101101011101011110100011010111111001011000'
        '111001011100011110110111111001111100101011'
    )}, {'target': 2, 'state': 6, 'panel': (
        '3331111222222000000000000111110000000022000000002211111111'
        '110001100000000000000000000000000000000000'
    ), 'mismatch': (
        '0110110111111000101101001100110111111011010101110100010010'
        '101111111111111111001011110010111110000111'
    )}, {'target': 2, 'state': 7, 'panel': (
        '0000000000011111111200000000002221110001222222222211111111'
        '110000000000000001000022200000000000333300'
    ), 'mismatch': (
        '0101111010010110101110101111011110101011001101111010001100'
        '011111111111011011011001110101000100111101'
    )}, {'target': 2, 'state': 8, 'panel': (
        '2222222222000000001100000000001111111111222333333302222222'
        '220111111111001111111100000000001111111111'
    ), 'mismatch': (
        '0100111110011110111101011100111001111111011011101100001111'
        '001001101111010111100000001111001101011111'
    )}, {'target': 3, 'state': 0, 'panel': (
        '0000000000111113333212222222222222222222332222220022222222'
        '220000000000111100000000000000000000000000'
    ), 'mismatch': (
        '0011110000011001110011111111110111011111111100000010110100'
        '001010111100011011100001100111111001111000'
    )}, {'target': 3, 'state': 1, 'panel': (
        '0000000000222222222211222222223333333333000333333322222222'
        '220000000000111111100001111111110000000000'
    ), 'mismatch': (
        '1100111011110011111101100100000110111111011001111010011000'
        '011101100001111101111111111101100110000111'
    )}, {'target': 3, 'state': 2, 'panel': (
        '0000111111333333333333333333333222233333000000000022222233'
        '330011000000000000000100000000000000000000'
    ), 'mismatch': (
        '1101010101001100011101101001111111011111010101100100010101'
        '111110101101000101001001010011111001110100'
    )}, {'target': 3, 'state': 3, 'panel': (
        '0000000000333333333222222222223333333333333333333333333333'
        '330000000000000000000000000000000000111110'
    ), 'mismatch': (
        '0101000001101011110010110011111111011000110111111000010101'
        '101010011110111101001011010111111100001100'
    )}, {'target': 3, 'state': 4, 'panel': (
        '0000010000222222222222222222223333333333333333333332222222'
        '220011111111222222222200000000003322222222'
    ), 'mismatch': (
        '0111011101110010100101101011111011011111101001111011010000'
        '000110000001011101100011111011010010111010'
    )}, {'target': 3, 'state': 5, 'panel': (
        '3333333300333333333333333311113333333333111113333333333333'
        '331111111111111111111122222222000000000000'
    ), 'mismatch': (
        '0110001111100000100110111000010110001111111000010101001000'
        '011100111000100010011110001010111000010100'
    )}, {'target': 3, 'state': 6, 'panel': (
        '0000000011222223333322222222222222223333333333333333333333'
        '330111111000000000000000000000000000000000'
    ), 'mismatch': (
        '1111110101100101101010010110110001110111001111010101010101'
        '001110110011010100010101111011110100011011'
    )}, {'target': 4, 'state': 0, 'panel': (
        '1111111111000000011111111122220000000000000011111100000000'
        '001111222223222222111102211111101110222222'
    ), 'mismatch': (
        '1111101010111001111100111010100001010111110011110010100010'
        '100010001011000101110110110011111001101011'
    )}, {'target': 4, 'state': 1, 'panel': (
        '1111111111000000011122222222222222200000322222222100000000'
        '000000000111000000000200001100331111111111'
    ), 'mismatch': (
        '0111000110111111111110101101111111101011000101011111011110'
        '011010011101011111110111110110010100110000'
    )}, {'target': 4, 'state': 2, 'panel': (
        '1111111133111111111100011100000000000000110000000000000000'
        '001111111111222222222121111222222222222222'
    ), 'mismatch': (
        '1001011011110111010011111010100001011000001111111110101111'
        '111001100111001101011111110011111111110101'
    )}, {'target': 4, 'state': 3, 'panel': (
        '0000000000000000000000000000001111111111111100000011111111'
        '111112222222002222222211111111110010111111'
    ), 'mismatch': (
        '1101001100101000011111001100011000000010000011111101001001'
        '000111101110101000100101110000010010001110'
    )}, {'target': 4, 'state': 4, 'panel': (
        '0000000000000000000000003000000000000000000000011100000000'
        '002222222222111111111122222222221111111111'
    ), 'mismatch': (
        '1011111011000000100111101111110100011011000111101100000110'
        '001001110001111101101011000000101111011011'
    )}, {'target': 4, 'state': 5, 'panel': (
        '2000000000000000000011111111110000100000111111111100000000'
        '221111111111222222222200000000001111222222'
    ), 'mismatch': (
        '1101010110101010001011010101000001110110110101111111011011'
        '111011011010001001011101010010111010101010'
    )}, {'target': 4, 'state': 6, 'panel': (
        '1111111111000000000000000011110000000000111111111100000000'
        '000000111111222222222210000000001111000000'
    ), 'mismatch': (
        '0110111111111110111101100010100101111110100011101011110000'
        '100110000100011111110111000011011111110101'
    )}, {'target': 4, 'state': 7, 'panel': (
        '0000000022333333333300000000000111100000000000000000000000'
        '021111111111000000000011000000002222222222'
    ), 'mismatch': (
        '1110101110110100110100001111010111101110111100110101010101'
        '100110010010001011101101011001011101000001'
    )}, {'target': 5, 'state': 0, 'panel': (
        '0000000000111111111122222221221111111111222222000111111111'
        '110000000000333333221122222111113333333333'
    ), 'mismatch': (
        '1001110101110010110111011110101000100100101111010011111010'
        '110110101001001001000111001011100010100110'
    )}, {'target': 5, 'state': 1, 'panel': (
        '0000000000000000000211111111121111111111000000000111111111'
        '121111111111111111111022222222223333333333'
    ), 'mismatch': (
        '1011101110010000111101101100010010111000010110111011100110'
        '110111110110001110101111011001010011111000'
    )}, {'target': 5, 'state': 2, 'panel': (
        '2222222222111111110022222222222222222220111111100033000000'
        '002222200001222222222233332222222111222222'
    ), 'mismatch': (
        '1001101110111110010001001010111100100111111110000001110101'
        '010001100101111110111111010111001001111010'
    )}, {'target': 5, 'state': 3, 'panel': (
        '0000022221222222222222222222221111111111000033322211111111'
        '111111111001333333333322222222223333333333'
    ), 'mismatch': (
        '1101011011101111111110110111011111111011101111110001110010'
        '000111010101100111100111010011001111101010'
    )}, {'target': 5, 'state': 4, 'panel': (
        '2222222222222222222200000000021111111100111111111101111112'
        '221111111222222221111122222222333333333333'
    ), 'mismatch': (
        '1100100110110110111100110000101111111111110010110010110110'
        '101001101011110111110110011100101101101101'
    )}, {'target': 5, 'state': 5, 'panel': (
        '0000000000333333333321100000001111111000220022222211111111'
        '111111111111222222222211111111123222222222'
    ), 'mismatch': (
        '1011011000101100101011111100011001110010110111111011011100'
        '100011101100110101110111001111010100010111'
    )}, {'target': 5, 'state': 6, 'panel': (
        '2220222222220000000022222222220222222111221111111133333333'
        '300022222211223222222233333333332222222222'
    ), 'mismatch': (
        '1101011111110001101100010010000010001111011110011001101111'
        '011001010100101100111101111011110011111100'
    )}, {'target': 5, 'state': 7, 'panel': (
        '0000000022222222222200002222330000000010022222000011110000'
        '002222222222222231111121111111112222222223'
    ), 'mismatch': (
        '1110110110101110011111110011010111001100001101110111010011'
        '011110101001010011100011111111101111011111'
    )}, {'target': 5, 'state': 8, 'panel': (
        '2222222222222222222200000000001111111111111111111122222333'
        '330000000000333333333333333222222232111222'
    ), 'mismatch': (
        '0011111110111101001101101011101001101100110110111111111100'
        '111100110001101110010111011000000000010010'
    )}, {'target': 6, 'state': 0, 'panel': (
        '2222200000000000000000000000011111111110112222222200000000'
        '000000002220111111111111111111112222111111'
    ), 'mismatch': (
        '1111101100000001000111100101111101000111101010101110110011'
        '100111010010111000100110100111101111110100'
    )}, {'target': 6, 'state': 1, 'panel': (
        '1100000000000000000000000000000000011111001000000000000222'
        '110000000000000000000022222222221111110003'
    ), 'mismatch': (
        '1111011011011000000011111111011101100111110001101100011000'
        '100010110001111111110110100011010010110001'
    )}, {'target': 6, 'state': 2, 'panel': (
        '0000000000333220000000000000001111111111110022222100011111'
        '110000000000000022222200000000002000000002'
    ), 'mismatch': (
        '1000101011100100101110111000010110010111101010111110011000'
        '101011110111110110011111001101100111111111'
    )}, {'target': 6, 'state': 3, 'panel': (
        '0000000000111111111133333000000000000111111211111100000111'
        '110000000000000000000022222222220000000000'
    ), 'mismatch': (
        '1111111011011100010011110101101111111110111111101111000000'
        '111110111011110101110111111111111111000111'
    )}, {'target': 6, 'state': 4, 'panel': (
        '1111111111000000000011111111111111110000000001111100000000'
        '001111110000000000000022222222222222200000'
    ), 'mismatch': (
        '1110010001001111101011010011110110011110000011100101011011'
        '100010010111011100111101111100111110010111'
    )}, {'target': 6, 'state': 5, 'panel': (
        '1111111000000000000022222201110000000000000000222200000000'
        '003333000000000000000000222222222221111111'
    ), 'mismatch': (
        '0111101111100111011110000010111110111101111100011000001010'
        '101101101010101101111111011111001101001101'
    )}, {'target': 6, 'state': 6, 'panel': (
        '1111111111000000200022222000000000000000000000000000000000'
        '000000000000300222222200000000000000000000'
    ), 'mismatch': (
        '1101011101101101011101001001111000000000111110100111100111'
        '010111001011011101110001110111110000111110'
    )}, {'target': 7, 'state': 0, 'panel': (
        '2222222111000000000000000000002200000000000000001000000000'
        '000000000000000000002222222222223333323333'
    ), 'mismatch': (
        '1111100100011010111000100011101100111111010101100111101111'
        '110001011001111100111011101100010110010100'
    )}, {'target': 7, 'state': 1, 'panel': (
        '2222333333000000000000000000002200000000222222222200001111'
        '111111111111111111111100000000022222222222'
    ), 'mismatch': (
        '0111010101101100010101111101000000011000111101110011101011'
        '101011111111011101011111101001101111011111'
    )}, {'target': 7, 'state': 2, 'panel': (
        '3332222333111111100000000000000000000000011110000000001000'
        '000000111111111111111111111100002222222222'
    ), 'mismatch': (
        '1110110001011111100111011111111111100001011110011111110010'
        '101011100110011001011100111011100001110111'
    )}, {'target': 7, 'state': 3, 'panel': (
        '3333333333111110000000000000000000000000000000000000000000'
        '000000000000100000000111111111113333333333'
    ), 'mismatch': (
        '0000000111110110110111010101110011011111011100111000111010'
        '110001111001001111101111011111110011101110'
    )}, {'target': 7, 'state': 4, 'panel': (
        '3333333333000000000000000000000000000000000000000030000111'
        '001000111111000000000000111111112222222233'
    ), 'mismatch': (
        '0100111111010011111100110101111110001001101010010001100101'
        '001011111101110110101111110011111110111111'
    )}, {'target': 7, 'state': 5, 'panel': (
        '3333333333111111110022220000001111111222000000000000112222'
        '220000000000222333333322222222213333333333'
    ), 'mismatch': (
        '0111110110110001111111111111111110000111111101111101101111'
        '111111111011111010110110111100011101010010'
    )}, {'target': 7, 'state': 6, 'panel': (
        '3333332333300000111100000000001111111111000000111222220000'
        '000000000000000000000011111100003333333222'
    ), 'mismatch': (
        '1101001111101101010101100011100110111011111101111101000011'
        '110000101000001011111100011110101001101110'
    )}, {'target': 7, 'state': 7, 'panel': (
        '2222222222000000011111111111000000022222000000333000000000'
        '000000000000110000000011111111113333333333'
    ), 'mismatch': (
        '0011011011101011111111001100111100001000011110100110110010'
        '101011111111101111001001001011111100001011'
    )}, {'target': 8, 'state': 0, 'panel': (
        '0000222222000000000000000000000000000002000000011122222222'
        '221111111111220000000011111111101111111111'
    ), 'mismatch': (
        '1000001110001110100100010100100101101100011101111101011111'
        '100010011100010011100111010110000100111011'
    )}, {'target': 8, 'state': 1, 'panel': (
        '0000000000000000000000000000000000000000000000000033333333'
        '330000000000000011111133333333220000000000'
    ), 'mismatch': (
        '0010000001111010110100111111101001110101011111011100010111'
        '101111011010011111111011011011011101110111'
    )}, {'target': 8, 'state': 2, 'panel': (
        '0000000000000000000200001000000000000011000000000011111111'
        '110000000222330000000011111111111111111111'
    ), 'mismatch': (
        '1010110101011011101110110011111101011010101011011101011110'
        '110110111101011010110110100001001110111000'
    )}, {'target': 8, 'state': 3, 'panel': (
        '0000000000333333330000000000110011110000000000000022222222'
        '221111111111111111111122222221331111111111'
    ), 'mismatch': (
        '0111111010010011100101101011111011101111001111001111001110'
        '001100110110111111111010100101111011110010'
    )}, {'target': 8, 'state': 4, 'panel': (
        '1111111111111222222200000000001111110000111111111111110000'
        '002222222233222222223311100000030000000000'
    ), 'mismatch': (
        '1010110110011001110111110101110111000110011110011011111101'
        '101000011110110110111110111101100011101000'
    )}, {'target': 8, 'state': 5, 'panel': (
        '2211111000111111111011111100001111111111000000000022222111'
        '111111111222111111111122222211110000000000'
    ), 'mismatch': (
        '1111101000000011111101011111111111110011001111101111101101'
        '101001001111001101001100101010101111101011'
    )}, {'target': 8, 'state': 6, 'panel': (
        '0000000000222222222200000002221111111111000011111122222211'
        '110011122211000000000011111111110000000000'
    ), 'mismatch': (
        '1011110101001111000010001011111110110100110101110111110111'
        '100011011110101001010110001101101111111111'
    )}, {'target': 8, 'state': 7, 'panel': (
        '0000000002000000000022000000001111111111111111111111111111'
        '111111111113122222222110033333303333333331'
    ), 'mismatch': (
        '1111010101111001001011111111001111111111111101100110101000'
        '111010111011001010111001111001111100111101'
    )}, {'target': 8, 'state': 8, 'panel': (
        '0000000020011111111100000000001111111111222220200011111111'
        '221111111122000000000011111111112222222211'
    ), 'mismatch': (
        '1011011011111111010110000011110001011101010010001011100001'
        '010110000010011011111001010100111111000001'
    )}, {'target': 9, 'state': 0, 'panel': (
        '2222223333000010001111111111110000001112112221222222222222'
        '222220000011111111111111110000002222211110'
    ), 'mismatch': (
        '1111110001010011001111110011110101101100101011100001111101'
        '001011101111011100100001010111111111101001'
    )}, {'target': 9, 'state': 1, 'panel': (
        '1111111122222222222221111111111100000000111111111111111111'
        '110000011111111100222201111100002222222222'
    ), 'mismatch': (
        '1010111110111010110001111110100111101011111001011101111011'
        '111011111110101011111100011000010011101111'
    )}, {'target': 9, 'state': 2, 'panel': (
        '2222222222000000000022222222220022222222222222100000000000'
        '000000111111222221111100011111110000000011'
    ), 'mismatch': (
        '0100010000010101111111111011011111111110110111110000101010'
        '110001001100110011010110011000001011100110'
    )}, {'target': 9, 'state': 3, 'panel': (
        '0000000000222222033322211111111111111111222200000210000000'
        '000000000000333333333311111111113333311111'
    ), 'mismatch': (
        '0111111101010101111110011011011001111011010100111111100101'
        '110111110000101111011110111011111101000101'
    )}, {'target': 9, 'state': 4, 'panel': (
        '2222211111000011111111112222220001111111333311111133333333'
        '331111111112111111111111111111111111111111'
    ), 'mismatch': (
        '0010000101111111110100110011100111101111111110010110000111'
        '100010110000101101011011100110111110101011'
    )}, {'target': 9, 'state': 5, 'panel': (
        '1111111112222222222223333333331122222200000000000022222222'
        '221111111111111111111100000001112222222222'
    ), 'mismatch': (
        '0110111011111110011011101000010111110111011101010001010011'
        '011111111011001100110101111001110110000011'
    )}, {'target': 9, 'state': 6, 'panel': (
        '0222222222000000001100000002220111111111000011111122222222'
        '220000000000100000000000000000000111110000'
    ), 'mismatch': (
        '1101101111000100011111110010101000010111010000011111101111'
        '100111011011011110000111110010101111100101'
    )}], 'counts': [310, 270, 240, 180], 'ne': 100000, 'width': 0.5, 'min_markers': 9, \
        'cycles': 80, 'ridge': 1e-06, 'ancestry': 0, 'threshold': 0.1, 'candidates': [ \
        {'k': 2, 'centers': [[17, 38], [24, 52], [22, 64], [97, 2], [52, 83], [34, 58] \
        , [94, 85], [74, 7], [4, 13], [57, 48]]}, {'k': 3, 'centers': [[99, 20, 70], [ \
        7, 63, 87], [68, 74, 54], [26, 11, 35], [87, 69, 89], [87, 90, 95], [99, 60, \
        13], [79, 53, 82], [32, 43, 96], [76, 17, 91]]}, {'k': 4, 'centers': [[66, 46, \
        75, 23], [34, 8, 66, 32], [25, 8, 49, 11], [23, 22, 71, 38], [79, 7, 29, 11], \
        [14, 47, 32, 19], [92, 35, 79, 10], [88, 75, 53, 41], [96, 29, 76, 94], [32, 9 \
        , 73, 28]]}]}

def solve_task(fixture: dict | None = None) -> float:
    import numpy as np
    d = _fixture() if fixture is None else fixture
    if not (isinstance(d, dict) and len(d.get('states', [])) > 0):
        raise ValueError('input outside the declared domain')
    g = d['map_cm']
    qnum = max(row['target'] for row in d['states'])+1
    if not (set(row['target'] for row in d['states']) == set(range(qnum))):
        raise ValueError('input outside the declared domain')
    if not (np.isfinite(d['threshold']) and -1 <= d['threshold'] <= 1):
        raise ValueError('input outside the declared domain')
    if not (len(d['candidates']) > 0):
        raise ValueError('input outside the declared domain')
    if not (len(set(c['k'] for c in d['candidates'])) == len(d['candidates'])):
        raise ValueError('input outside the declared domain')
    panels, mismatches, posts, taus = [], [], [], []
    for target in range(qnum):
        rows = sorted([r for r in d['states'] if r['target'] == target], key=lambda r: \
            r['state'])
        if not ([r['state'] for r in rows] == list(range(len(rows)))):
            raise ValueError('input outside the declared domain')
        if not (all(len(r['panel']) == len(g) == len(r['mismatch']) for r in rows)):
            raise ValueError('input outside the declared domain')
        panel = np.array([[int(v) for v in r['panel']] for r in rows]).T
        z = np.array([[int(v) for v in r['mismatch']] for r in rows]).T
        gamma, same = copying_state_posteriors(g,z,sum(d['counts']),d['ne'])
        post,tau = panel_statistics(gamma,same,panel,len(d['counts']))
        panels.append(panel); mismatches.append(z); posts.append(post); taus.append(tau)
    profiles,rates,ids,spans = window_records(g,posts,taus,d['width'],d[ \
        'min_markers'])
    eligible = []
    for candidate in d['candidates']:
        if not (all(len(row) == candidate['k'] for row in candidate['centers'])):
            raise ValueError('input outside the declared domain')
        means,resp,ll,win,drop = fit_profiles(profiles,candidate['centers'],d[ \
            'cycles'],d['ridge'])
        try:
            continuity = ancestry_continuity(resp.argmax(1),len(profiles),qnum \
                ,candidate['k'])
        except ValueError:
            continue
        if continuity[-1] >= d['threshold']:
            eligible.append((candidate['k'],means,resp,drop))
    if not (len(eligible) > 0):
        raise ValueError('input outside the declared domain')
    k,means,resp,drop = max(eligible,key=lambda item:item[0])
    p,rho = initial_model(means,resp,drop,rates)
    mu = np.full(k,1./k)
    before = [ancestry_posterior(g,z,j,d['counts'],p,rho,mu,10.)
              for z,j in zip(mismatches,panels)]
    updated = update_copying(before,panels,p)
    after = [ancestry_posterior(g,z,j,d['counts'],updated,rho,mu,10.)
             for z,j in zip(mismatches,panels)]
    return float(np.mean([post[:,0,:].sum(1) for post in after]))
SCICODE_GOLD_EOF
