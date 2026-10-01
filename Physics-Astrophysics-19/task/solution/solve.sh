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

def _indices(p):
    return [(a,b,d-a-b) for d in range(p+1)
            for a in range(d,-1,-1) for b in range(d-a,-1,-1)]

def _factorial(n):
    return math.prod(math.factorial(int(a)) for a in n)

def _binomial(n, k):
    return math.prod(math.comb(int(a), int(b)) for a,b in zip(n,k))

def _power(x, n):
    return np.prod(np.asarray(x) ** np.asarray(n))

def kernel_derivatives(r: "np.ndarray", epsilon: float, p: int) -> "np.ndarray":
    r = np.asarray(r, dtype=float)
    rho = np.dot(r, r) + epsilon**2
    radial = [-1.0 / np.sqrt(rho)]
    for q in range(1, p+1):
        radial.append(-(2*q-1) * radial[-1] / rho)
    values = []
    for n in _indices(p):
        value = 0.0
        for a in range(n[0]//2+1):
            for b in range(n[1]//2+1):
                for c in range(n[2]//2+1):
                    pairs = (a,b,c)
                    rest = tuple(n[j]-2*pairs[j] for j in range(3))
                    coefficient = _factorial(n) / (2**sum(pairs)*_factorial(pairs)*_factorial(rest))
                    value += coefficient * _power(r, rest) * radial[sum(n)-sum(pairs)]
        values.append(value)
    return np.array(values, dtype=float)

import numpy as np

def particle_moments(x: "np.ndarray", masses: "np.ndarray", center: "np.ndarray", p: int) -> "np.ndarray":
    offsets = np.asarray(x, dtype=float) - np.asarray(center)
    masses = np.asarray(masses, dtype=float)
    return np.array([np.sum(masses*np.prod(offsets**np.array(n),axis=1)) for n in _indices(p)])

import numpy as np

def translate_moments(moments: "np.ndarray", displacement: "np.ndarray", p: int) -> "np.ndarray":
    indices = _indices(p)
    result = np.zeros(len(indices))
    for i,n in enumerate(indices):
        for j,k in enumerate(indices):
            if all(k[a]<=n[a] for a in range(3)):
                result[i] += _binomial(n,k)*_power(displacement, np.subtract(n,k))*moments[j]
    return result

import numpy as np

def multipole_to_local(moments: "np.ndarray", displacement: "np.ndarray", epsilon: float, p: int, G: float) -> "np.ndarray":
    indices = _indices(p)
    lookup = {n:i for i,n in enumerate(indices)}
    derivatives = kernel_derivatives(displacement, epsilon, p)
    result = np.zeros(len(indices))
    for i,k in enumerate(indices):
        for j,n in enumerate(indices):
            if sum(k)+sum(n) <= p:
                result[i] += derivatives[lookup[tuple(np.add(k,n))]]*moments[j]/_factorial(n)
        result[i] *= G*(-1)**sum(k)/_factorial(k)
    return result

import numpy as np

def translate_locals(locals_in: "np.ndarray", displacement: "np.ndarray", p: int) -> "np.ndarray":
    indices = _indices(p)
    result = np.zeros(len(indices))
    for i,n in enumerate(indices):
        for j,k in enumerate(indices):
            if all(k[a]>=n[a] for a in range(3)):
                result[i] += _binomial(k,n)*_power(displacement,np.subtract(k,n))*locals_in[j]
    return result

import numpy as np

def evaluate_fixed_fmm(x: "np.ndarray", moments: "np.ndarray", tree: dict, epsilon: float, p: int, G: float) -> "np.ndarray":
    x = np.asarray(x, dtype=float)
    moments = np.asarray(moments, dtype=float)
    leaf = np.asarray(tree['leaf_ids'])
    parent = np.asarray(tree['parent_ids'])
    lc = np.asarray(tree['leaf_centers'])
    pc = np.asarray(tree['parent_centers'])
    size = len(_indices(p))
    parent_q = np.zeros((len(pc),size))
    for b in range(len(lc)):
        ids = np.flatnonzero(leaf == b)
        q = particle_moments(x[ids],moments[ids,0],lc[b],p)
        for i in ids:
            higher = moments[i].copy()
            higher[0] = 0.0
            q += translate_moments(higher,x[i]-lc[b],p)
        parent_q[parent[b]] += translate_moments(q,lc[b]-pc[parent[b]],p)
    parent_l = np.zeros_like(parent_q)
    for d in range(len(pc)):
        for s in range(len(pc)):
            if s != d:
                parent_l[d] += multipole_to_local(parent_q[s],pc[s]-pc[d],epsilon,p,G)
    result = np.zeros_like(moments)
    for b in range(len(lc)):
        local = translate_locals(parent_l[parent[b]],lc[b]-pc[parent[b]],p)
        for i in np.flatnonzero(leaf == b):
            result[i] = translate_locals(local,x[i]-lc[b],p)
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j and parent[leaf[i]] == parent[leaf[j]]:
                result[i] += multipole_to_local(moments[j],x[j]-x[i],epsilon,p,G)
    return result

import numpy as np

def pullback_fixed_fmm(x: "np.ndarray", masses: "np.ndarray", local_bar: "np.ndarray", tree: dict, epsilon: float, p: int, G: float) -> "np.ndarray":
    indices = _indices(p)
    lookup = {n:i for i,n in enumerate(indices)}
    q = np.zeros((len(x),len(indices)))
    q[:,0] = masses
    forward = evaluate_fixed_fmm(x,q,tree,epsilon,p,G)
    reverse = evaluate_fixed_fmm(x,local_bar,tree,epsilon,p,G)
    result = np.zeros((len(x),4))
    result[:,3] = reverse[:,0]
    for axis in range(3):
        e = tuple(int(a==axis) for a in range(3))
        result[:,axis] = np.asarray(masses)*reverse[:,lookup[e]]
        for j,n in enumerate(indices):
            raised = tuple(n[a]+e[a] for a in range(3))
            if sum(raised)<=p:
                result[:,axis] += (n[axis]+1)*local_bar[:,j]*forward[:,lookup[raised]]
    return result

import numpy as np

def _trajectory_adjoint(config):
    x = np.array(config['x'],dtype=float,copy=True)
    v = np.array(config['v'],dtype=float,copy=True)
    m = np.asarray(config['masses'],dtype=float)
    tree,eps,p,G = (config[k] for k in ('tree','epsilon','p','G'))
    h = config['h']
    q = np.zeros((len(x),len(_indices(p))))
    q[:,0] = m
    midpoints = []
    for _ in range(config['steps']):
        midpoint = x + .5*h*v
        local = evaluate_fixed_fmm(midpoint,q,tree,eps,p,G)
        v = v-h*local[:,1:4]
        x = midpoint+.5*h*v
        midpoints.append(midpoint)
    local = evaluate_fixed_fmm(x,q,tree,eps,p,G)
    residual = x-np.asarray(config['target'])
    weights = np.asarray(config['weights'])
    lam = config['lam']
    loss = .5*np.sum(weights[:,None]*residual**2)+.5*lam*np.dot(m,local[:,0])
    b = np.zeros_like(q)
    b[:,0] = .5*lam*m
    terminal = pullback_fixed_fmm(x,m,b,tree,eps,p,G)
    bx = weights[:,None]*residual+terminal[:,:3]
    bv = np.zeros_like(v)
    bm = .5*lam*local[:,0]+terminal[:,3]
    for midpoint in reversed(midpoints):
        bv_new = bv+.5*h*bx
        b = np.zeros_like(q)
        b[:,1:4] = -h*bv_new
        force_pullback = pullback_fixed_fmm(midpoint,m,b,tree,eps,p,G)
        bx = bx+force_pullback[:,:3]
        bm = bm+force_pullback[:,3]
        bv = bv_new+.5*h*bx
    derivative = np.sum(bx*config['dx'])+np.sum(bv*config['dv'])+np.dot(bm,config['dm'])
    return float(derivative),float(loss),bx,bv,bm,x,v

def trajectory_sensitivity(config: dict) -> float:
    return _trajectory_adjoint(config)[0]
SCICODE_GOLD_EOF
