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

def repair_axis(raw):
    out = np.array(raw, dtype=float, copy=True)
    if out.shape != (2,2,3,3,4) or not np.isfinite(out).all():
        raise ValueError('Expected finite four-cell Q2 stored states')
    out[0,:,1:,:,:] += out[0,:,:1,:,:]/5.0
    out[0,:,0,:,:] = 0.0
    return out

def _s1_fixture(constant=False):
    values = np.empty((2,2,3,3,4))
    x = np.array([0.,.5,1.])
    for a in range(2):
        for b in range(2):
            r,z = np.meshgrid(a+x,b+x,indexing='ij')
            u = np.stack([1+.1*z+.04*a+.01*b,.15*r,
                          .2+.05*z+.03*a,3+.2*z+.07*a+.11*b],axis=-1)
            if constant:
                u[:] = [1.,.1,.2,3.]
            values[a,b] = r[...,None]*u
    if not constant:
        values[0,0,0,1] = [.01,0,.005,.03]
        values[0,0,1,1] = [-.05,.8,.1,.65]
    return values

def _s1_noise(seed):
    return np.random.default_rng(seed).normal(size=(2,2,3,3,4))

import numpy as np

def regular_local_state(stored, radial_cell):
    w = np.asarray(stored,dtype=float)
    if w.shape != (3,3,4) or not np.isfinite(w).all() or radial_cell not in (0,1):
        raise ValueError('Invalid cell states or radial index')
    a = int(radial_cell)
    radius = a+np.array([0.,.5,1.])
    out = np.empty_like(w)
    if a == 0:
        if np.any(w[0] != 0):
            raise ValueError('Stored axis values must vanish')
        out[1:] = w[1:]/radius[1:,None,None]
        out[0] = -3*w[0]+4*w[1]-w[2]
    else:
        out[:] = w/radius[:,None,None]
    return out

def _s2_noise(seed):
    return np.random.default_rng(seed).normal(size=(3,3,4))

import numpy as np

def directional_jump_amplitudes(local_states):
    u = np.asarray(local_states,dtype=float)
    if u.shape != (2,2,3,3,4) or not np.isfinite(u).all():
        raise ValueError('Expected finite local patch states')
    derivative = np.array([[-3.,4.,-1.],[-1.,0.,1.],[1.,-4.,3.]])
    weights = np.array([1.,4.,1.])/6
    jumps = np.empty((2,2,3,2,4))
    for order in range(3):
        d = np.linalg.matrix_power(derivative,order)
        for a in range(2):
            for b in range(2):
                radial = np.einsum('i,ijk->jk',d[2],u[0,b])-np.einsum('i,ijk->jk',d[0],u[1,b])
                axial = np.einsum('j,ijk->ik',d[2],u[a,0])-np.einsum('j,ijk->ik',d[0],u[a,1])
                jumps[a,b,order,0] = np.sqrt(np.einsum('i,ik->k',weights,radial**2))
                jumps[a,b,order,1] = np.sqrt(np.einsum('i,ik->k',weights,axial**2))
    return jumps

def _s3_fixture():
    w = repair_axis(_s1_fixture())
    return np.array([[regular_local_state(w[a,b],a) for b in range(2)] for a in range(2)])

def _s3_step():
    u = np.zeros((2,2,3,3,4))
    u[1] = 2
    return u

import math

import numpy as np

def normalized_damping_rates(local_states, jump_amplitudes):
    u = np.asarray(local_states,dtype=float)
    jumps = np.asarray(jump_amplitudes,dtype=float)
    if (u.shape != (2,2,3,3,4) or jumps.shape != (2,2,3,2,4)
            or not np.isfinite(u).all() or not np.isfinite(jumps).all() or np.any(jumps < 0)):
        raise ValueError('Invalid local states or jumps')
    weights = np.array([1.,4.,1.])/6
    mass = weights[:,None]*weights[None,:]
    mean = np.einsum('ij,abijk->k',mass,u)/4
    amp = np.max(abs(u-mean),axis=(0,1,2,3))
    absolute = 1000*np.finfo(float).eps
    component = max(1e-8*np.max(np.linalg.norm(u,axis=-1)),absolute)
    rates = np.zeros((2,2,3))
    for a in range(2):
        for b in range(2):
            for direction in range(2):
                line = u[:,b] if direction == 0 else u[a,:]
                line_mean = np.einsum('ij,cijk->k',mass,line)/2
                denominator = np.maximum(np.max(abs(line-line_mean),axis=(0,1,2)),1e-6*amp)
                retained = (amp > component) & (denominator > absolute)
                for order in range(3):
                    values = np.zeros(4)
                    values[retained] = ((2*order+1)*jumps[a,b,order,direction,retained]
                                        /(2*math.factorial(order)*denominator[retained]))
                    rates[a,b,order] += 5*np.max(values)
    return rates

def _s4_flat_line():
    """One coordinate line nearly constant in m_r while the patch spans 3e3."""
    u = _s3_fixture()
    u[:,0,:,:,1] = 0.3
    u[:,0,0,0,1] = 0.3+3e-7
    u[:,1,:,:,1] = 3.0e3
    return u

def _s4_tall():
    u = _s1_noise(417)
    u[...,2] = 100
    return u

import numpy as np

def conservative_local_filter(local_states, rates, dt=0.02, strength=0.02):
    u = np.asarray(local_states,dtype=float)
    rates = np.asarray(rates,dtype=float)
    if (u.shape != (2,2,3,3,4) or rates.shape != (2,2,3)
            or not np.isfinite(u).all() or not np.isfinite(rates).all() or np.any(rates < 0)
            or not np.isfinite([dt,strength]).all() or min(dt,strength) < 0):
        raise ValueError('Invalid states, rates, timestep or strength')
    x = np.array([0.,.5,1.])
    weights = np.array([1.,4.,1.])/6
    mass = weights[:,None]*weights[None,:]
    xx,yy = np.meshgrid(x,x,indexing='ij')
    basis = np.stack([np.ones_like(xx),2*yy-1,2*xx-1,(2*xx-1)*(2*yy-1)],axis=-1).reshape(9,4)
    alpha = np.exp(-strength*dt*np.cumsum(rates,axis=-1)[...,1:])
    out = np.empty_like(u)
    for a in range(2):
        chi = np.broadcast_to((a+x)[:,None],(3,3))
        wm = (mass*chi).ravel()
        gram = basis.T@(wm[:,None]*basis)
        for b in range(2):
            values = u[a,b].reshape(9,4)
            p0 = np.sum(wm[:,None]*values,axis=0)/np.sum(wm)
            coefficients = np.linalg.solve(gram,basis.T@(wm[:,None]*values))
            p1 = (basis@coefficients).reshape(3,3,4)
            out[a,b] = chi[...,None]*(p0+alpha[a,b,0]*(p1-p0)+alpha[a,b,1]*(u[a,b]-p1))
    return out

def _s5_rates(u):
    return normalized_damping_rates(u,directional_jump_amplitudes(u))

import numpy as np

def _s6_geometry_anchor(stored, radial_cell):
    w = np.array(stored,dtype=float,copy=True)
    if (w.shape != (3,3,4) or not np.isfinite(w).all()
            or radial_cell not in (0,1)):
        raise ValueError('Invalid stored cell or radial index')
    a = int(radial_cell)
    if a == 0 and np.any(w[0] != 0):
        raise ValueError('Stored axis face must vanish')
    chi = np.broadcast_to((a+np.array([0.,.5,1.]))[:,None],(3,3))
    weights = np.array([1.,4.,1.])/6
    mass = weights[:,None]*weights[None,:]
    anchor = np.einsum('ij,ijk->k',mass,w)/np.sum(mass*chi)
    qa = anchor[3]-np.linalg.norm(anchor[:3])
    if anchor[0] <= 0 or qa <= 0:
        raise ValueError('Cell mean is outside the admissible cone')
    return w,chi,anchor,qa

def absolute_physical_scaling(stored, radial_cell, density_floor=1e-11, cone_floor=1e-11):
    if (not np.isfinite([density_floor,cone_floor]).all()
            or min(density_floor,cone_floor) <= 0):
        raise ValueError('Floors must be positive and finite')
    out,chi,anchor,qa = _s6_geometry_anchor(stored,radial_cell)
    reference = chi[...,None]*anchor
    local = regular_local_state(out,radial_cell)
    de,qe = min(density_floor,anchor[0]),min(cone_floor,qa)
    minimum = np.min(local[...,0])
    theta_d = 1.0 if minimum >= de else (anchor[0]-de)/(anchor[0]-minimum)
    out[...,0] = reference[...,0]+theta_d*(out[...,0]-reference[...,0])
    local = regular_local_state(out,radial_cell)
    minimum = np.min(local[...,3]-np.linalg.norm(local[...,:3],axis=-1))
    theta_q = 1.0 if minimum >= qe else (qa-qe)/(qa-minimum)
    return reference+theta_q*(out-reference)

def _s6_filtered():
    u = _s3_fixture()
    rates = normalized_damping_rates(u,directional_jump_amplitudes(u))
    return conservative_local_filter(u,rates)

import numpy as np

def relative_physical_scaling(stored, radial_cell, fraction=0.1):
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('Relative fraction must lie in [0,1)')
    out,chi,anchor,qa = _s6_geometry_anchor(stored,radial_cell)
    local = regular_local_state(out,radial_cell)
    radius = np.linalg.norm(local[...,:3],axis=-1)
    if np.any(local[...,0] < 0) or np.any(local[...,3]-radius < 0):
        raise ValueError('Input must satisfy absolute physical constraints')
    delta = fraction*qa/(anchor[3]+np.linalg.norm(anchor[:3]))
    minimum = np.min((1-delta)*local[...,3]-(1+delta)*radius)
    ga = (1-fraction)*qa
    theta = 1.0 if minimum >= 0 else ga/(ga-minimum)
    reference = chi[...,None]*anchor
    return reference+theta*(out-reference)

def _s7_absolute():
    return absolute_physical_scaling(_s6_filtered()[0,0],0)

def _s7_near_cone():
    """An a=1 cell whose corner node sits close to the cone, so theta<1 there."""
    base = absolute_physical_scaling(_s6_filtered()[1,0],1)
    _,chi,anchor,qa = _s6_geometry_anchor(base,1)
    out = chi[...,None]*anchor
    local = regular_local_state(out,1)
    d,mr,mz,_ = local[0,0]
    out[0,0,3] = chi[0,0]*(np.sqrt(d*d+mr*mr+mz*mz)+0.02*qa)
    return out

import numpy as np

from scipy.optimize import brentq

def primitive_pressure(local_state, gamma=5/3):
    u = np.asarray(local_state,dtype=float)
    if (u.shape != (4,) or not np.isfinite(u).all() or not np.isfinite(gamma)
            or not 1 < gamma <= 2):
        raise ValueError('Invalid local state or adiabatic index')
    scale = u[3]
    if u[0] <= 0 or scale <= 0:
        raise ValueError('State is not strictly admissible')
    d,mr,mz,e = u/scale
    if e <= np.linalg.norm(np.array([d,mr,mz])):
        raise ValueError('State is not strictly admissible')
    momentum_squared = mr*mr+mz*mz
    def residual(p):
        ep = e+p
        return p/(gamma-1)-e+momentum_squared/ep+d*np.sqrt(1-momentum_squared/ep**2)
    return float(scale*brentq(residual,0.,(gamma-1)*e,
                             xtol=np.nextafter(0.,1.),rtol=1e-14))

import numpy as np

def stabilized_patch_pressure(raw=None, dt=0.02, strength=0.02):
    if not np.isfinite([dt,strength]).all() or min(dt,strength) < 0:
        raise ValueError('Invalid timestep or filter strength')
    if raw is None:
        raw = np.empty((2,2,3,3,4))
        x = np.array([0.,.5,1.])
        for a in range(2):
            for b in range(2):
                r,z = np.meshgrid(a+x,b+x,indexing='ij')
                local = np.stack([1+.1*z+.04*a+.01*b,.15*r,
                                  .2+.05*z+.03*a,3+.2*z+.07*a+.11*b],axis=-1)
                raw[a,b] = r[...,None]*local
        raw[0,0,0,1] = [.01,0,.005,.03]
        raw[0,0,1,1] = [-.05,.8,.1,.65]
    repaired = repair_axis(raw)
    weights = np.array([1.,4.,1.])/6
    means = np.einsum('i,j,abijk->abk',weights,weights,repaired)
    if np.any(means[...,0] <= 0) or np.any(means[...,3] <= np.linalg.norm(means[...,:3],axis=-1)):
        raise ValueError('An input cell mean is inadmissible')
    local = np.array([[regular_local_state(repaired[a,b],a)
                       for b in range(2)] for a in range(2)])
    jumps = directional_jump_amplitudes(local)
    rates = normalized_damping_rates(local,jumps)
    filtered = conservative_local_filter(local,rates,dt=dt,strength=strength)
    final = np.empty_like(filtered)
    for a in range(2):
        for b in range(2):
            absolute = absolute_physical_scaling(filtered[a,b],a)
            final[a,b] = relative_physical_scaling(absolute,a)
    target = regular_local_state(final[0,0],0)[2,1]
    return primitive_pressure(target)

def _s9_perturbed():
    w = _s1_fixture()
    return w+np.random.default_rng(610).normal(scale=1e-5,size=w.shape)

def _s9_inadmissible():
    w = _s1_fixture()
    w[1,1,:,:,3] = 0.1
    return w
SCICODE_GOLD_EOF
