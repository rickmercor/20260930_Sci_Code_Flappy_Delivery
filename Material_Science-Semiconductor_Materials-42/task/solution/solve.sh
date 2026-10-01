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

def bloch_frames(kpoints, mass):
    k = np.asarray(kpoints, dtype=float)
    if k.ndim != 2 or k.shape[1] != 2 or not len(k) or not np.isfinite(k).all() or not np.isfinite(mass) or mass <= 0:
        raise ValueError('finite nonempty (K,2) momenta and positive mass required')
    x, y = 3.2 * k[:, 0], 3.2 * k[:, 1]
    h = np.zeros((len(k), 3, 3), dtype=complex)
    h[:, 0, 0] = -mass + .17 * (np.cos(x) + np.cos(y))
    h[:, 1, 1] = mass + .11 * np.cos(x) - .08 * np.cos(y)
    h[:, 2, 2] = mass + 1.15 + .09 * np.cos(x + y)
    h[:, 0, 1] = .44 + .31 * np.exp(-1j*x) + .23 * np.exp(-1j*y)
    h[:, 0, 2] = .29 - .19 * np.exp(1j*x) + .21 * np.exp(-1j*(x+y))
    h[:, 1, 2] = .16 + .13 * np.exp(-1j*y)
    h += np.triu(h, 1).conj().swapaxes(-1, -2)
    energy, vectors = np.linalg.eigh(h)
    # A documented gauge fixes the interface, not a physical observable.
    for i in range(len(k)):
        for band in range(3):
            j = int(np.argmax(np.abs(vectors[i, :, band])))
            vectors[i, :, band] *= np.exp(-1j*np.angle(vectors[i, j, band]))
    return np.concatenate((energy[:, None, :], vectors), axis=1)

import numpy as np

def slab_averages(lengths, thickness, z_fractions):
    q = np.asarray(lengths, dtype=float)
    z = np.asarray(z_fractions, dtype=float)
    if q.ndim < 1 or not q.size or z.ndim != 1 or not z.size or not np.isfinite(q).all() or np.any(q < 0) or not np.isfinite(thickness) or thickness < 0 or not np.isfinite(z).all() or np.any(np.abs(z) > .5):
        raise ValueError('nonnegative momenta/thickness and slab-contained sites required')
    x = q * thickness
    result = np.ones(x.shape + (len(z)+1,), dtype=float)
    regular = x > 1e-4
    a, b = .5-z, .5+z
    xx = x[regular, None]
    result[regular, :-1] = (-np.expm1(-xx*a)-np.expm1(-xx*b))/xx
    result[regular, -1] = 2*(np.expm1(-x[regular])+x[regular])/x[regular]**2
    xx = x[~regular, None]
    result[~regular, :-1] = (1-xx*(a*a+b*b)/2 + xx**2*(a**3+b**3)/6
                            -xx**3*(a**4+b**4)/24 + xx**4*(a**5+b**5)/120)
    xx = x[~regular]
    result[~regular, -1] = 1-xx/3+xx**2/12-xx**3/60+xx**4/360
    return result

import numpy as np

def density_vertices(left_vectors, right_vectors, phases, orbital_averages):
    left = np.asarray(left_vectors, dtype=complex)
    right = np.asarray(right_vectors, dtype=complex)
    phase = np.asarray(phases, dtype=complex)
    averages = np.asarray(orbital_averages, dtype=float)
    if left.ndim != 3 or left.shape[1:] != (3,3) or right.ndim != 4 or right.shape[1:] != left.shape or phase.ndim != 3 or phase.shape[0] != len(right) or phase.shape[2] != 3 or averages.shape != phase.shape or np.any(averages < 0) or not all(np.isfinite(x).all() for x in (left,right,phase,averages)):
        raise ValueError('incompatible vertex tensors or negative form factors')
    form = np.sqrt(averages)
    return np.einsum('kan,qkab,qga->qkgnb', left.conj(), right, phase*form, optimize=True)

import numpy as np

def dielectric_matrices(left_energies, right_energies, vertices, bare, spin=2.):
    el = np.asarray(left_energies, dtype=float)
    er = np.asarray(right_energies, dtype=float)
    vv = np.asarray(vertices, dtype=complex)
    potential = np.asarray(bare, dtype=float)
    if el.ndim != 2 or el.shape[1] != 3 or er.ndim != 3 or er.shape[1:] != el.shape or potential.ndim != 2 or potential.shape[0] != len(er) or vv.shape != (len(er),len(el),potential.shape[1],3,3) or np.any(potential < 0) or spin < 0 or not np.isfinite(spin) or not all(np.isfinite(x).all() for x in (el,er,vv,potential)):
        raise ValueError('incompatible response inputs')
    if np.any(np.max(el[:,0]) >= er[:,:,1:]) or np.any(np.max(er[:,:,0]) >= el[:,1:]):
        raise ValueError('occupied and empty spectra must be separated')
    occ = np.array([1., 0., 0.])
    weights = np.zeros(er.shape[:2] + (3, 3), dtype=float)
    for n in range(3):
        for b in range(3):
            if occ[n] != occ[b]:
                weights[:, :, n, b] = (spin/len(el)*(occ[n]-occ[b])
                                       /(el[None, :, n]-er[:, :, b]))
    chi = np.einsum('qkgnb,qkhnb,qknb->qgh', vv, vv.conj(), weights, optimize=True)
    sq = np.sqrt(potential)
    return np.eye(potential.shape[1])[None, :, :] - sq[:, :, None]*chi*sq[:, None, :]

import numpy as np

def screening_lengths(dielectric_samples, delta):
    e = np.asarray(dielectric_samples, dtype=complex)
    if e.ndim != 3 or e.shape[0] != 4 or e.shape[1] != e.shape[2] or not e.shape[1] or not np.isfinite(e).all() or not np.isfinite(delta) or delta <= 0:
        raise ValueError('four finite square dielectric samples and positive delta required')
    if not np.allclose(e,e.conj().swapaxes(-1,-2),atol=1e-10) or np.any(np.linalg.eigvalsh(e) <= 0):
        raise ValueError('positive Hermitian dielectric samples required')
    macro = e[:,0,0].real.copy()
    if e.shape[1] > 1:
        body_solution = np.linalg.solve(e[:,1:,1:],e[:,1:,0,None])[:,:,0]
        macro -= np.einsum('qg,qg->q',e[:,0,1:],body_solution).real
    r = (macro-1)/np.array([delta,delta/2,delta,delta/2])
    return np.array([2*r[1]-r[0],2*r[3]-r[2]],dtype=float)

import numpy as np

def screened_potentials(dielectrics, bare, pair_averages, zero_index,
                                q0, lengths, kappa, thickness):
    eps = np.asarray(dielectrics, dtype=complex)
    b = np.asarray(bare, dtype=float)
    f = np.asarray(pair_averages, dtype=float)
    if eps.ndim != 3 or eps.shape[1] != eps.shape[2] or b.shape != eps.shape[:2] or f.shape != b.shape or not all(np.isfinite(x).all() for x in (eps,b,f)) or np.any(b < 0) or np.any(f < 0) or zero_index < -1 or zero_index >= len(eps) or q0 <= 0 or kappa <= 0 or thickness < 0 or np.asarray(lengths).shape != (2,) or not np.isfinite([q0,kappa,thickness,*lengths]).all():
        raise ValueError('invalid screened-potential data')
    if not np.allclose(eps,eps.conj().swapaxes(-1,-2),atol=1e-10) or np.any(np.linalg.eigvalsh(eps) <= 0):
        raise ValueError('positive Hermitian dielectric matrices required')
    sq = np.sqrt(b*f)
    screened = sq[:, :, None]*np.linalg.inv(eps)*sq[:, None, :]
    if zero_index >= 0:
        c = 2*np.pi*14.3996454784255/(3.2**2*kappa)
        screened[zero_index, 0, :] = 0
        screened[zero_index, :, 0] = 0
        screened[zero_index, 0, 0] = c*(2/q0-np.sum(lengths)/2-thickness/3)
    return screened

import numpy as np

def exciton_hamiltonian(energies, pair_vertices, screened, q_indices,
                                head_only=False):
    e = np.asarray(energies, dtype=float)
    vv = np.asarray(pair_vertices, dtype=complex)
    ws = np.asarray(screened, dtype=complex)
    qi = np.asarray(q_indices)
    if e.ndim != 2 or e.shape[1] != 3 or not len(e) or ws.ndim != 3 or ws.shape[1] != ws.shape[2] or vv.shape != (len(e),len(e),ws.shape[1],3,3) or qi.shape != (len(e),len(e)) or not np.issubdtype(qi.dtype,np.integer) or np.any(qi < 0) or np.any(qi >= len(ws)) or not all(np.isfinite(x).all() for x in (e,vv,ws)):
        raise ValueError('incompatible BSE data')
    w = ws[qi]
    if head_only:
        w = w.copy()
        w[:, :, 1:, :] = 0
        w[:, :, :, 1:] = 0
    # Each vertex has left state at k_j and right state at k_i.
    electron = vv[:, :, :, 1:, 1:]
    hole = vv[:, :, :, 0, 0]
    direct = np.einsum('ijgdc,ijgh,ijh->icjd', electron.conj(), w, hole, optimize=True)/len(e)
    gaps = (e[:, 1:]-e[:, 0, None]).ravel()
    return np.diag(gaps) - direct.reshape(2*len(e), 2*len(e))

import numpy as np
from scipy.linalg import eigh

def exciton_levels(hamiltonian, count=3):
    h = np.asarray(hamiltonian, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.isfinite(h).all() or not isinstance(count,(int,np.integer)) or not 1 <= count <= len(h) or not np.allclose(h,h.conj().T,atol=1e-10):
        raise ValueError('finite Hermitian matrix and valid count required')
    return eigh(h,
                subset_by_index=[0, count-1], eigvals_only=True)

import numpy as np
from scipy.optimize import brentq

def calibration_branches(forward, observed, bounds):
    observed = np.asarray(observed, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if observed.shape != (2,) or bounds.shape != (2,2) or not np.isfinite(observed).all() or not np.isfinite(bounds).all() or np.any(bounds[:,0] >= bounds[:,1]):
        raise ValueError('two observations and nondegenerate rectangular bounds required')
    roots = []
    cache = {}
    def reduced(d):
        if d in cache:
            return cache[d]
        def first(k):
            return float(forward(np.array([k, d]))[0]-observed[0])
        lo, hi = bounds[0]
        flo, fhi = first(lo), first(hi)
        if flo*fhi > 0:
            cache[d] = (np.nan, np.nan)
        else:
            k = brentq(first, lo, hi, xtol=1e-10)
            cache[d] = (float(forward(np.array([k, d]))[1]-observed[1]), k)
        return cache[d]
    ds = np.linspace(*bounds[1], 65)
    values = [reduced(d)[0] for d in ds]
    for i in range(len(ds)-1):
        if not np.isfinite(values[i]) or not np.isfinite(values[i+1]):
            continue
        if values[i] == 0:
            d = ds[i]
        elif values[i]*values[i+1] < 0:
            d = brentq(lambda d: reduced(d)[0], ds[i], ds[i+1], xtol=2e-9)
        else:
            continue
        candidate = np.array([reduced(d)[1], d])
        if not any(np.max(np.abs(candidate-x)) < 2e-4 for x in roots):
            roots.append(candidate)
    if values[-1] == 0:
        roots.append(np.array([reduced(ds[-1])[1], ds[-1]]))
    if not roots:
        return np.empty((0, 2), dtype=float)
    result = np.round(np.asarray(roots), 6)
    return result[np.lexsort((result[:, 0], result[:, 1]))]

import numpy as np

def predict_exciton_splitting(observed, witness, bounds=((6., 9.), (.6, 3.2)),
                                      prediction=(7, 1.35, 2)):
    if np.asarray(observed).shape != (2,) or not np.isfinite(observed).all() or not np.isfinite(witness) or witness <= 0 or len(prediction) != 3:
        raise ValueError('invalid experiment')
    n, mass, cutoff = prediction
    if n not in (3,5,7) or cutoff not in (1,2) or not np.isfinite(mass) or not .8 <= mass <= 2.:
        raise ValueError('prediction requires n=3,5,7; mass in [0.8,2]; cutoff=1,2')
    bb = np.asarray(bounds,dtype=float)
    if bb.shape != (2,2) or not np.isfinite(bb).all() or np.any(bb[:,0] >= bb[:,1]) or bb[0,0] < 6 or bb[0,1] > 9 or bb[1,0] < .6 or bb[1,1] > 3.2:
        raise ValueError('bounds must be a nondegenerate subrectangle of the experiment box')
    a = 3.2
    c0 = 2*np.pi*14.3996454784255/a**2
    tau = a*np.array([[0., 0.], [.37, .12], [.16, .43]])
    z_fractions = np.array([0., .29, -.23])
    delta = 2e-3/a

    def prepare(n, mass, cutoff):
        n, cutoff = int(n), int(cutoff)
        kk = (np.arange(n)-(n-1)/2)*2*np.pi/(a*n)
        k = np.array([(x, y) for x in kk for y in kk])
        indices = [(i, j) for i in range(-cutoff, cutoff+1)
                   for j in range(-cutoff, cutoff+1) if i*i+j*j <= cutoff*cutoff]
        indices.sort(key=lambda ij: (ij[0]**2+ij[1]**2, ij[0], ij[1]))
        gs = np.asarray(indices)*2*np.pi/a
        zz = np.arange(-(n-1), n)
        q = np.array([(x, y) for x in zz for y in zz])*2*np.pi/(a*n)
        zero = (len(q)-1)//2
        q = np.vstack((q, [[delta, 0], [delta/2, 0], [0, delta], [0, delta/2], [.27, .13]]))
        lengths = np.linalg.norm(q[:, None, :]+gs[None, :, :], axis=-1)
        phases = np.exp(-1j*np.einsum('qgi,ai->qga', q[:, None, :]+gs, tau))
        pp = (np.arange(7)-3)*2*np.pi/(a*7)
        kp = np.array([(x, y) for x in pp for y in pp])
        left = bloch_frames(kp, mass)
        right = bloch_frames((kp[None, :, :]+q[:, None, :]).reshape(-1, 2), mass)
        right = right.reshape(len(q), len(kp), 4, 3)
        frames = bloch_frames(k, mass)
        i, j = np.indices((len(k), len(k)))
        qi = (i//n-j//n+n-1)*(2*n-1)+(i % n-j % n+n-1)
        pair_phases = phases[qi].reshape(len(k), len(k), len(gs), 3)
        # Pair phases vary with both i and j. Evaluate each i batch with Q=j,K=1.
        pair = np.empty((len(k), len(k), len(gs), 3, 3), dtype=complex)
        for jj in range(len(k)):
            pair[:, jj] = density_vertices(frames[jj:jj+1, 1:], frames[:, None, 1:],
                                                   pair_phases[:, jj],
                                                   np.ones((len(k), len(gs), 3)))[:, 0]
        return dict(n=n, mass=mass, lengths=lengths, phases=phases, zero=zero,
                    left=left, right=right, frames=frames, pair=pair, qi=qi)

    geometries = [prepare(5, 1.1, 1), prepare(5, 1.6, 1)]
    future = prepare(*prediction)

    def evaluate(g, parameters, full=False):
        kappa, thickness = parameters
        averages = slab_averages(g['lengths'], thickness, z_fractions)
        vertices = density_vertices(g['left'][:, 1:], g['right'][:, :, 1:],
                                            g['phases'], averages[:, :, :3])
        bare = np.zeros_like(g['lengths'])
        np.divide(c0/kappa, g['lengths'], out=bare, where=g['lengths'] > 1e-13)
        eps = dielectric_matrices(g['left'][:, 0].real, g['right'][:, :, 0].real,
                                          vertices, bare)
        inv = np.linalg.inv(eps)
        r = screening_lengths(eps[-5:-1], delta)
        q0 = .35*2*np.pi/(a*g['n'])
        w = screened_potentials(eps, bare, averages[:, :, 3], g['zero'], q0,
                                        r, kappa, thickness)
        h = exciton_hamiltonian(g['frames'][:, 0].real, g['pair'], w, g['qi'])
        levels = exciton_levels(h)
        if not full:
            return levels[0]
        hhead = exciton_hamiltonian(g['frames'][:, 0].real, g['pair'], w, g['qi'], True)
        head_e0 = exciton_levels(hhead, 1)[0]
        return levels, 1/inv[-1, 0, 0].real, w[g['zero'], 0, 0].real, head_e0-levels[0]

    def forward(parameters):
        return np.array([evaluate(g, parameters) for g in geometries])

    branches = calibration_branches(forward, observed, bounds)
    if not len(branches):
        return np.full(7, -1., dtype=float)
    details = [evaluate(future, parameters, True) for parameters in branches]
    choice = min(range(len(branches)), key=lambda i: (abs(details[i][1]-witness),
                                                    branches[i, 1], branches[i, 0]))
    levels, dielectric, head, bias = details[choice]
    return np.array([levels[1]-levels[0], branches[choice, 0], branches[choice, 1],
                     dielectric, head, levels[0], bias], dtype=float)
SCICODE_GOLD_EOF
