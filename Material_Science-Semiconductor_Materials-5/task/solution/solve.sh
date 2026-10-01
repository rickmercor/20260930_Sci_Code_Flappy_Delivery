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

def joint_operators(delta: float, omega: float, detunings: "np.ndarray", directions: "np.ndarray") -> "np.ndarray":
    n=len(detunings)
    low=np.array([[0.,1.],[0.,0.]],complex)
    ops=[]
    for k in range(n+1):
        op=np.ones((1,1),complex)
        for j in range(n+1):
            op=np.kron(op,low if j==k else np.eye(2))
        ops.append(op)
    a=ops[0].conj().T@ops[0]
    h0=delta*a+omega*(ops[0]+ops[0].conj().T)/2
    exchange=np.zeros_like(h0)
    for d,r,v in zip(detunings,directions,ops[1:]):
        h0+=d*v.conj().T@v
        exchange+=r*(ops[0].conj().T@v+ops[0]@v.conj().T)
    return np.array([h0,exchange,a,*ops])

import numpy as np
from functools import lru_cache
from scipy.special import roots_legendre

@lru_cache(maxsize=None)
def _gauss_nodes(count):
    return roots_legendre(count)

def bath_response(frequencies: "np.ndarray", alpha: float, cutoff: float, temperature: float) -> "np.ndarray":
    z = np.asarray(frequencies, complex)
    flat = z.ravel()
    theta = .1309203391 * temperature
    bound = 12 * cutoff
    x, weights = _gauss_nodes(240)
    # Split at zero to resolve the thermal frequency scale near the endpoint.
    v = np.concatenate(((x + 1) * bound / 2, -(x + 1) * bound / 2))
    weights = np.tile(weights * bound / 2, 2)

    def _thermal_weight(w):
        w = np.asarray(w, complex)
        out = np.zeros_like(w)
        nz = np.abs(w) > 1e-14
        out[nz] = alpha * w[nz]**3 * np.exp(-(w[nz]/cutoff)**2) / (-np.expm1(-w[nz]/theta))
        return out

    hz = _thermal_weight(flat)
    denom = flat[:, None] - v[None, :]
    diff = _thermal_weight(v)[None, :] - hz[:, None]
    integrand = np.divide(diff, denom, out=np.zeros_like(diff), where=np.abs(denom) > 1e-7)
    close = np.abs(denom) <= 1e-7
    if np.any(close):
        mid = (flat[:, None] + v[None, :]) / 2
        w = mid[close]
        derivative = _thermal_weight(w) * (3/w - 2*w/cutoff**2 - np.exp(-w/theta)/(theta*(-np.expm1(-w/theta))))
        integrand[close] = -derivative
    value = np.pi*hz + 1j*(integrand @ weights + hz*np.log((bound+flat)/(bound-flat)))
    return value.reshape(z.shape)

import numpy as np

def phonon_rate_series(hamiltonian0: "np.ndarray", exchange: "np.ndarray", coupling_operator: "np.ndarray", alpha: float, cutoff: float, temperature: float, order: int) -> "np.ndarray":
    def _evaluate(h):
        energies,vectors=np.linalg.eig(h)
        if np.linalg.cond(vectors)>1e6:
            raise np.linalg.LinAlgError('Ill-conditioned contour eigensystem')
        inverse=np.linalg.inv(vectors)
        f=bath_response(energies[None,:]-energies[:,None],alpha,cutoff,temperature)
        return vectors@((inverse@coupling_operator@vectors)*f)@inverse
    norm=np.linalg.norm(exchange,2)
    if norm==0:
        result=np.zeros((order+1,*hamiltonian0.shape),complex)
        result[0]=_evaluate(hamiltonian0)
        return result
    span=np.ptp(np.linalg.eigvalsh(hamiltonian0))
    radius=min(4.,np.pi*.1309203391*temperature/(2*norm),(6*cutoff-span)/(2*norm))
    samples=64
    while True:
        try:
            values=np.array([_evaluate(hamiltonian0+radius*np.exp(2j*np.pi*j/samples)*exchange) for j in range(samples)])
            break
        except np.linalg.LinAlgError:
            radius*=.8
    result=np.fft.fft(values,axis=0)[:order+1]/samples
    result/=radius**np.arange(order+1)[:,None,None]
    return result

import numpy as np

def phonon_superoperator(coupling_operator: "np.ndarray", rate_series: "np.ndarray") -> "np.ndarray":
    a=coupling_operator;eye=np.eye(len(a));result=[]
    for z in rate_series:
        result.append(-np.kron(eye,a@z)+np.kron(a.T,z)+np.kron(z.conj(),a)-np.kron((z.conj().T@a).T,eye))
    return np.array(result)

import numpy as np

def total_liouvillian(operators: "np.ndarray", phonon_series: "np.ndarray", gamma: float, widths: "np.ndarray", alpha: float, cutoff: float) -> "np.ndarray":
    h0,v,a=operators[:3];lowering=operators[3:];eye=np.eye(len(h0))
    delta_p=-alpha*np.sqrt(np.pi)*cutoff**3/4
    h=h0-delta_p*a
    result=phonon_series.copy()
    result[0]+=-1j*(np.kron(eye,h)-np.kron(h.T,eye))
    if len(result)>1:
        result[1]+=-1j*(np.kron(eye,v)-np.kron(v.T,eye))
    for decay,op in zip([gamma,*widths],lowering):
        population=op.conj().T@op
        result[0]+=decay*(np.kron(op.conj(),op)-.5*np.kron(eye,population)-.5*np.kron(population.T,eye))
    return result

import numpy as np
from scipy.linalg import lu_factor, lu_solve

def stationary_density(generators: "np.ndarray") -> "np.ndarray":
    order = len(generators)-1
    d = int(round(np.sqrt(generators.shape[1])))
    matrix = generators[0].copy()
    matrix[0] = np.eye(d).ravel(order='F')
    factor = lu_factor(matrix)
    states = []
    for k in range(order+1):
        rhs = np.zeros(d*d, complex)
        if k == 0:
            rhs[0] = 1
        else:
            for j in range(1, k+1):
                rhs -= generators[j] @ states[k-j]
            rhs[0] = 0
        state = lu_solve(factor, rhs).reshape((d,d), order='F')
        states.append(((state+state.conj().T)/2).ravel(order='F'))
    return np.array([x.reshape((d,d),order='F') for x in states])

import numpy as np

def subset_spectra(density_series: "np.ndarray", lowering_operators: "np.ndarray", widths: "np.ndarray", directions: "np.ndarray") -> "np.ndarray":
    n=len(directions);result=[]
    for mask in range(1,2**n):
        observable=np.eye(density_series.shape[1],dtype=complex);factor=1.;size=0
        for j in range(n):
            if mask&(1<<j):
                observable=observable@(lowering_operators[j].conj().T@lowering_operators[j])
                factor*=widths[j]/(2*np.pi*directions[j]**2);size+=1
        result.append(float(np.trace(observable@density_series[2*size]).real)*factor)
    return np.array(result)

import numpy as np

def connected_coincidence(parameters: dict) -> float:
    p=parameters
    operators=joint_operators(p['delta'],p['omega'],np.asarray(p['detunings']),np.asarray(p['directions']))
    z=phonon_rate_series(*operators[:3],p['alpha'],p['cutoff'],p['temperature'],6)
    k=phonon_superoperator(operators[2],z)
    liouvillian=total_liouvillian(operators,k,p['gamma'],np.asarray(p['widths']),p['alpha'],p['cutoff'])
    rho=stationary_density(liouvillian)
    spectra=subset_spectra(rho,operators[4:],np.asarray(p['widths']),np.asarray(p['directions']))
    g12=spectra[2]/spectra[0]/spectra[1]
    g13=spectra[4]/spectra[0]/spectra[3]
    g23=spectra[5]/spectra[1]/spectra[3]
    g123=spectra[6]/spectra[0]/spectra[1]/spectra[3]
    return float(g123-g12-g13-g23+2)
SCICODE_GOLD_EOF
