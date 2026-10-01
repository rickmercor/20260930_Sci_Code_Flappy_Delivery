#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def displaced_factor(omega: "np.ndarray", shift: "np.ndarray", duration: float, energy: float) -> tuple:
    import numpy as np
    omega, shift = np.asarray(omega, float), np.asarray(shift, float)
    angle = omega * duration
    alpha = shift * np.expm1(-1j * angle)
    phase = float(-energy * duration - np.sum(shift**2 * np.sin(angle)))
    return phase, angle, np.stack((alpha.real, alpha.imag), -1)

def contour_sources(path: "np.ndarray", times: "np.ndarray", energies: "np.ndarray", shifts: "np.ndarray", omega: "np.ndarray", coordinate: "np.ndarray") -> tuple:
    import numpy as np
    path, times = np.asarray(path, int), np.asarray(times, float)
    energies, shifts = np.asarray(energies, float), np.asarray(shifts, float)
    omega, coordinate = np.asarray(omega, float), np.asarray(coordinate, float)
    order, modes = len(times), len(omega)
    r, count = order + 1, 2 * order + 2
    u = np.zeros((count, modes, r+1), complex)
    v = np.zeros_like(u)
    angle = np.zeros(modes)
    phase, row = 0., 0
    for j in range(order, -1, -1):
        state = 0 if j == order else path[j]
        duration = -sum(times) if j == order else times[j]
        ph, theta, packed = displaced_factor(omega, shifts[state], duration, energies[state])
        alpha = packed[:, 0] + 1j * packed[:, 1]
        u[row, :, 0] = alpha * np.exp(-1j * angle)
        v[row, :, 0] = -alpha.conj() * np.exp(1j * angle)
        phase += ph
        angle += theta
        row += 1
        u[row, :, j+1] = coordinate * np.exp(-1j * angle)
        v[row, :, j+1] = coordinate * np.exp(1j * angle)
        row += 1
    return np.stack((u.real, u.imag), -1), np.stack((v.real, v.imag), -1), float(phase)

def gaussian_generator(u: "np.ndarray", v: "np.ndarray", phase: float, omega: "np.ndarray", beta: float) -> tuple:
    import numpy as np
    u, v = np.asarray(u, float), np.asarray(v, float)
    u, v = u[..., 0]+1j*u[..., 1], v[..., 0]+1j*v[..., 1]
    nu = .5 / np.tanh(.5*beta*np.asarray(omega, float))
    dimension = u.shape[-1]
    A = np.zeros((dimension, dimension), complex)
    for i in range(len(u)):
        for j in range(i+1, len(u)):
            A += .5*(v[i].T @ u[j] - u[i].T @ v[j])
    U, V = u.sum(axis=0), v.sum(axis=0)
    A += U.T @ (nu[:, None] * V)
    c = A[0, 0] + 1j*phase
    linear = A[0, 1:] + A[1:, 0]
    quadratic = A[1:, 1:] + A[1:, 1:].T
    return np.array([c.real, c.imag]), np.stack((linear.real, linear.imag), -1), np.stack((quadratic.real, quadratic.imag), -1)

def squarefree_moments(c: "np.ndarray", linear: "np.ndarray", quadratic: "np.ndarray") -> "np.ndarray":
    import numpy as np
    c, linear, quadratic = np.asarray(c), np.asarray(linear), np.asarray(quadratic)
    c = c[0]+1j*c[1]
    linear = linear[..., 0]+1j*linear[..., 1]
    quadratic = quadratic[..., 0]+1j*quadratic[..., 1]
    r = len(linear)
    values = np.zeros(1 << r, complex)
    values[0] = np.exp(c)
    for mask in range(1, len(values)):
        bit = mask & -mask
        i = bit.bit_length()-1
        rest = mask ^ bit
        value = linear[i] * values[rest]
        for j in range(r):
            if rest & (1 << j):
                value += quadratic[i, j] * values[rest ^ (1 << j)]
        values[mask] = value
    return np.stack((values.real, values.imag), -1)

def pathway_polynomial(path: "np.ndarray", mu0: "np.ndarray", mu1: "np.ndarray", moments: "np.ndarray") -> "np.ndarray":
    import numpy as np
    path, mu0, mu1 = np.asarray(path, int), np.asarray(mu0), np.asarray(mu1)
    moments = np.asarray(moments)
    moments = moments[:, 0]+1j*moments[:, 1]
    states = [0] + list(path) + [0]
    r = len(states)-1
    coeff = np.zeros(r+1, complex)
    for mask, moment in enumerate(moments):
        factor = 1.
        for j in range(r):
            matrix = mu1 if mask & (1 << j) else mu0
            factor *= matrix[states[j+1], states[j]]
        coeff[mask.bit_count()] += factor * moment
    return np.stack((coeff.real, coeff.imag), -1)

def response_polynomial(times: "np.ndarray", model: dict) -> "np.ndarray":
    import numpy as np
    from itertools import product
    mu0, mu1 = np.asarray(model['mu0']), np.asarray(model['mu1'])
    order, n = len(times), len(mu0)
    coeff = np.zeros((order+2, 2))
    for path in product(range(n), repeat=order):
        states = (0,) + path + (0,)
        if any(mu0[b,a] == 0 and mu1[b,a] == 0 for a,b in zip(states[:-1], states[1:])):
            continue
        u,v,phase = contour_sources(path, times, model['energies'], model['shifts'], model['omega'], model['coordinate'])
        c,l,B = gaussian_generator(u,v,phase,model['omega'],model['beta'])
        moments = squarefree_moments(c,l,B)
        coeff += pathway_polynomial(path,mu0,mu1,moments)
    return coeff

def spectral_polynomial(nodes: "np.ndarray", weights: "np.ndarray", frequencies: "np.ndarray", waits: "np.ndarray", damping: float, model: dict) -> "np.ndarray":
    import numpy as np
    nodes, weights = np.asarray(nodes), np.asarray(weights)
    frequencies, waits = np.asarray(frequencies), np.asarray(waits)
    total = np.zeros(len(waits)+4, complex)
    for i, t1 in enumerate(nodes):
        for j, tlast in enumerate(nodes):
            times = np.r_[t1, waits, tlast]
            c = response_polynomial(times,model)
            c = c[:,0]+1j*c[:,1]
            factor = weights[i]*weights[j]*np.exp(1j*(frequencies[0]*t1+frequencies[1]*tlast)-damping*(t1+tlast))
            total += factor*c
    return np.stack((total.real, total.imag), -1)

def solve_ht(nodes: "np.ndarray", weights: "np.ndarray", frequencies: "np.ndarray", waits: "np.ndarray", damping: float, model: dict) -> float:
    import numpy as np
    coeff = spectral_polynomial(nodes,weights,frequencies,waits,damping,model)
    coeff = coeff[:,0]+1j*coeff[:,1]
    if abs(coeff[1]) <= 1e-14:
        raise ValueError('linear HT coefficient is zero')
    answer = float((coeff[3]*coeff[1].conjugate()).imag / abs(coeff[1])**2)
    return answer
SCICODE_GOLD_EOF
