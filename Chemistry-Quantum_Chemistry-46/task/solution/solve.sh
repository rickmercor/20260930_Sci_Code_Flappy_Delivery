#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def transition_blocks(gaps, factors):
    import numpy as np
    gaps = np.asarray(gaps, dtype=float)
    factors = np.asarray(factors, dtype=float)
    interaction = factors @ factors.T
    return np.diag(gaps) + interaction, interaction

def cavity_jets(electronic_a, electronic_b, dipoles, frequencies, couplings, orders):
    import numpy as np
    from math import factorial
    ea, eb = np.asarray(electronic_a, float), np.asarray(electronic_b, float)
    d, w, lam = np.asarray(dipoles, float), np.asarray(frequencies, float), np.asarray(couplings, float)
    orders = tuple(map(int, orders))
    ne, nm = d.shape
    zero = (0,)*len(orders)
    a = np.zeros(tuple(k+1 for k in orders)+(ne+nm, ne+nm))
    b = np.zeros_like(a)
    a[zero][:ne,:ne], b[zero][:ne,:ne] = ea, eb
    a[zero][ne:,ne:] = np.diag(w)
    for mode in range(nm):
        for degree in range(orders[mode]+1):
            index = tuple(degree if axis == mode else 0 for axis in range(nm))
            dse = lam[mode]**2*2.0**degree/factorial(degree)*np.outer(d[:,mode],d[:,mode])
            g = -np.sqrt(w[mode]/2)*lam[mode]/factorial(degree)*d[:,mode]
            for tensor in (a,b):
                tensor[index][:ne,:ne] += dse
                tensor[index][:ne,ne+mode] += g
                tensor[index][ne+mode,:ne] += g
    return a, b

def stable_ring(a, b):
    import numpy as np
    from scipy.linalg import eig
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if min(np.linalg.eigvalsh(a-b).min(), np.linalg.eigvalsh(a+b).min()) <= 1e-12:
        raise ValueError('The RPA quadratic form is not strictly positive.')
    n = len(a)
    rpa = np.block([[a, b], [-b, -a]])
    values, vectors = eig(rpa)
    idx = np.where(values.real > 0)[0]
    idx = idx[np.argsort(values[idx].real)]
    omega = values[idx].real
    x, y = vectors[:n, idx].real, vectors[n:, idx].real
    t = np.linalg.solve(x.T, y.T).T
    t = (t+t.T)/2
    return omega, t

def ring_jets(a, b, t0):
    import numpy as np
    from scipy.linalg import solve_sylvester
    a, b, t0 = np.asarray(a, float), np.asarray(b, float), np.asarray(t0, float)
    shape = a.shape[:-2]
    zero = (0,)*len(shape)
    t = np.zeros_like(a)
    t[zero] = t0
    left, right = a[zero]+t0@b[zero], a[zero]+b[zero]@t0
    for alpha in sorted(np.ndindex(shape), key=sum)[1:]:
        residual = b[alpha].copy()
        for beta in np.ndindex(tuple(k+1 for k in alpha)):
            remainder = tuple(x-y for x,y in zip(alpha,beta))
            residual += a[beta]@t[remainder]+t[beta]@a[remainder]
            for gamma in np.ndindex(tuple(k+1 for k in remainder)):
                delta = tuple(x-y for x,y in zip(remainder,gamma))
                residual += t[beta]@b[gamma]@t[delta]
        t[alpha] = solve_sylvester(left,right,-residual)
    return t

def inverse_jets(matrix):
    import numpy as np
    matrix = np.asarray(matrix, float)
    shape = matrix.shape[:-2]
    zero = (0,)*len(shape)
    out = np.zeros_like(matrix)
    out[zero] = np.linalg.inv(matrix[zero])
    for alpha in sorted(np.ndindex(shape), key=sum)[1:]:
        residual = np.zeros_like(matrix[zero])
        for beta in np.ndindex(tuple(k+1 for k in alpha)):
            if beta == zero:
                continue
            remainder = tuple(x-y for x,y in zip(alpha,beta))
            residual += matrix[beta]@out[remainder]
        out[alpha] = -out[zero]@residual
    return out

def photon_covariance_jets(t, electronic_count):
    import numpy as np
    t = np.asarray(t, float)
    shape = t.shape[:-2]
    zero = (0,)*len(shape)
    minus, plus = -t.copy(), t.copy()
    minus[zero] += np.eye(t.shape[-1])
    plus[zero] += np.eye(t.shape[-1])
    iminus, iplus = inverse_jets(minus), inverse_jets(plus)
    ne = int(electronic_count)
    nm = t.shape[-1]-ne
    covariance = np.zeros(shape+(2,nm,nm))
    for alpha in np.ndindex(shape):
        for beta in np.ndindex(tuple(k+1 for k in alpha)):
            rest = tuple(x-y for x,y in zip(alpha,beta))
            covariance[alpha][0] += .5*(plus[beta]@iminus[rest])[ne:,ne:]
            covariance[alpha][1] += .5*(minus[beta]@iplus[rest])[ne:,ne:]
    return covariance

def entropy_jets(covariance, renyi=2.0):
    import numpy as np
    from scipy.linalg import sqrtm, solve_sylvester
    covariance = np.asarray(covariance, float)
    shape = covariance.shape[:-3]
    zero = (0,)*len(shape)
    indices = sorted(np.ndindex(shape),key=sum)
    m = covariance.shape[-1]
    if renyi not in (1.5,2.0):
        raise ValueError('Supported Renyi indices are 1.5 and 2.0.')
    q, p = covariance[...,0,:,:], covariance[...,1,:,:]
    if min(np.linalg.eigvalsh(q[zero]).min(),np.linalg.eigvalsh(p[zero]).min()) <= 0:
        raise ValueError('Base quadrature blocks must be positive definite.')

    def product(x,y):
        out = np.zeros_like(x)
        for alpha in indices:
            for beta in np.ndindex(tuple(k+1 for k in alpha)):
                rest = tuple(a-b for a,b in zip(alpha,beta))
                out[alpha] += x[beta]@y[rest]
        return out

    def square_root(x):
        out = np.zeros_like(x)
        out[zero] = np.real_if_close(sqrtm(x[zero]))
        for alpha in indices[1:]:
            residual = x[alpha].copy()
            for beta in np.ndindex(tuple(k+1 for k in alpha)):
                if beta == zero or beta == alpha:
                    continue
                rest = tuple(a-b for a,b in zip(alpha,beta))
                residual -= out[beta]@out[rest]
            out[alpha] = solve_sylvester(out[zero],out[zero],residual)
        return out

    def logdet_series(x):
        sign, value = np.linalg.slogdet(x[zero])
        if sign <= 0:
            raise ValueError('Nonpositive base determinant.')
        inverse = inverse_jets(x)
        out = np.zeros(shape)
        out[zero] = value
        for alpha in indices[1:]:
            axis = next(i for i,k in enumerate(alpha) if k)
            previous = list(alpha)
            previous[axis] -= 1
            value = 0.
            for beta in np.ndindex(tuple(k+1 for k in previous)):
                rest = tuple(a-b for a,b in zip(alpha,beta))
                value += rest[axis]*np.trace(inverse[beta]@x[rest])
            out[alpha] = value/alpha[axis]
        return out

    if renyi == 2.0:
        return .5*(logdet_series(2*q)+logdet_series(2*p))
    base_q = np.real_if_close(sqrtm(q[zero]))
    squared_symplectic = np.linalg.eigvalsh(base_q@p[zero]@base_q)
    if squared_symplectic.min() <= (.5+1e-12)**2:
        raise ValueError('Renyi 3/2 jets require a base symplectic gap above 1e-12.')
    nu = square_root(product(q,p))
    plus, minus = nu.copy(), nu.copy()
    plus[zero] += .5*np.eye(m)
    minus[zero] -= .5*np.eye(m)
    kernel = product(plus,square_root(plus))-product(minus,square_root(minus))
    return 2*logdet_series(kernel)

def solve_cavity(gaps, factors, dipoles, frequencies, couplings, orders, renyi=2.0):
    from math import factorial, prod
    ea, eb = transition_blocks(gaps,factors)
    a, b = cavity_jets(ea,eb,dipoles,frequencies,couplings,orders)
    zero = (0,)*len(orders)
    omega, t0 = stable_ring(a[zero],b[zero])
    t = ring_jets(a,b,t0)
    covariance = photon_covariance_jets(t,len(gaps))
    entropy = entropy_jets(covariance,renyi)
    index = tuple(map(int,orders))
    return float(prod(factorial(k) for k in index)*entropy[index])
SCICODE_GOLD_EOF
