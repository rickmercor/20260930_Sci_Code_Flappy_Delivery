#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

def emission_coefficients(T, sigma, depth, degeneracy):
    import numpy as np
    sigma = np.asarray(sigma, float)
    depth = np.asarray(depth, float)
    degeneracy = np.asarray(degeneracy, float)
    velocity = 1.e7*np.sqrt(T/300.)
    Nc = 2.8e19*(T/300.)**1.5
    c = sigma*velocity
    e = c*Nc*degeneracy*np.exp(-depth/(8.617333262145e-5*T))
    return c, e

def barrier_state(p, weights, alpha, eta):
    import numpy as np
    from scipy.optimize import brentq
    p = np.asarray(p, float)
    w = np.repeat(np.asarray(weights, float), 2)*np.tile([1., 2.], len(weights))
    r = alpha*(w@p)
    if eta == 0 or r == 0:
        psi = r
    else:
        psi = brentq(lambda v: v+eta*v**3-r, min(0., r), max(0., r), xtol=5e-15)
    normal = alpha*w/(1.+3.*eta*psi**2)
    return float(psi), normal

def kinetic_lna(p, cfg, density):
    import numpy as np
    p = np.asarray(p, float)
    d = p.size
    c, e = emission_coefficients(cfg['T'], cfg['sigma'], cfg['depth'], cfg['degeneracy'])
    psi, normal = barrier_state(p, cfg['weights'], cfg['alpha'], cfg['eta'])
    u = c*density*np.exp(-psi)
    F = np.zeros(d)
    A = np.zeros((d, d))
    D = np.zeros((d, d))
    for i in range(d//2):
        j = 2*i
        p0, p1, p2 = 1.-p[j]-p[j+1], p[j], p[j+1]
        forward0, reverse0 = u[i, 0]*p0, e[i, 0]*p1
        forward1, reverse1 = u[i, 1]*p1, e[i, 1]*p2
        F[j:j+2] = [forward0-reverse0-forward1+reverse1, forward1-reverse1]
        A[j:j+2, j:j+2] = [[-u[i, 0]-e[i, 0]-u[i, 1], -u[i, 0]+e[i, 1]],
                            [u[i, 1], -e[i, 1]]]
        A[j:j+2] -= np.outer([forward0-forward1, forward1], normal)
        D[j:j+2, j:j+2] = (((forward0+reverse0)*np.array([[1., 0.], [0., 0.]])
                            +(forward1+reverse1)*np.array([[1., -1.], [-1., 1.]]))
                           /cfg['weights'][i])
    return F, A, D

def gaussian_flow(p, cfg, density, duration):
    import numpy as np
    from scipy.integrate import solve_ivp
    p = np.asarray(p, float)
    d = p.size
    if duration == 0:
        return p.copy(), np.eye(d), np.zeros((d, d))
    initial = np.r_[p, np.eye(d).ravel(), np.zeros(d*d)]
    def _rhs(t, value):
        q = value[:d]
        Phi = value[d:d+d*d].reshape(d, d)
        V = value[d+d*d:].reshape(d, d)
        f, A, D = kinetic_lna(q, cfg, density)
        return np.r_[f, (A@Phi).ravel(), (A@V+V@A.T+D).ravel()]
    sol = solve_ivp(_rhs, (0., duration), initial, method='DOP853', rtol=3e-12, atol=3e-14)
    if not sol.success:
        raise ValueError('Gaussian flow integration failed')
    end = sol.y[:, -1]
    V = end[d+d*d:].reshape(d, d)
    return end[:d], end[d:d+d*d].reshape(d, d), (V+V.T)/2.

def hybrid_cycle(p, cfg, protocol, gates):
    import numpy as np
    from scipy.integrate import solve_ivp
    p = np.asarray(p, float)
    gates = np.asarray(gates, float)
    d = p.size
    m = d//2
    period, level = protocol['period'], protocol['level']
    nf, nr = protocol['fill_density'], protocol['read_density']
    def _event(t, q):
        return barrier_state(q, cfg['weights'], cfg['alpha'], cfg['eta'])[0]-level
    _event.terminal = True
    _event.direction = 1
    if _event(0., p) >= 0:
        raise ValueError('initial barrier must be below level')
    sol = solve_ivp(lambda t, q: kinetic_lna(q, cfg, nf)[0],
                    (0., period), p, events=_event, method='DOP853', rtol=3e-12, atol=3e-14)
    if not sol.success or len(sol.t_events[0]) == 0:
        raise ValueError('no interior upward event')
    tau = float(sol.t_events[0][0])
    if not 0. < tau < period:
        raise ValueError('event must be strictly interior')
    q, Pf, Vf = gaussian_flow(p, cfg, nf, tau)
    fminus = kinetic_lna(q, cfg, nf)[0]
    normal = barrier_state(q, cfg['weights'], cfg['alpha'], cfg['eta'])[1]
    speed = normal@fminus
    if speed <= 0:
        raise ValueError('event must be transversal and upward')
    clock = normal/speed  # b=-sqrt(N)*delta_tau
    total = d+1+2*m
    E = np.zeros((total, d))
    E[:d] = np.eye(d)-np.outer(fminus, clock)
    E[d] = clock
    B = E@Pf
    V = E@Vf@E.T
    charge = np.zeros((m, d))
    for i in range(m):
        charge[i, 2*i:2*i+2] = [1., 2.]
    mean_gates = np.zeros(2*m)
    previous = 0.
    for j, fraction in enumerate([*gates, 1.]):
        q, Phi, W = gaussian_flow(q, cfg, nr, (fraction-previous)*(period-tau))
        transition = np.eye(total)
        transition[:d, :d] = Phi
        B = transition@B
        V = transition@V@transition.T
        V[:d, :d] += W
        f = kinetic_lna(q, cfg, nr)[0]
        if j < 2:
            rows = slice(d+1+j*m, d+1+(j+1)*m)
            sample = np.eye(total)
            sample[rows] = 0.
            sample[rows, :d] = charge
            sample[rows, d] = fraction*(charge@f)
            B = sample@B
            V = sample@V@sample.T
            mean_gates[j*m:(j+1)*m] = charge@q
        else:
            sample = np.eye(total)
            sample[:d, d] = f
            B = sample@B
            V = sample@V@sample.T
        previous = fraction
    V = (V+V.T)/2.
    keep = slice(d+1, total)
    return (q, B[:d], B[keep], V[:d, :d], V[:d, keep], V[keep, keep],
            mean_gates, tau)

def periodic_state(cfg, protocol, gates):
    import numpy as np
    from scipy.optimize import root
    d = 2*len(cfg['weights'])
    p = np.zeros(d)
    for _ in range(4):
        p = hybrid_cycle(p, cfg, protocol, gates)[0]
    cache = {}
    def _value(v):
        if 'p' not in cache or not np.array_equal(v, cache['p']):
            cycle = hybrid_cycle(v, cfg, protocol, gates)
            cache.update(p=v.copy(), f=cycle[0]-v, J=cycle[1]-np.eye(d))
        return cache['f'], cache['J']
    result = root(lambda v: _value(v)[0], p, jac=lambda v: _value(v)[1], tol=1e-10)
    p = result.x
    if np.max(np.abs(_value(p)[0])) > 2e-10:
        raise ValueError('periodic state did not converge')
    if np.min(p) < -1e-10 or np.max(p.reshape(-1, 2).sum(axis=1)) > 1.+1e-10:
        raise ValueError('periodic state outside probability simplex')
    return p

def stationary_covariance(M, Qxx):
    import numpy as np
    from scipy.linalg import solve_discrete_lyapunov
    P = solve_discrete_lyapunov(np.asarray(M, float), np.asarray(Qxx, float))
    return (P+P.T)/2.

def detector_coefficients(mean_gates, weights, beta, population):
    import numpy as np
    mean_gates = np.asarray(mean_gates, float)
    weights = np.asarray(weights, float)
    m = weights.size
    mean = mean_gates.reshape(2, m)
    first = -.5*beta*(1.+beta*mean)**(-1.5)
    second = .75*beta**2*(1.+beta*mean)**(-2.5)
    L = np.zeros((m, 2*m))
    H = np.zeros((m, 2*m, 2*m))
    for i in range(m):
        L[i, i], L[i, m+i] = weights[i]*first[0, i], -weights[i]*first[1, i]
        H[i, i, i] = weights[i]*second[0, i]/np.sqrt(population)
        H[i, m+i, m+i] = -weights[i]*second[1, i]/np.sqrt(population)
    return L, H

def quadratic_spectrum(M, C, Qxx, Qxz, Qzz, L, H, omega):
    import numpy as np
    M, C, Qxx, Qxz, Qzz, L, H = [np.asarray(v, float) for v in (M, C, Qxx, Qxz, Qzz, L, H)]
    P = stationary_covariance(M, Qxx)
    R0 = C@P@C.T+Qzz
    K = M@P@C.T+Qxz
    phase = np.exp(-1j*omega)
    d, g, o = M.shape[0], C.shape[0], L.shape[0]
    hh = H.reshape(o, g*g)
    zero = L@R0@L.T+.5*hh@np.kron(R0, R0)@hh.T
    linear = L@C@np.linalg.solve(np.eye(d)-phase*M, phase*K)@L.T
    lifted = np.kron(C, C)@np.linalg.solve(np.eye(d*d)-phase*np.kron(M, M),
                                          phase*np.kron(K, K))
    positive = linear+.5*hh@lifted@hh.T
    spectrum = zero+positive+positive.conj().T
    return spectrum.real, spectrum.imag

def dlts_noise(cfg, protocol, gates, beta, population, omega, numerator, denominator):
    import numpy as np
    if beta == 0 or gates[0] == gates[1]:
        raise ValueError('selected auto-spectra must be positive')
    p = periodic_state(cfg, protocol, gates)
    _, M, C, Qxx, Qxz, Qzz, mean_gates, _ = hybrid_cycle(p, cfg, protocol, gates)
    L, H = detector_coefficients(mean_gates, cfg['weights'], beta, population)
    real, imag = quadratic_spectrum(M, C, Qxx, Qxz, Qzz, L, H, omega)
    first, second = real[numerator, numerator], real[denominator, denominator]
    if first <= 0 or second <= 0:
        raise ValueError('selected auto-spectra must be positive')
    return float(imag[numerator, denominator]/np.sqrt(first*second))
SCICODE_GOLD_EOF
