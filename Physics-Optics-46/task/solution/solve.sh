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

def cavity_realization(n_modes: int, n_ports: int, seed: int) -> "np.ndarray":
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_modes, n_modes))
    y = rng.standard_normal((n_modes, n_modes))
    v = rng.standard_normal((n_modes, n_ports + 1))
    h = (x + x.T + 1j*(y-y.T))/(2*np.sqrt(n_modes))
    return np.concatenate((h, v/np.sqrt(n_modes)), axis=1)

import numpy as np
from scipy.linalg import lu_factor, lu_solve

def scattering_jet(cavity: "np.ndarray", frequencies: "np.ndarray", log_coupling: float, detuning: float) -> "np.ndarray":
    p = cavity.shape[0]
    h = cavity[:, :p]
    w0 = cavity[:, p:]
    k = w0.shape[1]
    scale = np.exp(2*log_coupling)
    d = scale*(w0@w0.conj().T)
    adjoint = w0.conj().T
    eye = np.eye(p)
    result = np.empty((6, len(frequencies), k, k), dtype=complex)
    for n, frequency in enumerate(frequencies):
        factors = lu_factor((frequency+detuning)*eye-h+0.5j*d)
        # Powers of the resolvent applied to the coupling columns only.
        x1 = lu_solve(factors, w0)
        x2 = lu_solve(factors, x1)
        x3 = lu_solve(factors, x2)
        dx = lu_solve(factors, d@x1)
        dxd = lu_solve(factors, d@dx)
        mixed = lu_solve(factors, d@x2)+lu_solve(factors, dx)
        g = adjoint@x1
        gq = -1j*(adjoint@dx)
        gs = -(adjoint@x2)
        gqq = -2*(adjoint@dxd)-2j*(adjoint@dx)
        gqs = 1j*(adjoint@mixed)
        gss = 2*(adjoint@x3)
        result[0, n] = np.eye(k)-1j*scale*g
        result[1, n] = -1j*scale*(2*g+gq)
        result[2, n] = -1j*scale*gs
        result[3, n] = -1j*scale*(4*g+4*gq+gqq)
        result[4, n] = -1j*scale*(2*gs+gqs)
        result[5, n] = -1j*scale*gss
    return result

import numpy as np

def transmission_jet(scattering: "np.ndarray") -> "np.ndarray":
    s, sq, ss, sqq, sqs, sss = scattering[:, :, 1:, 0]
    a = np.abs(s)**2
    aq = 2*np.real(np.conj(s)*sq)
    as_ = 2*np.real(np.conj(s)*ss)
    aqq = 2*(np.abs(sq)**2+np.real(np.conj(s)*sqq))
    aqs = 2*(np.real(np.conj(sq)*ss)+np.real(np.conj(s)*sqs))
    ass = 2*(np.abs(ss)**2+np.real(np.conj(s)*sss))
    return np.stack((a.T, aq.T, as_.T, aqq.T, aqs.T, ass.T))

import numpy as np

def inverse_gram_jet(transmission: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    a, aq, as_, aqq, aqs, ass = transmission
    w = np.asarray(weights, dtype=float)
    first = (aq, as_)
    second = ((aqq, aqs), (aqs, ass))
    if a.shape[0] < a.shape[1]:
        # Underdetermined: (A^T A)^+ = A^T (A A^T)^-2 A, so F = Tr[(A A^T)^-2 A diag(w) A^T].
        pair = lambda u, v: u@v.T
        weighted = lambda u, v: (u*w)@v.T
    else:
        # Full column rank: G^+ = G^-1 and F = Tr[diag(w) G^-1].
        pair = lambda u, v: u.T@v
        weighted = None
    if weighted is None:
        gram = pair(a, a)
        gram_first = [pair(first[i], a)+pair(a, first[i]) for i in range(2)]
        inverse = np.linalg.solve(gram, np.eye(gram.shape[0]))
        inv_first = [-inverse@gram_first[i]@inverse for i in range(2)]
        value = float(w@np.diag(inverse))
        out = [value]+[float(w@np.diag(inv_first[i])) for i in range(2)]
        for i, j in ((0, 0), (0, 1), (1, 1)):
            gram_second = (pair(second[i][j], a)+pair(first[i], first[j])
                           +pair(first[j], first[i])+pair(a, second[i][j]))
            inv_second = (-inverse@gram_second@inverse
                          +inverse@gram_first[i]@inverse@gram_first[j]@inverse
                          +inverse@gram_first[j]@inverse@gram_first[i]@inverse)
            out.append(float(w@np.diag(inv_second)))
        return np.array(out)
    port = pair(a, a)
    band = weighted(a, a)
    port_first = [pair(first[i], a)+pair(a, first[i]) for i in range(2)]
    band_first = [weighted(first[i], a)+weighted(a, first[i]) for i in range(2)]
    inverse = np.linalg.solve(port, np.eye(port.shape[0]))
    inv_first = [-inverse@port_first[i]@inverse for i in range(2)]
    kernel = inverse@inverse
    kernel_first = [inv_first[i]@inverse+inverse@inv_first[i] for i in range(2)]
    out = [np.trace(kernel@band)]
    out += [np.trace(kernel_first[i]@band+kernel@band_first[i]) for i in range(2)]
    for i, j in ((0, 0), (0, 1), (1, 1)):
        port_second = (pair(second[i][j], a)+pair(first[i], first[j])
                       +pair(first[j], first[i])+pair(a, second[i][j]))
        band_second = (weighted(second[i][j], a)+weighted(first[i], first[j])
                       +weighted(first[j], first[i])+weighted(a, second[i][j]))
        inv_second = (-inverse@port_second@inverse
                      +inverse@port_first[i]@inverse@port_first[j]@inverse
                      +inverse@port_first[j]@inverse@port_first[i]@inverse)
        kernel_second = (inv_second@inverse+inv_first[i]@inv_first[j]
                         +inv_first[j]@inv_first[i]+inverse@inv_second)
        out.append(np.trace(kernel_second@band+kernel_first[i]@band_first[j]
                            +kernel_first[j]@band_first[i]+kernel@band_second))
    return np.array([float(v) for v in out])

import numpy as np

def spectral_correlation(transmission: "np.ndarray", max_lag: int) -> "np.ndarray":
    centered = transmission-transmission.mean(axis=0,keepdims=True)
    unit = centered/np.linalg.norm(centered,axis=0,keepdims=True)
    n = transmission.shape[1]
    rho = [np.mean(np.sum(unit[:,:n-k]*unit[:,k:],axis=0)) for k in range(max_lag+1)]
    return np.r_[transmission.sum(axis=0).mean(),rho]

import numpy as np
from scipy.optimize import brentq

def fit_lorentzian(profile: "np.ndarray", width_interval: "np.ndarray") -> "np.ndarray":
    rho = profile[2:]
    k2 = np.arange(1,len(rho)+1,dtype=float)**2
    lo, hi = np.log(width_interval)
    def _loss_derivative(u):
        a2 = np.exp(2*u)
        h = a2/(k2+a2)
        hp = 2*h*(1-h)
        return float(2*np.dot(h-rho,hp))
    points = [lo,hi]
    if _loss_derivative(lo)*_loss_derivative(hi)<0:
        points.append(brentq(_loss_derivative,lo,hi,xtol=1e-13))
    values = [np.sum((np.exp(2*u)/(k2+np.exp(2*u))-rho)**2) for u in points]
    index = int(np.argmin(values))
    a = width_interval[index] if index<2 else np.exp(points[index])
    return np.array([a,values[index]])

import numpy as np

def asymptotic_trace(normalized_width: float, throughput: float, n_ports: int, n_channels: int) -> float:
    a = normalized_width
    beta = n_ports/n_channels
    j = 2*np.sinh(np.sqrt(2)*beta*np.pi*a)*np.arctan(np.tanh(np.pi*a/2))/(np.pi**2*a*a)
    return float(n_ports*n_ports*n_channels*j/((n_channels-n_ports)*throughput*throughput))

import numpy as np
from scipy.optimize import brentq

def _split(jet):
    """Gradient and Hessian from a six-component jet."""
    return np.array([jet[1], jet[2]]), np.array([[jet[3], jet[4]], [jet[4], jet[5]]])

def _inside(point, bounds):
    return (bounds[0][0] <= point[0] <= bounds[0][1]) and (bounds[1][0] <= point[1] <= bounds[1][1])

def _newton(residual, start, bounds, tolerance):
    """Damped-free Newton confined to the rectangle; None if it leaves or stalls."""
    point = np.array(start, dtype=float)
    for _ in range(60):
        values, jacobian = residual(point)
        try:
            step = np.linalg.solve(jacobian, values)
        except np.linalg.LinAlgError:
            return None
        point = point-step
        if not _inside(point, bounds):
            return None
        if np.max(np.abs(step)) < 1e-14:
            break
    if np.max(np.abs(residual(point)[0])) > tolerance:
        return None
    return point

def _distinct(points, tolerance=1e-6):
    unique = []
    for point in points:
        if not any(np.max(np.abs(point-other)) < tolerance for other in unique):
            unique.append(point)
    return unique

def stationary_design(cavity: "np.ndarray", frequencies: "np.ndarray", weights: "np.ndarray", bounds: "np.ndarray", throughput_limit: float, scan_points: "np.ndarray") -> "np.ndarray":
    def _jets(point):
        scattering = scattering_jet(cavity, frequencies, float(point[0]), float(point[1]))
        transmission = transmission_jet(scattering)
        return inverse_gram_jet(transmission, weights), transmission.sum(axis=1).mean(axis=1)

    def _stationary(point):
        gradient, hessian = _split(_jets(point)[0])
        return gradient, hessian

    def _constrained(point):
        objective, throughput = _jets(point)
        grad_f, hess_f = _split(objective)
        grad_t, hess_t = _split(throughput)
        parallel = grad_f[0]*grad_t[1]-grad_f[1]*grad_t[0]
        row = np.array([hess_f[0, 0]*grad_t[1]+grad_f[0]*hess_t[1, 0]-hess_f[1, 0]*grad_t[0]-grad_f[1]*hess_t[0, 0],
                        hess_f[0, 1]*grad_t[1]+grad_f[0]*hess_t[1, 1]-hess_f[1, 1]*grad_t[0]-grad_f[1]*hess_t[0, 1]])
        return np.array([throughput[0]-throughput_limit, parallel]), np.array([grad_t, row])

    nodes = [np.linspace(bounds[i][0], bounds[i][1], int(scan_points[i])) for i in (0, 1)]
    interior, constrained, edge = [], [], []
    for first in nodes[0]:
        for second in nodes[1]:
            start = (float(first), float(second))
            found = _newton(_stationary, start, bounds, 1e-8)
            if found is not None:
                interior.append(found)
            found = _newton(_constrained, start, bounds, 1e-7)
            if found is not None:
                constrained.append(found)
    for axis in (0, 1):
        free = 1-axis
        for fixed in bounds[axis]:
            def _at(value):
                point = np.empty(2)
                point[axis] = fixed
                point[free] = value
                return point
            # Edges are one-dimensional and cheap, so subdivide each scan cell before bracketing.
            line = np.linspace(bounds[free][0], bounds[free][1], 8*(len(nodes[free])-1)+1)
            slopes = [_split(_jets(_at(value))[0])[0][free] for value in line]
            levels = [_jets(_at(value))[1][0]-throughput_limit for value in line]
            for i in range(len(line)-1):
                if slopes[i]*slopes[i+1] < 0:
                    edge.append(_at(brentq(lambda v: _split(_jets(_at(v))[0])[0][free], line[i], line[i+1], xtol=1e-13)))
                if levels[i]*levels[i+1] < 0:
                    edge.append(_at(brentq(lambda v: _jets(_at(v))[1][0]-throughput_limit, line[i], line[i+1], xtol=1e-13)))
    corners = [np.array([a, b]) for a in bounds[0] for b in bounds[1]]
    bounds = np.asarray(bounds, dtype=float)
    best = None
    for kind, points in (("interior", _distinct(interior)), ("constrained", _distinct(constrained)),
                         ("edge", _distinct(edge)), ("corner", corners)):
        for point in points:
            objective, throughput = _jets(point)
            on_limit = kind == "constrained" or abs(throughput[0]-throughput_limit) <= 1e-12
            if not (on_limit or throughput[0] <= throughput_limit):
                continue
            if best is None or objective[0] < best[0]:
                best = (objective[0], point, 1.0 if on_limit else 0.0)
    pinned = [1.0 if min(abs(best[1][i]-bounds[i][0]), abs(best[1][i]-bounds[i][1])) <= 1e-12 else 0.0
              for i in (0, 1)]
    return np.array([best[1][0], best[1][1], best[0], best[2], pinned[0], pinned[1]])

import numpy as np

def constrained_sensitivity(transmission: "np.ndarray", weights: "np.ndarray", throughput_active: bool, pinned_axes: "np.ndarray") -> "np.ndarray":
    if not throughput_active:
        return np.zeros(3)
    objective = inverse_gram_jet(transmission, weights)
    throughput = transmission.sum(axis=1).mean(axis=1)
    grad_f, hess_f = _split(objective)
    grad_t, hess_t = _split(throughput)
    free = ~np.asarray(pinned_axes, dtype=bool)
    # Only the directions the rectangle leaves free can respond to the limit.
    multiplier = -float(grad_f[free]@grad_t[free]/(grad_t[free]@grad_t[free]))
    if free.sum() < 2:
        return np.array([multiplier, -multiplier, 0.0])
    tangent = np.array([-grad_t[1], grad_t[0]])
    tangent = tangent/np.linalg.norm(tangent)
    curvature = float(tangent@(hess_f+multiplier*hess_t)@tangent)
    return np.array([multiplier, -multiplier, curvature])

import numpy as np

def optimized_bound_comparison(n_modes: int, n_ports: int, n_channels: int, seed: int, bandwidth: float, bounds: "np.ndarray", throughput_limit: float, band_channels: int, max_lag: int, width_interval: "np.ndarray", scan_points: "np.ndarray") -> "np.ndarray":
    cavity = cavity_realization(n_modes, n_ports, seed)
    frequencies = np.linspace(-bandwidth/2, bandwidth/2, n_channels)
    weights = np.zeros(n_channels)
    start = (n_channels-band_channels)//2
    weights[start:start+band_channels] = 1.0
    design = stationary_design(cavity, frequencies, weights, bounds, throughput_limit, scan_points)
    scattering = scattering_jet(cavity, frequencies, design[0], design[1])
    transmission = transmission_jet(scattering)
    profile = spectral_correlation(transmission[0], max_lag)
    fit = fit_lorentzian(profile, width_interval)
    predicted = asymptotic_trace(fit[0], profile[0], n_ports, n_channels)*band_channels/n_channels
    sensitivity = constrained_sensitivity(transmission, weights, bool(design[3]),
                                                  np.array([bool(design[4]), bool(design[5])]))
    discrepancy = 100*(predicted/design[2]-1)
    return np.array([discrepancy, design[0], design[1], design[2], profile[0], fit[0], fit[1],
                     predicted, sensitivity[0], sensitivity[2]])
SCICODE_GOLD_EOF
