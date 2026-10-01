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

from numpy.polynomial.legendre import leggauss

def _cdf_real_array(value, shape, name):
    try:
        a = np.asarray(value)
        if np.iscomplexobj(a):
            raise ValueError(name + ' must be real')
        a = np.asarray(a, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + ' must be a real numeric array') from exc
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(name + ' has invalid shape or nonfinite entries')
    return a

def _cdf_dt(value):
    try:
        dt = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('dt must be a positive scalar') from exc
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be a positive scalar')
    return dt

def _cdf_rule(_cache={}):
    if 'rule' in _cache:
        return _cache['rule']
    xi, wx = leggauss(24)
    tau, wt = leggauss(10)
    tau, wt = (tau + 1) / 2, wt / 2
    phi = np.column_stack((xi*(xi-1)/2, 1-xi*xi, xi*(xi+1)/2))
    dphi = np.column_stack((xi-.5, -2*xi, xi+.5))
    nodal = np.zeros((6*len(xi), 12))
    deriv = np.zeros_like(nodal)
    for e in range(6):
        for local, global_node in enumerate((2*e, 2*e+1, (2*e+2) % 12)):
            nodal[e*len(xi):(e+1)*len(xi), global_node] = phi[:, local]
            deriv[e*len(xi):(e+1)*len(xi), global_node] = dphi[:, local]
    time_basis = np.column_stack((np.ones(len(tau)), 2*tau-1))
    tests = (time_basis[:, None, :, None]*nodal[None, :, None, :]).reshape(-1, 24)
    gradients = (time_basis[:, None, :, None]*deriv[None, :, None, :]).reshape(-1, 24)
    weights = np.outer(wt, np.tile(wx, 6)).ravel()
    out = tau, nodal, deriv, tests, gradients, weights
    _cache['rule'] = out
    return out

def _cdf_history_domain(history):
    # The projection is affine in xi and quadratic in normalized time.
    # Its global minimum is among spatial endpoints and the endpoints
    # or interior stationary point of each scalar time quadratic.
    endpoint_derivatives = np.array([[-1.5, 2., -.5], [.5, -2., 1.5]])
    for e in range(6):
        ids = [2*e, 2*e+1, (2*e+2) % 12]
        chord = history[0, ids[2]]-history[0, ids[0]]
        chord_length = np.linalg.norm(chord)
        if not np.isfinite(chord_length) or chord_length == 0:
            raise ValueError('initial element chord has no direction')
        direction = chord/chord_length
        coefficients = np.einsum('ji,aic->ajc', endpoint_derivatives,
                                 history[:, ids, :]) @ direction
        for endpoint in range(2):
            c0, c1, c2 = coefficients[:, endpoint]
            minimum = min(c0, c0+c1+c2)
            if c2 > 0:
                stationary = -c1/(2*c2)
                if 0 < stationary < 1:
                    minimum = min(minimum, c0+stationary*(c1+stationary*c2))
            if not np.isfinite(minimum) or minimum <= 0:
                raise ValueError('history violates the positive tangent-projection domain')

def _cdf_forms(history, dt, _cache={}):
    h = _cdf_real_array(history, (3, 12, 2), 'history')
    dt = _cdf_dt(dt)
    key = (h.tobytes(), dt)
    if _cache.get('key') == key:
        return _cache['value']
    _cdf_history_domain(h)
    tau, nodal, deriv, tests, gradients, weights = _cdf_rule()
    x = h[0] + tau[:, None, None]*h[1] + tau[:, None, None]**2*h[2]
    tangent = np.einsum('qi,tic->tqc', deriv, x).reshape(-1, 2)
    jac = np.sqrt(np.sum(tangent*tangent, axis=1))
    if not np.all(np.isfinite(jac)) or np.min(jac) <= 0:
        raise ValueError('degenerate curve geometry')
    normal_density = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    physical_weights = dt*weights
    scalar = gradients.T @ ((physical_weights/jac)[:, None]*gradients)
    blocks = [tests.T @ ((physical_weights*normal_density[:, c])[:, None]*tests)
              for c in range(2)]
    normal = np.stack(blocks, axis=-1).reshape(24, 48)
    vector = np.kron(scalar, np.eye(2))
    work = (gradients.T @ ((physical_weights/jac)[:, None]*tangent)).ravel()
    out = vector, normal, scalar, work
    _cache.clear()
    _cache.update(key=key, value=out)
    return out

def _cdf_saddle(vector, normal, rhs):
    block = np.block([[vector, normal.T], [normal, np.zeros((24, 24))]])
    try:
        solution = np.linalg.solve(block, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError('constrained projection is not uniquely solvable') from exc
    error = np.max(np.abs(block @ solution-rhs))
    if not np.all(np.isfinite(solution)) or error > 2e-9*max(1., float(np.max(np.abs(rhs)))):
        raise ValueError('constrained projection is numerically singular')
    return solution

def _cdf_lengths(history):
    h = _cdf_real_array(history, (3, 12, 2), 'history')
    xi, weights = leggauss(48)
    derivative = np.column_stack((xi-.5, -2*xi, xi+.5))
    values = []
    for nodes in (h[0], h[0]+h[1]+h[2]):
        total = 0.
        for e in range(6):
            local = nodes[[2*e, 2*e+1, (2*e+2) % 12]]
            tangent = derivative @ local
            slope = local[0]-2*local[1]+local[2]
            intercept = (local[2]-local[0])/2
            den = np.dot(slope, slope)
            loc = np.clip(-np.dot(slope, intercept)/den, -1, 1) if den else 0.
            if np.linalg.norm(intercept+loc*slope) <= 0:
                raise ValueError('degenerate endpoint curve')
            total += float(weights @ np.sqrt(np.sum(tangent*tangent, axis=1)))
        values.append(total)
    return np.asarray(values)

def normal_diffusion_cost(history: "np.ndarray", velocity: "np.ndarray", dt: float) -> float:
    v = _cdf_real_array(velocity, (2, 12, 2), 'velocity')
    _, normal, scalar, _ = _cdf_forms(history, dt)
    action = normal @ v.ravel()
    kernel = np.zeros((24, 2))
    kernel[:12, 0] = 1.
    kernel[12:, 1] = 1.
    if np.max(np.abs(kernel.T @ action)) > 5e-11*max(1., float(np.linalg.norm(action))):
        raise ValueError('normal action is outside the scalar diffusion range')
    augmented = np.block([[scalar, kernel], [kernel.T, np.zeros((2, 2))]])
    try:
        potential = np.linalg.solve(augmented, np.r_[action, 0., 0.])[:24]
    except np.linalg.LinAlgError as exc:
        raise ValueError('scalar diffusion norm is not defined') from exc
    cost = float(action @ potential)
    if not np.isfinite(cost) or cost < -1e-10:
        raise ValueError('invalid diffusion cost')
    return max(0., cost)

import numpy as np

def minimum_deformation_lift(history: "np.ndarray", normal_action: "np.ndarray", dt: float) -> "np.ndarray":
    if np.shape(normal_action) != (2, 12):
        raise ValueError('normal_action must have shape (2,12)')
    action = _cdf_real_array(normal_action, (2, 12), 'normal_action')
    vector, normal, _, _ = _cdf_forms(history, dt)
    solution = _cdf_saddle(vector, normal, np.r_[np.zeros(48), action.ravel()])
    return solution[:48].reshape(2, 12, 2)

import numpy as np

def normal_length_force(history: "np.ndarray", dt: float) -> "np.ndarray":
    if np.shape(history) != (3, 12, 2):
        raise ValueError('history must have shape (3,12,2)')
    vector, normal, _, work = _cdf_forms(history, dt)
    auxiliary = _cdf_saddle(vector, normal, np.r_[-work, np.zeros(24)])
    return -auxiliary[48:].reshape(2, 12)

import numpy as np

def fixed_geometry_response(history: "np.ndarray", dt: float) -> "np.ndarray":
    if np.shape(history) != (3, 12, 2):
        raise ValueError('history must have shape (3,12,2)')
    force = normal_length_force(history, dt)
    _, _, scalar, _ = _cdf_forms(history, dt)
    action = -(scalar @ force.ravel()).reshape(2, 12)
    return minimum_deformation_lift(history, action, dt)

import numpy as np

from scipy.optimize import root

def _cdf_history_from_velocity(nodes, velocity, dt):
    v = np.reshape(velocity, (2, 12, 2))
    return np.array([nodes, dt*(v[0]-v[1]), dt*v[1]])

def discrete_curve_trajectory(initial_nodes: "np.ndarray", dt: float) -> "np.ndarray":
    nodes = _cdf_real_array(initial_nodes, (12, 2), 'initial_nodes')
    dt = _cdf_dt(dt)
    initial_history = np.array([nodes, np.zeros_like(nodes), np.zeros_like(nodes)])
    guess = fixed_geometry_response(initial_history, dt).ravel()
    # A static path has exactly zero linear temporal response coefficient.
    # Remove roundoff there before MINPACK chooses relative difference steps.
    guess[24:] = 0.
    # Continue from a short regular slab using the same finite spaces.
    schedule = []
    slab = min(dt, .001)
    while slab < dt:
        schedule.append(slab)
        slab = min(dt, 2*slab)
    schedule.append(dt)
    for slab in schedule:
        def _residual(coefficients):
            path = _cdf_history_from_velocity(nodes, coefficients, slab)
            return coefficients-fixed_geometry_response(path, slab).ravel()
        try:
            result = root(_residual, guess, method='hybr', options={'xtol': 2e-11})
            error = np.max(np.abs(_residual(result.x)))
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            raise ValueError('regular finite trajectory solve failed') from exc
        if not np.all(np.isfinite(result.x)) or error > 2e-8:
            raise ValueError('regular finite trajectory solve did not converge')
        guess = result.x
    return _cdf_history_from_velocity(nodes, guess, dt)

import numpy as np

from numpy.polynomial.legendre import leggauss

def relative_curve_length_loss(history: "np.ndarray") -> float:
    lengths = _cdf_lengths(history)
    answer = float(100*(lengths[0]-lengths[1])/lengths[0])
    if not np.isfinite(answer):
        raise ValueError('curve-length loss is not finite')
    return answer

import numpy as np

def curve_diffusion_length_loss(initial_nodes: "np.ndarray", dt: float) -> float:
    history = discrete_curve_trajectory(initial_nodes, dt)
    response = fixed_geometry_response(history, dt)
    force = normal_length_force(history, dt)
    _, normal, _, _ = _cdf_forms(history, dt)
    action = (normal @ response.ravel()).reshape(2, 12)
    lifted = minimum_deformation_lift(history, action, dt)
    dissipation = normal_diffusion_cost(history, lifted, dt)
    answer = relative_curve_length_loss(history)
    lengths = _cdf_lengths(history)
    physical_velocity = np.array([(history[1]+history[2])/dt, history[2]/dt])
    scale = max(1., float(np.max(np.abs(response))))
    if np.max(np.abs(physical_velocity-response)) > 3e-8*scale:
        raise ValueError('trajectory and response disagree')
    if np.max(np.abs(lifted-response)) > 3e-9*scale:
        raise ValueError('response is not its minimum-deformation representative')
    work = float(force.ravel() @ action.ravel())
    if abs(work+dissipation) > 2e-10*max(1., dissipation):
        raise ValueError('diffusion response and length work disagree')
    if abs(lengths[0]-lengths[1]-dissipation) > 2e-10*max(1., lengths[0]):
        raise ValueError('geometric and dissipative length losses disagree')
    return float(answer)
SCICODE_GOLD_EOF
