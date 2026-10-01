"""
Measure diffusion dissipation for a prescribed weak normal motion.

Unit mobility is used. All scalar and vector response and test fields use this continuous spatial quadratic space and are linear in time, while geometry is quadratic in time; all variational pairings are exact physical space-time integrals on the evolving curve. Use the whole-slab dual formulation of surface diffusion whose vector motion minimizes the integrated squared surface gradient for each weak normal action, where that action pairs the vector field's normal component with every scalar test through the evolving curve measure. Its discrete curvature force is the exact dual representation of the negative length variation through this minimum-deformation relation on the full weak-normal action space. The normal H-minus-one norm is dual to the square root of the integrated squared scalar surface gradient on the whole scalar test space, on actions annihilating its kernel.

Returns
-------
return cost
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def normal_diffusion_cost(history: "np.ndarray", velocity: "np.ndarray", dt: float) -> float:
    """Return the squared normal H-minus-one norm in the stated curve model.

    Parameters
    ----------
    history : np.ndarray
        Shape (3,12,2), with coefficients of (1,tau,tau**2).
    velocity : np.ndarray
        Shape (2,12,2), with coefficients of (1,2*tau-1).
    dt : float
        Positive physical slab length, where tau=t/dt.

    Returns
    -------
    float
        A finite, nonnegative squared normal H-minus-one norm.

    Raises
    ------
    ValueError
        If shapes or values are invalid, dt is not positive, the history
        is outside the stated domain, or the weak normal action is incompatible.

    Notes
    -----
    Elements e connect nodes (2e,2e+1,(2e+2)%12) at local coordinates
    (-1,0,1), using quadratic interpolation. Include the physical time
    measure, and use outward normals for counterclockwise node ordering.
    Each element tangent must have positive projection onto its initial
    endpoint chord direction throughout the entire element and time slab.

    The normal norm is dual to the square root of the physical
    space-time integral of squared scalar surface gradient on the
    whole degree-one temporal, continuous quadratic spatial scalar
    test space. Require the velocity's weak normal action to annihilate
    every test field constant in space.
    The supplied velocity need not be the history's derivative or a
    flow solution.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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

def _oracle_normal_diffusion_cost(history: "np.ndarray", velocity: "np.ndarray", dt: float) -> float:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = '''import numpy as np
theta = np.arange(12)*np.pi/6
radius = 1 + .16*np.cos(2*theta) + .07*np.sin(3*theta) + .04*np.cos(theta)
nodes = np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))
history = np.zeros((3,12,2))
history[0] = nodes
dt = .001
'''
    def _exception_case(setup, function_name, argument_text):
        for check_name, target in (('_check_public', function_name),
                                   ('_check_gold', '_oracle_'+function_name)):
            setup += ('\ndef '+check_name+'():\n'
                      '    try:\n'
                      '        '+target+'('+argument_text+')\n'
                      '    except ValueError:\n'
                      '        return 1\n'
                      '    except Exception:\n'
                      '        return 2\n'
                      '    return 0\n')
        return {'setup': setup, 'call': '_check_public()', 'gold_call': '_check_gold()'}
    return [
        {'setup': base + 'velocity=np.zeros((2,12,2)); velocity[0,:,0]=.3; velocity[0,:,1]=-.2',
         'call': 'normal_diffusion_cost(history.copy(), velocity.copy(), dt)',
         'gold_call': '_oracle_normal_diffusion_cost(history.copy(), velocity.copy(), dt)'},
        {'setup': base + 'velocity=np.zeros((2,12,2))',
         'call': 'normal_diffusion_cost(history.copy(), velocity.copy(), dt)',
         'gold_call': '_oracle_normal_diffusion_cost(history.copy(), velocity.copy(), dt)'},
        {'setup': base + 'history[1,:,0]=.02; velocity=np.zeros((2,12,2)); velocity[1,:,1]=.4',
         'call': 'normal_diffusion_cost(history.copy(), velocity.copy(), dt)',
         'gold_call': '_oracle_normal_diffusion_cost(history.copy(), velocity.copy(), dt)'},
        _exception_case(base + 'velocity=np.zeros((2,12,2)); velocity[0]=nodes',
                        'normal_diffusion_cost', 'history.copy(), velocity.copy(), dt'),
        _exception_case(base + 'history[1]=-16*nodes; history[2]=64*nodes; velocity=np.zeros((2,12,2))',
                        'normal_diffusion_cost', 'history.copy(), velocity.copy(), dt')]
