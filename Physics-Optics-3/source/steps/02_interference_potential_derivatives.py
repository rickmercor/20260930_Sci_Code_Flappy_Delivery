"""
Evaluate the common time-dependent optical potential and its spatial derivatives.

Evaluate the prescribed localized traveling interference potential, its spatial gradient, and its symmetric spatial Hessian at all supplied points. Differentiate the Gaussian envelope and the oscillatory phase together; the mixed Hessian is returned once. Parameter order is `[amplitude,xc,wx,wy,kx,ky,omega,phase]`.

Returns
-------
One float array containing value, gradient, and independent Hessian entries.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def interference_potential_derivatives(points, time, potential_parameters):
    """Return a float array of shape (N,6).

    Columns are [V,dV_dx,dV_dy,d2V_dxx,d2V_dxy,d2V_dyy].

    Raises
    ------
    ValueError
        If inputs are non-finite, points are not (N,2), or widths are non-positive.
    """
    return np.empty((len(points), 6), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_interference_potential_derivatives(points, time, potential_parameters):
    import numpy as np
    t = time
    params = potential_parameters
    points = np.asarray(points, float)
    params = np.asarray(params, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or params.shape != (8,):
        raise ValueError('points must be nonempty (N,2) and parameters length eight')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(params)) or not np.isfinite(t):
        raise ValueError('points, time, and parameters must be finite')
    amplitude, xc, wx, wy, kx, ky, omega, phase = params
    if wx <= 0 or wy <= 0:
        raise ValueError('potential widths must be positive')
    x = points[:, 0]
    y = points[:, 1]
    dx = x - xc
    env = np.exp(-0.5 * ((dx / wx) ** 2 + (y / wy) ** 2))
    arg = kx * x + ky * y - omega * t + phase
    co = np.cos(arg)
    si = np.sin(arg)
    avec = np.column_stack((-dx / wx**2, -y / wy**2))
    kval = np.array([kx, ky])
    value = amplitude * env * co
    grad = amplitude * env[:, None] * (avec * co[:, None] - kval * si[:, None])
    hlog = np.diag([-1.0 / wx**2, -1.0 / wy**2])
    aa = np.einsum('ni,nj->nij', avec, avec)
    ak = np.einsum('ni,j->nij', avec, kval) + np.einsum('i,nj->nij', kval, avec)
    hess = amplitude * env[:, None, None] * (
        (aa + hlog[None, :, :] - np.outer(kval, kval)[None, :, :]) * co[:, None, None]
        - ak * si[:, None, None]
    )
    return np.column_stack((value, grad, hess[:, 0, 0], hess[:, 0, 1], hess[:, 1, 1])).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\np=np.array([[0.,0.],[1.,-.5],[-2.,.3]]);a=np.array([.4,0.,1.2,.8,1.5,.6,3.2,.2])", "call":"interference_potential_derivatives(p,.3,a)", "gold_call":"_oracle_interference_potential_derivatives(p,.3,a)"},
        {"setup":"import numpy as np\np=np.array([[0.,0.],[2.,1.] ]);a=np.array([0.,0.,1.,1.,2.,1.,3.,0.])", "call":"interference_potential_derivatives(p,0.,a)", "gold_call":"_oracle_interference_potential_derivatives(p,0.,a)"},
        {"setup":"import numpy as np\np=np.array([[-1.2,.7]]);a=np.array([.9,-.4,.5,2.1,-1.3,.2,.7,-.8])", "call":"interference_potential_derivatives(p,1.1,a)", "gold_call":"_oracle_interference_potential_derivatives(p,1.1,a)"}
    ]
