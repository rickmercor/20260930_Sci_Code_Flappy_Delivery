"""
Propagate the same state with the independent periodic-grid Strang reference.

Propagate the same initial state and potential with the paper's independent periodic-grid reference. At each interval use the potential at the temporal midpoint for equal half potential phases around the kinetic FFT phase `exp[-i*hbar*(kx^2+ky^2)*dt/(2m)]`; use NumPy FFT ordering with `2*pi*fftfreq`. Solver implementations call the public derivative function for potential values; the private oracle chains its `_oracle_` twin.

Returns
-------
One packed complex split-step field on the y-major grid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def strang_split_step_reference(grid_x, grid_y, q0, p0, initial_gamma_diag,
                                 final_time, step_count, potential_parameters,
                                 mass=1.0, hbar=1.0):
    """Return a float array of shape (len(grid_y),len(grid_x),2).

    The last axis is [real(psi),imag(psi)].

    Raises
    ------
    ValueError
        If the uniform grids, state, time, steps, constants, or potential are invalid.
    """
    return np.empty((len(grid_y), len(grid_x), 2), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _initial_grid_gaussian(points, q0, p0, gamma_diag, hbar=1.0):
    import numpy as np
    points = np.asarray(points, float)
    q0 = np.asarray(q0, float)
    p0 = np.asarray(p0, float)
    gamma = np.diag(np.asarray(gamma_diag, float))
    dr = points - q0
    pref = (np.linalg.det(gamma) / (np.pi * hbar) ** 2) ** 0.25
    exponent = -0.5 * np.einsum('ni,ij,nj->n', dr, gamma, dr) / hbar + 1j * (dr @ p0) / hbar
    return pref * np.exp(exponent)

def _grid_potential_tuple(points, time, potential_parameters):
    import numpy as np
    packed = _oracle_interference_potential_derivatives(points, time, potential_parameters)
    value = packed[:, 0]
    gradient = packed[:, 1:3]
    hessian = np.empty((len(packed), 2, 2), dtype=float)
    hessian[:, 0, 0] = packed[:, 3]
    hessian[:, 0, 1] = packed[:, 4]
    hessian[:, 1, 0] = packed[:, 4]
    hessian[:, 1, 1] = packed[:, 5]
    return value, gradient, hessian

def _oracle_strang_split_step_reference(grid_x, grid_y, q0, p0, initial_gamma_diag, final_time, step_count, potential_parameters, mass=1.0, hbar=1.0):
    import numpy as np
    initial_gamma = initial_gamma_diag
    steps = step_count
    params = potential_parameters
    x=np.asarray(grid_x,float); y=np.asarray(grid_y,float)
    q0=np.asarray(q0,float); p0=np.asarray(p0,float); initial_gamma=np.asarray(initial_gamma,float)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 3 or len(y) < 3 or q0.shape != (2,) or p0.shape != (2,) or initial_gamma.shape != (2,):
        raise ValueError('grid and initial-state shapes are invalid')
    if not np.all(np.isfinite(np.r_[x,y,q0,p0,initial_gamma,final_time,mass,hbar])) or np.any(initial_gamma <= 0) or final_time <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError('grid and physical inputs must be finite and positive where required')
    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
        raise ValueError('step_count must be a positive integer')
    if not np.allclose(np.diff(x),x[1]-x[0]) or not np.allclose(np.diff(y),y[1]-y[0]):
        raise ValueError('grids must be uniform')
    _oracle_interference_potential_derivatives(np.array([[x[0],y[0]]]),0.0,params)
    X,Y=np.meshgrid(x,y,indexing='xy')
    pts=np.column_stack((X.ravel(),Y.ravel()))
    psi=_initial_grid_gaussian(pts,q0,p0,initial_gamma,hbar).reshape(len(y),len(x))
    dx=x[1]-x[0]; dy=y[1]-y[0]; dt=final_time/steps
    kx=2*np.pi*np.fft.fftfreq(len(x),d=dx)
    ky=2*np.pi*np.fft.fftfreq(len(y),d=dy)
    KX,KY=np.meshgrid(kx,ky,indexing='xy')
    kinetic=np.exp(-1j*hbar*(KX*KX+KY*KY)*dt/(2*mass))
    for s in range(steps):
        tm=(s+0.5)*dt
        V=_grid_potential_tuple(pts,tm,params)[0].reshape(len(y),len(x))
        half=np.exp(-0.5j*dt*V/hbar)
        psi=half*psi
        psi=np.fft.ifft2(kinetic*np.fft.fft2(psi))
        psi=half*psi
    return np.stack((psi.real, psi.imag), axis=-1).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nx=np.linspace(-4,4,16,endpoint=False);y=np.linspace(-3,3,12,endpoint=False);v=np.array([.3,0.,1.,1.,1.2,.4,2.1,.2])", "call":"strang_split_step_reference(x,y,np.array([-1.,0.]),np.array([1.,0.]),np.ones(2),1.,12,v)", "gold_call":"_oracle_strang_split_step_reference(x,y,np.array([-1.,0.]),np.array([1.,0.]),np.ones(2),1.,12,v)"},
        {"setup":"import numpy as np\nx=np.linspace(-3,3,12,endpoint=False);y=np.linspace(-2,2,8,endpoint=False);v=np.array([0.,0.,1.,1.,1.,1.,1.,0.])", "call":"strang_split_step_reference(x,y,np.zeros(2),np.array([.5,-.2]),np.ones(2),.5,4,v)", "gold_call":"_oracle_strang_split_step_reference(x,y,np.zeros(2),np.array([.5,-.2]),np.ones(2),.5,4,v)"},
        {"setup":"import numpy as np\nx=np.linspace(-5,5,20,endpoint=False);y=np.linspace(-2,2,10,endpoint=False);v=np.array([.8,.2,.6,1.7,-1.1,.9,3.,-.4])", "call":"strang_split_step_reference(x,y,np.array([-2.,.4]),np.array([2.,.1]),np.array([.7,1.2]),1.4,17,v,.8)", "gold_call":"_oracle_strang_split_step_reference(x,y,np.array([-2.,.4]),np.array([2.,.1]),np.array([.7,1.2]),1.4,17,v,.8)"}
    ]
