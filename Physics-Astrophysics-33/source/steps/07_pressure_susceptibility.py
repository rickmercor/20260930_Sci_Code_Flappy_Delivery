"""
Final orchestrator. Combine steps 1-6 for the optimal induced-pressure RMS

from an initially pressure-balanced, zero-mean density disturbance in a

two-dimensional periodic transverse layer. Each field is normalized by its

prescribed spatial mean. Only transverse conservative turbulent fluxes evolve.

Final orchestrator. Combine steps 1-6 for the optimal induced-pressure RMS

from an initially pressure-balanced, zero-mean density disturbance in a

two-dimensional periodic transverse layer. Each field is normalized by its

prescribed spatial mean. Only transverse conservative turbulent fluxes evolve.



Let theta=2\*pi\*x/period_x and phi=2\*pi\*y/period_y. The fixed profiles are

b=exp(field_x\*cos(theta)+field_y\*sin(phi)+field_mix\*cos(theta+2\*phi+.3)),

u=flow_scale\*(.9+.3\*sin(theta+.4)+.2\*cos(2\*phi)+.15\*sin(theta-phi+.2)),

z=z_scale\*exp(.15\*sin(2\*theta-phi-.3)),

ell=lpar_scale\*(1+.12\*cos(theta+phi+.2)). Gradients use x and y.

Use nx\*ny uniform tensor-grid nodes, flattened with y varying fastest.

Equilibrium normalizations and RMS use the arithmetic nodal mean.



Inputs: params=None selects the benchmark; otherwise a dictionary overrides

any subset of defaults: nx=21, ny=23, period_x=period_y=2\*pi, time=1200,

field_x=.8, field_y=.65, field_mix=.4, flow_scale=1, z_scale=.04,

lpar_scale=.8, rho_mean=p_mean=chi=1, gamma=5/3.

Returns: out, one finite nonnegative dimensionless pressure susceptibility.

Returns
-------
out, one finite nonnegative dimensionless pressure susceptibility.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pressure_susceptibility(params: dict | None = None) -> float:
    """Return the complete two-dimensional equilibrium/response scalar.

    params is None or a dictionary overriding the named benchmark inputs
    above. Return one native Python float: induced-pressure RMS divided by
    initial-density RMS, each expressed relative to its prescribed mean.
    Raise ValueError for unknown keys, non-dicts, nonscalar/nonfinite values,
    nonintegral/even node counts or counts<3, negative time/flow_scale/z_scale,
    nonpositive periods/lpar_scale/rho_mean/p_mean/chi, gamma<=1, or a
    nonfinite derived state. Field amplitudes may be finite signed reals.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_pressure_susceptibility(params: dict | None = None) -> float:
    p=dict(nx=21,ny=23,period_x=2*np.pi,period_y=2*np.pi,time=1200.,
           field_x=.8,field_y=.65,field_mix=.4,flow_scale=1.,z_scale=.04,
           lpar_scale=.8,rho_mean=1.,p_mean=1.,chi=1.,gamma=5/3)
    if params is not None:
        if not isinstance(params,dict) or set(params)-set(p): raise ValueError('invalid parameters')
        p.update(params)
    for key,val in p.items():
        a=np.asarray(val,dtype=float)
        if a.ndim or not np.isfinite(a): raise ValueError('nonfinite or nonscalar parameter')
        p[key]=float(a)
    for key in ['nx','ny']:
        if p[key]!=int(p[key]) or p[key]<3 or int(p[key])%2!=1: raise ValueError('invalid grid count')
    if any(p[k]<=0 for k in ['period_x','period_y','lpar_scale','rho_mean','p_mean','chi']) or p['gamma']<=1:
        raise ValueError('invalid positive parameter')
    if any(p[k]<0 for k in ['time','flow_scale','z_scale']): raise ValueError('invalid nonnegative parameter')
    nx,ny=int(p['nx']),int(p['ny'])
    x,y=np.meshgrid(2*np.pi*np.arange(nx)/nx,2*np.pi*np.arange(ny)/ny,indexing='ij')
    sx,sy=2*np.pi/p['period_x'],2*np.pi/p['period_y']
    b=np.exp(p['field_x']*np.cos(x)+p['field_y']*np.sin(y)+p['field_mix']*np.cos(x+2*y+.3))
    u=p['flow_scale']*(.9+.3*np.sin(x+.4)+.2*np.cos(2*y)+.15*np.sin(x-y+.2))
    z=p['z_scale']*np.exp(.15*np.sin(2*x-y-.3))
    ell=p['lpar_scale']*(1+.12*np.cos(x+y+.2))
    kx=sx*(-p['field_x']*np.sin(x)-p['field_mix']*np.sin(x+2*y+.3))
    ky=sy*(p['field_y']*np.cos(y)-2*p['field_mix']*np.sin(x+2*y+.3))
    ux=sx*p['flow_scale']*(.3*np.cos(x+.4)+.15*np.cos(x-y+.2))
    uy=sy*p['flow_scale']*(-.4*np.sin(2*y)-.15*np.cos(x-y+.2))
    bg=np.column_stack([v.ravel() for v in (b,u,z,ell,kx,ky,ux,uy)])
    eq=_oracle_stationary_density(bg[:,0],bg[:,1],p['rho_mean'])
    rho=eq[:-1]
    press=_oracle_stationary_pressure(rho,p['p_mean'],p['gamma'])
    tangent=_oracle_transport_tangent(bg,rho,press,p['gamma'],p['chi'])
    generator=_oracle_flux_generator(tangent,(nx,ny),(p['period_x'],p['period_y']))
    response=_oracle_pressure_response(generator,p['time'])*p['rho_mean']/p['p_mean']
    out=_oracle_optimal_response(response)
    if not np.isfinite(out): raise ValueError('nonfinite susceptibility')
    return out

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    return [
        {'setup': '', 'call': 'pressure_susceptibility(None)', 'gold_call': '_oracle_pressure_susceptibility(None)'},
        {'setup': '', 'call': "pressure_susceptibility({'nx': 5, 'ny': 7, 'field_mix': 0.2, 'time': 400.0})", 'gold_call': "_oracle_pressure_susceptibility({'nx': 5, 'ny': 7, 'field_mix': 0.2, 'time': 400.0})"},
        {'setup': '', 'call': "pressure_susceptibility({'nx': 7, 'ny': 5, 'flow_scale': 1.5, 'time': 900.0})", 'gold_call': "_oracle_pressure_susceptibility({'nx': 7, 'ny': 5, 'flow_scale': 1.5, 'time': 900.0})"},
        {'setup': '', 'call': "pressure_susceptibility({'nx': 3, 'ny': 5, 'z_scale': 0.0})", 'gold_call': "_oracle_pressure_susceptibility({'nx': 3, 'ny': 5, 'z_scale': 0.0})"},
        {'setup': '', 'call': "pressure_susceptibility({'nx': 5, 'ny': 3, 'time': 0.0})", 'gold_call': "_oracle_pressure_susceptibility({'nx': 5, 'ny': 3, 'time': 0.0})"},
        {'setup': 'def _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n', 'call': "_status(pressure_susceptibility, {'nx': 12})", 'gold_call': "_status(_oracle_pressure_susceptibility, {'nx': 12})"},
    ]
