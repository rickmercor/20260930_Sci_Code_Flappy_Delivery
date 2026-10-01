"""
Obtain the local tangent fluxes about the simultaneous zero-flux equilibrium.

The geometry, outward flow, dominant Elsasser amplitude z and parallel scale

ell are externally maintained, while density r and pressure p may respond.

Use v_A=b/sqrt(r), M_A=u/v_A, eta=z\*l_perp/4, and critical balance

chi=z\*ell/(v_A\*l_perp). Mass and thermal fluxes are those in steps 1 and 2.

Obtain the local tangent fluxes about the simultaneous zero-flux equilibrium.

The geometry, outward flow, dominant Elsasser amplitude z and parallel scale

ell are externally maintained, while density r and pressure p may respond.

Use v_A=b/sqrt(r), M_A=u/v_A, eta=z\*l_perp/4, and critical balance

chi=z\*ell/(v_A\*l_perp). Mass and thermal fluxes are those in steps 1 and 2.



Inputs: background has shape (N,8), columns [b,u,z,ell,k_x,k_y,u_x,u_y],

where k is the transverse gradient of ln b and (u_x,u_y) is the gradient of u;

rho and pressure are the positive equilibrium arrays of length N; gamma>1;

chi>0. Zero z and u are allowed. The gradients are dimensional in the chosen

length units; gradient(u)/v_A remains defined when u=0.



Returns: out with shape (N,7), columns [eta,a_x,a_y,h_x,h_y,c_x,c_y], defined

by delta Gamma_r=eta\*(-gradient(delta r) + a\*delta r) and

delta Gamma_p=eta\*(-gradient(delta p) + h\*delta p + c\*delta r), where

Gamma_p=(gamma-1)\*Gamma_th. This is linearization of the continuum flux

at zero flux, prior to numerical differentiation of the perturbations.

Returns
-------
out with shape (N,7), columns [eta,a_x,a_y,h_x,h_y,c_x,c_y], defined by delta Gamma_r=eta*(-gradient(delta r) + a*delta r) and delta Gamma_p=eta*(-gradient(delta p) + h*delta p + c*delta r), where Gamma_p=(gamma-1)*Gamma_th. This is linearization of the continuum flux at zero flux, prior to numerical differentiation of the perturbations.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def transport_tangent(background: np.ndarray, rho: np.ndarray, pressure: np.ndarray,
                      gamma: float, chi: float) -> np.ndarray:
    """Return the local scalar diffusivity and three transverse tangent vectors.

    background is (N,8) with columns [b,u,z,ell,k_x,k_y,u_x,u_y]; rho and
    pressure are (N,) equilibrium vectors. gamma is the heat-capacity ratio
    and chi is the critical-balance parameter. Return an (N,7) float array
    with columns [eta,a_x,a_y,h_x,h_y,c_x,c_y] defined above.

    Raise ValueError for wrong shapes, nonfinite inputs, nonpositive b/ell/rho/
    pressure/chi, negative u/z, gamma<=1, or a nonfinite derived output.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_transport_tangent(background: np.ndarray, rho: np.ndarray, pressure: np.ndarray,
                              gamma: float, chi: float) -> np.ndarray:
    bg=np.asarray(background,dtype=float)
    r,p=np.asarray(rho,dtype=float),np.asarray(pressure,dtype=float)
    g,ch=np.asarray(gamma,dtype=float),np.asarray(chi,dtype=float)
    if bg.ndim!=2 or bg.shape[1]!=8 or not len(bg) or r.shape!=(len(bg),) or p.shape!=r.shape or g.ndim or ch.ndim:
        raise ValueError('invalid shapes')
    if not all(np.all(np.isfinite(v)) for v in [bg,r,p,g,ch]): raise ValueError('nonfinite input')
    if np.any(bg[:,[0,3]]<=0) or np.any(bg[:,[1,2]]<0) or np.any(r<=0) or np.any(p<=0) or g<=1 or ch<=0:
        raise ValueError('invalid physical state')
    va=bg[:,0]/np.sqrt(r)
    eta=bg[:,2]**2*bg[:,3]/(4*va*ch)
    a=2*bg[:,4:6]+3*bg[:,6:8]/va[:,None]
    h=2*g*(bg[:,4:6]+bg[:,6:8]/va[:,None])
    c=g*p[:,None]*bg[:,6:8]/(va*r)[:,None]
    out=np.column_stack((eta,a,h,c))
    if not np.all(np.isfinite(out)): raise ValueError('nonfinite output')
    return out

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    return [
        {'setup': 'import numpy as np\nbg=np.array([[1.5,.4,.12,.7,.2,-.3,.1,.5],[.7,.9,.08,1.1,-.4,.5,-.7,.2],[1.2,0.,0.,.5,0.,.2,.3,0.]])\nr=np.array([.8,1.7,1.1]); p=np.array([.6,2.,.9])\n', 'call': 'transport_tangent(bg,r,p,1.6666666666666667,1.0)', 'gold_call': '_oracle_transport_tangent(bg,r,p,1.6666666666666667,1.0)'},
        {'setup': 'import numpy as np\nbg=np.array([[1.5,.4,.12,.7,.2,-.3,.1,.5],[.7,.9,.08,1.1,-.4,.5,-.7,.2],[1.2,0.,0.,.5,0.,.2,.3,0.]])\nr=np.array([.8,1.7,1.1]); p=np.array([.6,2.,.9])\nbg[:,2]=0.\n', 'call': 'transport_tangent(bg,r,p,1.4,0.8)', 'gold_call': '_oracle_transport_tangent(bg,r,p,1.4,0.8)'},
        {'setup': 'import numpy as np\nbg=np.array([[1.5,.4,.12,.7,.2,-.3,.1,.5],[.7,.9,.08,1.1,-.4,.5,-.7,.2],[1.2,0.,0.,.5,0.,.2,.3,0.]])\nr=np.array([.8,1.7,1.1]); p=np.array([.6,2.,.9])\nbg[:,6:8]=0.\n', 'call': 'transport_tangent(bg,r,p,1.6666666666666667,1.2)', 'gold_call': '_oracle_transport_tangent(bg,r,p,1.6666666666666667,1.2)'},
        {'setup': 'import numpy as np\nbg=np.array([[1.5,.4,.12,.7,.2,-.3,.1,.5],[.7,.9,.08,1.1,-.4,.5,-.7,.2],[1.2,0.,0.,.5,0.,.2,.3,0.]])\nr=np.array([.8,1.7,1.1]); p=np.array([.6,2.,.9])\nr*=2.; p*=.3\n', 'call': 'transport_tangent(bg,r,p,1.1,0.4)', 'gold_call': '_oracle_transport_tangent(bg,r,p,1.1,0.4)'},
        {'setup': 'import numpy as np\nbg=np.array([[1.5,.4,.12,.7,.2,-.3,.1,.5],[.7,.9,.08,1.1,-.4,.5,-.7,.2],[1.2,0.,0.,.5,0.,.2,.3,0.]])\nr=np.array([.8,1.7,1.1]); p=np.array([.6,2.,.9])\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_status(transport_tangent, bg, r, p, 5 / 3, 0.0)', 'gold_call': '_status(_oracle_transport_tangent, bg, r, p, 5 / 3, 0.0)'},
    ]
