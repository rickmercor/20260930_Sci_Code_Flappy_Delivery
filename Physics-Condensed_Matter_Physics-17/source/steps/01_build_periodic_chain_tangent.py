"""
Construct the periodic distances, overlap, bare Hamiltonian, and their frozen-support derivative with respect to one orbital position.

A force in a nonorthogonal localized basis depends on derivatives of both the Hamiltonian and the overlap. Periodic image branches and sparsity masks are held fixed while the smooth orbital matrix elements are differentiated.

Returns
-------
np.ndarray of shape (5,n,n), ordered D, S, H0, dS/dx_a, dH0/dx_a.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def build_periodic_chain_tangent(positions: np.ndarray, length: float, force_index: int) -> np.ndarray:
    """Construct the ring operators and the displacement tangent.

    Parameters
    ----------
    positions : np.ndarray
        Finite, strictly increasing real vector x of length n >= 2 in [0, length).
    length : float
        Finite positive ring length, in bohr.
    force_index : int
        Orbital index a in [0,n); Python and NumPy integers are accepted, not bool.

    Returns
    -------
    operators : np.ndarray
        Real array of shape (5,n,n), ordered [D,S,H0,dS,dH0]. D is in bohr,
        S is dimensionless, H0 is in eV, dS in bohr**(-1), and dH0 in eV/bohr.

    Raises
    ------
    ValueError
        If positions are not a finite strictly increasing real vector in the
        stated interval, length is not a positive finite scalar, or the index
        is not an integer in the stated range.

    Notes
    -----
    Set k=2*pi/length, delta_ij=x_i-x_j-length*floor((x_i-x_j)/length+1/2),
    D_ij=abs(delta_ij), and phase_ij=k*(x_i+x_j). The negative image is chosen
    at a half-ring tie. Differentiate each i<j pair once and mirror its
    derivative to j,i; this defines the one-sided tangent at a half-ring tie.
    For i != j:
    S_ij=0.16*exp(-D_ij/0.85)*(1+0.05*cos(phase_ij)) for D_ij<=2.25;
    H0_ij=-0.90*exp(-D_ij/1.25)*(1+0.10*cos(phase_ij)) for D_ij<=3.25.
    Other offdiagonal entries vanish. S_ii=1+0.03*cos(k*x_i) and
    H0_ii=-0.25+0.35*cos(k*x_i)+0.08*sin(2*k*x_i)+0.025*(-1)**i.
    The tangents differentiate these smooth expressions with respect to x_a,
    holding image integers and support masks fixed, including at a mask tie.
    No derivative of a cutoff or of a minimum-image switch is included.
    """
    return np.zeros((5,np.asarray(positions).size,np.asarray(positions).size),dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_periodic_chain_tangent(positions: np.ndarray, length: float, force_index: int) -> np.ndarray:
    import numpy as np
    from numbers import Real, Integral
    if np.iscomplexobj(positions):
        raise ValueError("positions must be real")
    x=np.asarray(positions,dtype=float)
    if isinstance(length,(bool,np.bool_)) or not isinstance(length,Real) or not np.isfinite(length) or length<=0:
        raise ValueError("length must be positive and finite")
    length=float(length)
    if x.ndim!=1 or x.size<2 or not np.all(np.isfinite(x)) or np.any(x<0) or np.any(x>=length) or np.any(np.diff(x)<=0):
        raise ValueError("positions must be finite, ordered and distinct in [0,length)")
    if isinstance(force_index,(bool,np.bool_)) or not isinstance(force_index,Integral) or not 0<=force_index<x.size:
        raise ValueError("force_index is out of range")
    n=x.size; k=2*np.pi/length
    raw=x[:,None]-x[None,:]
    delta=raw-length*np.floor(raw/length+0.5)
    d=np.abs(delta)
    phase=k*(x[:,None]+x[None,:])
    unit=np.zeros(n); unit[int(force_index)]=1.0
    dd=np.sign(delta)*(unit[:,None]-unit[None,:])
    dd=np.triu(dd,1); dd=dd+dd.T
    dphase=k*(unit[:,None]+unit[None,:])
    sfactor=np.exp(-d/0.85); hfactor=np.exp(-d/1.25)
    s=0.16*sfactor*(1.0+0.05*np.cos(phase))*(d<=2.25)
    h=-0.90*hfactor*(1.0+0.10*np.cos(phase))*(d<=3.25)
    ds=0.16*sfactor*(-dd/0.85*(1.0+0.05*np.cos(phase))-0.05*np.sin(phase)*dphase)*(d<=2.25)
    dh=-0.90*hfactor*(-dd/1.25*(1.0+0.10*np.cos(phase))-0.10*np.sin(phase)*dphase)*(d<=3.25)
    np.fill_diagonal(s,1.0+0.03*np.cos(k*x))
    np.fill_diagonal(h,-0.25+0.35*np.cos(k*x)+0.08*np.sin(2*k*x)+0.025*(-1.0)**np.arange(n))
    np.fill_diagonal(ds,-0.03*k*np.sin(k*x)*unit)
    np.fill_diagonal(dh,(-0.35*k*np.sin(k*x)+0.16*k*np.cos(2*k*x))*unit)
    return np.stack((d,s,h,ds,dh))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge, and invalid cases."""
    return [{'setup': 'import numpy as np\nx=np.array([0.,.7,1.9,3.1,5.2])',
      'call': 'build_periodic_chain_tangent(x,6.0,0)',
      'gold_call': '_oracle_build_periodic_chain_tangent(x,6.0,0)'},
     {'setup': 'import numpy as np\nx=np.array([0.,1.,3.25,8.])',
      'call': 'build_periodic_chain_tangent(x,10.0,3)',
      'gold_call': '_oracle_build_periodic_chain_tangent(x,10.0,3)'},
     {'setup': 'import numpy as np\nx=np.array([0.1,4.9])',
      'call': 'build_periodic_chain_tangent(x,5.0,0)',
      'gold_call': '_oracle_build_periodic_chain_tangent(x,5.0,0)'},
     {'setup': 'import numpy as np\nx=np.array([0.,2.25,3.25,8.])',
      'call': 'build_periodic_chain_tangent(x,12.0,1)',
      'gold_call': '_oracle_build_periodic_chain_tangent(x,12.0,1)'},
     {'setup': 'import numpy as np\nx=np.array([0.,2.])',
      'call': 'build_periodic_chain_tangent(x,4.0,0)',
      'gold_call': '_oracle_build_periodic_chain_tangent(x,4.0,0)'},
     {'setup': 'import numpy as np\n'
               'x=np.array([0.,0.,1.])\n'
               'def run_model():\n'
               '    try:\n'
               '        build_periodic_chain_tangent(x,6.0,0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_build_periodic_chain_tangent(x,6.0,0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'x=np.array([0.,1.])\n'
               'def run_model():\n'
               '    try:\n'
               '        build_periodic_chain_tangent(x,6.0,2)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_build_periodic_chain_tangent(x,6.0,2)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
