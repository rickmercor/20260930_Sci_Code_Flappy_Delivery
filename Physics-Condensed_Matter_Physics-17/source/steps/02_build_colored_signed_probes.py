"""
Greedily color a distance graph in orbital order and form its unnormalized signed probe block.

A color class aggregates distant orbitals into one probe. Color assignment is a deterministic geometric operation; the supplied signs retain unit magnitude and must not be divided by the color population.

Returns
-------
tuple (colors, probes), with integer shape (n,) and real shape (n,1+max(colors)).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def build_colored_signed_probes(distances: np.ndarray, exclusion: float, signs: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Construct fixed-order distance colors and their signed probes.

    Parameters
    ----------
    distances : np.ndarray
        Finite real, symmetric, nonnegative (n,n) distances; n>=1 and zero diagonal.
        Symmetry and zero-diagonal tolerance is absolute 1e-12.
    exclusion : float
        Nonnegative finite distance; conflicts occur strictly below this value.
    signs : np.ndarray
        Real vector of shape (n,) with values exactly +1 or -1. Integer and
        floating dtypes are both valid; no normalization is performed.

    Returns
    -------
    colors : np.ndarray
        Integer vector (n,) in orbital order. Assign the smallest nonnegative
        label absent from all earlier conflicting vertices.
    probes : np.ndarray
        Real (n,m) block, m=1+max(colors), whose columns follow increasing labels.
        Row i has signs[i] at column colors[i] and zero elsewhere.

    Raises
    ------
    ValueError
        If the distance, exclusion, or sign constraints fail.
    """
    return np.zeros(np.asarray(signs).size,dtype=int),np.zeros((np.asarray(signs).size,0),dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_colored_signed_probes(distances: np.ndarray, exclusion: float, signs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    from numbers import Real
    if np.iscomplexobj(distances) or np.iscomplexobj(signs):
        raise ValueError("inputs must be real")
    d=np.asarray(distances,dtype=float);s=np.asarray(signs,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or d.shape[0]<1 or not np.all(np.isfinite(d)) or np.any(d<0):
        raise ValueError("distances must be finite nonnegative square matrix")
    if not np.allclose(d,d.T,rtol=0,atol=1e-12) or not np.allclose(np.diag(d),0,rtol=0,atol=1e-12):
        raise ValueError("distances must be symmetric with zero diagonal")
    if isinstance(exclusion,(bool,np.bool_)) or not isinstance(exclusion,Real) or not np.isfinite(exclusion) or exclusion<0:
        raise ValueError("exclusion must be finite and nonnegative")
    n=d.shape[0]
    if s.shape!=(n,) or not np.all((s==1)|(s==-1)):
        raise ValueError("signs must have unit magnitude")
    c=np.zeros(n,dtype=int)
    for i in range(n):
        used={int(c[j]) for j in range(i) if d[i,j]<exclusion}
        label=0
        while label in used:label+=1
        c[i]=label
    p=np.zeros((n,int(c.max())+1))
    p[np.arange(n),c]=s
    return c,p

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge, and invalid cases."""
    return [{'setup': 'import numpy as np\n'
               'x=np.array([0.,1.,2.,3.]);D=np.minimum(abs(x[:,None]-x[None,:]),4-abs(x[:,None]-x[None,:]));s=np.array([1.,-1.,-1.,1.])\n'
               'def _pack_result(value):\n'
               '    colors,probes=value\n'
               '    return '
               'np.concatenate((np.array([colors.size,*probes.shape],dtype=float),colors.ravel(),probes.ravel()))\n',
      'call': '_pack_result(build_colored_signed_probes(D,1.1,s))',
      'gold_call': '_pack_result(_oracle_build_colored_signed_probes(D,1.1,s))'},
     {'setup': 'import numpy as np\n'
               'D=np.array([[0.,1.],[1.,0.]]);s=np.array([1.,-1.])\n'
               'def _pack_result(value):\n'
               '    colors,probes=value\n'
               '    return '
               'np.concatenate((np.array([colors.size,*probes.shape],dtype=float),colors.ravel(),probes.ravel()))\n',
      'call': '_pack_result(build_colored_signed_probes(D,1.0,s))',
      'gold_call': '_pack_result(_oracle_build_colored_signed_probes(D,1.0,s))'},
     {'setup': 'import numpy as np\n'
               'D=np.ones((5,5))-np.eye(5);s=np.array([1,-1,1,-1,1])\n'
               'def _pack_result(value):\n'
               '    colors,probes=value\n'
               '    return '
               'np.concatenate((np.array([colors.size,*probes.shape],dtype=float),colors.ravel(),probes.ravel()))\n',
      'call': '_pack_result(build_colored_signed_probes(D,2.0,s))',
      'gold_call': '_pack_result(_oracle_build_colored_signed_probes(D,2.0,s))'},
     {'setup': 'import numpy as np\n'
               'D=np.array([[0.,1.],[1.,0.]]);s=np.array([1.,-1.])\n'
               'def _pack_result(value):\n'
               '    colors,probes=value\n'
               '    return '
               'np.concatenate((np.array([colors.size,*probes.shape],dtype=float),colors.ravel(),probes.ravel()))\n',
      'call': '_pack_result(build_colored_signed_probes(D,0.0,s))',
      'gold_call': '_pack_result(_oracle_build_colored_signed_probes(D,0.0,s))'},
     {'setup': 'import numpy as np\n'
               'D=np.array([[0.,1.],[2.,0.]]);s=np.array([1.,-1.])\n'
               'def run_model():\n'
               '    try:\n'
               '        build_colored_signed_probes(D,1.1,s)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_build_colored_signed_probes(D,1.1,s)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'D=np.array([[0.,1.],[1.,0.]]);s=np.array([1.,0.])\n'
               'def run_model():\n'
               '    try:\n'
               '        build_colored_signed_probes(D,1.1,s)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_build_colored_signed_probes(D,1.1,s)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
