"""
Reconstruct a symmetric operator from a signed probe-action block, then apply the spatial and offdiagonal-magnitude masks.

Sparse extraction is linear before magnitude thresholding but is not multiplicative. Averaging the two directional estimates before discarding small entries is consequential when compressed actions contain color contamination.

Returns
-------
np.ndarray of shape (n,n), symmetric after the declared spatial and magnitude masks.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral, Real
import numpy as np

def extract_sparse_operator(action: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, keep_radius: float, magnitude_threshold: float=0.0) -> np.ndarray:
    """Extract one real symmetric locally supported operator.

    Parameters
    ----------
    action : np.ndarray
        Finite real action block (n,m), with m equal to the number of colors.
    distances : np.ndarray
        Finite nonnegative symmetric (n,n) distances, diagonal zero; n>=1.
        Absolute symmetry and zero-diagonal tolerance is 1e-12.
    colors : np.ndarray
        Integer-valued (n,) labels with all labels 0,...,m-1 represented.
        Integer-valued floating inputs are also valid.
    signs : np.ndarray
        Real (n,) values exactly +1 or -1; integer and float dtypes accepted.
    keep_radius : float
        Nonnegative finite inclusive spatial support radius.
    magnitude_threshold : float
        Nonnegative finite threshold applied to offdiagonal entries AFTER
        symmetrization. Equality is retained; diagonals are exempt.

    Returns
    -------
    matrix : np.ndarray
        Real symmetric (n,n) matrix. Before masking,
        T_ij=(signs[j]*action[i,colors[j]]+signs[i]*action[j,colors[i]])/2.
        Entries with distance>keep_radius vanish. Offdiagonal entries also
        vanish when abs(T_ij)<magnitude_threshold. No normalization or
        element thresholding of action precedes the symmetric reconstruction.

    Raises
    ------
    ValueError
        If shapes, finite-real data, distance conditions, labels, signs,
        keep_radius, or magnitude_threshold violate these constraints.
    """
    return np.zeros(np.asarray(distances).shape,dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_extract_sparse_operator(action: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, keep_radius: float, magnitude_threshold: float=0.0) -> np.ndarray:
    import numpy as np
    from numbers import Real
    if any(np.iscomplexobj(v) for v in (action,distances,colors,signs)):
        raise ValueError("all arrays must be real")
    z=np.asarray(action,dtype=float);d=np.asarray(distances,dtype=float)
    craw=np.asarray(colors);s=np.asarray(signs,dtype=float)
    if d.ndim!=2 or d.shape[0]!=d.shape[1] or d.shape[0]<1 or not np.all(np.isfinite(d)) or np.any(d<0):
        raise ValueError("distances must be finite nonnegative square matrix")
    n=d.shape[0]
    if not np.allclose(d,d.T,rtol=0,atol=1e-12) or not np.allclose(np.diag(d),0,rtol=0,atol=1e-12):
        raise ValueError("distances must be symmetric with zero diagonal")
    if craw.shape!=(n,) or not np.issubdtype(craw.dtype,np.number) or not np.all(np.isfinite(craw)) or np.any(craw<0) or np.any(craw!=np.floor(craw)):
        raise ValueError("colors must be nonnegative integer-valued labels")
    c=craw.astype(int);m=int(c.max())+1
    if not np.array_equal(np.unique(c),np.arange(m)):
        raise ValueError("colors must be contiguous")
    if z.shape!=(n,m) or not np.all(np.isfinite(z)):
        raise ValueError("action shape or values invalid")
    if s.shape!=(n,) or not np.all((s==1)|(s==-1)):
        raise ValueError("signs must have magnitude one")
    for value in (keep_radius,magnitude_threshold):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not np.isfinite(value) or value<0:
            raise ValueError("cutoffs must be finite nonnegative scalars")
    one_sided=z[:,c]*s[None,:]
    t=0.5*(one_sided+one_sided.T)
    keep=(d<=keep_radius)&((np.abs(t)>=magnitude_threshold)|np.eye(n,dtype=bool))
    return np.where(keep,t,0.0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary, edge, and invalid cases."""
    return [{'setup': 'import numpy as np\n'
               'D=np.array([[0.,1.,4.,5.],[1.,0.,3.,4.],[4.,3.,0.,1.],[5.,4.,1.,0.]])\n'
               'c=np.array([0,1,0,1]);s=np.array([1.,-1.,-1.,1.]);Z=np.array([[2.,1.],[-1.5,-3.],[-4.,-.4],[.5,5.]])',
      'call': 'extract_sparse_operator(Z,D,c,s,1.0)',
      'gold_call': '_oracle_extract_sparse_operator(Z,D,c,s,1.0)'},
     {'setup': 'import numpy as np\n'
               'D=np.array([[0.,1.],[1.,0.]]);c=np.array([0,1]);s=np.ones(2);Z=np.array([[.01,.125],[.375,.02]])',
      'call': 'extract_sparse_operator(Z,D,c,s,1.0,0.25)',
      'gold_call': '_oracle_extract_sparse_operator(Z,D,c,s,1.0,0.25)'},
     {'setup': 'import numpy as np\n'
               'D=np.ones((3,3))-np.eye(3);c=np.arange(3);s=np.ones(3);Z=np.array([[.01,.2,.5],[.4,.02,-.6],[.1,.2,.03]])',
      'call': 'extract_sparse_operator(Z,D,c,s,1.0,0.31)',
      'gold_call': '_oracle_extract_sparse_operator(Z,D,c,s,1.0,0.31)'},
     {'setup': 'import numpy as np\n'
               'D=np.ones((3,3))-np.eye(3);c=np.arange(3);s=np.ones(3);Z=np.arange(9.).reshape(3,3)',
      'call': 'extract_sparse_operator(Z,D,c,s,0.0,100.0)',
      'gold_call': '_oracle_extract_sparse_operator(Z,D,c,s,0.0,100.0)'},
     {'setup': 'import numpy as np\n'
               'D=np.array([[0.,1.],[1.,0.]]);c=np.array([0,2]);s=np.ones(2);Z=np.ones((2,3))\n'
               'def run_model():\n'
               '    try:\n'
               '        extract_sparse_operator(Z,D,c,s,1.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_extract_sparse_operator(Z,D,c,s,1.0)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'},
     {'setup': 'import numpy as np\n'
               'D=np.array([[0.,1.],[1.,0.]]);c=np.arange(2);s=np.ones(2);Z=np.eye(2)\n'
               'def run_model():\n'
               '    try:\n'
               '        extract_sparse_operator(Z,D,c,s,1.0,-0.1)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_extract_sparse_operator(Z,D,c,s,1.0,-0.1)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
