"""
Curvature-dependent target spacing with a finite physical neighborhood.

curvature supplies the scalar curvature measure of the previous step at each sample. kref is a reference value carried in the same inverse-length units as curvature, and tau weights the departure of the local curvature from it. radius is a physical Euclidean distance in the same length units as points, never a multiple of a spacing. Column 0 is the spacing this task attributes to a sample from its own curvature alone; column 1 is the spacing it attributes to the same sample once the samples lying within radius of it are taken into account.

Definitions used by this task (all arrays indexed in input row order, and d_ij the Euclidean distance between samples i and j):

raw_i = h0 / sqrt(1 + tau * |curvature_i - kref|) h_i = min over all j with d_ij <= radius of raw_j (j = i included)

Column 0 is raw_i and column 1 is h_i. The neighborhood includes its center and its boundary.

Return an (n,2) float array in the input row order.

Columns are [own-curvature spacing, characteristic spacing], both strictly
positive lengths in the units of points. points has shape (n,3) with n>=1
and curvature has shape (n,) with every entry nonnegative. h0 and radius
are strictly positive scalars; tau and kref are nonnegative scalars. All
array entries and all scalars are finite.

Raises
------
ValueError
    If points or curvature has the wrong shape, if n<1, if any entry of
    points or curvature is not finite, if any curvature entry is negative,
    if h0 or radius is not strictly positive, if tau or kref is negative,
    or if any of h0, tau, kref or radius is not finite.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def characteristic_lengths(points: 'np.ndarray | list | tuple', curvature: 'np.ndarray | list | tuple', h0: float, tau: float, kref: float, radius: float) -> 'np.ndarray':
    """Return an (n,2) float array in the input row order.

    Columns are [own-curvature spacing, characteristic spacing], both strictly
    positive lengths in the units of points. points has shape (n,3) with n>=1
    and curvature has shape (n,) with every entry nonnegative. h0 and radius
    are strictly positive scalars; tau and kref are nonnegative scalars. All
    array entries and all scalars are finite.

    Raises
    ------
    ValueError
        If points or curvature has the wrong shape, if n<1, if any entry of
        points or curvature is not finite, if any curvature entry is negative,
        if h0 or radius is not strictly positive, if tau or kref is negative,
        or if any of h0, tau, kref or radius is not finite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_characteristic_lengths(points: 'np.ndarray | list | tuple', curvature: 'np.ndarray | list | tuple', h0: float, tau: float, kref: float, radius: float) -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,float);c=np.asarray(curvature,float)
    if x.ndim!=2 or x.shape[1:]!=(3,) or len(x)<1 or c.shape!=(len(x),):
        raise ValueError('invalid length-field shapes')
    if not np.isfinite(x).all() or not np.isfinite(c).all() or np.any(c<0):
        raise ValueError('invalid length-field data')
    if not np.isfinite([h0,tau,kref,radius]).all() or h0<=0 or radius<=0 or tau<0 or kref<0:
        raise ValueError('invalid length parameters')
    raw=h0/np.sqrt(1+tau*abs(c-kref))
    d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    h=np.min(np.where(d<=radius,raw[None,:],np.inf),axis=1)
    return np.column_stack((raw,h))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.4,0.,0.],[2.,0.,0.]]);c=np.array([1.,3.,2.])',
      'call': 'characteristic_lengths(x,c,.42,.5,0.,.5)',
      'gold_call': '_oracle_characteristic_lengths(x,c,.42,.5,0.,.5)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nx=np.array([[0.,0.,0.],[.5,0.,0.]]);c=np.array([1.,4.])',
      'call': 'characteristic_lengths(x,c,1.,1.,0.,.5)',
      'gold_call': '_oracle_characteristic_lengths(x,c,1.,1.,0.,.5)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nx=[[0.,0.,0.]];c=[2.]',
      'call': 'characteristic_lengths(x,c,.8,0.,0.,1.)',
      'gold_call': '_oracle_characteristic_lengths(x,c,.8,0.,0.,1.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.35,0.,0.],[1.5,0.,0.]]);c=np.array([1.,4.,2.5])',
      'call': 'characteristic_lengths(x,c,.5,.8,2.,.4)',
      'gold_call': '_oracle_characteristic_lengths(x,c,.5,.8,2.,.4)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'x=np.array([[0.,0.,0.],[.5,0.,0.],[3.,0.,0.]]);c=np.array([0.,9.,4.])',
      'call': 'characteristic_lengths(x,c,1.,1.,0.,.5)',
      'gold_call': '_oracle_characteristic_lengths(x,c,1.,1.,0.,.5)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _bad_public():\n'
               '    try: characteristic_lengths([[0.,0.,0.]],[1.],.42,.5,0.,0.); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2\n'
               'def _bad_gold():\n'
               '    try: _oracle_characteristic_lengths([[0.,0.,0.]],[1.],.42,.5,0.,0.); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': '_bad_public()',
      'gold_call': '_bad_gold()',
      'tol': 1e-09}]
