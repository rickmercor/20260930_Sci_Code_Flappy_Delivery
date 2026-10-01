"""
Occupied areas and local integral support for variable target resolution.

lengths supplies the characteristic spacing of the previous step at each sample. packing is a dimensionless positive coefficient of this task's reference-area convention. Column 0 is the area this task attributes to a sample on the surface; column 1 is the dimensionless support that measures that attribution against the sample's own target resolution. Both columns are evaluated on the positions supplied in this call, never on cached positions carried over from an earlier state.

Definitions used by this task (d_ij the Euclidean distance between samples, h_i the characteristic spacing of sample i, sums over every sample j including j = i, whose distance is 0):

r_i = 2 * h_i W_ij = 3 / (pi * r_i^2) * max(1 - d_ij / r_i, 0) rho_i = sum_j (h_j / h_i)^2 * W_ij A_i = 1 / rho_i S_i = packing * h_i^2 * rho_i

Column 0 is A_i and column 1 is S_i. Every kernel sum centered on sample i uses r_i, the cutoff built from that sample's own spacing, for every term; a neighbor at d_ij >= r_i contributes exactly zero.

Return an (n,2) float array in the input row order.

Columns are [occupied area A_i, dimensionless support S_i]; column 0
carries squared-length units and column 1 is dimensionless. points has
shape (n,3) with n>=1 and lengths has shape (n,) with every entry strictly
positive. packing is a strictly positive finite scalar. All entries are
finite. Two distinct samples closer than 1e-12 lie outside this contract.
Any finite-range sum evaluated for a sample takes its range from that
sample's own characteristic spacing, never from another sample's spacing
or from any combination of two samples' spacings.

Raises
------
ValueError
    If points or lengths has the wrong shape, if n<1, if any entry of
    points, lengths or packing is not finite, if any entry of lengths or
    packing is not strictly positive, or if two distinct samples are
    separated by less than 1e-12.

Returns
-------
return
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def occupied_support(points: 'np.ndarray | list | tuple', lengths: 'np.ndarray | list | tuple', packing: float=0.933) -> 'np.ndarray':
    """Return an (n,2) float array in the input row order.

    Columns are [occupied area A_i, dimensionless support S_i]; column 0
    carries squared-length units and column 1 is dimensionless. points has
    shape (n,3) with n>=1 and lengths has shape (n,) with every entry strictly
    positive. packing is a strictly positive finite scalar. All entries are
    finite. Two distinct samples closer than 1e-12 lie outside this contract.
    Any finite-range sum evaluated for a sample takes its range from that
    sample's own characteristic spacing, never from another sample's spacing
    or from any combination of two samples' spacings.

    Raises
    ------
    ValueError
        If points or lengths has the wrong shape, if n<1, if any entry of
        points, lengths or packing is not finite, if any entry of lengths or
        packing is not strictly positive, or if two distinct samples are
        separated by less than 1e-12.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_occupied_support(points: 'np.ndarray | list | tuple', lengths: 'np.ndarray | list | tuple', packing: float=0.933) -> 'np.ndarray':
    import numpy as np
    x=np.asarray(points,float);h=np.asarray(lengths,float)
    if x.ndim!=2 or x.shape[1:]!=(3,) or len(x)<1 or h.shape!=(len(x),):
        raise ValueError('invalid support shapes')
    if not np.isfinite(x).all() or not np.isfinite(h).all() or np.any(h<=0) or not np.isfinite(packing) or packing<=0:
        raise ValueError('invalid support values')
    d=np.linalg.norm(x[:,None,:]-x[None,:,:],axis=2)
    if np.any(d[np.triu_indices(len(x),1)]<1e-12):
        raise ValueError('distinct coincident samples')
    r=2*h
    kernel=3/(np.pi*r[:,None]**2)*np.maximum(1-d/r[:,None],0)
    rho=np.sum(kernel*(h[None,:]/h[:,None])**2,axis=1)
    return np.column_stack((1/rho,packing*h*h*rho))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nx=[[0.,0.,0.],[.3,0.,0.],[.1,.3,0.]];h=[.3,.4,.25]',
      'call': 'occupied_support(x,h)',
      'gold_call': '_oracle_occupied_support(x,h)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nx=[[0.,0.,0.],[2.,0.,0.]];h=[1.,1.]',
      'call': 'occupied_support(x,h)',
      'gold_call': '_oracle_occupied_support(x,h)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nx=[[0.,0.,0.]];h=[.2]',
      'call': 'occupied_support(x,h,.9)',
      'gold_call': '_oracle_occupied_support(x,h,.9)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nx=[[0.,0.,0.],[.45,0.,0.],[0.,.9,0.]];h=[.6,.25,.5]',
      'call': 'occupied_support(x,h)',
      'gold_call': '_oracle_occupied_support(x,h)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nx=[[0.,0.,0.],[1.,0.,0.]];h=[.5,.6]',
      'call': 'occupied_support(x,h)',
      'gold_call': '_oracle_occupied_support(x,h)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _bad_public():\n'
               '    try: occupied_support([[0.,0.,0.],[0.,0.,0.]],[1.,1.]); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2\n'
               'def _bad_gold():\n'
               '    try: _oracle_occupied_support([[0.,0.,0.],[0.,0.,0.]],[1.,1.]); return 0\n'
               '    except ValueError: return 1\n'
               '    except Exception: return 2',
      'call': '_bad_public()',
      'gold_call': '_bad_gold()',
      'tol': 1e-09}]
