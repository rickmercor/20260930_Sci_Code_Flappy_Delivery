"""
Select potential cosmic-void seeds from density ratio and divergence. A cell is a candidate when density_ratio-1 is strictly below density_threshold and divergence is strictly positive, independently of the later growth divergence threshold. Return integer coordinate rows (i,j,k) in decreasing divergence order, breaking ties by ascending lexicographic (k,j,i). Inputs are finite real cubic arrays of identical shape with side 5 through 65 and nonnegative density ratio; density_threshold is a finite real scalar. An empty selection has shape (0,3). Invalid inputs raise ValueError.

A strongly expanding underdensity is visited before a weaker one because an accepted cube later excludes seeds inside its cell box. Candidate selection does not itself exclude the grid boundary or already claimed cells. These are distinct operations in the subsequent growth stage.

Returns
-------
integer coordinate array, ordered candidate seeds
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def void_candidates(density_ratio, divergence, density_threshold):
    """Return sorted candidate seed coordinates.

    Parameters
    ----------
    density_ratio : array_like
        Nonnegative finite real cubic array with side 5 through 65.
    divergence : array_like
        Finite real array with the same shape.
    density_threshold : float
        Finite density-contrast cutoff, not a density-ratio cutoff.

    Returns
    -------
    ndarray
        Integer array of shape (m,3) in candidate visitation order.

    Raises
    ------
    ValueError
        If any array or scalar violates the stated contract.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_void_candidates(density_ratio, divergence, density_threshold):
    arrays = []
    for value in (density_ratio, divergence):
        try:
            raw = np.asarray(value)
            if raw.dtype.kind not in 'iuf' or raw.ndim != 3 or len(set(raw.shape)) != 1 or not 5 <= raw.shape[0] <= 65:
                raise ValueError("real cubic field required")
            arr = raw.astype(float)
        except (TypeError, OverflowError) as exc:
            raise ValueError("invalid field") from exc
        if not np.isfinite(arr).all():
            raise ValueError("finite fields required")
        arrays.append(arr)
    d,v = arrays
    if d.shape != v.shape or np.any(d < 0):
        raise ValueError("invalid density or field shape")
    if isinstance(density_threshold,(bool,np.bool_)) or not np.isscalar(density_threshold) or np.iscomplexobj(density_threshold):
        raise ValueError("real scalar required")
    try:
        threshold = float(density_threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("real scalar required") from exc
    if not np.isfinite(threshold):
        raise ValueError("finite threshold required")
    cells = np.argwhere((d-1 < threshold) & (v > 0))
    result = sorted(cells.tolist(),key=lambda p:(-v[tuple(p)],p[2],p[1],p[0]))
    return np.array(result,dtype=int).reshape(-1,3)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base = 'd=np.ones((7,7,7)); v=np.zeros_like(d)\nd[2,3,4]=.2;d[4,3,2]=.1;v[2,3,4]=2;v[4,3,2]=3\n'
    cases = [
        {'setup':base,'call':'void_candidates(d,v,-.6)','gold_call':'_oracle_void_candidates(d,v,-.6)'},
        {'setup':'d=np.ones((5,5,5));v=np.ones_like(d)\n','call':'void_candidates(d,v,-.6)','gold_call':'_oracle_void_candidates(d,v,-.6)'},
        {'setup':base+'v[4,3,2]=2\n','call':'void_candidates(d,v,-.6)','gold_call':'_oracle_void_candidates(d,v,-.6)'},
        {'setup':base+'d[2,3,4]=.4;v[4,3,2]=0\n','call':'void_candidates(d,v,-.6)','gold_call':'_oracle_void_candidates(d,v,-.6)'}]
    for args in ('d,v,float("inf")','d[:2],v,-.6','-d,v,-.6'):
        setup=base
        for name, function in (('run_model','void_candidates'),('run_gold','_oracle_void_candidates')):
            setup+=f'def {name}():\n    try:\n        {function}({args})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
