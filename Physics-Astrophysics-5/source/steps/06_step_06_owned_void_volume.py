"""
Measure the largest void from its unique cell ownership map. Accept a nonnegative integer cubic array with side 5 through 65 and labels at most the number of grid cells. Ignore label zero, count each positive label and multiply the maximum count by cell_side**3; no positive label gives zero. The physical cell side is a finite nonnegative real scalar. Return a native float in cubic length units, raising ValueError for invalid inputs or a positive volume outside the finite float range.

The measurement uses cells actually assigned to each owner, not enclosing-box volume or the sum of overlapping primitive volumes. All cells have the same physical side, so the final ranking depends only on integer owned-cell counts. No minimum-radius rejection or cavity filling is applied at this measurement stage.

Returns
-------
native float, maximum owned-cell physical volume
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def owned_void_volume(ownership, cell_side):
    """Return the largest deduplicated physical void volume.

    Parameters
    ----------
    ownership : array_like
        Nonnegative integer cubic ownership grid, side 5 through 65.
    cell_side : float
        Nonnegative finite physical cell side.

    Returns
    -------
    float
        Maximum positive-label cell count times the physical cell volume.

    Raises
    ------
    ValueError
        If input values are invalid or the result is not representable.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_owned_void_volume(ownership, cell_side):
    try:o=np.asarray(ownership)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError("invalid grid") from exc
    if o.ndim!=3 or len(set(o.shape))!=1 or not 5<=o.shape[0]<=65 or o.dtype.kind not in 'iu':
        raise ValueError("integer cubic grid required")
    if np.any(o<0) or np.any(o>o.size):raise ValueError("invalid ownership labels")
    if isinstance(cell_side,(bool,np.bool_)) or not np.isscalar(cell_side) or np.iscomplexobj(cell_side):
        raise ValueError("real cell side required")
    try:h=float(cell_side)
    except (TypeError,ValueError,OverflowError) as exc:raise ValueError("invalid cell side") from exc
    if not np.isfinite(h) or h<0:raise ValueError("nonnegative finite cell side required")
    _,counts=np.unique(o[o>0],return_counts=True)
    maximum=int(counts.max(initial=0))
    from fractions import Fraction
    value=maximum*Fraction(h)**3
    try:answer=float(value)
    except OverflowError as exc:raise ValueError("volume is not representable") from exc
    if not np.isfinite(answer):raise ValueError("volume is not finite")
    return answer

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    base='o=np.zeros((7,7,7),dtype=int);o[1:4,1:4,1:4]=2;o[4:6,4:6,4:6]=1\n'
    cases=[
        {'setup':base,'call':'owned_void_volume(o,.37)','gold_call':'_oracle_owned_void_volume(o,.37)'},
        {'setup':base,'call':'owned_void_volume(o,0.)','gold_call':'_oracle_owned_void_volume(o,0.)'},
        {'setup':base+'o[:]=0\n','call':'owned_void_volume(o,1e200)','gold_call':'_oracle_owned_void_volume(o,1e200)'},
        {'setup':base,'call':'owned_void_volume(o,1e-100)','gold_call':'_oracle_owned_void_volume(o,1e-100)'},
    ]
    for args in ('o,-1.','o,1e200','o.astype(float),1.'):
        setup=base
        for name,function in (('run_model','owned_void_volume'),('run_gold','_oracle_owned_void_volume')):
            setup+=f'def {name}():\n    try:\n        {function}({args})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})
    return cases
