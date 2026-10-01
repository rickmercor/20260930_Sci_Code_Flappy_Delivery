"""
Compose the synthetic fields, candidate ordering, face barriers, isotropic cube growth, ordered merge and ownership-volume measurement. Use density ratio 1-0.92 exp(-s) and divergence 1.2-s+0.000001*(i+n*j+n*n*k), where s is the minimum scaled squared distance over supplied wells. Candidate contrast threshold is -0.6 with strictly positive divergence. Face thresholds are gradient 0.25, contrast 10 and divergence 0; gradient lookahead is off and derivatives use unit index spacing. Cube growth rejects current bounds at most 1 or at least n-2 and rejects radius zero. Merge uses both boxes expanded by one, fixed descending raw-volume order with acceptance-order ties, nearest-cell adoption, permanent majors and first-assignment ownership. Return the maximum owned-cell count times cell_side**3 before radius cuts or cavity filling. Accept integer n from 5 through 65, nonempty (m,3) real centers in [0,n-1], matching scales in [0.25,n] and finite nonnegative cell_side. Invalid inputs or unrepresentable final volumes raise ValueError.

This deterministic catalogue separates the effects of dynamical seed selection from the geometry of ordered merging. A connected-component union would erase the protected major owners and can join otherwise distinct underdense regions. Counting the final ownership map removes primitive overlap while retaining the nonpercolating partition.

Returns
-------
native float, largest void volume after the single-level merge pass
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def largest_void(n, centers, scales, cell_side):
    """Return the largest post-merge physical ownership volume.

    Parameters
    ----------
    n : int
        Integer grid side from 5 through 65.
    centers : array_like
        Nonempty real (m,3) well centers in [0,n-1].
    scales : array_like
        Matching real (m,3) scales in [0.25,n].
    cell_side : float
        Finite nonnegative physical cell side.

    Returns
    -------
    float
        Maximum owned-cell volume in cubic physical length units.

    Raises
    ------
    ValueError
        If any upstream contract fails or final volume is unrepresentable.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_largest_void(n, centers, scales, cell_side):
    d,v=_oracle_void_fields(n,centers,scales)
    seeds=_oracle_void_candidates(d,v,-.6)
    barriers=_oracle_face_barriers(d,v,.25,10.,0.)
    cubes=_oracle_grow_void_cubes(seeds,barriers)
    ownership=_oracle_merge_void_cubes(cubes,n)
    return _oracle_owned_void_volume(ownership,cell_side)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    fixture='centers=[(8,20,20),(32,20,20),(16,20,20),(24,20,20),(20,24,20),(16,30,22),(8,8,8),(22,18,26)]\nscales=[(6,6,6),(5.8,5.8,5.8),(3.8,3.8,3.8),(3.8,3.8,3.8),(3.5,3.5,3.5),(4.5,4.5,4.5),(4,4,4),(3.5,3.5,3.5)]\n'
    cases=[{'setup':fixture,'call':'largest_void(41,centers,scales,.37)','gold_call':'_oracle_largest_void(41,centers,scales,.37)'}]
    for args in ('13,[[6,6,6]],[[3,3,3]],1.','9,[[4,4,4]],[[.25,.25,.25]],.37','13,[[6,6,6]],[[3,3,3]],0.'):
        cases.append({'setup':'','call':f'largest_void({args})','gold_call':f'_oracle_largest_void({args})'})
    for args in ('4,[[2,2,2]],[[1,1,1]],.37','13,[[6,6,6]],[[3,3,3]],float("inf")'):
        setup=''
        for name,function in (('run_model','largest_void'),('run_gold','_oracle_largest_void')):
            setup+=f'def {name}():\n    try:\n        {function}({args})\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n'
        cases.append({'setup':setup,'call':'run_model()','gold_call':'run_gold()'})

    def _final_answer_case():
        return {'setup':'centers=[(8,20,20),(32,20,20),(16,20,20),(24,20,20),(20,24,20),(16,30,22),(8,8,8),(22,18,26)]\nscales=[(6,6,6),(5.8,5.8,5.8),(3.8,3.8,3.8),(3.8,3.8,3.8),(3.5,3.5,3.5),(4.5,4.5,4.5),(4,4,4),(3.5,3.5,3.5)]\n',
                'gold_call':'_oracle_largest_void(41,centers,scales,.37)','extract':'round(result,6)'}

    return cases
