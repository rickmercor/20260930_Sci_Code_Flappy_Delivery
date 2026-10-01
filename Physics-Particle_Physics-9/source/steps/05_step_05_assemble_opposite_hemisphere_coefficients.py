"""
Integrate the opposite-hemisphere contributions II1 and II2.

For sectors II1 and II2, in that order, call assemble_laurent_sector_rule, evaluate assemble_qqbar_sector_jets on the rule's first five columns, pass those points and jets into assemble_nonabelian_sector_jets with the requested moments, and contract with combine_laurent_sector_coefficients. Return these two coefficient rows before the symmetry multiplier. These contributions use the double-collinear sector maps of Table V with the source corrections stated in Step 1 and exact hemisphere domains. Process all rule rows, using contiguous slices to keep memory bounded; any partition into nonoverlapping slices is allowed. Sum the five contraction outputs over slices. The returned coefficients must agree componentwise with the exact value `ref` of the prescribed rule within `1e-5*(1+abs(ref))`; dtype and summation order are not prescribed. Propagate predecessor ValueError for invalid input.

Returns
-------
coefficients : ndarray, shape (2,5) Sector rows II1,II2; eps powers -4 through zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_opposite_hemisphere_coefficients(order: int = 16, moments: tuple = (0, 0, 0, 0, 0)) -> np.ndarray:
    """Integrate the opposite-hemisphere contributions II1 and II2.

    Parameters
    ----------
    order : int, optional
        Common quadrature order, default sixteen, one through sixteen.
    moments : sequence of five integers, optional
        Coordinate powers, default all zero.

    Returns
    -------
    coefficients : ndarray, shape (2,5)
        Sector rows II1,II2; eps powers -4 through zero.

    Raises
    ------
    ValueError
        Invalid order or moments, propagated from the predecessors.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_opposite_hemisphere_coefficients(order: int = 16, moments: tuple = (0, 0, 0, 0, 0)) -> np.ndarray:
    import numpy as np
    coefficients = []
    for sector in ('II1','II2'):
        _oracle_assemble_laurent_sector_rule(sector, order, 0, 0)
        count = (order+1)**3*(2*order+1 if sector == 'II2' else order+1)*order
        parts = []
        for start in range(0,count,16384):
            rule = _oracle_assemble_laurent_sector_rule(sector, order, start, min(start+16384,count))
            qqbar = _oracle_assemble_qqbar_sector_jets(sector, rule[:,:5])
            jets = _oracle_assemble_nonabelian_sector_jets(sector, rule[:,:5], qqbar, moments)
            parts.append(_oracle_combine_laurent_sector_coefficients(rule,jets))
        coefficients.append(np.sum(parts,axis=0,dtype=np.longdouble))
    return np.asarray(coefficients, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'order=4\nmoments=(0,0,0,0,0)\n', 'call': '#case:normal\nassemble_opposite_hemisphere_coefficients(order,moments)', 'gold_call': '_oracle_assemble_opposite_hemisphere_coefficients(order,moments)', 'tol': 1e-05},
     {'setup': 'order=2\nmoments=(0, 0, 0, 0, 0)\n', 'call': '#case:boundary\nassemble_opposite_hemisphere_coefficients(order,moments)', 'gold_call': '_oracle_assemble_opposite_hemisphere_coefficients(order,moments)', 'tol': 1e-05},
     {'setup': 'order=3\nmoments=(1, 0, 0, 0, 0)\n', 'call': '#case:edge\nassemble_opposite_hemisphere_coefficients(order,moments)', 'gold_call': '_oracle_assemble_opposite_hemisphere_coefficients(order,moments)', 'tol': 1e-05},
     {'setup': 'order=2\nmoments=(0, 1, 0, 0, 1)\n', 'call': '#case:normal\nassemble_opposite_hemisphere_coefficients(order,moments)', 'gold_call': '_oracle_assemble_opposite_hemisphere_coefficients(order,moments)', 'tol': 1e-05},
     {'setup': 'def check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n', 'call': '#case:edge\ncheck(lambda:assemble_opposite_hemisphere_coefficients(True))', 'gold_call': 'check(lambda:_oracle_assemble_opposite_hemisphere_coefficients(True))'}]
