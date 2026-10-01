"""
Sum the seven sector finite parts and apply the factor four for the other energy order and hard-leg hemisphere.

For each same-hemisphere sector I1,I2,I3,I4,I5, call assemble_laurent_sector_rule with the requested order, evaluate assemble_qqbar_sector_jets on its first five columns, pass those points and jets into assemble_nonabelian_sector_jets with the requested moments, and contract with combine_laurent_sector_coefficients, keeping each sector's finite coefficient. Obtain the II1 and II2 rows from assemble_opposite_hemisphere_coefficients with the same order and moments. Pass the five same-hemisphere coefficients and the two opposite rows to combine_sector_finite_parts and return its Python float. The result is the fixed-quadrature estimate of the finite coefficient of the integrated scalar nonabelian dipole, normalized as in Eq. (18). The moment convention for all four symmetry images is stated in Step 2. Process the I1-I5 rule rows in contiguous slices as in Step 5; any partition into nonoverlapping slices is allowed. The returned float must agree with the exact value `ref` of the prescribed rule within `1e-5*(1+abs(ref))`. Propagate predecessor ValueError for invalid input.

Returns
-------
finite_part : float Four times the sum of the seven finite-coefficient estimates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_nonabelian_integrated_finite_part(order: int = 16, moments: tuple = (0, 0, 0, 0, 0)) -> float:
    """Sum the seven sector finite parts and apply the factor four for the other energy order and hard-leg hemisphere.

    Parameters
    ----------
    order : int, optional
        Common quadrature order, default sixteen, one through sixteen.
    moments : sequence of five integers, optional
        Coordinate moment powers, default all zero.

    Returns
    -------
    finite_part : float
        Four times the sum of the seven finite-coefficient estimates.

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

def _oracle_compute_nonabelian_integrated_finite_part(order: int = 16, moments: tuple = (0, 0, 0, 0, 0)) -> float:
    import numpy as np
    _oracle_assemble_laurent_sector_rule('I1',order,0,0)
    count = (order+1)**4*order
    same = []
    for sector in ('I1','I2','I3','I4','I5'):
        parts = []
        for start in range(0,count,16384):
            rule = _oracle_assemble_laurent_sector_rule(sector, order, start, min(start+16384,count))
            qqbar = _oracle_assemble_qqbar_sector_jets(sector, rule[:,:5])
            jets = _oracle_assemble_nonabelian_sector_jets(sector, rule[:,:5], qqbar, moments)
            parts.append(_oracle_combine_laurent_sector_coefficients(rule,jets)[4])
        same.append(np.sum(parts,dtype=np.longdouble))
    opposite = _oracle_assemble_opposite_hemisphere_coefficients(order,moments)
    return _oracle_combine_sector_finite_parts(same, opposite)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'order=16\nmoments=(0,0,0,0,0)\n', 'call': '#case:normal\ncompute_nonabelian_integrated_finite_part(order,moments)', 'gold_call': '_oracle_compute_nonabelian_integrated_finite_part(order,moments)', 'tol': 1e-05},
     {'setup': 'order=2\nmoments=(0, 0, 0, 0, 0)\n', 'call': '#case:boundary\ncompute_nonabelian_integrated_finite_part(order,moments)', 'gold_call': '_oracle_compute_nonabelian_integrated_finite_part(order,moments)', 'tol': 1e-05},
     {'setup': 'order=3\nmoments=(1, 0, 0, 0, 0)\n', 'call': '#case:edge\ncompute_nonabelian_integrated_finite_part(order,moments)', 'gold_call': '_oracle_compute_nonabelian_integrated_finite_part(order,moments)', 'tol': 1e-05},
     {'setup': 'order=2\nmoments=(0, 1, 0, 0, 1)\n', 'call': '#case:normal\ncompute_nonabelian_integrated_finite_part(order,moments)', 'gold_call': '_oracle_compute_nonabelian_integrated_finite_part(order,moments)', 'tol': 1e-05},
     {'setup': 'def check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n', 'call': '#case:edge\ncheck(lambda:compute_nonabelian_integrated_finite_part(True))', 'gold_call': 'check(lambda:_oracle_compute_nonabelian_integrated_finite_part(True))'}]
