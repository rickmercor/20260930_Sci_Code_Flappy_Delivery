"""
Calculate the composite spatial factor for a larva and its adult sibling.

Full-sibling offspring share a mother, but their laying dates can occur in either order. The adult sibling also moves after emerging, so the spatial term differs by laying order.

Returns
-------
float: mixed-stage sibling spatial factor for fixed laying days.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def larva_adult_movement(M: "np.ndarray", x1: int, x2: int, y1: int, y2: int, t2: int, te: int, tl: int, tp: int) -> float:
    """Calculate the composite spatial factor for a larva and its adult sibling.

Parameters
----------
M : square row-stochastic adult daily movement matrix
x1 : known laying site of reference larva 1
x2 : sampled location of adult sibling 2
y1,y2 : egg-laying days of siblings 1 and 2
t2 : sampling day of adult sibling 2
te,tl,tp : subadult stage durations

Returns
-------
float, conditional spatial probability of adult 2 at x2, given larva 1 at x1 and the two laying dates.

Notes
-----
Valid supplied histories have t2 >= y2+te+tl+tp. When y2 >= y1, maternal movement after laying egg 1 and movement of adult offspring 2 combine as the inclusive adult transition from x1 on day y1+te+tl+tp to x2 on day t2. When y2 < y1, condition the earlier maternal node on the known later node x1 using the normalized destination column of the inclusive transition from y2 to y1. Average the offspring transition from each earlier node on day y2+te+tl+tp to sampled node x2 at t2 using those conditional origin probabilities. This is the two-order movement law in S1 Text Eqs. 26-27."""
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_larva_adult_movement(M: "np.ndarray", x1: int, x2: int, y1: int, y2: int, t2: int, te: int, tl: int, tp: int) -> float:
    dev=te+tl+tp
    if y2>=y1:
        return float(_oracle_adult_transition(M,y1+dev,t2)[x1,x2])
    earlier=_oracle_reverse_origin(M,x1,y2,y1)
    offspring=_oracle_adult_transition(M,y2+dev,t2)
    return float(earlier @ offspring[:,x2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup='import numpy as np\nM=np.array([[.73,.14,.08,.05],[.13,.73,.05,.09],[.1,.07,.73,.1],[.05,.1,.12,.73]],dtype=float)\n'
    return [
        {"setup":setup,"call":"larva_adult_movement(M,0,2,1,3,15,2,5,1)","gold_call":"_oracle_larva_adult_movement(M,0,2,1,3,15,2,5,1)","tol":1e-9},
        {"setup":setup,"call":"larva_adult_movement(M,2,0,3,0,12,2,5,1)","gold_call":"_oracle_larva_adult_movement(M,2,0,3,0,12,2,5,1)","tol":1e-9},
        {"setup":setup,"call":"larva_adult_movement(M,1,1,2,2,10,2,5,1)","gold_call":"_oracle_larva_adult_movement(M,1,1,2,2,10,2,5,1)","tol":1e-9},
    ]
