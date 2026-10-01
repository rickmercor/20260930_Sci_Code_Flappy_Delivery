"""
Return next-product event statistics for every microscopic starting state.

Product events can leave the catalyst in either empty conformation, so turnover retains conformation-dependent information.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def product_first_passage(generator: "np.typing.ArrayLike", u2: float, alpha2: float) -> "np.ndarray":
    """Return the mean time and post-product conformation probabilities for the next product event.

    ``generator`` is the six-state row generator in the fixed state order. Product
    release occurs only on ``CS -> C`` with rate ``u2`` and ``CS* -> C*`` with
    rate ``alpha2``. These two product channels terminate the current product interval.

    Returns
    -------
    ndarray
        Real ``(6,3)`` array. Column 0 contains mean first-passage times; columns
        1 and 2 contain probabilities that the product event leaves the catalyst
        in ``C`` and ``C*`` respectively.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_product_first_passage(generator: "np.typing.ArrayLike", u2: float, alpha2: float) -> "np.ndarray":
    import numpy as np
    q = np.asarray(generator, dtype=float)
    transient = q.copy()
    absorb = np.zeros((6, 2), dtype=float)
    transient[1,0] -= u2
    transient[4,3] -= alpha2
    absorb[1,0] = u2
    absorb[4,1] = alpha2
    times = np.linalg.solve(transient, -np.ones(6))
    probs = np.linalg.solve(transient, -absorb)
    return np.column_stack((times, probs))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'q=_oracle_master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9); q_gold=_oracle_master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9)',
            'call': 'product_first_passage(q,1.,10.)',
            'gold_call': '_oracle_product_first_passage(q_gold,1.,10.)',
            'tol': 1e-10,
        },
        {
            'setup': 'q=_oracle_master_generator(4.,2.,.5,1.2,.25,1.7,1.,1.,.4,.4); q_gold=_oracle_master_generator(4.,2.,.5,1.2,.25,1.7,1.,1.,.4,.4)',
            'call': 'product_first_passage(q,.5,.5)',
            'gold_call': '_oracle_product_first_passage(q_gold,.5,.5)',
            'tol': 1e-10,
        },
        {
            'setup': 'q=_oracle_master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05); q_gold=_oracle_master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05)',
            'call': 'product_first_passage(q,2.4,60.)',
            'gold_call': '_oracle_product_first_passage(q_gold,2.4,60.)',
            'tol': 1e-10,
        },
        {
            'setup': 'q=_oracle_master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.); q_gold=_oracle_master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.)',
            'call': 'product_first_passage(q,.2,.05)',
            'gold_call': '_oracle_product_first_passage(q_gold,.2,.05)',
            'tol': 1e-10,
        },
    ]
