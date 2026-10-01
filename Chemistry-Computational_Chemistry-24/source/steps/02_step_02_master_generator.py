"""
Return the six-state transition-rate generator for one active site.

The six chemical states represent occupancy and conformation of one dynamically switching catalytic site.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def master_generator(u0: float, u1: float, u2: float, w0: float, w1: float, beta0: float, x: float, y: float, gamma1: float, gamma2: float) -> "np.ndarray":
    """Build the six-state continuous-time master-equation generator.

    State order is ``[C, CS, CI, C*, CS*, CI*]``. Use a row generator: off-diagonal
    entries are transition rates from row state to column state and every row sums to
    zero. Substrate dissociation and product formation are distinct microscopic
    channels but share the same state-to-state edge. ``gamma1`` applies A->B and
    ``gamma2`` B->A in all three chemical configurations.

    Returns
    -------
    ndarray
        Real ``(6,6)`` generator.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_master_generator(u0: float, u1: float, u2: float, w0: float, w1: float, beta0: float, x: float, y: float, gamma1: float, gamma2: float) -> "np.ndarray":
    import numpy as np
    alpha0, alpha1, alpha2, beta1, _, _ = _oracle_parallel_rates(u0, u1, u2, w1, beta0, x, y)
    q = np.zeros((6, 6), dtype=float)
    q[0,1] += u0;           q[0,2] += u1;          q[0,3] += gamma1
    q[1,0] += w0 + u2;      q[1,4] += gamma1
    q[2,0] += w1;           q[2,5] += gamma1
    q[3,4] += alpha0;       q[3,5] += alpha1;      q[3,0] += gamma2
    q[4,3] += beta0 + alpha2; q[4,1] += gamma2
    q[5,3] += beta1;        q[5,2] += gamma2
    q[np.diag_indices(6)] = -q.sum(axis=1)
    return q

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np', 'call': 'master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9)', 'gold_call': '_oracle_master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9)', 'tol': 1e-12},
        {'setup': 'import numpy as np', 'call': 'master_generator(4.,2.,.5,1.2,.25,1.7,1.,1.,.4,.4)', 'gold_call': '_oracle_master_generator(4.,2.,.5,1.2,.25,1.7,1.,1.,.4,.4)', 'tol': 1e-12},
        {'setup': 'import numpy as np', 'call': 'master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05)', 'gold_call': '_oracle_master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05)', 'tol': 1e-12},
        {'setup': 'import numpy as np', 'call': 'master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.)', 'gold_call': '_oracle_master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.)', 'tol': 1e-12},
    ]
