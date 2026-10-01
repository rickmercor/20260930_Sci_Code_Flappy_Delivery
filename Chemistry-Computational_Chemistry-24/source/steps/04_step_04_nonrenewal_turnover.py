"""
Return long-run turnover statistics from the next-product summaries.

Successive product intervals need not share the same initial conformation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nonrenewal_turnover(first_passage: "np.typing.ArrayLike") -> "np.ndarray":
    """Combine next-product statistics across successive turnovers.

    ``first_passage`` is the ``(6,3)`` output of ``product_first_passage``. Product
    events leave the next turnover in either state ``C`` (row 0) or ``C*`` (row 3).

    Returns
    -------
    ndarray
        ``[tau_dynamic, pi_C, pi_Cstar, P_CC, P_CCstar, P_CstarC, P_CstarCstar]``.
        ``P`` is the embedded post-product conformation transition matrix and
        ``pi`` is its stationary row distribution.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_nonrenewal_turnover(first_passage: "np.typing.ArrayLike") -> "np.ndarray":
    import numpy as np
    fp = np.asarray(first_passage, dtype=float)
    p = fp[[0,3], 1:3]
    a = p.T - np.eye(2)
    a[-1,:] = 1.0
    b = np.array([0.0, 1.0])
    pi = np.linalg.solve(a, b)
    tau = float(pi @ fp[[0,3], 0])
    return np.array([tau, pi[0], pi[1], p[0,0], p[0,1], p[1,0], p[1,1]], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'q=_oracle_master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9); q_gold=_oracle_master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9); fp=_oracle_product_first_passage(q,1.,10.); fp_gold=_oracle_product_first_passage(q_gold,1.,10.)',
            'call': 'nonrenewal_turnover(fp)',
            'gold_call': '_oracle_nonrenewal_turnover(fp_gold)',
            'tol': 1e-10,
        },
        {
            'setup': 'import numpy as np; fp=np.zeros((6,3)); fp[:,0]=[2,1,3,4,1,2]; fp[:,1:]=[.5,.5]; fp[3,1:]=[.5,.5]; fp_gold=fp.copy()',
            'call': 'nonrenewal_turnover(fp)',
            'gold_call': '_oracle_nonrenewal_turnover(fp_gold)',
            'tol': 1e-12,
        },
        {
            'setup': 'q=_oracle_master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05); q_gold=_oracle_master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05); fp=_oracle_product_first_passage(q,2.4,60.); fp_gold=_oracle_product_first_passage(q_gold,2.4,60.)',
            'call': 'nonrenewal_turnover(fp)',
            'gold_call': '_oracle_nonrenewal_turnover(fp_gold)',
            'tol': 1e-10,
        },
        {
            'setup': 'q=_oracle_master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.); q_gold=_oracle_master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.); fp=_oracle_product_first_passage(q,.2,.05); fp_gold=_oracle_product_first_passage(q_gold,.2,.05)',
            'call': 'nonrenewal_turnover(fp)',
            'gold_call': '_oracle_nonrenewal_turnover(fp_gold)',
            'tol': 1e-10,
        },
    ]
