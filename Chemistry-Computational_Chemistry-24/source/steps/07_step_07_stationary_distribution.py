"""
Return the long-run six-state occupancies of the dynamic catalyst.

Steady occupancies quantify how the catalyst distributes among chemical and conformational states.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def stationary_distribution(generator: "np.typing.ArrayLike") -> "np.ndarray":
    """Return the normalized stationary distribution of a finite row generator.

    ``generator`` is a real square continuous-time Markov generator with a unique
    stationary distribution.

    Returns
    -------
    ndarray
        Stationary row probabilities in the generator's state order.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_stationary_distribution(generator: "np.typing.ArrayLike") -> "np.ndarray":
    import numpy as np
    q = np.asarray(generator, dtype=float)
    a = q.T.copy()
    a[-1,:] = 1.0
    b = np.zeros(q.shape[0], dtype=float)
    b[-1] = 1.0
    return np.linalg.solve(a, b)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary and edge cases with isolated inputs."""
    return [
        {
            'setup': 'q=_oracle_master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9); q_gold=_oracle_master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9)',
            'call': 'stationary_distribution(q)',
            'gold_call': '_oracle_stationary_distribution(q_gold)',
            'tol': 1e-11,
        },
        {
            'setup': 'import numpy as np; q=np.array([[-2.,2.],[1.,-1.]]); q_gold=q.copy()',
            'call': 'stationary_distribution(q)',
            'gold_call': '_oracle_stationary_distribution(q_gold)',
            'tol': 1e-12,
        },
        {
            'setup': 'q=_oracle_master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05); q_gold=_oracle_master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05)',
            'call': 'stationary_distribution(q)',
            'gold_call': '_oracle_stationary_distribution(q_gold)',
            'tol': 1e-11,
        },
        {
            'setup': 'q=_oracle_master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.); q_gold=_oracle_master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.)',
            'call': 'stationary_distribution(q)',
            'gold_call': '_oracle_stationary_distribution(q_gold)',
            'tol': 1e-11,
        },
        {
            'setup': 'q=_oracle_master_generator(10.,3.,1.,4.,1/3,4.,1.,1.,2.3,.9)\nq_gold=_oracle_master_generator(10.,3.,1.,4.,1/3,4.,1.,1.,2.3,.9)',
            'call': 'stationary_distribution(q)',
            'gold_call': '_oracle_stationary_distribution(q_gold)',
            'tol': 1e-11,
        },
    ]
