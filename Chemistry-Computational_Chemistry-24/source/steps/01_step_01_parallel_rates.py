"""
Return the type-B kinetic and inhibition-equilibrium quantities for a supplied two-landscape catalytic instance.

The catalytic conformations have different kinetic landscapes whose parameters are related by the source model.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def parallel_rates(
    u0: float,
    u1: float,
    u2: float,
    w1: float,
    beta0: float,
    x: float,
    y: float,
) -> "np.ndarray":
    """Return the transformed type-B and inhibition-equilibrium quantities.

    Parameters are positive finite rates and positive dimensionless ``x`` and
    ``y``. Use the fixed landscape transformations:

    ``alpha0 = u0 * x``
    ``alpha1 = u1 * y``
    ``alpha2 = u2 / x``
    ``beta1 = w1 / y``

    Define ``Keq = u1 / w1`` and ``Keq_star = alpha1 / beta1``.

    Returns
    -------
    ndarray
        ``[alpha0, alpha1, alpha2, beta1, Keq, Keq_star]`` as finite reals,
        in exactly that order.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_parallel_rates(u0: float, u1: float, u2: float, w1: float, beta0: float, x: float, y: float) -> "np.ndarray":
    import numpy as np
    alpha0 = u0 * x
    alpha1 = u1 * y
    alpha2 = u2 / x
    beta1 = w1 / y
    keq = u1 / w1
    keq_star = alpha1 / beta1
    return np.array([alpha0, alpha1, alpha2, beta1, keq, keq_star], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': '', 'call': 'parallel_rates(10.,3.,1.,1/3,1.,.1,.1)', 'gold_call': '_oracle_parallel_rates(10.,3.,1.,1/3,1.,.1,.1)', 'tol': 1e-12},
        {'setup': '', 'call': 'parallel_rates(4.,2.,.5,.25,1.7,1.,1.)', 'gold_call': '_oracle_parallel_rates(4.,2.,.5,.25,1.7,1.,1.)', 'tol': 1e-12},
        {'setup': '', 'call': 'parallel_rates(8.,.7,2.4,.9,.3,.04,3.2)', 'gold_call': '_oracle_parallel_rates(8.,.7,2.4,.9,.3,.04,3.2)', 'tol': 1e-12},
        {'setup': '', 'call': 'parallel_rates(.8,6.,.2,2.,5.,4.,.25)', 'gold_call': '_oracle_parallel_rates(.8,6.,.2,2.,5.,4.,.25)', 'tol': 1e-12},
    ]
