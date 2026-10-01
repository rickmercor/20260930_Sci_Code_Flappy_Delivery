"""
Return the single-site loop currents, affinities, product flux, and dissipation per product.

Externally driven conformational switching can sustain nonequilibrium cycle currents and energetic costs.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cycle_dissipation(stationary: "np.typing.ArrayLike", u0: float, u1: float, u2: float, w0: float, w1: float, beta0: float, x: float, y: float, gamma1: float, gamma2: float) -> "np.ndarray":
    """Return the source-defined loop quantities and dissipation per product.

    Parameters
    ----------
    stationary : array-like
        Normalized time-stationary probabilities in the order
        [C, CS, CI, C*, CS*, CI*]. The first three states are empty,
        substrate-bound and inhibitor-bound conformation A; stars denote B.
    u0, u1, u2 : float
        Positive finite type-A substrate-binding, inhibitor-binding and
        product-formation rates in inverse seconds.
    w0, w1 : float
        Positive finite type-A substrate and inhibitor dissociation rates
        in inverse seconds.
    beta0 : float
        Positive finite type-B substrate-dissociation rate in inverse seconds.
    x, y : float
        Positive finite dimensionless landscape parameters.
    gamma1, gamma2 : float
        Positive finite A-to-B and B-to-A switching rates in inverse seconds,
        common to all three chemical occupancies.

    Returns
    -------
    ndarray
        Shape (7,), ordered as
        [J_sub, J_inh, A_sub, A_inh, J_product, sigma, DeltaW].
        J_sub and J_inh are positive for A-to-B flow on the substrate-bound
        and inhibitor-bound switching edges; affinities use those orientations.
        The two currents, product flux and sigma have units of inverse seconds;
        sigma is the dissipation rate divided by k_B T. The affinities are
        dimensionless. DeltaW is the numerical dissipation per product with
        k_B T as the energy unit.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_cycle_dissipation(stationary: "np.typing.ArrayLike", u0: float, u1: float, u2: float, w0: float, w1: float, beta0: float, x: float, y: float, gamma1: float, gamma2: float) -> "np.ndarray":
    p = np.asarray(stationary, dtype=float)
    alpha0, alpha1, alpha2, beta1, _, _ = _oracle_parallel_rates(u0, u1, u2, w1, beta0, x, y)
    j_sub = p[1] * gamma1 - p[4] * gamma2
    j_inh = p[2] * gamma1 - p[5] * gamma2
    a_sub = np.log((u0 * (beta0 + alpha2)) / ((w0 + u2) * alpha0))
    a_inh = np.log((u1 * beta1) / (w1 * alpha1))
    j_product = p[1] * u2 + p[4] * alpha2
    sigma = j_sub * a_sub + j_inh * a_inh
    delta_w = sigma / j_product
    return np.array([j_sub, j_inh, a_sub, a_inh, j_product, sigma, delta_w], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            'setup': 'q=_oracle_master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9); q_gold=_oracle_master_generator(10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9); p=_oracle_stationary_distribution(q); p_gold=_oracle_stationary_distribution(q_gold)',
            'call': 'cycle_dissipation(p,10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9)',
            'gold_call': '_oracle_cycle_dissipation(p_gold,10.,3.,1.,4.,1/3,1.,.1,.1,2.3,.9)',
            'tol': 1e-10,
        },
        {
            'setup': 'q=_oracle_master_generator(4.,2.,.5,1.2,.25,1.7,1.,1.,.4,.4); q_gold=_oracle_master_generator(4.,2.,.5,1.2,.25,1.7,1.,1.,.4,.4); p=_oracle_stationary_distribution(q); p_gold=_oracle_stationary_distribution(q_gold)',
            'call': 'cycle_dissipation(p,4.,2.,.5,1.2,.25,1.7,1.,1.,.4,.4)',
            'gold_call': '_oracle_cycle_dissipation(p_gold,4.,2.,.5,1.2,.25,1.7,1.,1.,.4,.4)',
            'tol': 1e-10,
        },
        {
            'setup': 'q=_oracle_master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05); q_gold=_oracle_master_generator(8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05); p=_oracle_stationary_distribution(q); p_gold=_oracle_stationary_distribution(q_gold)',
            'call': 'cycle_dissipation(p,8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05)',
            'gold_call': '_oracle_cycle_dissipation(p_gold,8.,.7,2.4,.2,.9,.3,.04,3.2,7.,.05)',
            'tol': 1e-10,
        },
        {
            'setup': 'q=_oracle_master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.); q_gold=_oracle_master_generator(.8,6.,.2,5.,2.,5.,4.,.25,.08,9.); p=_oracle_stationary_distribution(q); p_gold=_oracle_stationary_distribution(q_gold)',
            'call': 'cycle_dissipation(p,.8,6.,.2,5.,2.,5.,4.,.25,.08,9.)',
            'gold_call': '_oracle_cycle_dissipation(p_gold,.8,6.,.2,5.,2.,5.,4.,.25,.08,9.)',
            'tol': 1e-10,
        },
    ]
