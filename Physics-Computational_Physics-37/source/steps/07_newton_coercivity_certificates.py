"""
Evaluate both normalized Newton-Jacobian coercivity certificates from the paper.

Let tau be the time step, mu the permeability, c_m the mass constant, rho the density lower bound, J the current bound, V_e the electron-speed bound, and r_min the effective-resistivity lower bound. Define



A0 = c_m^2 * rho,

A = A0 - tau*mu*J/4.



For the B.6 estimate, compute



B6 = mu * (1 - (J*tau + V_e*sqrt(tau))/4),

C6 = (tau/2) * (r_min - mu*V_e*sqrt(tau)/2).



For the B.7 estimate, use the coefficient obtained from the preceding Young-inequality proof, rather than the additional inner-mu factor in the printed display:



B7 = mu * (1 - (J + V_e)*tau/4),

C7 = (tau/2) * (r_min - mu*V_e/2).



Normalize the two triples by [A0, mu, tau*r_min/2] and return



[A/A0, B6/mu, C6/(tau*r_min/2),

 A/A0, B7/mu, C7/(tau*r_min/2)].



Do not clip negative coefficients.

Returns
-------
Return one length-6 real NumPy array without clipping negative entries.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def newton_coercivity_certificates(time_step, permeability, mass_constant,
                                    density_min, current_bound,
                                    electron_speed_bound, resistivity_min):
    """Evaluate the two normalized coercivity coefficient triples.

    Parameters
    ----------
    time_step : float
        Positive time step.
    permeability : float
        Positive magnetic permeability.
    mass_constant : float
        Positive mass-norm constant.
    density_min : float
        Positive density lower bound.
    current_bound : float
        Nonnegative projected-current bound.
    electron_speed_bound : float
        Nonnegative electron-speed bound.
    resistivity_min : float
        Positive effective-resistivity lower bound.
    Returns
    -------
    ndarray, shape (6,)
        Normalized [A6,B6,C6,A7,B7,C7] without clipping.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_newton_coercivity_certificates(time_step, permeability,
                                            mass_constant, density_min,
                                            current_bound, electron_speed_bound,
                                            resistivity_min):
    """Return the normalized Appendix-B.6 and Appendix-B.7 coefficients."""
    import math
    import numpy as np
    tau = float(time_step)
    mu = float(permeability)
    cm = float(mass_constant)
    rho = float(density_min)
    ch = float(current_bound)
    ce = float(electron_speed_bound)
    rmin = float(resistivity_min)
    values = (tau, mu, cm, rho, ch, ce, rmin)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("inputs must be finite")
    if tau <= 0 or mu <= 0 or cm <= 0 or rho <= 0 or ch < 0 or ce < 0 or rmin <= 0:
        raise ValueError("invalid certificate domain")
    base_a = cm * cm * rho
    a = base_a - 0.25 * tau * mu * ch
    b6 = mu * (1.0 - 0.25 * (ch * tau + ce * math.sqrt(tau)))
    c6 = 0.5 * tau * (rmin - 0.5 * mu * ce * math.sqrt(tau))
    b7 = mu * (1.0 - 0.25 * (ch + ce) * tau)
    c7 = 0.5 * tau * (rmin - 0.5 * mu * ce)
    return np.array([
        a / base_a, b6 / mu, c6 / (0.5 * tau * rmin),
        a / base_a, b7 / mu, c7 / (0.5 * tau * rmin),
    ], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'newton_coercivity_certificates(.001,1.,.92,.8,1.2,2.5,.01)', 'gold_call': '_oracle_newton_coercivity_certificates(.001,1.,.92,.8,1.2,2.5,.01)'}, {'setup': '', 'call': 'newton_coercivity_certificates(.01,2.,.8,.4,3.,1.,.2)', 'gold_call': '_oracle_newton_coercivity_certificates(.01,2.,.8,.4,3.,1.,.2)'}, {'setup': '', 'call': 'newton_coercivity_certificates(.0001,.5,1.1,1.5,.2,.3,.005)', 'gold_call': '_oracle_newton_coercivity_certificates(.0001,.5,1.1,1.5,.2,.3,.005)'}, {'setup': '', 'call': 'newton_coercivity_certificates(.02,1.5,.7,.25,5.,4.,.5)', 'gold_call': '_oracle_newton_coercivity_certificates(.02,1.5,.7,.25,5.,4.,.5)'}, {'setup': '', 'call': 'newton_coercivity_certificates(.002,1.,.92,1.,.8,1.4,.02)', 'gold_call': '_oracle_newton_coercivity_certificates(.002,1.,.92,1.,.8,1.4,.02)'}, {'setup': '', 'call': 'newton_coercivity_certificates(.0004,3.,1.2,.6,2.,6.,.04)', 'gold_call': '_oracle_newton_coercivity_certificates(.0004,3.,1.2,.6,2.,6.,.04)'}, {'setup': '', 'call': 'newton_coercivity_certificates(.03,.8,.85,2.,.1,.2,.09)', 'gold_call': '_oracle_newton_coercivity_certificates(.03,.8,.85,2.,.1,.2,.09)'}]
