"""
Compute the diffusion coefficient of hot-excited interlayer excitons in a twisted R-stacked MoSe2/WSe2 heterobilayer after phonon relaxation through the moire mini-bands.

The chain composes every earlier step for the interlayer K-K exciton, electron in MoSe2 and hole in WSe2:
the moire reciprocal lattice from the MoSe2 lattice constant 0.327 nm; the exciton moire amplitude from the
R-type band parameters (gamma1, gamma2) = (-4.389, -6.178) meV for the MoSe2 conduction band and
(-1.467, -5.856) meV for the WSe2 valence band, with m_e = 0.64, m_h = 0.51 and the 1s binding energy
173 meV; the lowest 6 mini-bands on the shifted nk x nk grid with plane waves up to n_shell = 2; the scattering
weights at the lattice temperature; self-consistent dephasings by plain iteration from 1.0 meV to a largest
update below 1e-8 meV (at most 2000 updates); the rate matrix at those dephasings; the occupation after
t_eval from a uniform window of half-width half_width about e_center above the lowest mini-band energy; and
the relaxation-time diffusion coefficient of that occupation with the self-consistent rates.

Returns
-------
d : float -- Diffusion coefficient in cm^2/s.
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def moire_exciton_diffusion(theta_deg: float, temperature: float, nk: int, e_center: float, half_width: float,
                            t_eval: float) -> float:
    """Return the diffusion coefficient of the relaxed moire exciton population.

    Parameters
    ----------
    theta_deg : float
        Twist angle, degrees.
    temperature : float
        Lattice temperature, K.
    nk : int
        Grid points per moire reciprocal direction.
    e_center : float
        Centre of the uniform excitation window above the lowest mini-band energy, meV.
    half_width : float
        Half-width of the excitation window, meV.
    t_eval : float
        Time after excitation at which the occupation is evaluated, ps.

    Returns
    -------
    d : float
        Diffusion coefficient in cm^2/s.

    Raises
    ------
    ValueError
        If no state lies in the excitation window or the dephasing iteration does not converge.
    """
    return d

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_moire_exciton_diffusion(theta_deg: float, temperature: float, nk: int, e_center: float,
                                    half_width: float, t_eval: float) -> float:
    m_e, m_h, e_b = 0.64, 0.51, 173.0
    b = _oracle_moire_reciprocal_lattice(theta_deg, 0.327)
    theta = _oracle_interlayer_moire_coupling(np.array([-4.389, -6.178]), np.array([-1.467, -5.856]),
                                              m_e, m_h, e_b, float(np.linalg.norm(b[0])))
    q_pts, e, c, v2, basis = _oracle_moire_exciton_bands(b, theta, m_e, m_h, 2, nk, 6)
    weights = _oracle_scattering_weights(q_pts, c, basis, b, m_e, m_h, e_b, temperature)
    gam = _oracle_self_consistent_dephasing(e, weights, 1.0, 1e-8, 2000)
    rates = _oracle_transition_rate_matrix(e, weights, gam)
    n = _oracle_relaxed_distribution(rates, e, e_center, half_width, t_eval)
    return _oracle_rta_diffusion_coefficient(v2, rates, n)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np",
         "call": "moire_exciton_diffusion(3.0, 10.0, 3, 60.0, 6.0, 100.0)",
         "gold_call": "_oracle_moire_exciton_diffusion(3.0, 10.0, 3, 60.0, 6.0, 100.0)"},
        {"setup": "import numpy as np",
         "call": "moire_exciton_diffusion(4.0, 70.0, 3, 40.0, 10.0, 20.0)",
         "gold_call": "_oracle_moire_exciton_diffusion(4.0, 70.0, 3, 40.0, 10.0, 20.0)"},
        # coarse grid, early time: the population is still hot
        {"setup": "import numpy as np",
         "call": "moire_exciton_diffusion(3.0, 30.0, 2, 50.0, 15.0, 1.0)",
         "gold_call": "_oracle_moire_exciton_diffusion(3.0, 30.0, 2, 50.0, 15.0, 1.0)"},
    ]
