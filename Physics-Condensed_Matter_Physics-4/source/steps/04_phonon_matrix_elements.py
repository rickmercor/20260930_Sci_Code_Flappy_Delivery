"""
Evaluate the squared exciton-phonon coupling of one carrier's layer at a set of momentum transfers, for the longitudinal acoustic and the optical deformation-potential channels.

Within the deformation-potential approximation a carrier couples to the long-wavelength acoustic branch
through a first-order constant D1 (energy) and to a dispersionless optical branch through a zeroth-order
constant D0 (energy per length). The acoustic branch is linear, hbar Omega = hbar v_LA q, the optical branch
has a fixed energy E_op. The squared matrix element of a phonon mode of momentum q is hbar D^2/(2 rho A omega_q)
for the full deformation (with D -> D1 q for the acoustic mode), where rho is the areal mass density of the
layer and A the sample area; the returned quantities are multiplied by A. The areal mass density follows from
one formula unit per hexagonal cell of lattice constant a, with Avogadro's number 6.02214076e23 and
1 g/cm^2 = 6.241509074e10 meV ps^2 nm^-4. Because the carrier is bound in a 1s exciton and displaced from its
centre of mass by a fraction mass_ratio of the relative coordinate, the carrier's coupling is multiplied by
the form factor of the 2D hydrogenic 1s density evaluated at mass_ratio * q, whose amplitude decays as
exp(-r/a_B). Units: meV, nm, ps; hbar = 0.6582119569 meV ps.

Returns
-------
elements : np.ndarray -- Array of shape (3,) + q.shape: [A |g_ac|^2 (meV^2 nm^2), hbar Omega_ac (meV), A |g_op|^2 (meV^2 nm^2)].
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phonon_matrix_elements(q: "np.ndarray", mass_ratio: float, a_b: float, v_la: float, d1: float, d0: float,
                           e_op: float, a_lattice: float, molar_mass: float) -> "np.ndarray":
    """Return area-scaled squared couplings and the acoustic phonon energy at momentum transfers q.

    Parameters
    ----------
    q : np.ndarray
        Momentum-transfer magnitudes, nm^-1, any shape, non-negative.
    mass_ratio : float
        Fraction of the relative coordinate by which this carrier sits away from the exciton centre of mass.
    a_b : float
        Exciton radius a_B of the 1s state, nm.
    v_la : float
        Longitudinal acoustic sound velocity, nm/ps.
    d1 : float
        Acoustic deformation potential, meV.
    d0 : float
        Optical deformation potential, meV/nm.
    e_op : float
        Optical phonon energy, meV.
    a_lattice : float
        Lattice constant of the carrier's layer, nm.
    molar_mass : float
        Molar mass of one formula unit of the layer, g/mol.

    Returns
    -------
    elements : np.ndarray
        Array of shape (3,) + q.shape: [A |g_ac|^2 (meV^2 nm^2), hbar Omega_ac (meV), A |g_op|^2 (meV^2 nm^2)].
    """
    return elements

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_phonon_matrix_elements(q: "np.ndarray", mass_ratio: float, a_b: float, v_la: float, d1: float,
                                   d0: float, e_op: float, a_lattice: float, molar_mass: float) -> "np.ndarray":
    hbar = 0.6582119569
    q = np.asarray(q, dtype=float)
    rho = molar_mass / 6.02214076e23 / (np.sqrt(3.0) / 2.0 * (a_lattice * 1e-7) ** 2) * 6.241509074e10
    ff2 = _exciton_form_factor(mass_ratio * q, a_b) ** 2
    m_ac = hbar * d1 ** 2 * q / (2.0 * rho * v_la) * ff2
    omega = hbar * v_la * q
    m_op = hbar ** 2 * d0 ** 2 / (2.0 * rho * e_op) * ff2
    return np.array([m_ac, omega, m_op])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\nq = np.array([0.05, 0.4, 1.2, 3.0])",
         "call": "phonon_matrix_elements(q, 0.4435, 0.8809, 4.1, 3.4e3, 5.2e4, 36.6, 0.327, 253.86)",
         "gold_call": "_oracle_phonon_matrix_elements(q, 0.4435, 0.8809, 4.1, 3.4e3, 5.2e4, 36.6, 0.327, 253.86)"},
        {"setup": "import numpy as np\nq = np.array([[0.2, 0.9], [1.7, 2.6]])",
         "call": "phonon_matrix_elements(q, 0.5565, 0.8809, 3.3, 2.1e3, 3.1e4, 30.8, 0.325, 341.76)",
         "gold_call": "_oracle_phonon_matrix_elements(q, 0.5565, 0.8809, 3.3, 2.1e3, 3.1e4, 30.8, 0.325, 341.76)"},
        # q = 0: acoustic coupling and energy vanish, optical does not
        {"setup": "import numpy as np\nq = np.array([0.0, 1e-3])",
         "call": "phonon_matrix_elements(q, 0.5, 1.5, 5.0, 1.0e3, 1.0e4, 45.0, 0.316, 160.07)",
         "gold_call": "_oracle_phonon_matrix_elements(q, 0.5, 1.5, 5.0, 1.0e3, 1.0e4, 45.0, 0.316, 160.07)"},
    ]
