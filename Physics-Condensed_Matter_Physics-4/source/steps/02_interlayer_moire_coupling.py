"""
Compute the complex Fourier amplitude of the moire potential felt by the centre of mass of a 1s interlayer exciton.

In an R-stacked heterobilayer the band edge of each layer at the K valley is shifted periodically by the
atoms of the other layer. For one band the shift is parametrised by two energies (gamma1, gamma2), the
interactions of the K-point orbital with the metal and chalcogen atoms of the neighbouring layer, which
combine into the complex amplitude v = gamma1 + gamma2 exp(2 pi i / 3) of a potential built from the first
shell of moire reciprocal vectors. The interlayer exciton has its electron in the conduction band of one
layer and its hole in the valence band of the other. Its centre of mass therefore sees the conduction-band
modulation minus the valence-band modulation, where the valence-layer amplitude enters with the opposite
chirality (complex conjugate) because the registry of the second layer is mirrored relative to the first.
Each carrier is displaced from the centre of mass R along the relative coordinate r (electron at
R + (m_h/M) r, hole at R - (m_e/M) r, M = m_e + m_h), so its modulation is averaged over the 1s
relative-motion density, taken as the 2D hydrogenic state with amplitude proportional to exp(-r/a_B).
The radius a_B is fixed by the published 1s binding energy through the virial relation of that state,
E_b = hbar^2 / (2 mu a_B^2), mu = m_e m_h / M, with hbar^2/(2 m0) = 38.09982 meV nm^2.

Returns
-------
theta : np.ndarray -- Float array [Re Theta, Im Theta] in meV. The exciton potential is
"""

import numpy as np
from scipy.linalg import expm

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interlayer_moire_coupling(gamma_c: "np.ndarray", gamma_v: "np.ndarray", m_e: float, m_h: float,
                              e_b: float, g0: float) -> "np.ndarray":
    """Return the exciton moire amplitude Theta on the first-shell moire vectors b1, b2 and -(b1 + b2).

    Parameters
    ----------
    gamma_c : np.ndarray
        (gamma1, gamma2) of the conduction band of the electron layer, meV.
    gamma_v : np.ndarray
        (gamma1, gamma2) of the valence band of the hole layer, meV.
    m_e, m_h : float
        Electron and hole effective masses in units of the free-electron mass.
    e_b : float
        1s binding energy of the interlayer exciton, meV, positive.
    g0 : float
        Length of a first-shell moire reciprocal vector, nm^-1.

    Returns
    -------
    theta : np.ndarray
        Float array [Re Theta, Im Theta] in meV. The exciton potential is
        V(R) = sum over the three vectors g of Theta exp(i g.R) + complex conjugate.
    """
    return theta

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _exciton_form_factor(q, a_b):
    return (1.0 + (np.asarray(q, dtype=float) * a_b / 2.0) ** 2) ** -1.5


def _bohr_radius(m_e, m_h, e_b):
    mu = m_e * m_h / (m_e + m_h)
    return float(np.sqrt(38.09982 / (mu * e_b)))


def _oracle_interlayer_moire_coupling(gamma_c: "np.ndarray", gamma_v: "np.ndarray", m_e: float, m_h: float,
                                      e_b: float, g0: float) -> "np.ndarray":
    a_b = _bohr_radius(m_e, m_h, e_b)
    w = np.exp(2j * np.pi / 3.0)
    v_c = gamma_c[0] + gamma_c[1] * w
    v_v = gamma_v[0] + gamma_v[1] * w
    m = m_e + m_h
    theta = v_c * _exciton_form_factor(m_h / m * g0, a_b) - np.conj(v_v) * _exciton_form_factor(m_e / m * g0, a_b)
    return np.array([theta.real, theta.imag])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {"setup": "import numpy as np\ngc = np.array([-4.389, -6.178]); gv = np.array([-1.467, -5.856])",
         "call": "interlayer_moire_coupling(gc, gv, 0.64, 0.51, 173.0, 1.1616)",
         "gold_call": "_oracle_interlayer_moire_coupling(gc, gv, 0.64, 0.51, 173.0, 1.1616)"},
        # unequal masses with a large g0: the two form factors differ strongly
        {"setup": "import numpy as np\ngc = np.array([-4.389, -6.178]); gv = np.array([-1.467, -5.856])",
         "call": "interlayer_moire_coupling(gc, gv, 1.2, 0.3, 90.0, 3.5)",
         "gold_call": "_oracle_interlayer_moire_coupling(gc, gv, 1.2, 0.3, 90.0, 3.5)"},
        # purely imaginary valence amplitude (2 + 4 exp(2 pi i/3) = 3.464i) isolates the conjugation and the sign
        {"setup": "import numpy as np\ngc = np.array([0.0, 0.0]); gv = np.array([2.0, 4.0])",
         "call": "interlayer_moire_coupling(gc, gv, 0.5, 0.5, 200.0, 0.8)",
         "gold_call": "_oracle_interlayer_moire_coupling(gc, gv, 0.5, 0.5, 200.0, 0.8)"},
        # g0 -> 0 limit: form factors equal one
        {"setup": "import numpy as np\ngc = np.array([3.0, -1.0]); gv = np.array([-2.0, 5.0])",
         "call": "interlayer_moire_coupling(gc, gv, 0.64, 0.51, 173.0, 0.0)",
         "gold_call": "_oracle_interlayer_moire_coupling(gc, gv, 0.64, 0.51, 173.0, 0.0)"},
    ]
