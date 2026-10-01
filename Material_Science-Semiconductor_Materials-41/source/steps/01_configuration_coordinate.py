"""
Reduce one deep level to its configuration coordinate diagram. From the lattice relaxation that accompanies the charge-state change, the effective phonon energy and the electronic transition energy, return the oscillator length of the mass-weighted coordinate, the Huang-Rhys factor, the relaxation energy, the classical capture barrier of the diagram, and the dimensionless displacement between the two parabolas, the last of these defined so that its square is the Huang-Rhys factor. Work in the mass-weighted coordinate in which the kinetic operator carries no mass, so that a relaxation quoted in amu^(1/2) Angstrom and a phonon energy quoted in eV fix the oscillator length with no further input.

The two charge states of a deep level are represented by two harmonic potential energy surfaces of the same curvature, displaced along a single mass-weighted coordinate and offset in energy. Every quantity in the model follows from that picture: the Huang-Rhys factor counts the phonons emitted in the relaxation, and the intersection of the two parabolas fixes the classical barrier that the older capture models place in the exponent.

Returns
-------
A (5,) float64 array holding [oscillator length in amu^(1/2) Angstrom, Huang-Rhys factor, relaxation energy in eV, classical capture barrier in eV, dimensionless displacement whose square is the Huang-Rhys factor].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def configuration_coordinate(delta_q, hw, delta_e):
    """Reduce one deep level to its configuration coordinate diagram. A (5,) float64 array
    holding [oscillator length in amu^(1/2) Angstrom, Huang-Rhys factor, relaxation energy
    in eV, classical capture barrier in eV, dimensionless displacement whose square is the
    Huang-Rhys factor]."""
    return np.zeros(5)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


HBAR2   = 4.18005956e-3

def _oracle_configuration_coordinate(delta_q, hw, delta_e):
    delta_q = float(delta_q); hw = float(hw); delta_e = float(delta_e)
    if hw <= 0.0 or delta_q <= 0.0:
        raise ValueError("hw and delta_q must be positive")
    ell2 = HBAR2/hw
    s_factor = delta_q*delta_q/(2.0*ell2)
    relax = s_factor*hw
    barrier = (delta_e - relax)**2/(4.0*relax)
    return np.array([np.sqrt(ell2), s_factor, relax, barrier, delta_q/np.sqrt(2.0*ell2)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup': 'import numpy as np\ndelta_q = 4.90\nhw = 0.038\ndelta_e = 1.02\n',
         'call': 'configuration_coordinate(delta_q, hw, delta_e)',
         'gold_call': '_oracle_configuration_coordinate(delta_q, hw, delta_e)'},
        {'setup': 'import numpy as np\ndelta_q = 1.20\nhw = 0.038\ndelta_e = 1.02\n',
         'call': 'configuration_coordinate(delta_q, hw, delta_e)',
         'gold_call': '_oracle_configuration_coordinate(delta_q, hw, delta_e)'},
        {'setup': 'import numpy as np\ndelta_q = 4.66\nhw = 0.040\ndelta_e = 1.00\n',
         'call': 'configuration_coordinate(delta_q, hw, delta_e)',
         'gold_call': '_oracle_configuration_coordinate(delta_q, hw, delta_e)'},
    ]
