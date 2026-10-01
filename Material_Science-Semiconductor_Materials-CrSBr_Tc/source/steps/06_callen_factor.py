"""
Compute the Callen-decoupling renormalisation of the single-ion anisotropy. Returns a dimensionless real scalar.

The random phase approximation assumes fluctuations in the local longitudinal
spin operator are small, which fails as temperature rises. Callen's decoupling replaces it,
writing the longitudinal operator as a combination of its exact representations weighted by
a parameter alpha = <S''z_a>/(2 S_a^2).

Carrying that through the equation of motion multiplies the RPA single-ion anisotropy
contribution by

    Upsilon = 1 - <S''z_a> (1 + 2 Phi_a) / (2 S_a^2),

where Phi_a is the site-resolved magnon occupation. The factor scales the whole anisotropy
contribution, A, B and C alike.

For CrSBr, S = 3/2.

Returns
-------
float: the dimensionless Callen renormalisation factor.
"""

import functools
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def callen_factor(sz: float, phi: float) -> float:
    """Compute the Callen-decoupling renormalisation of the single-ion anisotropy. Returns a dimensionless real scalar.

    Args:
        sz: sublattice magnetisation.
        phi: site-resolved magnon occupation.

    Returns:
        float: the dimensionless Callen renormalisation factor.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _spin():
    """The Cr3+ spin of CrSBr."""
    return 1.5


def _callen(sz, phi):
    """Callen renormalisation of the single-ion anisotropy."""
    S = _spin()
    return 1.0 - sz * (1.0 + 2.0 * phi) / (2.0 * S ** 2)


def _oracle_callen_factor(sz: float, phi: float) -> float:
    """Compute the Callen-decoupling renormalisation of the single-ion anisotropy."""
    return _callen(sz, phi)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # normal: saturation at zero occupation
            "setup": "",
            "call": 'callen_factor(1.5, 0.0)',
            "gold_call": '_oracle_callen_factor(1.5, 0.0)',
        },
        {
            # normal: a partly demagnetised, thermally populated state
            "setup": "",
            "call": 'callen_factor(1.0, 0.5)',
            "gold_call": '_oracle_callen_factor(1.0, 0.5)',
        },
        {
            # boundary: zero magnetisation, where the factor is unity
            "setup": "",
            "call": 'callen_factor(0.0, 0.0)',
            "gold_call": '_oracle_callen_factor(0.0, 0.0)',
        },
        {
            # edge: the state reached just below the transition
            "setup": "",
            "call": 'callen_factor(0.1388530302, 8.4800470)',
            "gold_call": '_oracle_callen_factor(0.1388530302, 8.4800470)',
        },
    ]
