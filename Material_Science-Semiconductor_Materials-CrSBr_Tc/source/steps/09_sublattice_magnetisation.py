"""
Evaluate Callen's expression for the sublattice magnetisation. Returns a dimensionless real scalar.

Given the magnon occupation, the sublattice magnetisation follows from Callen's
expression

    <S''z_a> = [ (S_a - Phi_a)(1 + Phi_a)^(2 S_a + 1)
                 + (S_a + 1 + Phi_a) Phi_a^(2 S_a + 1) ]
               / [ (1 + Phi_a)^(2 S_a + 1) - Phi_a^(2 S_a + 1) ],

with S = 3/2 for CrSBr.

Returns
-------
float: the dimensionless sublattice magnetisation.
"""

import functools
import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sublattice_magnetisation(phi: float) -> float:
    """Evaluate Callen's expression for the sublattice magnetisation. Returns a dimensionless real scalar.

    Args:
        phi: site-resolved magnon occupation; must be non-negative.

    Returns:
        float: the dimensionless sublattice magnetisation.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _spin():
    """The Cr3+ spin of CrSBr."""
    return 1.5


def _magnetisation(phi):
    """Callen's expression for the sublattice magnetisation."""
    S = _spin()
    e = 2.0 * S + 1.0
    p = max(float(phi), 1e-15)
    return float(((S - p) * (1.0 + p) ** e + (S + 1.0 + p) * p ** e)
                 / ((1.0 + p) ** e - p ** e))


def _oracle_sublattice_magnetisation(phi: float) -> float:
    """Evaluate Callen's expression for the sublattice magnetisation."""
    return _magnetisation(phi)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases: normal, boundary and edge."""
    return [
        {
            # boundary: zero occupation, where the moment saturates
            "setup": "",
            "call": 'sublattice_magnetisation(0.0)',
            "gold_call": '_oracle_sublattice_magnetisation(0.0)',
        },
        {
            # normal: a lightly populated state
            "setup": "",
            "call": 'sublattice_magnetisation(0.3089500000)',
            "gold_call": '_oracle_sublattice_magnetisation(0.3089500000)',
        },
        {
            # normal: a strongly renormalised state
            "setup": "",
            "call": 'sublattice_magnetisation(2.8062300000)',
            "gold_call": '_oracle_sublattice_magnetisation(2.8062300000)',
        },
        {
            # edge: the large occupation reached just below the transition
            "setup": "",
            "call": 'sublattice_magnetisation(8.4800470000)',
            "gold_call": '_oracle_sublattice_magnetisation(8.4800470000)',
        },
    ]
