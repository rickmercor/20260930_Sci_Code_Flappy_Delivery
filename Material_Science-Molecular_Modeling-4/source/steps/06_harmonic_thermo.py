"""
From a potential energy value, the selected frequencies, and a temperature, compute the classical and quantum coarse-grained free energies and the total classical and quantum positional mode variances of the paper’s backmapping construction, using the exact forms the paper gives. Validate the inputs.

Reduced units with hbar and kB equal to one; the quantum variance must exceed the classical one at low temperature because of zero-point motion.

Returns
-------
float64 array (4,): [F classical, F quantum, total classical variance, total quantum variance]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def harmonic_thermo(V, omegas, T):
    """V: float potential energy; omegas: float array (L,) of selected
    frequencies; T: temperature, reduced units with hbar = kB = 1.

    Returns float64 array (4,): [classical free energy, quantum free
    energy, total classical positional mode variance, total quantum
    positional mode variance], forms per the source paper."""
    return np.zeros(4)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8: harmonic thermodynamic quantities from stiff-mode frequencies."""

import numpy as np

def _oracle_harmonic_thermo(V, omegas, T):
    om = np.asarray(omegas, dtype=np.float64)
    if om.ndim != 1 or om.size < 1 or not np.all(np.isfinite(om)) or np.any(om <= 0.0):
        raise ValueError("invalid frequencies")
    if not np.isfinite(V) or not np.isfinite(T) or T <= 0.0:
        raise ValueError("invalid scalar input")
    beta = 1.0 / float(T)
    F_cl = float(V) + float(np.sum(np.log(beta * om))) / beta
    F_qm = float(V) + float(np.sum(om / 2.0 + np.log(1.0 - np.exp(-beta * om)) / beta))
    v_cl = float(np.sum(1.0 / (beta * om * om)))
    v_qm = float(np.sum((1.0 / (2.0 * om)) / np.tanh(beta * om / 2.0)))
    return np.asarray([F_cl, F_qm, v_cl, v_qm], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'V = -2.35\nomegas = np.array([16.1, 18.4, 22.9, 27.5])\nT = 0.35', "call": "harmonic_thermo(V, omegas, T)", "gold_call": "_oracle_harmonic_thermo(V, omegas, T)", "tol": 1e-09},
        {"setup": 'V = 0.0\nomegas = np.array([16.1, 18.4, 22.9, 27.5])\nT = 0.05', "call": "harmonic_thermo(V, omegas, T)", "gold_call": "_oracle_harmonic_thermo(V, omegas, T)", "tol": 1e-09},
        {"setup": 'V = 1.5\nomegas = np.array([20.0])\nT = 2.0', "call": "harmonic_thermo(V, omegas, T)", "gold_call": "_oracle_harmonic_thermo(V, omegas, T)", "tol": 1e-09},
    ]
