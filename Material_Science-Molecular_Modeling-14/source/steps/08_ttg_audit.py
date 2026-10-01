"""
Run the full pipeline on the two declared trilayer configurations and report per configuration: the number of surviving reciprocal degrees of freedom; the Hamiltonian dimension; the Hermiticity residual of the assembled matrix; the momentum LDoS at the three declared energies at polynomial order P; and the largest eigenvalue of the assembled Hamiltonian. Assemble by calling the earlier sub-problem functions.

The audit certifies the truncated reciprocal construction end to end: the degree-of-freedom counts witness the truncation criteria, the Hermiticity residual witnesses the assembly, and the LDoS values depend on every convention in the pipeline.

Returns
-------
return (2, 7) float64: audit rows per configuration
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def ttg_audit(P):
    """P: Chebyshev polynomial order. Builds the two declared trilayer
    configurations, runs the truncation, assembly and kernel polynomial
    pipeline on each with the declared parameters, and returns a float64
    array (2, 7) with columns: degree-of-freedom count; Hamiltonian
    dimension; Hermiticity residual; the momentum LDoS at the three declared
    energies; and the largest eigenvalue. Assembled by calling the earlier
    sub-problem functions. Raises ValueError on an invalid order."""
    return np.zeros((2, 7))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 8 (final orchestrator): trilayer audit over the two configurations."""

import numpy as np

_VARIANTS = [(-0.06, 0.0, 0.11, 1.15, 5.0), (-0.09, 0.0, 0.07, 1.05, 4.6)]
_Q_OFF = np.array([0.03, 0.02])
_E_LIST = np.array([-0.5, 0.0, 0.5])
_NMAX = 3
_S_KPM = 0.25


def _oracle_ttg_audit(P):
    if isinstance(P, bool) or not isinstance(P, (int, np.integer)) or P < 2:
        raise ValueError("invalid polynomial order")
    rows = []
    for variant, (t1, t2, t3, W, L) in enumerate(_VARIANTS):
        geom = _oracle_layer_geometry(np.array([t1, t2, t3]))
        qb = (geom[1, 8:10] if variant == 0 else geom[1, 10:12]) + _Q_OFF
        dofs = _oracle_wl_dof(qb, geom, W, L, _NMAX)
        Hri = _oracle_assemble_hamiltonian(qb, dofs, geom)
        Hc = Hri[:, :, 0] + 1j * Hri[:, :, 1]
        herm = float(np.linalg.norm(Hc - Hc.conj().T))
        lam = np.linalg.eigvalsh(Hc)
        ld = _oracle_kpm_ldos(Hri, dofs, _E_LIST, P, _S_KPM)
        rows.append([float(dofs.shape[0]), float(Hc.shape[0]), herm,
                     float(ld[0]), float(ld[1]), float(ld[2]), float(lam.max())])
    return np.asarray(rows, dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'P = 64', "call": "ttg_audit(P)", "gold_call": "_oracle_ttg_audit(P)", "tol": 1e-08},
        {"setup": 'P = 48', "call": "ttg_audit(P)", "gold_call": "_oracle_ttg_audit(P)", "tol": 1e-08},
        {"setup": 'P = 80', "call": "ttg_audit(P)", "gold_call": "_oracle_ttg_audit(P)", "tol": 1e-08},
    ]
