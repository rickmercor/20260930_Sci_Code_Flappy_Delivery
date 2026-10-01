"""
Compose every public step into the end-to-end energy calculation.

Dispersion-root information reaches the observable through the sequential public closure chain.

Returns
-------
float: final base-10 logarithmic field-energy ratio.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def run_root_matched_energy(
    kappa: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    delta_t: float,
    final_time: float,
    residual_tol: float = 1e-12,
) -> float:
    """Run the complete root-matched closure pipeline.

    Parameters
    ----------
    kappa, amplitudes, phases : np.ndarray
        Aligned wave-number, amplitude, and phase vectors.
    delta_t, final_time, residual_tol : float
        Propagation step, final time, and positive root-residual tolerance.

    Returns
    -------
    float
        Base-10 logarithmic normalized field energy.

    Raises
    ------
    ValueError
        If an input is invalid or the pipeline does not produce a finite result.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_run_root_matched_energy(
    kappa: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    delta_t: float,
    final_time: float,
    residual_tol: float = 1e-12,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    seeds = _oracle_build_kinetic_root_seeds(kappa)
    roots = _oracle_solve_least_damped_roots(seeds, residual_tol)
    pade = _oracle_match_pade_coefficients(roots)
    closure = _oracle_derive_closure_coefficients(pade)
    generators = _oracle_assemble_moment_generators(closure)
    initial = _oracle_initialize_isothermal_modes(generators, amplitudes, phases)
    final = _oracle_propagate_midpoint_modes(generators, initial, delta_t, final_time)
    result = float(_oracle_compute_log_field_energy(kappa, initial, final))
    if not np.isfinite(result):
        raise ValueError("final energy diagnostic must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only tests with immutable end-to-end targets."""
    return [
        {
            "setup": "k=np.array([0.21,0.37,0.70]); a=np.array([0.012,0.018,0.010]); p=np.array([0.10,-0.20,0.40])",
            "call": "run_root_matched_energy(k,a,p,0.005,12.0)",
            "gold_call": "-0.26599141294849293",
        },
        {
            "setup": "k=np.array([0.27,0.40,0.58]); a=np.array([0.013,0.021,0.008]); p=np.array([-0.3,0.6,0.1])",
            "call": "run_root_matched_energy(k,a,p,0.005,18.75)",
            "gold_call": "-1.0704258214202154",
        },
        {
            "setup": "k=np.array([0.40]); a=np.array([0.02]); p=np.array([0.0])",
            "call": "run_root_matched_energy(k,a,p,0.004,10.0)",
            "gold_call": "-0.8395230455570646",
        },
    ]
