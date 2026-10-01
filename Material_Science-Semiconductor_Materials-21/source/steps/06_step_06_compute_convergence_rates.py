"""
Chain the sub-problem functions 01-05 to return both schemes' convergence rates at one Knudsen-number triple.

This step wires the analysis together for a single operating point. The three Knudsen numbers first give the scalar coefficients of the coupled system. The solid-angle integrals are then needed at two different arguments, one per species, each formed from that species' own relaxation scale and group speed - and the phonon scale is the reduced transport one, not either individual phonon channel. Substituting one of the individual channels there is the most common way to obtain a plausible but wrong answer, because the matrices stay well formed and the rate stays inside a believable range. The coefficients and the two pairs of integrals then give the unaccelerated amplification matrix, which the accelerated operator on the old amplitudes needs as a factor, while the same coefficients give the operator on the new amplitudes; the two spectral radii follow.

Returns
-------
np.ndarray of shape (2,), float: the unaccelerated then the accelerated convergence rate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_convergence_rates(kn_ep: float, kn_pe: float, kn_pp: float,
                              C_e: float = 1.0, C_p: float = 1.0,
                              v_e: float = 1.0, v_p: float = 1.0) -> np.ndarray:
    """Return both schemes' convergence rates at one Knudsen-number triple.

    Parameters
    ----------
    kn_ep, kn_pe, kn_pp : float
        The three scattering Knudsen numbers, finite and strictly positive.
    C_e, C_p, v_e, v_p : float
        Electron and phonon heat capacities and group speeds, positive.

    Returns
    -------
    rates : np.ndarray
        Shape (2,): the unaccelerated convergence rate followed by the
        accelerated one, at unit perturbation wave-vector magnitude.

    Notes
    -----
    This is an orchestrating step: call the public functions of sub-problems
    01-05, feeding each returned value into the next. Import inside the body.
    """
    return np.zeros(2, dtype=float)  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_convergence_rates(kn_ep: float, kn_pe: float, kn_pp: float,
                                      C_e: float = 1.0, C_p: float = 1.0,
                                      v_e: float = 1.0, v_p: float = 1.0) -> np.ndarray:
    import numpy as np

    scalars = [float(v) for v in (kn_ep, kn_pe, kn_pp, C_e, C_p, v_e, v_p)]
    if not all(np.isfinite(v) and v > 0.0 for v in scalars):
        raise ValueError("Knudsen numbers, capacities and speeds must be finite and > 0")
    kn_ep, kn_pe, kn_pp, C_e, C_p, v_e, v_p = scalars

    # -- Step 01: every scalar coefficient of the coupled system.
    coefficients = _oracle_compute_system_coefficients(kn_ep, kn_pe, kn_pp,
                                                       C_e, C_p, v_e, v_p)
    kn_p = float(np.asarray(coefficients, dtype=float).ravel()[0])
    # -- Step 02: the solid-angle integrals at the two distinct arguments.
    moments_e = _oracle_compute_angular_moments(kn_ep * v_e)
    moments_p = _oracle_compute_angular_moments(kn_p * v_p)
    # -- Steps 03-05: the three operators and their spectral radii.
    unaccelerated = _oracle_build_unaccelerated_matrix(coefficients, moments_e, moments_p)
    operators = _oracle_build_accelerated_operators(coefficients, moments_e, moments_p,
                                                    unaccelerated, kn_ep,
                                                    v_e=v_e, v_p=v_p)
    return _oracle_compute_spectral_radii(unaccelerated, operators)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: a transition-regime point of the benchmark sweep.
        {"setup": "import numpy as np\nkn = 1.5848931924611136\n",
         "call": "compute_convergence_rates(kn, kn, kn)",
         "gold_call": "_oracle_compute_convergence_rates(kn, kn, kn)"},
        # Normal: the benchmark's interior worst-case point.
        {"setup": "import numpy as np\nkn = 10.0 ** 0.5\n",
         "call": "compute_convergence_rates(kn, kn, kn)",
         "gold_call": "_oracle_compute_convergence_rates(kn, kn, kn)"},
        # Edge: diffusive electrons against ballistic phonons.
        {"setup": "import numpy as np\n",
         "call": "compute_convergence_rates(0.05, 20.0, 20.0)",
         "gold_call": "_oracle_compute_convergence_rates(0.05, 20.0, 20.0)"},
        # Edge: unequal capacities and speeds break the row symmetry.
        {"setup": "import numpy as np\n",
         "call": "compute_convergence_rates(0.4, 2.0, 0.08, C_e=0.6, C_p=1.4, v_e=1.2, v_p=0.8)",
         "gold_call": "_oracle_compute_convergence_rates(0.4, 2.0, 0.08, C_e=0.6, C_p=1.4, v_e=1.2, v_p=0.8)"},
        # Boundary: the two phonon channels four decades apart.
        {"setup": "import numpy as np\n",
         "call": "compute_convergence_rates(1.0, 100.0, 0.01)",
         "gold_call": "_oracle_compute_convergence_rates(1.0, 100.0, 0.01)"},
        # Boundary: deep diffusive corner, where the accelerated rate nearly vanishes.
        {"setup": "import numpy as np\n",
         "call": "compute_convergence_rates(1e-4, 1e-4, 1e-4)",
         "gold_call": "_oracle_compute_convergence_rates(1e-4, 1e-4, 1e-4)"},
        # Integration anchors: the two endpoints whose rates are reported in the rubric.
        {"setup": "import numpy as np\n",
         "call": "compute_convergence_rates(1e-2, 1e-2, 1e-2)",
         "gold_call": "_oracle_compute_convergence_rates(1e-2, 1e-2, 1e-2)"},
        {"setup": "import numpy as np\n",
         "call": "compute_convergence_rates(1e2, 1e2, 1e2)",
         "gold_call": "_oracle_compute_convergence_rates(1e2, 1e2, 1e2)"},
    ]
