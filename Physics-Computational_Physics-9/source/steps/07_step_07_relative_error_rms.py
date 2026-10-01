"""
Compute the RMS relative error between two potentials after shifting each so its minimum equals 1.

The gravitational potential is defined only up to an additive constant, so a raw pointwise comparison between a numerical and an analytic potential is gauge-dependent and ill-conditioned wherever the reference crosses zero. Following the benchmarking convention of the reference work, both fields are first shifted by a constant so that each has minimum value exactly 1; the pointwise relative error is then eps = |(Phi_num - Phi_ref) / Phi_ref| on the shifted fields, and the scalar accuracy metric is the root-mean-square norm

    ||eps||_2 = sqrt( (1 / N_total) * sum_ijk eps_ijk^2 ),

where N_total is the total number of grid cells. The shift makes the metric invariant under adding any constant to either input and keeps the denominator >= 1 everywhere.

Returns
-------
float: the root mean square of |(phi_num_shifted - phi_ref_shifted) / phi_ref_shifted| as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def relative_error_rms(phi_num: np.ndarray, phi_ref: np.ndarray) -> float:
    """RMS relative error between two potentials after min-to-1 shifting.

    Parameters
    ----------
    phi_num : np.ndarray
        Numerical potential, any shape with at least one element and only
        finite values.
    phi_ref : np.ndarray
        Reference potential with the same shape as phi_num and only finite
        values.

    Returns
    -------
    error_norm : float
        The root-mean-square of |(phi_num_shifted - phi_ref_shifted) /
        phi_ref_shifted|, where each field is shifted so its minimum is 1,
        as a native Python float.

    Notes
    -----
    The evaluation sandbox may execute this function in isolation: include
    every import your implementation needs (for example
    ``import numpy as np``) inside the function body.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_relative_error_rms(phi_num: np.ndarray, phi_ref: np.ndarray) -> float:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    phi_num = np.asarray(phi_num, dtype=float)
    phi_ref = np.asarray(phi_ref, dtype=float)
    if phi_num.size < 1:
        raise ValueError("phi_num and phi_ref must contain at least one element")
    if phi_num.shape != phi_ref.shape:
        raise ValueError("phi_num and phi_ref must have identical shapes")
    if not np.all(np.isfinite(phi_num)) or not np.all(np.isfinite(phi_ref)):
        raise ValueError("phi_num and phi_ref must contain only finite values")

    num_shifted = phi_num - phi_num.min() + 1.0
    ref_shifted = phi_ref - phi_ref.min() + 1.0
    eps = np.abs((num_shifted - ref_shifted) / ref_shifted)
    return float(np.sqrt(np.mean(eps**2)))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: generic 3D fields with a small perturbation (normal scenario) ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(21)
phi_ref = rng.normal(size=(6, 6, 6))
phi_num = phi_ref + 1.0e-4 * rng.normal(size=(6, 6, 6))
""",
            "call": "relative_error_rms(phi_num, phi_ref)",
            "gold_call": "_oracle_relative_error_rms(phi_num, phi_ref)",
        },
        # --- Boundary: identical inputs must give exactly zero ---
        {
            "setup": """import numpy as np
phi_ref = np.linspace(-5.0, 2.0, 64).reshape(4, 4, 4)
phi_num = phi_ref.copy()
""",
            "call": "relative_error_rms(phi_num, phi_ref)",
            "gold_call": "_oracle_relative_error_rms(phi_num, phi_ref)",
        },
        # --- Edge: inputs differing by a pure constant offset must give zero ---
        {
            "setup": """import numpy as np
rng = np.random.default_rng(5)
phi_ref = rng.normal(size=(5, 5, 5))
phi_num = phi_ref - 42.0
""",
            "call": "relative_error_rms(phi_num, phi_ref)",
            "gold_call": "_oracle_relative_error_rms(phi_num, phi_ref)",
        },
        # --- Edge: single-element arrays ---
        {
            "setup": """import numpy as np
phi_ref = np.array([3.5])
phi_num = np.array([-7.0])
""",
            "call": "relative_error_rms(phi_num, phi_ref)",
            "gold_call": "_oracle_relative_error_rms(phi_num, phi_ref)",
        },
    ]
