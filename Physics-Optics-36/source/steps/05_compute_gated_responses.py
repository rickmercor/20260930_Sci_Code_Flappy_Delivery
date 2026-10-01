"""
Compute each packet outcome’s incoherent contribution to each OCT optical-depth bin.

Bins have equally spaced one-way optical centres. Boundary packets are excluded as specified by the strict coherence gate.

Returns
-------
np.ndarray of shape (K, M), the incoherent intensity contributed by each outcome to each bin, preserving the supplied outcome and bin orders; outside-bin and exact-boundary contributions are zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_gated_responses(opl: "np.ndarray", detected: "np.ndarray", resolution: float, bins: "np.ndarray") -> "np.ndarray":
    """Parameters
    ----------
    opl : np.ndarray
        Shape (K,), nonnegative corrected round-trip optical paths in mm.
    detected : np.ndarray
        Shape (K,), nonnegative detected intensities.
    resolution : float
        Positive one-way optical bin resolution in mm.
    bins : np.ndarray
        Shape (M,), M >= 1, strictly increasing nonnegative integer bin
        indices. The one-way optical centre of index j is j * resolution.

    Returns
    -------
    response : np.ndarray
        Shape (K, M), intensity contributed by each outcome to each bin,
        preserving both supplied orders. Outside-bin and exact-boundary
        contributions are zero. Intensities are not squared or coherently
        combined at this stage.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_gated_responses(opl: "np.ndarray", detected: "np.ndarray", resolution: float, bins: "np.ndarray") -> "np.ndarray":
    opl = np.asarray(opl, dtype=float)
    centres = resolution * np.asarray(bins)
    accepted = np.abs(opl[:, None] / 2 - centres[None, :]) < resolution / 2
    return np.asarray(detected, dtype=float)[:, None] * accepted

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary and edge cases."""
    return [{'setup': 'import numpy as np\no=np.array([.5852,1.5561,2.4605])\nd=np.array([.001,.002,.003])\nb=np.arange(1,7)', 'call': 'compute_gated_responses(o.copy(), d.copy(), .25, b.copy())', 'gold_call': '_oracle_compute_gated_responses(o.copy(), d.copy(), .25, b.copy())', 'tol': 1e-12}, {'setup': 'import numpy as np\no=np.array([.25,.75])\nd=np.array([1.,2.])\nb=np.array([1])', 'call': 'compute_gated_responses(o.copy(), d.copy(), .25, b.copy())', 'gold_call': '_oracle_compute_gated_responses(o.copy(), d.copy(), .25, b.copy())', 'tol': 1e-12}, {'setup': 'import numpy as np\no=np.array([0.,3.6,1.])\nd=np.array([.01,.02,0.])\nb=np.arange(1,7)', 'call': 'compute_gated_responses(o.copy(), d.copy(), .25, b.copy())', 'gold_call': '_oracle_compute_gated_responses(o.copy(), d.copy(), .25, b.copy())', 'tol': 1e-12}]
