"""
Compute accepted hybrid detected packet intensities from transport histories and mode transfer.

Unit launch energy is used. Recorded probabilities account for transport scattering; this stage assigns the deterministic intensity of each outcome.

Returns
-------
np.ndarray of shape (K,), the hybrid detected intensities per unit launch energy, preserving the input record order; angularly rejected packets contribute zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_detected_weights(records: "np.ndarray", sites: "np.ndarray", transfer: "np.ndarray", absorption: float, scattering: float, angle_limit: float) -> "np.ndarray":
    """Parameters
    ----------
    records : np.ndarray
        Shape (K, 7), columns exit x, exit y, vx, vy, vz < 0, final
        segment length s, and full in-medium path L; directions are unit
        vectors and lengths are in mm.
    sites : np.ndarray
        Shape (K, 3), corresponding last-scatter positions in mm.
    transfer : np.ndarray
        Shape (K,), nonnegative spherical-wave power coupling.
    absorption, scattering : float
        Nonnegative coefficients in inverse mm.
    angle_limit : float
        Internal incidence cutoff in degrees, strictly between 0 and 90.

    Returns
    -------
    detected : np.ndarray
        Shape (K,), hybrid detected intensities per unit launch energy.
        Packets on or above the angular cutoff have zero intensity.
        No extra Fresnel transmission is included.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_detected_weights(records: "np.ndarray", sites: "np.ndarray", transfer: "np.ndarray", absorption: float, scattering: float, angle_limit: float) -> "np.ndarray":
    records = np.asarray(records, dtype=float)
    c = -records[:, 4]
    keep = c > np.cos(np.deg2rad(angle_limit))
    result = np.zeros(len(records), dtype=float)
    exponent = (-absorption * records[keep, 6]
                + (absorption + scattering) * (records[keep, 5] - sites[keep, 2]))
    result[keep] = transfer[keep] * c[keep] * np.exp(exponent)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary and edge cases."""
    return [{'setup': 'import numpy as np\nr=np.array([[.15,.02,.6,0,-.8,.25,.49],[0,0,0,0,-1,.6,1.2]])\nx=np.array([[0,.02,.2],[0,0,.6]])\nt=np.array([.0012,.0036])', 'call': 'compute_detected_weights(r.copy(), x.copy(), t.copy(), .15, 3., 50.)', 'gold_call': '_oracle_compute_detected_weights(r.copy(), x.copy(), t.copy(), .15, 3., 50.)', 'tol': 1e-12}, {'setup': 'import numpy as np\nr=np.array([[0,0,0,0,-1,0,0]])\nx=np.zeros((1,3))\nt=np.array([.0036])', 'call': 'compute_detected_weights(r.copy(), x.copy(), t.copy(), 0., 0., 50.)', 'gold_call': '_oracle_compute_detected_weights(r.copy(), x.copy(), t.copy(), 0., 0., 50.)', 'tol': 1e-12}, {'setup': 'import numpy as np\nr=np.array([[.55,0,.8,0,-.6,.7,1.7],[.01,.38,0,.6,-.8,.6,1.5]])\nx=np.array([[-.01,0,.42],[.01,.02,.48]])\nt=np.array([.0026,.0012])', 'call': 'compute_detected_weights(r.copy(), x.copy(), t.copy(), .15, 3., 50.)', 'gold_call': '_oracle_compute_detected_weights(r.copy(), x.copy(), t.copy(), .15, 3., 50.)', 'tol': 1e-12}]
