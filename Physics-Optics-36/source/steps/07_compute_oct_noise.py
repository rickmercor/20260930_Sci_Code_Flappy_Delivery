"""
Compute the RMS noise coefficient of a hybrid particle-wave OCT detector replay.

This is the final orchestrator. The calculation treats the supplied finite transport population as exact.

Returns
-------
float, the dimensionless RMS noise coefficient of the complete detector-replay pipeline against the exact population reference, as defined by compute_noise_coefficient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_oct_noise(records: "np.ndarray", probabilities: "np.ndarray", waist: float, rayleigh: float, focus: float, absorption: float, scattering: float, refractive_index: float, angle_limit: float, resolution: float, bins: "np.ndarray") -> float:
    """Parameters
    ----------
    records : np.ndarray
        Shape (K, 7), K >= 1, columns exit x, exit y, outward unit vx, vy,
        vz < 0, final segment s >= 0 and full in-medium physical path L >= s.
        All lengths are in mm, surface z = 0 and positive z is inward.
    probabilities : np.ndarray
        Shape (K,), unconditional nonnegative probabilities, sum <= 1;
        remaining probability has zero signal. These include scattering
        survival but exclude absorption, and launches have unit energy.
    waist, rayleigh : float
        Positive in-medium Gaussian waist radius and Rayleigh range in mm.
    focus : float
        In-medium focus depth in mm.
    absorption, scattering : float
        Nonnegative coefficients in inverse mm.
    refractive_index : float
        Positive homogeneous sample index.
    angle_limit : float
        Strict incidence cutoff in degrees, between 0 and 90; no additional
        Fresnel transmission is included.
    resolution : float
        Positive one-way optical bin resolution in mm.
    bins : np.ndarray
        Shape (M,), M >= 1, increasing nonnegative integer bin indices.
        Exact coherence-bin boundaries are excluded. The resulting
        population profile must have positive total intensity.

    Returns
    -------
    coefficient : float
        Dimensionless RMS noise coefficient against the exact population
        reference, as defined by compute_noise_coefficient.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_oct_noise(records: "np.ndarray", probabilities: "np.ndarray", waist: float, rayleigh: float, focus: float, absorption: float, scattering: float, refractive_index: float, angle_limit: float, resolution: float, bins: "np.ndarray") -> float:
    sites = _oracle_recover_scatter_sites(records)
    transfer = _oracle_compute_mode_transfer(sites, waist, rayleigh, focus)
    detected = _oracle_compute_detected_weights(records, sites, transfer, absorption, scattering, angle_limit)
    opl = _oracle_compute_hybrid_opl(records, sites, refractive_index)
    responses = _oracle_compute_gated_responses(opl, detected, resolution, bins)
    return _oracle_compute_noise_coefficient(responses, probabilities)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary and edge cases."""
    return [{'setup': 'import numpy as np\nr=np.array([[0.01, 0.0, 0.0, 0.0, -1.0, 0.2, 0.42], [0.15, 0.02, 0.6, 0.0, -0.8, 0.25, 0.49], [-0.13, 0.03, -0.6, 0.0, -0.8, 0.35, 0.84], [0.02, 0.17, 0.0, 0.6, -0.8, 0.4, 0.97], [0.22, 0.0, 0.6, 0.0, -0.8, 0.5, 1.18], [-0.24, 0.01, -0.6, 0.0, -0.8, 0.5, 1.27], [0.03, 0.34, 0.0, 0.6, -0.8, 0.65, 1.68], [0.43, 0.04, 0.6, 0.0, -0.8, 0.7, 1.84], [0.015, -0.02, 0.0, 0.0, -1.0, 0.8, 1.85], [0.55, 0.0, 0.8, 0.0, -0.6, 0.7, 1.7], [0.02, 0.01, 0.0, 0.0, -1.0, 0.1, 2.4], [0.02, -0.01, 0.0, 0.0, -1.0, 0.15, 0.3], [0.24, 0.26, 0.48, 0.64, -0.6, 0.4, 1.0], [0.01, 0.38, 0.0, 0.6, -0.8, 0.6, 1.5], [-0.35, 0.0, -0.6, 0.0, -0.8, 0.6, 1.61], [0.0, 0.0, 0.0, 0.0, -1.0, 0.6, 1.2]], dtype=float)\np=np.array([0.03125, 0.046875, 0.03125, 0.0625, 0.046875, 0.03125, 0.046875, 0.03125, 0.0625, 0.046875, 0.03125, 0.03125, 0.046875, 0.03125, 0.03125, 0.0625])\nb=np.arange(1,7)', 'call': 'compute_oct_noise(r.copy(), p.copy(), .03, .5, .6, .15, 3., 1.33, 50., .25, b.copy())', 'gold_call': '_oracle_compute_oct_noise(r.copy(), p.copy(), .03, .5, .6, .15, 3., 1.33, 50., .25, b.copy())', 'tol': 1e-11}, {'setup': 'import numpy as np\nr=np.array([[0,0,0,0,-1,.25,.5]])\np=np.array([1.])\nb=np.array([1])', 'call': 'compute_oct_noise(r.copy(), p.copy(), .03, .5, .6, 0., 0., 1., 50., .25, b.copy())', 'gold_call': '_oracle_compute_oct_noise(r.copy(), p.copy(), .03, .5, .6, 0., 0., 1., 50., .25, b.copy())', 'tol': 1e-11}, {'setup': 'import numpy as np\nr=np.array([[0.01, 0.0, 0.0, 0.0, -1.0, 0.2, 0.42], [0.15, 0.02, 0.6, 0.0, -0.8, 0.25, 0.49], [-0.13, 0.03, -0.6, 0.0, -0.8, 0.35, 0.84], [0.02, 0.17, 0.0, 0.6, -0.8, 0.4, 0.97], [0.22, 0.0, 0.6, 0.0, -0.8, 0.5, 1.18], [-0.24, 0.01, -0.6, 0.0, -0.8, 0.5, 1.27], [0.03, 0.34, 0.0, 0.6, -0.8, 0.65, 1.68], [0.43, 0.04, 0.6, 0.0, -0.8, 0.7, 1.84], [0.015, -0.02, 0.0, 0.0, -1.0, 0.8, 1.85], [0.55, 0.0, 0.8, 0.0, -0.6, 0.7, 1.7], [0.02, 0.01, 0.0, 0.0, -1.0, 0.1, 2.4], [0.02, -0.01, 0.0, 0.0, -1.0, 0.15, 0.3], [0.24, 0.26, 0.48, 0.64, -0.6, 0.4, 1.0], [0.01, 0.38, 0.0, 0.6, -0.8, 0.6, 1.5], [-0.35, 0.0, -0.6, 0.0, -0.8, 0.6, 1.61], [0.0, 0.0, 0.0, 0.0, -1.0, 0.6, 1.2]], dtype=float)\np=np.array([0.03125, 0.046875, 0.03125, 0.0625, 0.046875, 0.03125, 0.046875, 0.03125, 0.0625, 0.046875, 0.03125, 0.03125, 0.046875, 0.03125, 0.03125, 0.0625])\nb=np.arange(1,9)', 'call': 'compute_oct_noise(r.copy(), p.copy(), .04, .7, .4, .2, 1.1, 1.4, 40., .2, b.copy())', 'gold_call': '_oracle_compute_oct_noise(r.copy(), p.copy(), .04, .7, .4, .2, 1.1, 1.4, 40., .2, b.copy())', 'tol': 1e-11}]
