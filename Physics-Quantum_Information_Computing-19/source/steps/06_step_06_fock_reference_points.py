"""
Compute, for every photon number up to the cut-off, the conditional detection probability and the conditional error probability that the channel and detection model assigns to a signal carrying exactly that many photons.

Unlike the sifted rates, these Fock-resolved quantities are never measured: they are what the model predicts would be seen if the photon number could be fixed at will. Because a model of the channel is the only thing available before any parameter estimation has been carried out, they are the natural place to anchor the affine relaxation of the confinement constraints, and they are the anchor against which any later, data-driven anchor is judged. Both detectors share one efficiency, folded into the transmittance, and one per-gate dark-count probability, and the misalignment is the same fixed polarisation rotation seen by every photon of a signal.

Both returned arrays are indexed by photon number from zero up to and including the cut-off, and neither depends on the nominal intensity setting.

Returns
-------
yield_reference : np.ndarray — Shape (n_cut + 1,). Entry n is the conditional detection probability of a signal carrying exactly n photons.; error_reference : np.ndarray — Shape (n_cut + 1,). Entry n is the conditional error probability of a signal carrying exactly n photons in the basis used to estimate the phase error, averaged over the encoded bit.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_fock_reference_points(n_cut: int, transmittance: float, misalignment: float, dark_count: float) -> "tuple[np.ndarray, np.ndarray]":
    '''Compute the model's Fock-resolved detection and error probabilities.

    Parameters
    ----------
    n_cut : int
        Photon-number cut-off. A non-negative integer.
    transmittance : float
        Overall probability that an emitted photon reaches and fires a
        detector, combining channel loss and detector efficiency. In [0, 1].
    misalignment : float
        Polarisation misalignment angle in radians, in [0, pi / 4].
    dark_count : float
        Dark-count probability of each detector per gate, in [0, 1).

    Returns
    -------
    yield_reference : np.ndarray
        Shape (n_cut + 1,). Entry n is the conditional detection probability of
        a signal carrying exactly n photons.
    error_reference : np.ndarray
        Shape (n_cut + 1,). Entry n is the conditional error probability of a
        signal carrying exactly n photons in the basis used to estimate the
        phase error, averaged over the encoded bit.

    Raises
    ------
    ValueError
        If `n_cut` is negative, if `transmittance` is outside [0, 1], if
        `misalignment` is outside [0, pi / 4], or if `dark_count` is negative
        or is not below one.
    '''
    return yield_reference, error_reference

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_fock_reference_points(n_cut: int, transmittance: float, misalignment: float, dark_count: float) -> "tuple[np.ndarray, np.ndarray]":
    if isinstance(n_cut, bool) or not isinstance(n_cut, (int, np.integer)):
        raise ValueError("n_cut must be an integer")
    for value in (transmittance, misalignment, dark_count):
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("transmittance, misalignment and dark_count must be real scalars")
    n_cut = int(n_cut)
    transmittance = float(transmittance)
    misalignment = float(misalignment)
    dark_count = float(dark_count)
    if n_cut < 0:
        raise ValueError("n_cut must be non-negative")
    if not np.isfinite(transmittance) or transmittance < 0.0 or transmittance > 1.0:
        raise ValueError("transmittance must lie in [0, 1]")
    if not np.isfinite(misalignment) or misalignment < 0.0 or misalignment > 0.25 * np.pi:
        raise ValueError("misalignment must lie in [0, pi / 4]")
    if not np.isfinite(dark_count) or dark_count < 0.0 or dark_count >= 1.0:
        raise ValueError("dark_count must lie in [0, 1)")

    n = np.arange(n_cut + 1, dtype=float)
    dark = 1.0 - transmittance
    silent = dark ** n
    aligned = (transmittance * np.cos(misalignment) ** 2 + dark) ** n - silent
    crossed = (transmittance * np.sin(misalignment) ** 2 + dark) ** n - silent
    both = 1.0 - silent - aligned - crossed

    clean = crossed + 0.5 * both
    stray_correct = 0.5 * (crossed + both)
    stray_wrong = silent + crossed + 0.5 * (aligned + both)

    no_dark = (1.0 - dark_count) ** 2
    error_reference = (no_dark * clean
                       + dark_count * (1.0 - dark_count) * (stray_correct + stray_wrong)
                       + 0.5 * dark_count ** 2)
    yield_reference = 1.0 - no_dark * silent
    return np.asarray(yield_reference, dtype=float), np.asarray(error_reference, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    guard = """
def run_model(fn, *args):
    try:
        fn(*args)
        return 0.0
    except ValueError:
        return 1.0
"""
    return [
        # --- normal: the benchmark link at 35 km ------------------------------
        {
            "setup": "args = (10, 10 ** (-0.2 * 35.0 / 10.0) * 0.65, 0.08, 7.2e-8)\n",
            "call": "compute_fock_reference_points(*args)",
            "gold_call": "_oracle_compute_fock_reference_points(*args)",
        },
        # --- normal: a short, low-loss link -----------------------------------
        {
            "setup": "args = (10, 10 ** (-0.2 * 5.0 / 10.0) * 0.65, 0.08, 7.2e-8)\n",
            "call": "compute_fock_reference_points(*args)",
            "gold_call": "_oracle_compute_fock_reference_points(*args)",
        },
        # --- boundary: vacuum-only cut-off ------------------------------------
        {
            "setup": "args = (0, 0.1296920505, 0.08, 7.2e-8)\n",
            "call": "compute_fock_reference_points(*args)",
            "gold_call": "_oracle_compute_fock_reference_points(*args)",
            "tol": 1e-12,
        },
        # --- boundary: perfect alignment --------------------------------------
        {
            "setup": "args = (10, 0.1296920505, 0.0, 7.2e-8)\n",
            "call": "compute_fock_reference_points(*args)",
            "gold_call": "_oracle_compute_fock_reference_points(*args)",
            "tol": 1e-12,
        },
        # --- boundary: noiseless detectors ------------------------------------
        {
            "setup": "args = (10, 0.1296920505, 0.08, 0.0)\n",
            "call": "compute_fock_reference_points(*args)",
            "gold_call": "_oracle_compute_fock_reference_points(*args)",
            "tol": 1e-12,
        },
        # --- boundary: fully mixing misalignment ------------------------------
        {
            "setup": "args = (8, 0.1296920505, 0.25 * 3.141592653589793, 0.0)\n",
            "call": "compute_fock_reference_points(*args)",
            "gold_call": "_oracle_compute_fock_reference_points(*args)",
            "tol": 1e-12,
        },
        # --- boundary: fully opaque channel -----------------------------------
        {
            "setup": "args = (6, 0.0, 0.08, 7.2e-8)\n",
            "call": "compute_fock_reference_points(*args)",
            "gold_call": "_oracle_compute_fock_reference_points(*args)",
            "tol": 1e-12,
        },
        # --- edge: transparent channel with unit transmittance ----------------
        {
            "setup": "args = (6, 1.0, 0.12, 1e-6)\n",
            "call": "compute_fock_reference_points(*args)",
            "gold_call": "_oracle_compute_fock_reference_points(*args)",
        },
        # --- edge: reference values must stay strictly inside the unit interval
        {
            "setup": """import numpy as np
def interior_margins(fn):
    y, e = fn(14, 0.1296920505, 0.08, 7.2e-8)
    return np.array([float(np.min(y)), float(np.max(y)), float(np.min(e)), float(np.max(e))])
""",
            "call": "interior_margins(compute_fock_reference_points)",
            "gold_call": "interior_margins(_oracle_compute_fock_reference_points)",
            "tol": 1e-12,
        },
        # --- edge: dark counts dominating a very lossy link -------------------
        {
            "setup": "args = (10, 1e-6, 0.08, 1e-5)\n",
            "call": "compute_fock_reference_points(*args)",
            "gold_call": "_oracle_compute_fock_reference_points(*args)",
            "tol": 1e-12,
        },
        # --- invalid: negative cut-off ----------------------------------------
        {
            "setup": guard + "args = (-1, 0.13, 0.08, 7.2e-8)\n",
            "call": "run_model(compute_fock_reference_points, *args)",
            "gold_call": "run_model(_oracle_compute_fock_reference_points, *args)",
        },
        # --- invalid: negative transmittance ----------------------------------
        {
            "setup": guard + "args = (10, -0.1, 0.08, 7.2e-8)\n",
            "call": "run_model(compute_fock_reference_points, *args)",
            "gold_call": "run_model(_oracle_compute_fock_reference_points, *args)",
        },
        # --- invalid: misalignment beyond a quarter turn -----------------------
        {
            "setup": guard + "args = (10, 0.13, 2.0, 7.2e-8)\n",
            "call": "run_model(compute_fock_reference_points, *args)",
            "gold_call": "run_model(_oracle_compute_fock_reference_points, *args)",
        },
    ]
