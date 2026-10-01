"""
Run the complete TM channel-pole audit and return its diagnostic scalar.

This is the final orchestrator. Find all physical poles for the chosen

ell, n and x_cut in |Re(x)|<x_cut, -4<Im(x)<0, then obtain their R3 residues

and every derivative-normalization channel pole/residue. Compute both weighted

errors on [0.35,7.4] with a 400-point Gauss-Legendre rule, using exact TM

boundary data, the common subtraction constant -1 and both real-part signs.

The requested scalar is log10(E_physical/E_complete). Compose every earlier

step, directly or transitively, and use its output. The production instance

is n=2.7, ell=3, x_cut=18.5. This finite modal truncation defines the audit;

increasing the pole domain changes the experiment, not its quadrature accuracy.

Returns
-------
float64 ndarray, shape (5,), log10 error ratio, two errors and the two pole counts
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scattering_pole_audit(n: float, ell: int, x_cut: float) -> "np.ndarray":
    """Compute the complete derivative-normalized TM modal audit.

    Parameters
    ----------
    n : float
        Real refractive index in [2, 3].
    ell : int
        Angular momentum, 1 through 4.
    x_cut : float
        Real-part pole cutoff in [8, 22]; poles are simple and at least
        1e-5 from the boundary, as in the physical-pole contract.

    Returns
    -------
    ndarray
        Float64 array of shape (5,), ordered as log10 of the physical-only
        error divided by the complete error, physical-only error, complete
        error, physical-pole count and channel-pole count. Both errors use
        the fixed 400-node rule specified in the scientific background.

    Raises
    ------
    ValueError
        If an input lies outside the supported domain or the physical-pole
        set cannot be resolved numerically.

    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_scattering_pole_audit(
    n: float, ell: int, x_cut: float
) -> "np.ndarray":
    poles = _oracle_physical_tm_poles(ell, n, x_cut)
    residues = _oracle_tm_physical_residues(ell, n, poles)
    channels = _oracle_tm_channel_data(ell, n)
    errors = _oracle_tm_spectral_errors(
        ell, n, poles, residues, channels, 0.35, 7.4, 400
    )
    return np.array(
        [
            np.log10(errors[0] / errors[1]),
            errors[0],
            errors[1],
            len(poles),
            len(channels),
        ],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "setup": ("import numpy as np\n"),
            "call": ("scattering_pole_audit(2.6, 2, 12.7)\n"),
            "gold_call": ("_oracle_scattering_pole_audit(2.6, 2, 12.7)\n"),
            "tol": 1e-08,
        },
        {
            "setup": ("import numpy as np\n"),
            "call": ("scattering_pole_audit(2.0, 1, 8.2)\n"),
            "gold_call": ("_oracle_scattering_pole_audit(2.0, 1, 8.2)\n"),
            "tol": 1e-08,
        },
        {
            "setup": ("import numpy as np\n"),
            "call": ("scattering_pole_audit(3.0, 4, 20.4)\n"),
            "gold_call": ("_oracle_scattering_pole_audit(3.0, 4, 20.4)\n"),
            "tol": 1e-08,
        },
    ]
