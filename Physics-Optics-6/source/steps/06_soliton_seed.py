"""
Build the analytic pulse that the driven cavity's stationary solution grows out of.

Dropping loss, drive and the delayed response leaves a conservative problem whose localised solution is the classical hyperbolic-secant pulse. In the dimensionless variables its width is the inverse square root of the detuning and its peak amplitude is the square root of twice the detuning; no free parameter survives.

That pulse is not the stationary state of the driven, damped cavity, but integrating forward from it relaxes onto that state. The fast-time grid is centred so the pulse begins at the middle of the window.

Returns
-------
np.ndarray of shape (n_modes,), complex128: the conservative localised pulse at this detuning, on the centred fast-time grid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def soliton_seed(zeta: float, n_modes: int, window: float) -> "np.ndarray":
    '''Return the conservative hyperbolic-secant pulse used to start the integration.

    The fast-time grid holds n_modes uniformly spaced points of spacing window / n_modes, with
    the sample of index n_modes // 2 sitting at fast time zero, so the grid runs from
    -(n_modes // 2) * spacing upwards.

    The field this pipeline advances obeys, in its dimensionless variables,

        dpsi/dtau = -(1 + i*zeta)*psi + i*d2psi/dtheta2
                    + i*psi*((1 - f_R)*|psi|^2 + f_R*(h conv |psi|^2)) + f_pump,

    with theta the fast time in units of tau0 and no numerical factor on the curvature term.

    On that grid the field is the localised solution of the conservative reduction of that
    equation at this detuning: the equation with the loss, the drive and the delayed channel
    all removed, leaving only the detuning, the
    fast-time curvature and the instantaneous nonlinearity. That balance admits one localised
    solution, centred at fast time zero, with no free parameter beyond zeta. It is real and
    positive, and is returned as a complex array.

    Parameters
    ----------
    zeta : float
        Dimensionless cavity detuning, positive.
    n_modes : int
        Number of fast-time samples.
    window : float
        Length of the periodic fast-time window in dimensionless units.

    Returns
    -------
    psi0 : np.ndarray
        Complex array of shape (n_modes,) holding the seed field.
    '''
    return psi0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_soliton_seed(zeta: float, n_modes: int, window: float) -> "np.ndarray":
    spacing = window / n_modes
    theta = (np.arange(n_modes) - n_modes // 2) * spacing
    arg = np.clip(theta * np.sqrt(zeta), -700.0, 700.0)
    return (np.sqrt(2.0 * zeta) / np.cosh(arg)).astype(complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the benchmark grid and detuning.
        {
            "setup": "import numpy as np\n",
            "call": "soliton_seed(17.795878, 512, 10.07153)",
            "gold_call": "_oracle_soliton_seed(17.795878, 512, 10.07153)",
        },
        # Boundary: an odd number of samples, where the centre index and the grid offset
        # cannot be read off by symmetry.
        {
            "setup": "import numpy as np\n",
            "call": "soliton_seed(4.0, 33, 12.0)",
            "gold_call": "_oracle_soliton_seed(4.0, 33, 12.0)",
        },
        # Edge: a weak detuning, where the pulse is wide enough to reach the window edges and
        # the periodic seam is no longer negligible.
        {
            "setup": "import numpy as np\n",
            "call": "soliton_seed(0.04, 128, 8.0)",
            "gold_call": "_oracle_soliton_seed(0.04, 128, 8.0)",
        },
    ]
