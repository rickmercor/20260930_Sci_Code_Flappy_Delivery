"""
Precompute the exact solution operators for the linear half of a driven, damped, dispersive cavity so that the nonlinear integration can be split.

Building the linear operators once makes the linear half step exact for any step length, so that only the nonlinear phase is approximated. Each stored operator advances the field by half of the integration step, a full step being a linear half step, a nonlinear phase, and a second linear half step.

Returns
-------
np.ndarray of shape (2, n_modes), complex128: row zero the field factor and row one the drive factor of the exact linear half step.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lle_propagators(zeta: float, n_modes: int, window: float, dtau: float) -> "np.ndarray":
    '''Return the exact field and drive propagators for one linear half step.

    The mode offsets are the angular frequencies conjugate to the fast time on a periodic
    window of length window with n_modes samples, in the standard transform ordering, so the
    offset of index k is two pi times the discrete transform frequency for sample spacing
    window / n_modes.

    The field this pipeline advances obeys, in its dimensionless variables,

        dpsi/dtau = -(1 + i*zeta)*psi + i*d2psi/dtheta2
                    + i*psi*((1 - f_R)*|psi|^2 + f_R*(h conv |psi|^2)) + f_pump,

    with theta the fast time in units of tau0 and no numerical factor on the curvature term.

    The two returned rows carry out, exactly, the part of that equation which excludes the
    nonlinear term, over half of dtau: row zero multiplies the transformed field, row one
    multiplies the transformed drive, and their sum is the transformed field at the end of the
    half step.

    Parameters
    ----------
    zeta : float
        Dimensionless cavity detuning.
    n_modes : int
        Number of fast-time samples, equal to the number of modes retained.
    window : float
        Length of the periodic fast-time window in dimensionless units.
    dtau : float
        Full integration step in dimensionless slow time. Each returned operator advances by
        half of it.

    Returns
    -------
    propagators : np.ndarray
        Complex array of shape (2, n_modes). Row zero is the field propagator over a half
        step; row one is the drive propagator over a half step.
    '''
    return propagators  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_lle_propagators(zeta: float, n_modes: int, window: float, dtau: float) -> "np.ndarray":
    offsets = 2.0 * np.pi * np.fft.fftfreq(n_modes, d=window / n_modes)
    linear = -(1.0 + 1j * zeta) - 1j * offsets * offsets
    field = np.exp(linear * dtau / 2.0)
    # The real part of `linear` is -1 everywhere, so it never vanishes and the drive
    # propagator needs no removable-singularity branch.
    return np.vstack([field, (field - 1.0) / linear])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # Normal: the benchmark grid, detuning and step.
        {
            "setup": "import numpy as np\n",
            "call": "lle_propagators(17.795878, 512, 10.07153, 2.0e-4)",
            "gold_call": "_oracle_lle_propagators(17.795878, 512, 10.07153, 2.0e-4)",
        },
        # Boundary: a zero step, where the field propagator must be exactly one and the drive
        # propagator exactly zero for every mode.
        {
            "setup": "import numpy as np\n",
            "call": "lle_propagators(3.0, 32, 6.0, 0.0)",
            "gold_call": "_oracle_lle_propagators(3.0, 32, 6.0, 0.0)",
        },
        # Edge: a long step on a narrow window, where the highest mode offsets are large and
        # the propagators decay to the steady-state limit.
        {
            "setup": "import numpy as np\n",
            "call": "lle_propagators(0.5, 64, 2.0, 12.0)",
            "gold_call": "_oracle_lle_propagators(0.5, 64, 2.0, 12.0)",
        },
    ]
