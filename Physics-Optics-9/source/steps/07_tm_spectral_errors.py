"""
Integrate the two weighted spectral errors for supplied modal data.

The exact TM reference is T=(xi_+'/xi_-')*(N/D). Construct the

physical-only and physical-plus-channel finite expansions with subtraction

value -1. Return the integrals of x^2*abs(T-U)^2 over [left,right], in that

order, using an order-point Gauss-Legendre rule on this interval. Supplied

modal data may be incomplete for diagnostic uses; do not replace it by a new

pole search. Use the preceding boundary, radial-wave and modal-sum functions.

No modulus normalization or fitted background is applied to either sum.

Returns
-------
float64 ndarray, shape (2,), physical-only and channel-completed weighted spectral errors
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tm_spectral_errors(
    ell: int,
    n: float,
    poles: "np.ndarray",
    residues: "np.ndarray",
    channels: "np.ndarray",
    left: float,
    right: float,
    order: int,
) -> "np.ndarray":
    """Compute weighted squared errors with the specified quadrature.

    Parameters
    ----------
    ell : int
        Angular momentum, 1 through 4.
    n : float
        Real refractive index in [2, 3].
    poles : ndarray
        One-dimensional complex physical-pole data; may be empty.
    residues : ndarray
        Complex residue data with the same shape as poles.
    channels : ndarray
        Complex array of shape (number_of_channel_poles, 2), with each
        row containing a channel pole and its residue. May have zero rows.
    left, right : float
        Positive real band endpoints, 0.1 <= left < right <= 8.
    order : int
        Number of Gauss-Legendre nodes, an integer from 16 through 800.

    Returns
    -------
    ndarray
        Float64 array of shape (2,), physical-only error followed by the
        error including the supplied channel data.

    Raises
    ------
    ValueError
        If the band or order is outside its range, ell or n is unsupported,
        or the supplied pole/residue data violates the preceding sum contract.

    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import roots_legendre


def _oracle_tm_spectral_errors(
    ell: int,
    n: float,
    poles: "np.ndarray",
    residues: "np.ndarray",
    channels: "np.ndarray",
    left: float,
    right: float,
    order: int,
) -> "np.ndarray":
    if (
        not 0.1 <= left < right <= 8.0
        or not 16 <= order <= 800
        or int(order) != order
    ):
        raise ValueError("unsupported band or quadrature order")
    poles = np.asarray(poles, dtype=complex)
    residues = np.asarray(residues, dtype=complex)
    channels = np.asarray(channels, dtype=complex)
    nodes, weights = roots_legendre(order)
    x = left + 0.5 * (right - left) * (nodes + 1)
    weights = 0.5 * (right - left) * weights
    radial = _oracle_riccati_wave_data(ell, x)
    boundary = _oracle_tm_boundary_data(ell, n, x)
    exact = radial[:, 4] / radial[:, 7] * boundary[:, 0] / boundary[:, 1]
    physical = _oracle_subtracted_pole_sum(x, poles, residues, -1.0)
    complete = _oracle_subtracted_pole_sum(
        x,
        np.concatenate([poles, channels[:, 0]]),
        np.concatenate([residues, channels[:, 1]]),
        -1.0,
    )
    weighted = weights * x**2
    return np.array(
        [
            np.sum(weighted * np.abs(exact - physical) ** 2),
            np.sum(weighted * np.abs(exact - complete) ** 2),
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
            "setup": (
                "import numpy as np\n"
                "\n"
                "p = np.array(\n"
                "    [\n"
                "        -2.4021436406684367 - 0.0470498072800886j,\n"
                "        2.4021436406684367 - 0.0470498072800886j,\n"
                "    ]\n"
                ")\n"
                "r = np.array(\n"
                "    [\n"
                "        -0.0331678256666403 + 0.0860589876996838j,\n"
                "        0.0331678256666403 + 0.0860589876996838j,\n"
                "    ]\n"
                ")\n"
                "c = np.array(\n"
                "    [\n"
                "        [\n"
                "            0.8705692253836403 + 2.157137812402353j,\n"
                "            0.0433279494695307 + 0.1011190305771352j,\n"
                "        ]\n"
                "    ]\n"
                ")\n"
            ),
            "call": (
                "tm_spectral_errors(\n"
                "    3,\n"
                "    2.7,\n"
                "    p.copy(),\n"
                "    r.copy(),\n"
                "    c.copy(),\n"
                "    0.35,\n"
                "    7.4,\n"
                "    400,\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_tm_spectral_errors(\n"
                "    3,\n"
                "    2.7,\n"
                "    p.copy(),\n"
                "    r.copy(),\n"
                "    c.copy(),\n"
                "    0.35,\n"
                "    7.4,\n"
                "    400,\n"
                ")\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "p = np.array([], dtype=complex)\n"
                "r = np.array([], dtype=complex)\n"
                "c = np.empty((0, 2), dtype=complex)\n"
            ),
            "call": (
                "tm_spectral_errors(\n"
                "    1, 2.0, p.copy(), r.copy(), c.copy(), 0.1, 0.4, 16\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_tm_spectral_errors(\n"
                "    1, 2.0, p.copy(), r.copy(), c.copy(), 0.1, 0.4, 16\n"
                ")\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "p = np.array([2.0 - 0.3j])\n"
                "r = np.array([0.1 + 0.2j])\n"
                "c = np.empty((0, 2), dtype=complex)\n"
            ),
            "call": (
                "tm_spectral_errors(\n"
                "    4, 3.0, p.copy(), r.copy(), c.copy(), 2.0, 8.0, 800\n"
                ")\n"
            ),
            "gold_call": (
                "_oracle_tm_spectral_errors(\n"
                "    4, 3.0, p.copy(), r.copy(), c.copy(), 2.0, 8.0, 800\n"
                ")\n"
            ),
            "tol": 1e-09,
        },
    ]
