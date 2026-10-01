"""
Regularize the screened head at the zero transfer by the source's small-circle average.

The bare interaction diverges at zero transfer while the head of the inverse

dielectric matrix tends to one linearly. The source replaces W_00(0) by the

average of the interaction over a small circle about Gamma and evaluates the

linear correction through two screening parameters, one per axis, obtained

numerically from the slope of the inverse dielectric head. Its closed form

carries a factor one quarter on the slope term.

Returns
-------
a finite native float, the regularized head W_00(0) in eV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def head_regularization(
    inverse_head_x: float,
    inverse_head_y: float,
    transfer_x: float,
    transfer_y: float,
    fraction: float,
    area: float,
    coupling: float,
) -> float:
    """Return the regularized screened head W_00 at zero transfer, in eV.

    Parameters
    ----------
    inverse_head_x, inverse_head_y : float
        Finite real values in (0, 1], the head element of the full inverse
        dielectric matrix at the mesh-adjacent transfers (transfer_x, 0) and
        (0, transfer_y), respectively.
    transfer_x, transfer_y : float
        Finite strictly positive magnitudes, in inverse angstroms, of the
        mesh-adjacent transfers along the x and y axes.
    fraction : float
        Finite value in (0, 1]; the circle radius is q0 = fraction * k0 with
        k0 the smaller of the two transfers.
    area : float
        Positive finite unit-cell area in square angstroms.
    coupling : float
        Nonnegative finite Coulomb constant C in eV angstroms; the strict-2D
        Fourier interaction at positive p is 2*pi*C/(area*p).

    Returns
    -------
    head : float
        The source's small-circle regularization: the average of the strict-2D
        interaction over the circle of radius q0 about Gamma, 4*pi*C/(area*q0),
        multiplied by 1 - q0*(r0x + r0y)/4, where the screening parameters are
        the finite-difference slopes r0x = (1 - inverse_head_x)/transfer_x and
        r0y = (1 - inverse_head_y)/transfer_y of the inverse dielectric head.

    Raises
    ------
    ValueError
        If any input is not finite and real, if a transfer or the area is not
        strictly positive, if the fraction lies outside (0, 1], if the coupling
        is negative, or if an inverse head lies outside (0, 1].
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_head_regularization(
    inverse_head_x: float,
    inverse_head_y: float,
    transfer_x: float,
    transfer_y: float,
    fraction: float,
    area: float,
    coupling: float,
) -> float:
    """Source's small-circle regularization of the screened head at Gamma.

    The screening parameters are the finite-difference slopes of the head of the
    inverse dielectric matrix from Gamma to the mesh-adjacent transfer along each
    axis, r0x = (1 - inverse_head_x)/transfer_x and likewise for y. The circle
    radius is q0 = fraction * k0 with k0 the smaller of the two transfers. The
    average of the strict-2D Coulomb potential 2*pi*C/(area*p) over that circle is
    4*pi*C/(area*q0), and the source's closed form multiplies it by
    (1 - q0*(r0x + r0y)/4).
    """
    ex = _finite_scalar(inverse_head_x, "inverse_head_x", -np.inf)
    ey = _finite_scalar(inverse_head_y, "inverse_head_y", -np.inf)
    kx = _finite_scalar(transfer_x, "transfer_x", 0.0)
    ky = _finite_scalar(transfer_y, "transfer_y", 0.0)
    fr = _finite_scalar(fraction, "fraction", 0.0)
    ar = _finite_scalar(area, "area", 0.0)
    c = _finite_scalar(coupling, "coupling", 0.0)
    if kx <= 0.0 or ky <= 0.0:
        raise ValueError("transfers must be strictly positive")
    if not (0.0 < fr <= 1.0):
        raise ValueError("fraction must lie in (0, 1]")
    if ar <= 0.0:
        raise ValueError("area must be positive")
    if not (0.0 < ex <= 1.0 and 0.0 < ey <= 1.0):
        raise ValueError("inverse heads must lie in (0, 1]")
    r0x = (1.0 - ex) / kx
    r0y = (1.0 - ey) / ky
    q0 = fr * min(kx, ky)
    average = 4.0 * np.pi * c / (ar * q0)
    return float(average * (1.0 - q0 * (r0x + r0y) / 4.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Two-dimensional and Q2D heads at the prompt mesh, a coarse mesh, and rejections."""
    return [
        {
            "setup": "import numpy as np\n",
            "call": "head_regularization(0.958014716, 0.955394748, 2*np.pi/(9*3.2), 2*np.pi/(9*4.1), 0.5, 3.2*4.1, 3.59991137)",
            "gold_call": "_oracle_head_regularization(0.958014716, 0.955394748, 2*np.pi/(9*3.2), 2*np.pi/(9*4.1), 0.5, 3.2*4.1, 3.59991137)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "head_regularization(0.968446, 0.964182, 2*np.pi/(9*3.2), 2*np.pi/(9*4.1), 0.4, 3.2*4.1, 3.59991137)",
            "gold_call": "_oracle_head_regularization(0.968446, 0.964182, 2*np.pi/(9*3.2), 2*np.pi/(9*4.1), 0.4, 3.2*4.1, 3.59991137)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "head_regularization(0.91, 0.87, 2*np.pi/(3*2.8), 2*np.pi/(3*3.6), 1.0, 2.8*3.6, 2.8)",
            "gold_call": "_oracle_head_regularization(0.91, 0.87, 2*np.pi/(3*2.8), 2*np.pi/(3*3.6), 1.0, 2.8*3.6, 2.8)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "head_regularization(1.0, 1.0, 0.3, 0.2, 0.5, 10.0, 0.0)",
            "gold_call": "_oracle_head_regularization(1.0, 1.0, 0.3, 0.2, 0.5, 10.0, 0.0)",
        },
        {
            "setup": """import numpy as np


def rejected(fn):
    try:
        fn(0.95, 0.95, 0.2, 0.2, 1.5, 10.0, 3.0)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "rejected(head_regularization)",
            "gold_call": "rejected(_oracle_head_regularization)",
        },
    ]
