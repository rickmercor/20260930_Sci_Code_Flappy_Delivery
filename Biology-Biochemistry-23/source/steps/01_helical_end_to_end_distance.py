"""
Return the end-to-end distance in nm of an n_bp base-pair B-DNA duplex measured between its two force-bearing termini under shear, i.e. the 5' terminus of one strand and the 5' terminus of the other strand, which lie at opposite ends of the duplex on the two different backbones. Use the source's explicit three-dimensional parametrisation of the two backbone helices with the standard B-DNA parameters 2 nm diameter, 0.34 nm rise per base pair and 10.5 base pairs per turn, and the source's convention for where on the two helices the two termini sit; do not return the contour length along the helix axis.

For duplexes only a few base pairs long the axial contour length is comparable to the 2 nm helix diameter, so the distance over which a shear force acts depends on where on the helix the two pulled termini sit. This geometric length is the rod length that enters the force-extension response of the duplex and it controls how much mechanical work each base-pair transition costs.

Returns
-------
float, the helical end-to-end distance of the duplex in nm.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def helical_end_to_end_distance(n_bp: int) -> float:
    """Return the end-to-end distance in nm of an n_bp base-pair B-DNA duplex measured between its two force-bearing termini under shear, i.e. the 5' terminus of one strand and the 5' terminus of the other strand, which lie at opposite ends of the duplex on the two different backbones. Use the source's explicit three-dimensional parametrisation of the two backbone helices with the standard B-DNA parameters 2 nm diameter, 0.34 nm rise per base pair and 10.5 base pairs per turn, and the source's convention for where on the two helices the two termini sit; do not return the contour length along the helix axis.

    Parameters
    ----------
    n_bp : int
        Number of closed base pairs in the duplex (>= 1).

    Returns
    -------
    distance : float
        End-to-end distance in nm.

    Raises
    ------
    ValueError
        If n_bp is not an integer >= 1.
    """
    return distance

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, curve_fit


def _oracle_helical_end_to_end_distance(n_bp: int) -> float:
    """End-to-end distance (nm) of an n_bp duplex between its two force-bearing termini, SI eqs (15)-(17).

    Backbones r1(phi) = (R cos phi, R sin phi, p phi / 2 pi), r2(phi) = (-R cos phi, -R sin phi, p phi / 2 pi)
    with R = 1.0 nm, pitch p = 0.34 nm * 10.5, phi(n) = 2 pi n / 10.5. The termini are base 1 on strand 1
    and base n_bp on strand 2: axial separation 0.34 (n_bp - 1) nm and transverse separation
    2 R |cos(pi (n_bp - 1) / 10.5)| nm (2 nm for a single base pair).
    """
    if isinstance(n_bp, bool) or int(n_bp) != n_bp or int(n_bp) < 1:
        raise ValueError("n_bp must be an integer >= 1")
    n = int(n_bp)
    rise, radius, bp_per_turn = 0.34, 1.0, 10.5
    axial = rise * (n - 1)
    transverse = 2.0 * radius * abs(np.cos(np.pi * (n - 1) / bp_per_turn))
    return float(np.sqrt(axial * axial + transverse * transverse))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nn_bp = 9\n",
            "call": "helical_end_to_end_distance(n_bp)",
            "gold_call": "_oracle_helical_end_to_end_distance(n_bp)",
        },
        {
            "setup": "import numpy as np\nn_bp = 1\n",
            "call": "helical_end_to_end_distance(n_bp)",
            "gold_call": "_oracle_helical_end_to_end_distance(n_bp)",
        },
        {
            "setup": "import numpy as np\nn_bp = 22\n",
            "call": "helical_end_to_end_distance(n_bp)",
            "gold_call": "_oracle_helical_end_to_end_distance(n_bp)",
        },
        {
            "setup": "import numpy as np\ndef run_model():\n    try:\n        helical_end_to_end_distance(0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_helical_end_to_end_distance(0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
