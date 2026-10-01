"""
Return the two diet coefficients of a predator that feeds on exactly two prey, with trophic positions y_below and y_above (y_below < y_above), given that the predator's trophic position minus one equals centre and lies between the two prey positions: the coefficients are the unique proportions summing to one whose weighted mean of the prey positions equals centre. Return the array [coefficient of the lower prey, coefficient of the upper prey].

For a predator with two prey the trophic-position definition and the unit row sum give two equations in two unknowns, so the diet is fully determined by the three trophic positions; this two-prey solution is the building block of the reconstruction.

Returns
-------
numpy.ndarray of float64 with shape (2,): [coefficient of the lower prey, coefficient of the upper prey].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pair_diet_coefficients(y_below: float, y_above: float, centre: float) -> "numpy.ndarray":
    """Return the two diet coefficients of a predator that feeds on exactly two prey, with trophic positions y_below and y_above (y_below < y_above), given that the predator's trophic position minus one equals centre and lies between the two prey positions: the coefficients are the unique proportions summing to one whose weighted mean of the prey positions equals centre. Return the array [coefficient of the lower prey, coefficient of the upper prey].

    Parameters
    ----------
    y_below : float
        Trophic position of the lower prey.
    y_above : float
        Trophic position of the upper prey, strictly larger than y_below.
    centre : float
        The predator's trophic position minus one; must satisfy y_below <= centre <= y_above.

    Returns
    -------
    coefficients : numpy.ndarray
        Array of shape (2,): the diet coefficients of the lower and of the upper prey (float64), summing to one.

    Raises
    ------
    ValueError
        If any input is not finite, y_above does not exceed y_below, or centre lies outside [y_below, y_above].
    """
    return coefficients

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np


def _oracle_pair_diet_coefficients(y_below: float, y_above: float, centre: float) -> "numpy.ndarray":
    """Two-prey solution of Eq 7: the diet coefficients (below, above) of a predator whose prey mean position is centre."""
    yb, ya, c = float(y_below), float(y_above), float(centre)
    if not all(np.isfinite([yb, ya, c])):
        raise ValueError("all inputs must be finite")
    if not (yb <= c <= ya) or ya <= yb:
        raise ValueError("the centre must lie between the two prey positions, y_below < y_above")
    # the predator's prey mean c is a weighted average of the two prey positions (Eq 6a) with weights summing
    # to one (Eq 6b): linear interpolation of c between y_below and y_above
    return np.array([(ya - c) / (ya - yb), (c - yb) / (ya - yb)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "y_below, y_above, centre = 3.3, 3.54, 3.429\n",
            "call": "pair_diet_coefficients(y_below, y_above, centre)",
            "gold_call": "_oracle_pair_diet_coefficients(y_below, y_above, centre)",
        },
        {
            "setup": "y_below, y_above, centre = 2.0, 3.3, 2.75\n",
            "call": "pair_diet_coefficients(y_below, y_above, centre)",
            "gold_call": "_oracle_pair_diet_coefficients(y_below, y_above, centre)",
        },
        {
            "setup": "y_below, y_above, centre = 4.14, 5.00225, 4.14\n",
            "call": "pair_diet_coefficients(y_below, y_above, centre)",
            "gold_call": "_oracle_pair_diet_coefficients(y_below, y_above, centre)",
        },
        {
            "setup": "y_below, y_above, centre = 3.3, 3.54, 3.6\ndef run_model():\n    try:\n        pair_diet_coefficients(y_below, y_above, centre)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_pair_diet_coefficients(y_below, y_above, centre)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
