"""
Select the regularisation weight from a family of candidate solutions by locating the corner of the L-curve. The step receives two arrays of the same length, one entry per candidate in order of increasing weight, and returns the index of the candidate the criterion selects. It performs no solves of its own: the caller has already produced the family and evaluated both quantities for every member, and this step is the decision rule alone.

A regularised problem has no single right answer until the weight is fixed, and the weight cannot be read off the data because the noise level alone does not determine it. The standard device is to plot two competing quantities of the family against each other on logarithmic axes and take the point where the curve turns most sharply. The first argument is the residual quantity of each candidate and is placed on the horizontal axis; the second is the regularisation norm and is placed on the vertical axis. What enters each of the two quantities is fixed by the source paper's parameter-choice section and is the caller's responsibility; this step only needs them in that order.

The turn is measured by the signed curvature of the plane curve through the points (log of the first quantity, log of the second), traversed in the order the candidates are supplied: the product of the first derivative of the horizontal coordinate with the second derivative of the vertical one, minus the product of the second derivative of the horizontal coordinate with the first derivative of the vertical one, divided by the sum of the squared first derivatives raised to the power three halves. Take every derivative by central differences in the candidate index with unit spacing, consider only interior candidates, compare the signed values rather than their magnitudes, and break ties towards the lowest index. Swapping the axes or reversing the order of traversal flips the sign of every curvature, so either slip turns the most concave point into the selection.

The curvature is scale-invariant in a useful way: multiplying either quantity by a positive constant shifts its logarithm by a constant and leaves every difference unchanged, so the selected candidate does not depend on the units either quantity is measured in, nor on the base of the logarithm.

Choosing a regularisation weight is the part of an inverse problem that cannot be automated away by refining anything. Too little and the reconstruction is noise amplified; too much and it is a smooth curve that has forgotten the measurement. The difficulty is that both failure modes look plausible in isolation, so the choice has to be made by comparing a family of candidates against each other rather than by examining any one of them.

The L-curve is the best known of the comparison devices. Its shape is the point: for a wide class of discrete ill-posed problems a residual quantity and a regularisation norm traded against each other, plotted logarithmically, form a curve with two nearly straight arms meeting in a distinct corner, and the corner marks where the balance tips from one failure mode to the other. Locating it is then a question about the geometry of a plane curve, which is why curvature is the natural instrument: it is largest exactly where a curve turns most sharply, and it is invariant under the rescalings that the choice of units would otherwise introduce.

Discretely sampled curves need care. The curvature of a plane curve is built from first and second derivatives of both coordinates, and on a sampled curve those have to be approximated. Because the candidates are usually laid out on a geometric ladder, the natural parameter is the index rather than the weight itself, and treating the index as the parameter keeps the difference formulas uniform. Interior points are the only ones where a symmetric formula is available, which is why the corner is never reported at an endpoint and why a family should bracket the expected answer rather than start at it.

Curvature carries a sign as well as a size. Relative to the direction of travel a plane curve bends one way with positive curvature and the other way with negative curvature, and a sampled L-curve usually shows bends of both kinds, including sharp concave kinks near the heavily regularised end. A rule that ranks candidates by the magnitude of the curvature can therefore land on one of those kinks instead of the corner, which is why the orientation of the axes, the direction of travel and the use of the signed value all belong to the specification.

Returns
-------
float, the index of the selected candidate as a floating-point value, always strictly between 0 and one less than the number of candidates.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def lcurve_corner_index(residual_quantity, regularisation_norm):
    """Index of the L-curve corner over a family of candidate solutions.

    Both arguments hold one entry per candidate, ordered by increasing
    regularisation weight.  With x_j = log(residual_quantity[j]) on the
    horizontal axis and y_j = log(regularisation_norm[j]) on the vertical
    axis, the corner is the interior candidate that maximises the signed
    curvature
        kappa_j = (x'_j y''_j - x''_j y'_j) / (x'_j**2 + y'_j**2)**1.5,
    with x', y' the central first differences and x'', y'' the central
    second differences in the index j (unit spacing).  The signed value is
    compared, not its magnitude, and ties go to the lowest index.

    Args:
        residual_quantity (np.ndarray): shape (J,), strictly positive.
        regularisation_norm (np.ndarray): shape (J,), strictly positive.

    Expected return:
        float: the selected candidate index, strictly between 0 and J-1.

    Raises:
        ValueError: if the two arrays have different lengths, if fewer than
        three candidates are supplied, or if either array has a non-positive
        entry.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_lcurve_corner_index(residual_quantity, regularisation_norm):
    def _curvature(lx, ly, j):
        x1 = 0.5 * (lx[j + 1] - lx[j - 1])
        y1 = 0.5 * (ly[j + 1] - ly[j - 1])
        x2 = lx[j + 1] - 2.0 * lx[j] + lx[j - 1]
        y2 = ly[j + 1] - 2.0 * ly[j] + ly[j - 1]
        den = (x1 * x1 + y1 * y1) ** 1.5
        if den == 0.0:
            return 0.0
        return (x1 * y2 - x2 * y1) / den

    a = np.asarray(residual_quantity, dtype=float).reshape(-1)
    b = np.asarray(regularisation_norm, dtype=float).reshape(-1)
    if a.size != b.size:
        raise ValueError("the two quantities must have the same length")
    if a.size < 3:
        raise ValueError("at least three candidates are required")
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("both quantities must be strictly positive")
    lx = np.log(a)
    ly = np.log(b)
    best = -np.inf
    idx = 1
    for j in range(1, a.size - 1):
        kap = _curvature(lx, ly, j)
        if kap > best:
            best = kap
            idx = j
    return float(idx)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": "R = np.array([1.432947378321885e-06, 1.4326456455132533e-05, 0.00014296353458271348, 0.0014002187672507702, 0.011616288480568109, 0.04359077195432432, 0.07177512248029197, 0.10871723932922955, 0.3378390171046489, 0.9683872684888668, 1.4633445051830587])\nQ = np.array([10.008848130282349, 10.007002215718579, 9.98858838294412, 9.808876130824869, 8.36920318388763, 4.391949944409112, 2.7281224296245488, 2.3033540906105556, 1.7415030165518712, 0.7220910960349058, 0.13899744384472554])",
            "call": "lcurve_corner_index(R, Q)",
            "gold_call": "_oracle_lcurve_corner_index(R, Q)",
        },
        {
            "setup": "R = np.array([1.2042501571541096e-06, 1.2039859223919535e-05, 0.00012013499084189798, 0.0011756131791766592, 0.00968348679694632, 0.03531269532642973, 0.05337526730493257, 0.11543354618951013, 0.6583496778823867, 3.258949594655435, 6.307166529469901])\nQ = np.array([10.411400896392967, 10.410147641637218, 10.397652082479572, 10.276299764530423, 9.346782044761481, 7.3219171039197075, 6.87396506740155, 6.708503943073768, 6.028763114434793, 3.496637130901241, 0.6914798908637894])",
            "call": "lcurve_corner_index(R, Q)",
            "gold_call": "_oracle_lcurve_corner_index(R, Q)",
        },
        {
            "setup": "t = np.arange(11.0)\nx = np.exp(np.where(t <= 5.0, -3.0 + 0.05 * t, -2.75 + 1.20 * (t - 5.0)))\ny = np.exp(np.where(t <= 5.0, 2.0 - 1.20 * t, -4.0 - 0.05 * (t - 5.0)))",
            "call": "lcurve_corner_index(x, y)",
            "gold_call": "_oracle_lcurve_corner_index(x, y)",
        },
        {
            "setup": "t = np.arange(11.0)\nx = np.exp(np.where(t <= 5.0, -3.0 + 0.05 * t, -2.75 + 1.20 * (t - 5.0)))\ny = np.exp(np.where(t <= 5.0, 2.0 - 1.20 * t, -4.0 - 0.05 * (t - 5.0)))",
            "call": "lcurve_corner_index(137.0 * x, 0.004 * y)",
            "gold_call": "_oracle_lcurve_corner_index(x, y)",
        },
        {
            "setup": "pass",
            "call": "lcurve_corner_index(np.array([1.0, 2.0, 4.0]), np.array([4.0, 2.0, 1.0]))",
            "gold_call": "_oracle_lcurve_corner_index(np.array([1.0, 2.0, 4.0]), np.array([4.0, 2.0, 1.0]))",
        },
        {
            "setup": "R = np.array([1.432947378321885e-06, 1.4326456455132533e-05, 0.00014296353458271348, 0.0014002187672507702, 0.011616288480568109, 0.04359077195432432, 0.07177512248029197, 0.10871723932922955, 0.3378390171046489, 0.9683872684888668, 1.4633445051830587])\nQ = np.array([10.008848130282349, 10.007002215718579, 9.98858838294412, 9.808876130824869, 8.36920318388763, 4.391949944409112, 2.7281224296245488, 2.3033540906105556, 1.7415030165518712, 0.7220910960349058, 0.13899744384472554])",
            "call": "lcurve_corner_index(Q, R)",
            "gold_call": "_oracle_lcurve_corner_index(Q, R)",
        },
        {
            "setup": "def run_model():\n    try:\n        lcurve_corner_index(np.ones(5), np.ones(4))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_lcurve_corner_index(np.ones(5), np.ones(4))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        lcurve_corner_index(np.array([1.0, 0.0, 2.0]), np.ones(3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_lcurve_corner_index(np.array([1.0, 0.0, 2.0]), np.ones(3))\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
