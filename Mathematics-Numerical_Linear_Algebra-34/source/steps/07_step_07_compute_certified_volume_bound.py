"""
Orchestrate the full pipeline and return the certified upper bound on the volume.

Given the polynomial $g$ as coefficients and exponent vectors, together with the moment budget $K$, carry out the complete computation by calling the public functions of the preceding steps in sequence: identify_scaling_structure to recover the weights and weighted degree from the exponent pattern, compute_certified_maximum to obtain a certified estimate of the range of $g$ over the box, compute_box_moments to obtain the moments of $g$ within the budget, compute_reference_moments to obtain the constants relating moments of $g$ over $K$ to the volume, and compute_upper_bound together with compute_lower_bound to obtain the two endpoints of the certified bracket.

Verify that the endpoints are consistent, in the sense that the lower does not exceed the upper, and return the upper endpoint. This is the tightest upper bound on $\operatorname{vol}K$ that the permitted moment data forces.

This step must call the public functions defined in the preceding steps rather than reimplementing their contents.

The pipeline composes into a chain in which the ambient dimension appears exactly once. Characterization and the reference constants consume only the exponent pattern; the box moments consume the coefficients and constitute the sole stage whose cost grows with the dimension, performed once and independently of the budget thereafter. Everything downstream is univariate, with sizes determined by the budget alone.

The composition is what makes the result certified rather than approximate. Each stage is either exact, as with the rational moments and the rational constants, or produces a bound in a known direction, as with the range estimate and the two extremal values. No stage introduces a heuristic, a sample, or an asymptotic approximation, so the final inequality holds unconditionally rather than with high probability or in the limit.

Computing both endpoints when only one is reported is not redundant. The two are obtained from different constraints, so agreement that they bracket the volume in the correct order is a genuine consistency check on the whole chain: an error in the moments, the constants, or either extremal computation will typically break the ordering. A bracket that inverts indicates a defect upstream, not a tight bound.

Two structural facts govern the accuracy of the result. The bracket tightens monotonically in the budget, since enlarging the matrices shrinks the admissible set, so the reported bound decreases and never increases as more moments are supplied. And the tightening is geometric rather than algebraic, at a rate governed by the separation between the interval carrying the subtracted part and the interval carrying the residual, so a modest budget already resolves several significant figures. The reported value is a bound and not the volume itself: it strictly exceeds the volume at every finite budget, approaching it only as the budget grows without limit.

Returns
-------
float: the certified upper bound on the volume of the unit sublevel set of g, given the moment budget.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_certified_volume_bound(coeffs: np.ndarray, exponents: np.ndarray, max_order: int) -> float:
    """Compute the certified upper bound on the volume of the unit sublevel set.

    Parameters
    ----------
    coeffs : np.ndarray
        Array of shape (s,) and dtype float64 holding the coefficients of
        the s monomials of g.
    exponents : np.ndarray
        Integer array of shape (s, n). Row i holds the exponent vector of
        the i-th monomial, matching coeffs[i]. The polynomial must possess
        an exact scaling structure, be strictly positive away from the
        origin, and have unit sublevel set contained in [-1, 1]^n.
    max_order : int
        The moment budget K. The bound may depend on g only through the
        integrals of g^k over the box for k = 0, ..., K.

    Returns
    -------
    float
        The tightest upper bound on the volume of K = {x : g(x) <= 1}
        that the permitted moment data forces.

    Raises
    ------
    ValueError
        If the exponent array is not two-dimensional, if any upstream
        stage rejects its input, or if the computed bracket is
        inconsistent.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_certified_volume_bound(coeffs: np.ndarray, exponents: np.ndarray, max_order: int) -> float:
    """Reference implementation: compose the reference implementations of the preceding steps."""
    E = np.asarray(exponents, dtype=np.int64)
    if E.ndim != 2:
        raise ValueError("exponents must be two-dimensional")

    dimension = E.shape[1]

    structure = _oracle_identify_scaling_structure(E)
    weights = np.rint(structure[:dimension]).astype(np.int64)
    weighted_degree = int(round(float(structure[dimension])))

    gbar = _oracle_compute_certified_maximum(coeffs, E)

    box_moments = _oracle_compute_box_moments(coeffs, E, max_order)
    reference_moments = _oracle_compute_reference_moments(weights, weighted_degree, max_order)

    upper = _oracle_compute_upper_bound(box_moments, reference_moments, dimension)
    lower = _oracle_compute_lower_bound(box_moments, reference_moments, gbar, dimension)

    if lower > upper:
        raise ValueError("computed bracket is inconsistent")

    return float(upper)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential test cases: normal, boundary, and edge instances."""
    return [
        {
            # normal: the task instance at the pipeline budget k = 0..6
            "setup": "import numpy as np\ncoeffs = np.array([2.0, 2.0, 2.0, -2.0])\nexponents = np.array([[8, 0, 0], [0, 4, 0], [0, 0, 2], [4, 2, 0]])\nK = 6",
            "call": "compute_certified_volume_bound(coeffs, exponents, K)",
            "gold_call": "_oracle_compute_certified_volume_bound(coeffs, exponents, K)",
        },
        {
            # boundary: the same instance at the reduced budget k = 0..4,
            # which must return a strictly looser bound
            "setup": "import numpy as np\ncoeffs = np.array([2.0, 2.0, 2.0, -2.0])\nexponents = np.array([[8, 0, 0], [0, 4, 0], [0, 0, 2], [4, 2, 0]])\nK = 4",
            "call": "compute_certified_volume_bound(coeffs, exponents, K)",
            "gold_call": "_oracle_compute_certified_volume_bound(coeffs, exponents, K)",
        },
        {
            # edge: the isotropic Euclidean ball in three dimensions, where
            # the scaling data coincides with the ambient dimension and the
            # construction must reduce to the classical case
            "setup": "import numpy as np\ncoeffs = np.array([1.0, 1.0, 1.0])\nexponents = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])\nK = 6",
            "call": "compute_certified_volume_bound(coeffs, exponents, K)",
            "gold_call": "_oracle_compute_certified_volume_bound(coeffs, exponents, K)",
        },
        {
            # edge: the task instance at the extended budget k = 0..14, where
            # every downstream stage operates outside the range in which
            # fixed-precision dense linear algebra is adequate; the correct
            # value is 3.7657024354
            "setup": "import numpy as np\ncoeffs = np.array([2.0, 2.0, 2.0, -2.0])\nexponents = np.array([[8, 0, 0], [0, 4, 0], [0, 0, 2], [4, 2, 0]])\nK = 14",
            "call": "compute_certified_volume_bound(coeffs, exponents, K)",
            "gold_call": "_oracle_compute_certified_volume_bound(coeffs, exponents, K)",
        },
    ]
