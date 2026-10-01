"""
The fitted coefficient of an iteration is the constrained minimiser of the sketched loss on a closed interval. This step performs that whole minimisation and returns the coefficient itself; no intermediate set is reported.



A polynomial is smooth, so its minimum over a closed interval is attained either at one of the two endpoints or at a stationary point strictly inside, and only that finite set has to be examined. At order `$p$` the sketched loss has degree `$2 * p$`, so its derivative has degree `$2 * p - 1$`. From `$p = 3$` upward that derivative is a quintic or higher, which by Abel's theorem has no solution by radicals, so the stationary points cannot be written in closed form and must be obtained numerically as the eigenvalues of the derivative's companion matrix. That numerical route is why the conventions below are part of the specification rather than an implementation detail: a real stationary point arrives with a small spurious imaginary part, and a stationary point sitting on an endpoint arrives displaced from it.



The admissible set is built by these conventions, applied in this order.



1. Differentiate the ascending coefficient vector term by term. 2. Write the derivative in descending order and drop its leading zeros. If fewer than two coefficients remain, the derivative has no companion matrix and the polynomial contributes no stationary point at all. 3. Take the companion-matrix eigenvalues of what remains. An eigenvalue counts as real when its imaginary magnitude is at most `$1e-10$`, and is discarded otherwise; the surviving point is its real part. 4. A surviving point within `$1e-12$` of an endpoint is clipped onto that endpoint, and a point outside the closed interval after clipping is dropped. 5. Both endpoints are admissible unconditionally, whatever the derivative does. 6. Sort every admissible point into ascending order, then discard any point within `$1e-12$` of one already kept, so that the smaller representative of a cluster survives.



The coefficient is then chosen from that ascending set. Each candidate is scored by evaluating the loss at it with Horner's rule on the ascending coefficient vector. Horner is not decoration here: by the last iteration the sketched loss has fallen many orders of magnitude, its coefficients no longer share a scale, and evaluating the powers separately loses the cancellation that produces the small value.



The comparison is a single ascending scan rather than a global argument-minimum. The first candidate is provisionally the best. Each later candidate replaces the current best only when its score is smaller than the current best score by more than `$1e-14$`; otherwise the earlier, and therefore smaller, candidate stays in place. Scores within `$1e-14$` of each other are in that sense tied, and the fit never trades a numerically indistinguishable improvement for a larger coefficient. Because every comparison is made against the running best and not against the global minimum, the scan can settle on a candidate whose score is not the smallest in the set: once the running best has moved, a later candidate that improves on it by less than the tolerance is refused, even though a global argument-minimum would have selected it.



Uniformly scaling every loss coefficient does not move a stationary point, but it does interact with the *absolute* `$1e-14$` score tolerance. The implementation must therefore remain defined when finite coefficients are so large that forming ``j * c[j]`` directly would overflow, and when they are so small that a solver normalizes them to find the roots. Root finding may use a uniformly normalized derivative, but a normalized Horner scan must scale the comparison tolerance by the same factor; otherwise it silently replaces the specified absolute tie rule with a relative one. No restriction to a convenient coefficient magnitude is part of the public contract.

Returns
-------
float, the fitted coefficient chosen from the admissible set
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fit_bounded_coefficient(c: np.ndarray, lower: float, upper: float) -> float:
    """Minimise the sketched loss over the closed coefficient interval.

    Raises ``ValueError`` unless every one of the following holds: ``c`` is real
    rather than complex; ``c`` is one-dimensional with at least two entries;
    every entry of ``c`` is finite, including uniformly tiny or near-overflow
    vectors for which derivative formation must be scaled; and ``lower`` and
    ``upper`` are finite with ``lower < upper``, so an inverted or degenerate
    interval is rejected rather than reordered.

    Parameters
    ----------
    c : np.ndarray
        Finite real loss coefficients in ascending powers of ``alpha``, with at
        least two entries.
    lower : float
        Lower endpoint of the closed coefficient interval.
    upper : float
        Upper endpoint of the closed coefficient interval, strictly above
        ``lower``.

    Returns
    -------
    float
        The fitted coefficient: the admissible point selected by the ascending
        running-best scan.
    """
    return result  # noqa: F821 - required model stub

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_fit_bounded_coefficient(
    c: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    """Reference bounded minimisation under the published conventions."""
    coefficients = np.asarray(c)
    if np.iscomplexobj(coefficients):
        raise ValueError("c must be real")
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or coefficients.size < 2:
        raise ValueError("c must be one-dimensional with at least two entries")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("c must contain only finite values")
    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("bounds must be finite and satisfy lower < upper")
    coefficient_scale = float(np.max(np.abs(coefficients)))
    extreme_scale = coefficient_scale != 0.0 and (
        coefficient_scale < 1e-150 or coefficient_scale > 1e150
    )
    working = coefficients / coefficient_scale if extreme_scale else coefficients
    ascending = np.array(
        [(j + 1) * working[j + 1] for j in range(working.size - 1)],
        dtype=float,
    )
    descending = ascending[::-1]
    nonzero = np.flatnonzero(descending != 0.0)
    found = [lower, upper]
    if nonzero.size and descending[nonzero[0] :].size > 1:
        for root in np.roots(descending[nonzero[0] :]):
            if abs(root.imag) > 1e-10:
                continue
            point = float(root.real)
            if abs(point - lower) <= 1e-12:
                point = lower
            if abs(point - upper) <= 1e-12:
                point = upper
            if lower <= point <= upper:
                found.append(point)
    found.sort()
    kept: list[float] = []
    for point in found:
        if not kept or abs(point - kept[-1]) > 1e-12:
            kept.append(point)
    best = float(kept[0])
    best_value = 0.0
    for coefficient in working[::-1]:
        best_value = best_value * best + float(coefficient)
    score_tolerance = float(1e-14 / coefficient_scale) if extreme_scale else 1e-14
    for point in kept[1:]:
        value = 0.0
        for coefficient in working[::-1]:
            value = value * float(point) + float(coefficient)
        if value < best_value - score_tolerance:
            best = float(point)
            best_value = value
    return float(best)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return stationary, tie, uniformly extreme-scale and invalid cases."""
    return [
        {
            "setup": """import numpy as np
c = np.array([0.28359856992081955, -1.3328783554331864, 1.8826489702882505, -0.7757344846501709, 0.7338625066266097, -0.08200828917549519, 0.01244128703541562])
lower = 1.0 / 3.0
upper = 1.0
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
c = np.array([0.00012841869850001316, -0.0007686217050783954, 0.0011484645842911106, 5.0212995224972e-06, 2.522195649364186e-07, 1.169741299808406e-09, 3.769607340993229e-12])
lower = 1.0 / 3.0
upper = 1.0
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
c = np.array([7.3840898330407616e-09, -4.429962899933568e-08, 6.643717020477382e-08, 1.472568269349956e-11, 1.7025306880368127e-15, 1.0234736395930816e-19, 2.6251051599194346e-24])
lower = 1.0 / 3.0
upper = 1.0
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
base = np.array([0.28359856992081955, -1.3328783554331864, 1.8826489702882505, -0.7757344846501709, 0.7338625066266097, -0.08200828917549519, 0.01244128703541562])
c = base * (1.2e308 / np.max(np.abs(base)))
lower = 1.0 / 3.0
upper = 1.0
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
base = np.array([0.28359856992081955, -1.3328783554331864, 1.8826489702882505, -0.7757344846501709, 0.7338625066266097, -0.08200828917549519, 0.01244128703541562])
c = base * 1e-310
lower = 1.0 / 3.0
upper = 1.0
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
c = np.array([0.0, -1.0499999999999998e-13, 1.8e-13, -1e-13])
lower = 0.0
upper = 1.0
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
c = np.array([0.0, 0.99999999999994, -3.99999999999994, 4.0])
lower = 0.0
upper = 1.0
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
c = np.array([0.0, 0.99999999999998, -3.99999999999998, 4.0])
lower = 0.0
upper = 1.0
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
c = np.array([9.0, -6.0, 1.0])
lower = 0.0
upper = 1.0
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
c = np.array([2.5, 0.0, 0.0])
lower = 0.25
upper = 0.75
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
c = np.array([1.0, -2.0])
lower = 0.2
upper = 0.9
""",
            "call": "fit_bounded_coefficient(c, lower, upper)",
            "gold_call": "_oracle_fit_bounded_coefficient(c, lower, upper)",
        },
        {
            "setup": """import numpy as np
c = np.array([1.0, -2.0, 1.0])
lower = 0.8
upper = 0.8
def run_model_bounds():
    try:
        fit_bounded_coefficient(c, lower, upper)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_bounds():
    try:
        _oracle_fit_bounded_coefficient(c, lower, upper)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_bounds()",
            "gold_call": "run_oracle_bounds()",
        },
        {
            "setup": """import numpy as np
c = np.array([1.0 + 0.0j, -2.0 + 1e-18j, 1.0 + 0.0j])
lower = 0.0
upper = 1.0
def run_model_complex():
    try:
        fit_bounded_coefficient(c, lower, upper)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_complex():
    try:
        _oracle_fit_bounded_coefficient(c, lower, upper)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_complex()",
            "gold_call": "run_oracle_complex()",
        },
        {
            "setup": """import numpy as np
c = np.array([5.0])
lower = 0.0
upper = 1.0
def run_model_short():
    try:
        fit_bounded_coefficient(c, lower, upper)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_short():
    try:
        _oracle_fit_bounded_coefficient(c, lower, upper)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_short()",
            "gold_call": "run_oracle_short()",
        },
    ]
