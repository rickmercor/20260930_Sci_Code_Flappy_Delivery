"""
Construct positive quadrature rules for the tilted mark components.

For component $m$, integrate against $z^{k_m-1}e^{-z}$ on the positive half-line.

The physical mark nodes and mixture-weighted probability weights are



$$

x_{mq}=z_{mq}/b_m^\theta,\qquad

\omega_{mq}=p_m^\theta w_{mq}/\Gamma(k_m).

$$



Rows retain mixture order, and nodes increase within a component. The sum of all

weights is one up to roundoff. Quadrature does not incorporate the exponential

modal factor; that factor depends on frequency and is applied later.

Returns
-------
Shape (M, order, 2), final axis mark node in loss units and probability weight.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_mark_quadrature(
    mixture: np.ndarray,
    order: int,
) -> np.ndarray:
    r"""Construct positive quadrature rules for the tilted mark components.

    Parameters
    ----------
    mixture : np.ndarray
        Shape (M, 3), columns normalized positive probability, positive shape, positive
        tilted rate.
    order : int
        Positive quadrature order per component, at most 128.

    Returns
    -------
    rule : np.ndarray
        Shape (M, order, 2), final axis mark node in loss units and probability weight.

    Raises
    ------
    ValueError
        If mixture entries are invalid, probabilities are unnormalized, or order is
        outside the integer range 1 through 128.

    Notes
    -----
    Inputs are not modified. Numerical comparisons use rtol=1e-9 and atol=1e-11.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import gammaln, roots_genlaguerre


def _oracle_build_mark_quadrature(
    mixture: np.ndarray,
    order: int,
) -> np.ndarray:
    mixture = _real_array(mixture, "mixture")
    order = _positive_integer(order, "order")
    if mixture.ndim != 2 or mixture.shape[1] != 3 or order > 128:
        raise ValueError("invalid mixture shape or quadrature order")
    p, k, b = _checked_mixture(mixture[:, 0], mixture[:, 1], mixture[:, 2])
    result = np.empty((len(p), order, 2))
    for m, (probability, shape, rate) in enumerate(zip(p, k, b)):
        nodes, weights = roots_genlaguerre(order, shape - 1)
        result[m, :, 0] = nodes / rate
        result[m, :, 1] = probability * np.exp(np.log(weights) - gammaln(shape))
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError(
            "quadrature cannot be represented with positive finite entries"
        )
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """import numpy as np
from scipy.special import gammaln, roots_genlaguerre
mix = np.array([[0.6, 2.0, 3.8], [0.4, 6.0, 2.3]])
n = 8
""",
            "call": "build_mark_quadrature(mix.copy(), n)",
            "gold_call": "_oracle_build_mark_quadrature(mix.copy(), n)",
        },
        {
            "setup": """import numpy as np
from scipy.special import gammaln, roots_genlaguerre
mix = np.array([[1.0, 1.0, 2.0]])
n = 1
""",
            "call": "build_mark_quadrature(mix.copy(), n)",
            "gold_call": "_oracle_build_mark_quadrature(mix.copy(), n)",
        },
        {
            "setup": """import numpy as np
from scipy.special import gammaln, roots_genlaguerre
mix = np.array([[0.2, 0.5, 0.7], [0.8, 3.0, 4.0]])
n = 12
""",
            "call": "build_mark_quadrature(mix.copy(), n)",
            "gold_call": "_oracle_build_mark_quadrature(mix.copy(), n)",
        },
        {
            "setup": """import numpy as np
from scipy.special import gammaln, roots_genlaguerre
mix = np.array([[1.0, 1.0, 2.0]])
n = 0

def _model_exception():
    try:
        build_mark_quadrature(mix.copy(), n)
        return 0
    except ValueError:
        return 1

def _reference_exception():
    try:
        _oracle_build_mark_quadrature(mix.copy(), n)
        return 0
    except ValueError:
        return 1
""",
            "call": "_model_exception()",
            "gold_call": "_reference_exception()",
        },
    ]
