"""
Compute the exponentially fitted trace weights and three signed source-kernel moments.

Put $\ell>0$, $z=2\ell\beta_n/\alpha$ and $B(z)=z/(e^z-1)$, continuously extended by $B(0)=1$.

For $s\in[-\ell,\ell]$, define

$$

Q(s)=

*\begin*{cases}

-*\dfrac*{1-e^{-(s+\ell)\beta_n/\alpha}}{1-e^{-2\ell\beta_n/\alpha}},\&s<0,\\

*\dfrac*{e^{(\ell-s)\beta_n/\alpha}-1}{e^{2\ell\beta_n/\alpha}-1},\&s\geq0.

*\end*{cases}

$$

At zero drift the branches become $-(s+\ell)/(2\ell)$ and $(\ell-s)/(2\ell)$.

The required moments are $I_k=*\int_*{-\ell}^{\ell}Q(s)s^k\,ds$, $k=0,1,2$.

Return $[B(z),B(-z),I_0,I_1,I_2]$; split quadrature at the jump at zero and avoid growing exponentials.

The supported numerical domain is $|z|\leq1000$.

Returns
-------
A five-entry float array containing the two trace weights followed by the zeroth, first, and second signed kernel moments.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_normal_kernel(
    half_width: float, diffusion: float, normal_drift: float
) -> np.ndarray:
    r"""Compute the exponentially fitted trace weights and three signed source-kernel moments.

    Parameters
    ----------
    half_width : float
        Positive dimensionless normal half-width $\ell$.
    diffusion : float
        Positive dimensionless diffusion coefficient $\alpha$.
    normal_drift : float
        Finite signed normal drift $\beta_n$.

    Returns
    -------
    np.ndarray
        Shape $(5,)$, containing two trace weights and three moments.

    Raises
    ------
    ValueError
        If scalar data are non-finite, widths or diffusion are non-positive, or $|z|>1000$,
        or if the kernel moments are not representable as finite floats.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad


def _decaying_ratio(fraction, exponent):
    if abs(exponent) < 1e-14:
        return fraction
    if exponent > 0:
        return (
            np.exp((fraction - 1) * exponent)
            * (-np.expm1(-fraction * exponent))
            / (-np.expm1(-exponent))
        )
    return np.expm1(fraction * exponent) / np.expm1(exponent)


def _trace_weight(exponent):
    if abs(exponent) < 1e-5:
        return 1 - exponent / 2 + exponent**2 / 12 - exponent**4 / 720
    if exponent > 0:
        return exponent * np.exp(-exponent) / (-np.expm1(-exponent))
    return exponent / np.expm1(exponent)


def _oracle_compute_normal_kernel(
    half_width: float, diffusion: float, normal_drift: float
) -> np.ndarray:
    """Evaluate the reference numerical operation."""
    ell = _positive_scalar(half_width, "half_width")
    alpha = _positive_scalar(diffusion, "diffusion")
    beta = _finite_scalar(normal_drift, "normal_drift")
    z = 2 * ell * beta / alpha
    if not np.isfinite(z) or abs(z) > 1000:
        raise ValueError("normal Peclet number must satisfy abs(z) <= 1000")
    moments = []
    for degree in range(3):
        left = quad(
            lambda t: -_decaying_ratio((t + 1) / 2, -z) * t**degree,
            -1,
            0,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        right = quad(
            lambda t: _decaying_ratio((1 - t) / 2, z) * t**degree,
            0,
            1,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        moment = left + right
        for _ in range(degree + 1):
            moment *= ell
        moments.append(moment)
    result = np.array([_trace_weight(z), _trace_weight(-z), *moments])
    if not np.all(np.isfinite(result)):
        raise ValueError("kernel moments exceed the finite numerical range")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nell, alpha, beta = .0774, .03, 1.7\n",
            "call": "compute_normal_kernel(ell, alpha, beta)",
            "gold_call": "_oracle_compute_normal_kernel(ell, alpha, beta)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nell, alpha, beta = .2, .1, 0.\n",
            "call": "compute_normal_kernel(ell, alpha, beta)",
            "gold_call": "_oracle_compute_normal_kernel(ell, alpha, beta)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nell, alpha, beta = .5, .001, -.9\n",
            "call": "compute_normal_kernel(ell, alpha, beta)",
            "gold_call": "_oracle_compute_normal_kernel(ell, alpha, beta)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nell, alpha, beta = .5, .001, .9\n",
            "call": "compute_normal_kernel(ell, alpha, beta)",
            "gold_call": "_oracle_compute_normal_kernel(ell, alpha, beta)",
        },
        {
            "setup": "import numpy as np\nfrom scipy.integrate import quad\nell, alpha, beta = -.1, .03, 1.\ndef _exception_code(function):\n    try:\n        function(ell, alpha, beta)\n    except ValueError:\n        return 1\n    return 0\n\n",
            "call": "_exception_code(compute_normal_kernel)",
            "gold_call": "_exception_code(_oracle_compute_normal_kernel)",
        },
    ]
