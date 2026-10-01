"""
Average the first four derivatives of the anharmonic directional potential.

The fixed potential is



$$

V(x)=\frac12x^TD_0x+\sum_r\left[

\frac{g_r}{6}(s_r^3-3c_rs_r)

+\frac{h_r}{24}(s_r^4-6c_rs_r^2+3c_r^2)

+\frac{k_r}{720}(s_r^6-15c_rs_r^4+45c_r^2s_r^2-15c_r^3)

\right],\qquad s_r=a_r^Tx.

$$



Here $a_r^T$ is row $r$ of $A$, and $c_r$ is a fixed input coefficient,

not the variance of the Gaussian currently being optimized.

All coordinates are mass-rescaled and $\hbar=k_B=1$.



Each $s_r$ is Gaussian with mean $m_r$ and variance $v_r$; correlations

between different directions do not enter a single-direction average.

For the anharmonic summand $W_r(s_r)$, return



$$

J_{r,j}=\left\langle\frac{d^{j+1}W_r}{ds_r^{j+1}}\right\rangle,

\qquad j=0,1,2,3.

$$



The sextic contribution remains in these dressed lower-order vertices.

Zero variance means the deterministic limit $s_r=m_r$.

No harmonic contribution belongs in this returned directional array.

Returns
-------
A float64 matrix whose columns are the Gaussian averages of the first, second, third and fourth directional derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def gaussian_vertex_averages(
    means: np.ndarray,
    variances: np.ndarray,
    reference_variances: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
) -> np.ndarray:
    r"""Average the first four derivatives of the anharmonic directional potential.

    Parameters
    ----------
    means : np.ndarray, shape (R,)
        Finite Gaussian means, with $1\le R\le32$.
    variances : np.ndarray, shape (R,)
        Finite nonnegative current variances.
    reference_variances : np.ndarray, shape (R,)
        Finite nonnegative fixed $c_r$ coefficients.
    cubic : np.ndarray, shape (R,)
        Finite signed $g_r$, already multiplied by the current coupling.
    quartic : np.ndarray, shape (R,)
        Finite nonnegative $h_r$.
    sextic : np.ndarray, shape (R,)
        Finite nonnegative $k_r$.

    Returns
    -------
    vertices : np.ndarray, shape (R, 4)
        Averages of derivatives of orders one through four, in that order.

    Raises
    ------
    ValueError
        If arrays are not finite real vectors of the same allowed length, variances
        or even coefficients are negative, or a result is not representable.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_gaussian_vertex_averages(
    means: np.ndarray,
    variances: np.ndarray,
    reference_variances: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
) -> np.ndarray:
    means = _real(means, "means")
    if means.ndim != 1 or not 1 <= means.size <= 32:
        raise ValueError("means must contain one to 32 directions")
    size = means.size
    variances = _vector(variances, size, "variances")
    reference_variances = _vector(reference_variances, size, "reference_variances")
    cubic = _vector(cubic, size, "cubic")
    quartic = _vector(quartic, size, "quartic")
    sextic = _vector(sextic, size, "sextic")
    if (
        np.any(variances < 0)
        or np.any(reference_variances < 0)
        or np.any(quartic < 0)
        or np.any(sextic < 0)
    ):
        raise ValueError(
            "variances, quartic and sextic coefficients must be nonnegative"
        )
    shift = variances - reference_variances
    first = (
        cubic * (means**2 + shift) / 2
        + quartic * (means**3 + 3 * means * shift) / 6
        + sextic * (means**5 + 10 * means**3 * shift + 15 * means * shift**2) / 120
    )
    second = (
        cubic * means
        + quartic * (means**2 + shift) / 2
        + sextic * (means**4 + 6 * means**2 * shift + 3 * shift**2) / 24
    )
    third = cubic + quartic * means + sextic * (means**3 + 3 * means * shift) / 6
    fourth = quartic + sextic * (means**2 + shift) / 2
    result = np.column_stack((first, second, third, fourth))
    if not np.all(np.isfinite(result)):
        raise ValueError("vertex averages exceed the numerical range")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic numerical and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nmeans = directions@centroid\nvariances = reference_variances*0.85\n",
            "call": "gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "gold_call": "_oracle_gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nmeans = directions@centroid\nvariances = reference_variances*0.85\nmeans *= -1.5\nvariances *= 2.0\n",
            "call": "gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "gold_call": "_oracle_gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nmeans = directions@centroid\nvariances = reference_variances*0.85\nmeans[:] = 0.0\nvariances = reference_variances.copy()\n",
            "call": "gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "gold_call": "_oracle_gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nmeans = directions@centroid\nvariances = reference_variances*0.85\nvariances[:] = 0.0\n",
            "call": "gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "gold_call": "_oracle_gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nsextic[:] = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nmeans = directions@centroid\nvariances = reference_variances*0.85\n",
            "call": "gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "gold_call": "_oracle_gaussian_vertex_averages(means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nmeans = directions@centroid\nvariances = reference_variances*0.85\nvariances[0] = -1.0\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(gaussian_vertex_averages,means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "gold_call": "_raises(_oracle_gaussian_vertex_averages,means.copy(),variances.copy(),reference_variances.copy(),cubic.copy(),quartic.copy(),sextic.copy())",
            "tol": 1e-09,
        },
    ]
