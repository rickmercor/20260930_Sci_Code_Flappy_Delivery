"""
Construct a correlated Gaussian residual-force ensemble.

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



The supplied $\Phi$ need not satisfy stationarity.

Let $\Psi$ be its quantum covariance and let $L$ be the lower Cholesky factor

with positive diagonal; each configuration is $u=Lz$ and $x=b+u$.

Use a normalized five-node standard-normal product rule, with increasing

one-dimensional nodes and lexicographic tensor rows, last index fastest:



$$

z=(-\sqrt{5+\sqrt{10}},-\sqrt{5-\sqrt{10}},0,

\sqrt{5-\sqrt{10}},\sqrt{5+\sqrt{10}}),

$$



$$

w=((7-2\sqrt{10})/60,(7+2\sqrt{10})/60,8/15,

(7+2\sqrt{10})/60,(7-2\sqrt{10})/60).

$$



Return the precision $P=\Psi^{-1}$, rows $v=Pu$, residual forces

$f=-\nabla V(b+u)+\Phi u$, and product weights.

Preserve the residual force's mean and linear component; the subsequent response

calculation must handle them. There is no random seed.

Returns
-------
A tuple containing the float64 precision matrix, precision-displacement rows, residual-force rows and normalized weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_correlated_ensemble(
    auxiliary: np.ndarray,
    temperature: float,
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_variances: np.ndarray,
) -> tuple:
    r"""Construct a correlated Gaussian residual-force ensemble.

    Parameters
    ----------
    auxiliary : np.ndarray, shape (d, d)
        Finite real symmetric positive-definite auxiliary matrix, of the same dimension as bare.
    temperature : float
        Finite thermal energy $\theta\ge0$.
    bare : np.ndarray, shape (d, d)
        Finite real symmetric positive-definite fixed $D_0$, with $1\le d\le5$.
    centroid : np.ndarray, shape (d,)
        Finite fixed centroid $b$.
    directions : np.ndarray, shape (R, d)
        Finite direction rows, with $1\le R\le32$.
    cubic : np.ndarray, shape (R,)
        Finite signed cubic coefficients, already multiplied by the current coupling.
    quartic : np.ndarray, shape (R,)
        Finite nonnegative quartic coefficients.
    sextic : np.ndarray, shape (R,)
        Finite nonnegative sextic coefficients.
    reference_variances : np.ndarray, shape (R,)
        Finite nonnegative fixed potential coefficients $c_r$.

    Returns
    -------
    precision : np.ndarray, shape (d, d)
        Inverse quantum covariance.
    precision_displacements : np.ndarray, shape (5**d, d)
        Precision-weighted displacements in the declared row order.
    residual_forces : np.ndarray, shape (5**d, d)
        Uncentered residual force rows.
    weights : np.ndarray, shape (5**d,)
        Normalized nonnegative product weights.

    Raises
    ------
    ValueError
        If the auxiliary or model data violate the documented shapes, signs,
        finiteness or positive-definiteness conditions, or a numerical result is not
        representable.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_correlated_ensemble(
    auxiliary: np.ndarray,
    temperature: float,
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_variances: np.ndarray,
) -> tuple:
    data = _model_data(
        bare, centroid, directions, cubic, quartic, sextic, reference_variances
    )
    bare, centroid, directions, cubic, quartic, sextic, reference_variances = data
    auxiliary = _symmetric(auxiliary, "auxiliary", positive=True)
    if auxiliary.shape != bare.shape:
        raise ValueError("auxiliary and bare dimensions must agree")
    covariance, _ = _oracle_quantum_matrix_response(auxiliary, temperature)
    dimension = len(bare)
    root10 = np.sqrt(10.0)
    nodes = np.array(
        [
            -np.sqrt(5 + root10),
            -np.sqrt(5 - root10),
            0.0,
            np.sqrt(5 - root10),
            np.sqrt(5 + root10),
        ]
    )
    weights1d = np.array(
        [
            (7 - 2 * root10) / 60,
            (7 + 2 * root10) / 60,
            8 / 15,
            (7 + 2 * root10) / 60,
            (7 - 2 * root10) / 60,
        ]
    )
    indices = np.indices((5,) * dimension).reshape(dimension, -1).T
    displacements = nodes[indices] @ np.linalg.cholesky(covariance).T
    weights = np.prod(weights1d[indices], axis=1)
    positions = displacements + centroid
    projections = positions @ directions.T
    fixed = reference_variances
    radial = (
        cubic / 2 * (projections**2 - fixed)
        + quartic / 6 * (projections**3 - 3 * fixed * projections)
        + sextic
        / 120
        * (projections**5 - 10 * fixed * projections**3 + 15 * fixed**2 * projections)
    )
    residual_forces = (
        -positions @ bare.T - radial @ directions + displacements @ auxiliary.T
    )
    precision = np.linalg.solve(covariance, np.eye(dimension))
    precision_displacements = displacements @ precision.T
    if not np.all(np.isfinite(residual_forces)):
        raise ValueError("force ensemble exceeds the numerical range")
    return precision, precision_displacements, residual_forces, weights

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic numerical and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ndef _numeric_result(function, *args):\n    precision, vectors, forces, weights = function(*args)\n    dimension = len(args[0]); rows = 5**dimension\n    if (np.shape(precision)!=(dimension,dimension) or np.shape(vectors)!=(rows,dimension)\n        or np.shape(forces)!=(rows,dimension) or np.shape(weights)!=(rows,)):\n        raise ValueError('incorrect ensemble shapes')\n    return np.concatenate([np.asarray(x).ravel() for x in (precision,vectors,forces,weights)])\n",
            "call": "_numeric_result(build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "gold_call": "_numeric_result(_oracle_build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nrotation = np.array([[.8,-.6,0.,0.],[.6,.8,0.,0.],[0.,0.,.6,-.8],[0.,0.,.8,.6]])\nbare = rotation@bare@rotation.T\ncentroid = rotation@centroid\ndirections = directions@rotation.T\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ndef _numeric_result(function, *args):\n    precision, vectors, forces, weights = function(*args)\n    dimension = len(args[0]); rows = 5**dimension\n    if (np.shape(precision)!=(dimension,dimension) or np.shape(vectors)!=(rows,dimension)\n        or np.shape(forces)!=(rows,dimension) or np.shape(weights)!=(rows,)):\n        raise ValueError('incorrect ensemble shapes')\n    return np.concatenate([np.asarray(x).ravel() for x in (precision,vectors,forces,weights)])\n",
            "call": "_numeric_result(build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "gold_call": "_numeric_result(_oracle_build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare = bare[:3,:3].copy()\ncentroid = centroid[:3].copy()\ndirections = directions[:,:3].copy()\ntemperature = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ndef _numeric_result(function, *args):\n    precision, vectors, forces, weights = function(*args)\n    dimension = len(args[0]); rows = 5**dimension\n    if (np.shape(precision)!=(dimension,dimension) or np.shape(vectors)!=(rows,dimension)\n        or np.shape(forces)!=(rows,dimension) or np.shape(weights)!=(rows,)):\n        raise ValueError('incorrect ensemble shapes')\n    return np.concatenate([np.asarray(x).ravel() for x in (precision,vectors,forces,weights)])\n",
            "call": "_numeric_result(build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "gold_call": "_numeric_result(_oracle_build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = bare.copy()\ndef _numeric_result(function, *args):\n    precision, vectors, forces, weights = function(*args)\n    dimension = len(args[0]); rows = 5**dimension\n    if (np.shape(precision)!=(dimension,dimension) or np.shape(vectors)!=(rows,dimension)\n        or np.shape(forces)!=(rows,dimension) or np.shape(weights)!=(rows,)):\n        raise ValueError('incorrect ensemble shapes')\n    return np.concatenate([np.asarray(x).ravel() for x in (precision,vectors,forces,weights)])\n",
            "call": "_numeric_result(build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "gold_call": "_numeric_result(_oracle_build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nsextic[:] = 0.0\ncentroid[:] = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ndef _numeric_result(function, *args):\n    precision, vectors, forces, weights = function(*args)\n    dimension = len(args[0]); rows = 5**dimension\n    if (np.shape(precision)!=(dimension,dimension) or np.shape(vectors)!=(rows,dimension)\n        or np.shape(forces)!=(rows,dimension) or np.shape(weights)!=(rows,)):\n        raise ValueError('incorrect ensemble shapes')\n    return np.concatenate([np.asarray(x).ravel() for x in (precision,vectors,forces,weights)])\n",
            "call": "_numeric_result(build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "gold_call": "_numeric_result(_oracle_build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nstationary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\nauxiliary = stationary.copy()\nauxiliary[0,0] = -1.0\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "gold_call": "_raises(_oracle_build_correlated_ensemble,auxiliary.copy(),temperature,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy())",
            "tol": 1e-09,
        },
    ]
