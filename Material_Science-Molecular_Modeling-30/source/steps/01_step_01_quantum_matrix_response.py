"""
Compute the covariance and static pair response in a fixed coordinate frame.

For a positive auxiliary matrix $\Phi$, the thermal quantum covariance is the

matrix function



$$

\Psi(\Phi,\theta)=\frac12\Phi^{-1/2}

\coth\left(\frac{\Phi^{1/2}}{2\theta}\right),

\qquad \Psi(\Phi,0)=\frac12\Phi^{-1/2}.

$$



The symmetric-pair response $\Lambda$ is defined by its action on an arbitrary

symmetric perturbation $E$:



$$

\operatorname{pack}(\delta\Psi)=2\Lambda\operatorname{pack}(E),

\qquad \delta\Psi=\left.\frac{d}{dt}\Psi(\Phi+tE,\theta)\right|_{t=0}.

$$



This derivative must retain changes of polarization vectors; repeated and nearly

repeated auxiliary frequencies have their continuous limits.

Return physical-frame matrices, independent of choices of eigenvector signs.

Symmetric coordinates use $B^{ii}=e_ie_i^T$ and

$B^{ij}=(e_ie_j^T+e_je_i^T)/\sqrt2$ for $i<j$, in lexicographic pair order.

For $p=d(d+1)/2$, packing means $s_a=\operatorname{tr}((B^a)^TS)$;

the coordinate frame is fixed in physical space, including inside degenerate eigenspaces.

Returns
-------
A tuple containing the float64 covariance matrix and the float64 static pair-response matrix.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def quantum_matrix_response(auxiliary: np.ndarray, temperature: float) -> tuple:
    r"""Compute the covariance and static pair response in a fixed coordinate frame.

    Parameters
    ----------
    auxiliary : np.ndarray, shape (d, d)
        Finite real symmetric positive-definite $\Phi$, with $1\le d\le5$.
    temperature : float
        Finite thermal energy $\theta\ge0$.

    Returns
    -------
    covariance : np.ndarray, shape (d, d)
        Quantum covariance $\Psi$.
    pair_response : np.ndarray, shape (p, p)
        Symmetric negative-definite $\Lambda$ in the specified pair basis.

    Raises
    ------
    ValueError
        If the matrix is not finite real symmetric positive definite, its dimension
        is outside 1 to 5, temperature is invalid, or the response is not
        representable.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _real(value, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numerical") from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _symmetric(value, name, positive=False):
    result = _real(value, name)
    if (
        result.ndim != 2
        or result.shape[0] != result.shape[1]
        or not 1 <= len(result) <= 15
    ):
        raise ValueError(f"{name} must be a nonempty square matrix of order at most 15")
    if np.linalg.norm(result - result.T) > 1e-12 * max(1.0, np.linalg.norm(result)):
        raise ValueError(f"{name} must be symmetric")
    result = (result + result.T) * 0.5
    if positive and np.linalg.eigvalsh(result)[0] <= 0:
        raise ValueError(f"{name} must be positive definite")
    return result


def _vector(value, size, name):
    result = _real(value, name)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return result


def _temperature(value):
    result = _real(value, "temperature")
    if result.ndim != 0 or result < 0:
        raise ValueError("temperature must be a nonnegative scalar")
    return float(result)


def _basis(dimension):
    result = []
    for row in range(dimension):
        for col in range(row, dimension):
            entry = np.zeros((dimension, dimension))
            entry[row, col] = entry[col, row] = (
                1.0 if row == col else 1.0 / np.sqrt(2.0)
            )
            result.append(entry)
    return np.asarray(result)


def _pack(matrix, basis):
    return np.einsum("aij,ij->a", basis, matrix)


def _unpack(vector, basis):
    return np.einsum("a,aij->ij", vector, basis)


def _oracle_quantum_matrix_response(auxiliary: np.ndarray, temperature: float) -> tuple:
    auxiliary = _symmetric(auxiliary, "auxiliary", positive=True)
    dimension = len(auxiliary)
    if dimension > 5:
        raise ValueError("the optical dimension must not exceed five")
    temperature = _temperature(temperature)
    values, modes = np.linalg.eigh(auxiliary)
    frequencies = np.sqrt(values)
    if temperature == 0.0:
        occupations = np.zeros(dimension)
    else:
        decay = np.exp(-frequencies / temperature)
        occupations = decay / (-np.expm1(-frequencies / temperature))
    variances = (occupations + 0.5) / frequencies
    covariance = (modes * variances) @ modes.T
    spectral = np.empty((dimension, dimension))
    for row, left in enumerate(frequencies):
        for col, right in enumerate(frequencies):
            low, high = min(left, right), max(left, right)
            gap = high - low
            quotient = 0.0
            if temperature > 0.0:
                if gap == 0.0:
                    quotient = (
                        -occupations[row] * (1.0 + occupations[row]) / temperature
                    )
                else:
                    quotient = (
                        np.exp(-low / temperature)
                        * np.expm1(-gap / temperature)
                        / (
                            gap
                            * (-np.expm1(-low / temperature))
                            * (-np.expm1(-high / temperature))
                        )
                    )
            spectral[row, col] = -(
                (1.0 + occupations[row] + occupations[col]) / (left + right) - quotient
            ) / (4.0 * left * right)
    basis = _basis(dimension)
    rotated = np.einsum("ia,kij,jb->kab", modes, basis, modes)
    pair_response = np.einsum("aij,ij,bij->ab", rotated, spectral, rotated)
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(pair_response)):
        raise ValueError("quantum response exceeds the numerical range")
    return covariance, (pair_response + pair_response.T) * 0.5

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic numerical and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ndef _numeric_result(function, *args):\n    covariance, response = function(*args)\n    dimension = len(args[0]); pairs = dimension*(dimension+1)//2\n    if np.shape(covariance) != (dimension,dimension) or np.shape(response) != (pairs,pairs):\n        raise ValueError('incorrect returned matrix shapes')\n    return np.concatenate((np.asarray(covariance).ravel(),np.asarray(response).ravel()))\n",
            "call": "_numeric_result(quantum_matrix_response,bare.copy(),temperature)",
            "gold_call": "_numeric_result(_oracle_quantum_matrix_response,bare.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nrotation = np.array([[.8,-.6,0.,0.],[.6,.8,0.,0.],[0.,0.,.6,-.8],[0.,0.,.8,.6]])\nbare = rotation@bare@rotation.T\ncentroid = rotation@centroid\ndirections = directions@rotation.T\ndef _numeric_result(function, *args):\n    covariance, response = function(*args)\n    dimension = len(args[0]); pairs = dimension*(dimension+1)//2\n    if np.shape(covariance) != (dimension,dimension) or np.shape(response) != (pairs,pairs):\n        raise ValueError('incorrect returned matrix shapes')\n    return np.concatenate((np.asarray(covariance).ravel(),np.asarray(response).ravel()))\n",
            "call": "_numeric_result(quantum_matrix_response,bare.copy(),temperature)",
            "gold_call": "_numeric_result(_oracle_quantum_matrix_response,bare.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare[1,1] += 2e-12\ndef _numeric_result(function, *args):\n    covariance, response = function(*args)\n    dimension = len(args[0]); pairs = dimension*(dimension+1)//2\n    if np.shape(covariance) != (dimension,dimension) or np.shape(response) != (pairs,pairs):\n        raise ValueError('incorrect returned matrix shapes')\n    return np.concatenate((np.asarray(covariance).ravel(),np.asarray(response).ravel()))\n",
            "call": "_numeric_result(quantum_matrix_response,bare.copy(),temperature)",
            "gold_call": "_numeric_result(_oracle_quantum_matrix_response,bare.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ntemperature = 0.0\nrotation = np.array([[.8,-.6,0.,0.],[.6,.8,0.,0.],[0.,0.,.6,-.8],[0.,0.,.8,.6]])\nbare = rotation@bare@rotation.T\ncentroid = rotation@centroid\ndirections = directions@rotation.T\ndef _numeric_result(function, *args):\n    covariance, response = function(*args)\n    dimension = len(args[0]); pairs = dimension*(dimension+1)//2\n    if np.shape(covariance) != (dimension,dimension) or np.shape(response) != (pairs,pairs):\n        raise ValueError('incorrect returned matrix shapes')\n    return np.concatenate((np.asarray(covariance).ravel(),np.asarray(response).ravel()))\n",
            "call": "_numeric_result(quantum_matrix_response,bare.copy(),temperature)",
            "gold_call": "_numeric_result(_oracle_quantum_matrix_response,bare.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare = np.diag([0.04,0.49,2.56])\ntemperature = 1.8\ndef _numeric_result(function, *args):\n    covariance, response = function(*args)\n    dimension = len(args[0]); pairs = dimension*(dimension+1)//2\n    if np.shape(covariance) != (dimension,dimension) or np.shape(response) != (pairs,pairs):\n        raise ValueError('incorrect returned matrix shapes')\n    return np.concatenate((np.asarray(covariance).ravel(),np.asarray(response).ravel()))\n",
            "call": "_numeric_result(quantum_matrix_response,bare.copy(),temperature)",
            "gold_call": "_numeric_result(_oracle_quantum_matrix_response,bare.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare[0,0] = -1.0\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(quantum_matrix_response,bare.copy(),temperature)",
            "gold_call": "_raises(_oracle_quantum_matrix_response,bare.copy(),temperature)",
            "tol": 1e-09,
        },
    ]
