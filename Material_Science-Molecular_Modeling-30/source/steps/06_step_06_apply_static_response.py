"""
Apply the coupled static response in the physical coordinate frame.

Symmetric coordinates use $B^{ii}=e_ie_i^T$ and

$B^{ij}=(e_ie_j^T+e_je_i^T)/\sqrt2$ for $i<j$, in lexicographic pair order.

For $p=d(d+1)/2$, packing means $s_a=\operatorname{tr}((B^a)^TS)$;

the coordinate frame is fixed in physical space, including inside degenerate eigenspaces.



The full static response has centroid and symmetric-pair sectors.

Its harmonic part acts with $\Phi$ in the centroid sector and $-\Lambda^{-1}$

in the pair sector; its anharmonic part is the fully permutation-symmetric

Gaussian score-moment action.

Both the auxiliary matrix and pair response may be nondiagonal.

The full operator can be singular or indefinite even when $\Phi$ is positive

definite, and these cases remain valid operator actions.

Returns
-------
A float64 vector containing the full static response of the input centroid-pair state.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def apply_static_response(
    auxiliary: np.ndarray,
    pair_response: np.ndarray,
    precision: np.ndarray,
    precision_displacements: np.ndarray,
    residual_forces: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    r"""Apply the coupled static response in the physical coordinate frame.

    Parameters
    ----------
    auxiliary : np.ndarray, shape (d, d)
        Finite real symmetric positive-definite $\Phi$, with $1\le d\le5$.
    pair_response : np.ndarray, shape (p, p)
        Finite real symmetric negative-definite $\Lambda$ in the physical-frame pair basis.
    precision : np.ndarray, shape (d, d)
        Finite real symmetric positive-definite $P$, with $1\le d\le5$.
    precision_displacements : np.ndarray, shape (K, d)
        Finite real $v=Pu$ rows, with $K\ge1$.
    residual_forces : np.ndarray, shape (K, d)
        Finite real residual force rows, with mean and linear components allowed.
    weights : np.ndarray, shape (K,)
        Finite nonnegative weights, normalized within absolute error $10^{-12}$.
    state : np.ndarray, shape (d + p,)
        Finite centroid entries followed by packed pair entries.

    Returns
    -------
    response : np.ndarray, shape (d + p,)
        Full static action in the declared basis.

    Raises
    ------
    ValueError
        If matrices, ensembles or state violate their documented shapes, signs,
        definiteness, realness or finiteness conditions.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _response_data(auxiliary, pair_response, precision, vectors, forces, weights):
    auxiliary = _symmetric(auxiliary, "auxiliary", positive=True)
    precision, vectors, forces, weights = _force_data(
        precision, vectors, forces, weights
    )
    if auxiliary.shape != precision.shape:
        raise ValueError("auxiliary and precision dimensions must agree")
    size = len(auxiliary) * (len(auxiliary) + 1) // 2
    pair_response = _symmetric(pair_response, "pair_response")
    if (
        pair_response.shape != (size, size)
        or np.linalg.eigvalsh(pair_response)[-1] >= 0
    ):
        raise ValueError("pair_response must have shape (p,p) and be negative definite")
    return auxiliary, pair_response, precision, vectors, forces, weights


def _oracle_apply_static_response(
    auxiliary: np.ndarray,
    pair_response: np.ndarray,
    precision: np.ndarray,
    precision_displacements: np.ndarray,
    residual_forces: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    auxiliary, pair_response, precision, vectors, forces, weights = _response_data(
        auxiliary,
        pair_response,
        precision,
        precision_displacements,
        residual_forces,
        weights,
    )
    dimension = len(auxiliary)
    state = _vector(state, dimension + len(pair_response), "state")
    harmonic = np.concatenate(
        (
            auxiliary @ state[:dimension],
            -np.linalg.solve(pair_response, state[dimension:]),
        )
    )
    return harmonic + _oracle_apply_anharmonic_response(
        precision, vectors, forces, weights, state
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic numerical and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ncubic *= 1.6\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare = bare[:3,:3].copy()\ncentroid = centroid[:3].copy()\ndirections = directions[:,:3].copy()\ntemperature = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ncubic[:] = 0.0\nquartic[:] = 0.0\nsextic[:] = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = bare.copy()\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\nstate[:len(bare)] = 0.0\n",
            "call": "apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nauxiliary = np.array([[1.8,.2,-.1],[.2,1.2,.25],[-.1,.25,2.4]])\nprecision = np.array([[2.,.3,-.2],[.3,1.5,.1],[-.2,.1,1.1]])\nvectors = np.sin(np.arange(21,dtype=float).reshape(7,3)*.47)\nforces = np.cos(np.arange(21,dtype=float).reshape(7,3)*.31)+0.3*vectors\nweights = np.arange(1.,8.)/28.\nstate = np.linspace(-.4,.7,9)\n_, pair_response = _oracle_quantum_matrix_response(auxiliary,.6)\n",
            "call": "apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_static_response(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, base_weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\nweights = base_weights * 2.0\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(apply_static_response,auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_raises(_oracle_apply_static_response,auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
    ]
