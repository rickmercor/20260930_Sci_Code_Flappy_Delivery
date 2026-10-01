"""
Recover positional curvature after covariance relaxation, including at a critical point.

Symmetric coordinates use $B^{ii}=e_ie_i^T$ and

$B^{ij}=(e_ie_j^T+e_je_i^T)/\sqrt2$ for $i<j$, in lexicographic pair order.

For $p=d(d+1)/2$, packing means $s_a=\operatorname{tr}((B^a)^TS)$;

the coordinate frame is fixed in physical space, including inside degenerate eigenspaces.



Let the full static operator be partitioned into centroid and pair sectors.

The positional Hessian is the curvature remaining after the pair variables have

relaxed while the centroid displacement is imposed.

The covariance-sector block must be positive definite for a local covariance

minimum; the full static operator and the positional Hessian are allowed to

be singular or indefinite.

Physical Hessian interpretation requires a stationary constrained Gaussian.

Return the covariance-eliminated centroid block, preserving all off-diagonal

mode mixing and finite negative eigenvalues.

Resolve pair-sector linear responses to relative residual at most $10^{-10}$,

using normalization $\max(1,\|\text{right-hand side}\|_F)$.

No inverse of a singular centroid susceptibility is required by this contract.

Returns
-------
A float64 symmetric positional Hessian that remains defined at a zero-curvature instability.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def relaxed_positional_hessian(
    auxiliary: np.ndarray,
    pair_response: np.ndarray,
    precision: np.ndarray,
    precision_displacements: np.ndarray,
    residual_forces: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    r"""Recover positional curvature after covariance relaxation, including at a critical point.

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

    Returns
    -------
    hessian : np.ndarray, shape (d, d)
        Full symmetric positional Hessian in reduced squared-frequency units.

    Raises
    ------
    ValueError
        If input matrix, ensemble or temperature-independent response data violate
        the documented domain.
    RuntimeError
        If the pair sector is not positive definite, reciprocity fails by more than
        $10^{-10}$ relatively, a solve fails or misses its residual tolerance, or
        the Hessian is not finite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_relaxed_positional_hessian(
    auxiliary: np.ndarray,
    pair_response: np.ndarray,
    precision: np.ndarray,
    precision_displacements: np.ndarray,
    residual_forces: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    data = _response_data(
        auxiliary,
        pair_response,
        precision,
        precision_displacements,
        residual_forces,
        weights,
    )
    auxiliary, pair_response, precision, vectors, forces, weights = data
    dimension = len(auxiliary)
    total = dimension + len(pair_response)
    columns = [_oracle_apply_static_response(*data, column) for column in np.eye(total)]
    operator = np.column_stack(columns)
    asymmetry = np.linalg.norm(operator - operator.T) / max(
        1.0, np.linalg.norm(operator)
    )
    if asymmetry > 1e-10:
        raise RuntimeError("static response violates reciprocity")
    operator = (operator + operator.T) * 0.5
    coupling = operator[:dimension, dimension:]
    pair_block = operator[dimension:, dimension:]
    if np.linalg.eigvalsh(pair_block)[0] <= 0:
        raise RuntimeError("the covariance sector is not locally stable")
    try:
        response = np.linalg.solve(pair_block, coupling.T)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("pair response is singular") from error
    residual = np.linalg.norm(pair_block @ response - coupling.T) / max(
        1.0, np.linalg.norm(coupling)
    )
    if not np.isfinite(residual) or residual > 1e-10:
        raise RuntimeError("pair response tolerance not reached")
    hessian = auxiliary - coupling @ response
    if not np.all(np.isfinite(hessian)):
        raise RuntimeError("positional Hessian is not finite")
    return (hessian + hessian.T) * 0.5

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic numerical and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "gold_call": "_oracle_relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ncubic *= 1.6\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "gold_call": "_oracle_relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare = bare[:3,:3].copy()\ncentroid = centroid[:3].copy()\ndirections = directions[:,:3].copy()\ntemperature = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "gold_call": "_oracle_relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ncubic[:] = 0.0\nquartic[:] = 0.0\nsextic[:] = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "gold_call": "_oracle_relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.35\nreference_temperature = 0.35\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "gold_call": "_oracle_relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare = np.array([[0.64]])\ncentroid = np.zeros(1)\ndirections = np.ones((1,1))\nquartic = np.array([0.9])\nsextic = np.zeros(1)\nauxiliary = bare.copy()\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nreference_variances = np.diag(covariance).copy()\ncubic = np.sqrt(np.array([0.64*(0.9-1/pair_response[0,0])]))\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "gold_call": "_oracle_relaxed_positional_hessian(auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, base_weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\nweights = base_weights * 2.0\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(relaxed_positional_hessian,auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "gold_call": "_raises(_oracle_relaxed_positional_hessian,auxiliary.copy(),pair_response.copy(),precision.copy(),vectors.copy(),forces.copy(),weights.copy())",
            "tol": 1e-09,
        },
    ]
