"""
Apply the permutation-symmetric anharmonic response from Gaussian force moments.

Symmetric coordinates use $B^{ii}=e_ie_i^T$ and

$B^{ij}=(e_ie_j^T+e_je_i^T)/\sqrt2$ for $i<j$, in lexicographic pair order.

For $p=d(d+1)/2$, packing means $s_a=\operatorname{tr}((B^a)^TS)$;

the coordinate frame is fixed in physical space, including inside degenerate eigenspaces.



For a Gaussian with precision $P$, define its score tensors by



$$

\mathcal H^{(n)}_{i_1\ldots i_n}(u)

=(-1)^n\rho(u)^{-1}\partial_{i_1}\cdots\partial_{i_n}\rho(u),

\qquad \rho(u)\propto\exp(-u^TPu/2).

$$



Construct the cubic and quartic vertices from the fully permutation-symmetrized

weighted moments $-\langle f\otimes\mathcal H^{(2)}\rangle$ and

$-\langle f\otimes\mathcal H^{(3)}\rangle$, respectively; symmetrization

means the arithmetic average over all permutations of the three or four indices.

This definition also fixes the returned action for finite supplied ensembles.

The normalization contributions in the score tensors are retained even when

the auxiliary matrix is not stationary or the centroid is externally constrained.

For a state $(r,s)$ and the symmetric matrix $S$ unpacked from $s$, return



$$

\left(C^{(3)}:S,\ \operatorname{pack}(C^{(3)}\cdot r+C^{(4)}:S)\right).

$$



The two cubic actions are transposes in the orthonormal pair coordinates.

The supplied arrays may be any finite weighted ensemble satisfying the interface;

no zero-force or stationarity identity may be assumed.

Returns
-------
A float64 vector containing the cubic centroid response and the cubic-plus-quartic symmetric-pair response.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def apply_anharmonic_response(
    precision: np.ndarray,
    precision_displacements: np.ndarray,
    residual_forces: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    r"""Apply the permutation-symmetric anharmonic response from Gaussian force moments.

    Parameters
    ----------
    precision : np.ndarray, shape (d, d)
        Finite real symmetric positive-definite $P$, with $1\le d\le5$.
    precision_displacements : np.ndarray, shape (K, d)
        Finite real $v=Pu$ rows, with $K\ge1$.
    residual_forces : np.ndarray, shape (K, d)
        Finite real residual force rows, with mean and linear components allowed.
    weights : np.ndarray, shape (K,)
        Finite nonnegative weights, normalized within absolute error $10^{-12}$.
    state : np.ndarray, shape (d + p,)
        Finite centroid entries followed by packed symmetric-pair entries.

    Returns
    -------
    response : np.ndarray, shape (d + p,)
        Anharmonic action in the input state order.

    Raises
    ------
    ValueError
        If the precision, array shapes, realness, finiteness, weight normalization
        or state dimensions violate the contract, or the response is not
        representable.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _force_data(precision, precision_displacements, residual_forces, weights):
    precision = _symmetric(precision, "precision", positive=True)
    dimension = len(precision)
    if dimension > 5:
        raise ValueError("the optical dimension must not exceed five")
    vectors = _real(precision_displacements, "precision_displacements")
    forces = _real(residual_forces, "residual_forces")
    if (
        vectors.ndim != 2
        or vectors.shape[1] != dimension
        or len(vectors) < 1
        or forces.shape != vectors.shape
    ):
        raise ValueError("force and displacement arrays must have equal shape (K,d)")
    weights = _vector(weights, len(vectors), "weights")
    if np.any(weights < 0) or abs(float(weights.sum()) - 1) > 1e-12:
        raise ValueError("weights must be nonnegative and normalized")
    return precision, vectors, forces, weights


def _oracle_apply_anharmonic_response(
    precision: np.ndarray,
    precision_displacements: np.ndarray,
    residual_forces: np.ndarray,
    weights: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    precision, vectors, forces, weights = _force_data(
        precision, precision_displacements, residual_forces, weights
    )
    dimension = len(precision)
    basis = _basis(dimension)
    state = _vector(state, dimension + len(basis), "state")
    centroid = state[:dimension]
    pair = _unpack(state[dimension:], basis)
    scalar2 = np.einsum("ki,ij,kj->k", vectors, pair, vectors) - np.trace(
        precision @ pair
    )
    mixed = np.einsum("ki,ij,kj->k", vectors, pair, forces)
    psforce = forces @ pair @ precision
    top = (
        -np.einsum(
            "k,ki->i",
            weights,
            forces * scalar2[:, None] + 2 * vectors * mixed[:, None] - 2 * psforce,
        )
        / 3
    )
    score2 = np.einsum("ki,kj->kij", vectors, vectors) - precision
    rf = forces @ centroid
    rv = vectors @ centroid
    pr = precision @ centroid
    cubic = (
        rf[:, None, None] * score2
        + rv[:, None, None]
        * (
            np.einsum("ki,kj->kij", forces, vectors)
            + np.einsum("ki,kj->kij", vectors, forces)
        )
        - np.einsum("i,kj->kij", pr, forces)
        - np.einsum("ki,j->kij", forces, pr)
    )
    score3 = vectors * scalar2[:, None] - 2 * vectors @ pair @ precision
    quartic = (
        np.einsum("ki,kj->kij", forces, score3)
        + np.einsum("ki,kj->kij", score3, forces)
        + 2
        * (
            mixed[:, None, None] * score2
            - np.einsum("ki,kj->kij", vectors, psforce)
            - np.einsum("ki,kj->kij", psforce, vectors)
        )
    )
    bottom = -np.einsum("k,kij->ij", weights, cubic / 3 + quartic / 4)
    result = np.concatenate((top, _pack(bottom, basis)))
    if not np.all(np.isfinite(result)):
        raise ValueError("anharmonic response exceeds the numerical range")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic numerical and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ncubic *= 1.6\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare = bare[:3,:3].copy()\ncentroid = centroid[:3].copy()\ndirections = directions[:,:3].copy()\ntemperature = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ncubic[:] = 0.0\nquartic[:] = 0.0\nsextic[:] = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\n",
            "call": "apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = bare.copy()\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\nstate[:len(bare)] = 0.0\n",
            "call": "apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nauxiliary = np.array([[1.8,.2,-.1],[.2,1.2,.25],[-.1,.25,2.4]])\nprecision = np.array([[2.,.3,-.2],[.3,1.5,.1],[-.2,.1,1.1]])\nvectors = np.sin(np.arange(21,dtype=float).reshape(7,3)*.47)\nforces = np.cos(np.arange(21,dtype=float).reshape(7,3)*.31)+0.3*vectors\nweights = np.arange(1.,8.)/28.\nstate = np.linspace(-.4,.7,9)\n_, pair_response = _oracle_quantum_matrix_response(auxiliary,.6)\n",
            "call": "apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_oracle_apply_anharmonic_response(precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncovariance, pair_response = _oracle_quantum_matrix_response(auxiliary,temperature)\nprecision, vectors, forces, base_weights = _oracle_build_correlated_ensemble(auxiliary,temperature,bare,centroid,directions,cubic,quartic,sextic,reference_variances)\nstate = np.linspace(-0.3,0.5,len(bare)+len(pair_response))\nweights = base_weights * 2.0\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(apply_anharmonic_response,precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "gold_call": "_raises(_oracle_apply_anharmonic_response,precision.copy(),vectors.copy(),forces.copy(),weights.copy(),state.copy())",
            "tol": 1e-09,
        },
    ]
