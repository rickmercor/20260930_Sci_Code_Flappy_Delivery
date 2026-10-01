"""
Differentiate the stationary Gaussian and its relaxed positional Hessian along a parameter path.

The supplied $\Phi$ is stationary for the constrained Gaussian at fixed centroid

$b$ and the directional polynomial potential of the earlier steps.

Only the current cubic coefficients and actual temperature vary:



$$

g_r(t)=g_r+t\dot g_r,\qquad \theta(t)=\theta+t\tau.

$$



The bare quadratic matrix, $b$, directions, quartic and sextic coefficients,

and reference variances $c_r$ remain fixed.

The optimized matrix follows its covariance-stable branch, so



$$

\Phi(t)=\left\langle\nabla^2 V_t(b+u)\right\rangle_{\Psi(\Phi(t),\theta(t))}.

$$



Return $\dot\Phi$, $\dot\Psi$ and $\dot H$ at $t=0$, where $H$ is the complete

covariance-relaxed positional Hessian from the earlier response steps.

Both explicit parameter dependence and implicit Gaussian relaxation are included.

A singular or indefinite positional Hessian is valid; the covariance sector must

remain locally stable and the stationary response must be resolvable.

A residual of at most $10^{-10}\max(1,\|\text{right-hand side}\|_2)$ is required

for the differentiated stationarity equation.

The bare matrix need not be supplied because it has zero derivative;

stationarity of the provided $\Phi$ is an input precondition.

At zero temperature use the same right-derivative convention as the preceding step.

Returns
-------
A float64 array containing the auxiliary, covariance and relaxed-Hessian directional derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def stationary_hessian_tangent(
    auxiliary: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_variances: np.ndarray,
    temperature: float,
    cubic_direction: np.ndarray,
    temperature_direction: float,
) -> np.ndarray:
    r"""Differentiate the stationary Gaussian and its relaxed positional Hessian along a parameter path.

    Parameters
    ----------
    auxiliary : np.ndarray, shape (d, d)
        Finite real positive-definite stationary $\Phi$, with $1\le d\le5$.
    centroid : np.ndarray, shape (d,)
        Finite fixed constrained centroid $b$.
    directions : np.ndarray, shape (R, d)
        Finite direction rows, with $1\le R\le32$.
    cubic : np.ndarray, shape (R,)
        Finite signed current cubic coefficients, already scaled by coupling.
    quartic : np.ndarray, shape (R,)
        Finite nonnegative fixed quartic coefficients.
    sextic : np.ndarray, shape (R,)
        Finite nonnegative fixed sextic coefficients.
    reference_variances : np.ndarray, shape (R,)
        Finite nonnegative fixed potential coefficients $c_r$.
    temperature : float
        Finite actual thermal energy $\theta\ge0$.
    cubic_direction : np.ndarray, shape (R,)
        Finite signed cubic derivatives $\dot g_r$.
    temperature_direction : float
        Finite thermal derivative $\tau$, nonnegative when $\theta=0$.

    Returns
    -------
    response : np.ndarray, shape (3, d, d)
        Symmetric $\dot\Phi$, $\dot\Psi$ and $\dot H$, in that order.

    Raises
    ------
    ValueError
        If model data or directional parameters violate the documented shapes,
        realness, finiteness, sign or symmetry constraints.
    RuntimeError
        If the covariance sector is unstable, the implicit response is singular,
        its residual tolerance is not reached, or a derivative is not finite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_stationary_hessian_tangent(
    auxiliary: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_variances: np.ndarray,
    temperature: float,
    cubic_direction: np.ndarray,
    temperature_direction: float,
) -> np.ndarray:
    data = _model_data(
        auxiliary, centroid, directions, cubic, quartic, sextic, reference_variances
    )
    auxiliary, centroid, directions, cubic, quartic, sextic, reference_variances = data
    cubic_direction = _vector(cubic_direction, len(cubic), "cubic_direction")
    temperature = _temperature(temperature)
    rate = _scalar_direction(temperature_direction, "temperature_direction")
    if temperature == 0 and rate < 0:
        raise ValueError(
            "a zero-temperature directional path cannot enter negative temperature"
        )
    covariance, pair_response = _oracle_quantum_matrix_response(auxiliary, temperature)
    basis = _basis(len(auxiliary))
    means = directions @ centroid
    variances = np.einsum("ri,ij,rj->r", directions, covariance, directions)
    vertices = _oracle_gaussian_vertex_averages(
        means, variances, reference_variances, cubic, quartic, sextic
    )
    pairs = np.einsum("aij,ri,rj->ra", basis, directions, directions)
    coupling = np.einsum("r,ri,ra->ia", vertices[:, 2], directions, pairs)
    quartic_block = np.einsum("r,ra,rb->ab", vertices[:, 3], pairs, pairs)
    inverse_pair = np.linalg.solve(pair_response, np.eye(len(pair_response)))
    pair_block = quartic_block - inverse_pair
    if np.linalg.eigvalsh(pair_block)[0] <= 0:
        raise RuntimeError("the covariance sector is not locally stable")
    thermal_covariance, _ = _oracle_quantum_response_tangent(
        auxiliary, temperature, np.zeros_like(auxiliary), rate
    )
    direct = np.einsum("r,r,ra->a", cubic_direction, means, pairs)
    direct += 0.5 * quartic_block @ _pack(thermal_covariance, basis)
    jacobian = np.eye(len(pair_response)) - quartic_block @ pair_response
    try:
        packed_direction = np.linalg.solve(jacobian, direct)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("implicit stationarity response is singular") from error
    if np.linalg.norm(jacobian @ packed_direction - direct) > 1e-10 * max(
        1.0, np.linalg.norm(direct)
    ):
        raise RuntimeError("implicit stationarity response did not converge")
    auxiliary_direction = _unpack(packed_direction, basis)
    covariance_direction, pair_direction = _oracle_quantum_response_tangent(
        auxiliary, temperature, auxiliary_direction, rate
    )
    variance_direction = np.einsum(
        "ri,ij,rj->r", directions, covariance_direction, directions
    )
    cubic_vertex_direction = cubic_direction + 0.5 * sextic * means * variance_direction
    quartic_vertex_direction = 0.5 * sextic * variance_direction
    coupling_direction = np.einsum(
        "r,ri,ra->ia", cubic_vertex_direction, directions, pairs
    )
    quartic_direction = np.einsum("r,ra,rb->ab", quartic_vertex_direction, pairs, pairs)
    block_direction = quartic_direction + inverse_pair @ pair_direction @ inverse_pair
    response = np.linalg.solve(pair_block, coupling.T)
    hessian_direction = (
        auxiliary_direction
        - coupling_direction @ response
        - response.T @ coupling_direction.T
        + response.T @ block_direction @ response
    )
    result = np.stack(
        [
            auxiliary_direction,
            covariance_direction,
            (hessian_direction + hessian_direction.T) * 0.5,
        ]
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("stationary Hessian tangent is not finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncubic_direction = cubic.copy()\nrate = .4\n",
            "call": "stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "gold_call": "_oracle_stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncubic_direction = cubic.copy()\nrate = .4\nrate=0.0\n",
            "call": "stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "gold_call": "_oracle_stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncubic_direction = cubic.copy()\nrate = .4\ncubic_direction[:]=0.0\nrate=1.0\n",
            "call": "stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "gold_call": "_oracle_stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncubic_direction = cubic.copy()\nrate = .4\ncubic_direction[:]=0.0\nrate=0.0\n",
            "call": "stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "gold_call": "_oracle_stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ntemperature=0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncubic_direction = cubic.copy()\nrate = .4\n",
            "call": "stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "gold_call": "_oracle_stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nsextic[:]=0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncubic_direction = cubic.copy()\nrate = .4\n",
            "call": "stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "gold_call": "_oracle_stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare = bare[:3,:3].copy()\ncentroid = centroid[:3].copy()\ndirections = directions[:,:3].copy()\ntemperature = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncubic_direction = cubic.copy()\nrate = .4\n",
            "call": "stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "gold_call": "_oracle_stationary_hessian_tangent(auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\nauxiliary = _oracle_relax_auxiliary_matrix(bare,centroid,directions,cubic,quartic,sextic,reference_variances,temperature)\ncubic_direction = cubic.copy()\nrate = .4\ncubic_direction=np.zeros(len(cubic)+1)\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(stationary_hessian_tangent,auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "gold_call": "_raises(_oracle_stationary_hessian_tangent,auxiliary.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature,cubic_direction.copy(),rate)",
            "tol": 1e-09,
        },
    ]
