"""
Find the stationary auxiliary matrix at a constrained centroid.

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



Optimize the quantum Gaussian at fixed $b$ and temperature $\theta$, allowing

all covariance entries to vary.

The sought positive-definite auxiliary matrix satisfies



$$

\Phi=\left\langle\nabla^2 V(b+u)\right\rangle,

\qquad u\sim\mathcal N(0,\Psi(\Phi,\theta)).

$$



Return the locally stable covariance branch connected to the initial positive

matrix $D_0$; supported successful inputs have a unique such branch.

The convergence criterion is



$$

\frac{\|\Phi-\langle\nabla^2V\rangle\|_F}{\max(1,\|\Phi\|_F)}

\le\text{tolerance}.

$$



A small residual of a diagonal-only restriction is insufficient.

The internal mean force may be nonzero because the centroid is constrained;

it is balanced by a linear external field with zero Hessian.

Any converged method satisfying this contract is admissible.

Returns
-------
A float64 symmetric auxiliary matrix satisfying full quantum Gaussian stationarity at the prescribed centroid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def relax_auxiliary_matrix(
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_variances: np.ndarray,
    temperature: float,
    tolerance: float = 1e-12,
    max_steps: int = 80,
) -> np.ndarray:
    r"""Find the stationary auxiliary matrix at a constrained centroid.

    Parameters
    ----------
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
    temperature : float
        Finite thermal energy $\theta\ge0$.
    tolerance : float, optional
        Relative stationarity tolerance in $[10^{-14},10^{-6}]$, default $10^{-12}$.
    max_steps : int, optional
        Positive iteration limit, default 80.

    Returns
    -------
    auxiliary : np.ndarray, shape (d, d)
        Symmetric positive-definite stationary auxiliary matrix.

    Raises
    ------
    ValueError
        If shapes, realness, finiteness, positive definiteness, coefficient signs,
        temperature, tolerance or iteration limit violate the documented domain.
    RuntimeError
        If a locally stable positive-definite stationary solution cannot be resolved
        within the requested iteration limit, or its numerical Jacobian is singular.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _model_data(
    bare, centroid, directions, cubic, quartic, sextic, reference_variances
):
    bare = _symmetric(bare, "bare", positive=True)
    dimension = len(bare)
    if dimension > 5:
        raise ValueError("the optical dimension must not exceed five")
    centroid = _vector(centroid, dimension, "centroid")
    directions = _real(directions, "directions")
    if (
        directions.ndim != 2
        or directions.shape[1] != dimension
        or not 1 <= len(directions) <= 32
    ):
        raise ValueError("directions must have shape (R,d), with one to 32 rows")
    size = len(directions)
    cubic = _vector(cubic, size, "cubic")
    quartic = _vector(quartic, size, "quartic")
    sextic = _vector(sextic, size, "sextic")
    reference_variances = _vector(reference_variances, size, "reference_variances")
    if np.any(quartic < 0) or np.any(sextic < 0) or np.any(reference_variances < 0):
        raise ValueError(
            "even coefficients and reference variances must be nonnegative"
        )
    return bare, centroid, directions, cubic, quartic, sextic, reference_variances


def _stationarity_data(
    auxiliary,
    temperature,
    bare,
    centroid,
    directions,
    cubic,
    quartic,
    sextic,
    reference_variances,
):
    covariance, pair_response = _oracle_quantum_matrix_response(auxiliary, temperature)
    variances = np.einsum("ri,ij,rj->r", directions, covariance, directions)
    vertices = _oracle_gaussian_vertex_averages(
        directions @ centroid, variances, reference_variances, cubic, quartic, sextic
    )
    target = bare + np.einsum("r,ri,rj->ij", vertices[:, 1], directions, directions)
    return target, covariance, pair_response, vertices


def _oracle_relax_auxiliary_matrix(
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_variances: np.ndarray,
    temperature: float,
    tolerance: float = 1e-12,
    max_steps: int = 80,
) -> np.ndarray:
    data = _model_data(
        bare, centroid, directions, cubic, quartic, sextic, reference_variances
    )
    bare, centroid, directions, cubic, quartic, sextic, reference_variances = data
    temperature = _temperature(temperature)
    if (
        not np.isscalar(tolerance)
        or not np.isfinite(tolerance)
        or not 1e-14 <= tolerance <= 1e-6
    ):
        raise ValueError("tolerance must lie between 1e-14 and 1e-6")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 1
    ):
        raise ValueError("max_steps must be a positive integer")
    basis = _basis(len(bare))
    directional_pairs = np.einsum("aij,ri,rj->ra", basis, directions, directions)
    auxiliary = bare.copy()
    for _ in range(max_steps):
        target, covariance, pair_response, vertices = _stationarity_data(
            auxiliary, temperature, *data
        )
        residual = _pack(auxiliary - target, basis)
        norm = np.linalg.norm(residual)
        if norm <= tolerance * max(1.0, np.linalg.norm(auxiliary)):
            return auxiliary
        quartic_matrix = np.einsum(
            "r,ra,rb->ab", vertices[:, 3], directional_pairs, directional_pairs
        )
        jacobian = np.eye(len(basis)) - quartic_matrix @ pair_response
        try:
            direction = _unpack(np.linalg.solve(jacobian, -residual), basis)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("stationarity Jacobian is singular") from error
        fraction = 1.0
        for _ in range(45):
            trial = auxiliary + fraction * direction
            if np.linalg.eigvalsh(trial)[0] > 0:
                trial_target = _stationarity_data(trial, temperature, *data)[0]
                if np.linalg.norm(trial - trial_target) <= (1 - 1e-4 * fraction) * norm:
                    auxiliary = trial
                    break
            fraction *= 0.5
        else:
            raise RuntimeError("positive-definite stationarity iteration failed")
    raise RuntimeError("stationarity tolerance not reached")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic numerical and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ncubic *= 1.2\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\n",
            "call": "relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "gold_call": "_oracle_relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare = bare[:3,:3].copy()\ncentroid = centroid[:3].copy()\ndirections = directions[:,:3].copy()\ntemperature = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\n",
            "call": "relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "gold_call": "_oracle_relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nrotation = np.array([[.8,-.6,0.,0.],[.6,.8,0.,0.],[0.,0.,.6,-.8],[0.,0.,.8,.6]])\nbare = rotation@bare@rotation.T\ncentroid = rotation@centroid\ndirections = directions@rotation.T\ncubic *= 1.7\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\n",
            "call": "relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "gold_call": "_oracle_relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ncentroid[:] = 0.0\nsextic[:] = 0.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\n",
            "call": "relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "gold_call": "_oracle_relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ntemperature = 0.05\ncubic *= 1.8\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\n",
            "call": "relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "gold_call": "_oracle_relax_auxiliary_matrix(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nquartic[0] = -1.0\nreference_covariance, _ = _oracle_quantum_matrix_response(bare, reference_temperature)\nreference_variances = np.einsum('ri,ij,rj->r',directions,reference_covariance,directions)\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(relax_auxiliary_matrix,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "gold_call": "_raises(_oracle_relax_auxiliary_matrix,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_variances.copy(),temperature)",
            "tol": 1e-09,
        },
    ]
