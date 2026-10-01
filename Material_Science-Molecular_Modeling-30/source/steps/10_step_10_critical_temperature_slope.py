"""
Compute the thermal slope of the positional-instability coupling with the reference potential fixed.

For the potential, centroid, quantum covariance and full response of the preceding

steps, define the reference variances once from $D_0$ and $\theta_{\rm ref}$.

At each actual temperature near $\theta$, let $\eta_c(\theta)$ be the unique simple

stable-to-unstable zero of the smallest algebraic positional-Hessian eigenvalue

in the supplied coupling bracket, after full covariance reoptimization.

Return



$$

\left.\frac{d\eta_c}{d\theta}\right|_{\theta},

$$



in inverse reduced thermal-energy units.

The centroid and every potential coefficient, including $c_r$, are fixed in this

derivative; only actual temperature and the critical coupling change.

The reference temperature is not varied.

Use the total stationary response rather than a partial derivative at frozen

auxiliary matrix, and include quartic feedback and changing dressed vertices.

The critical positional eigenvalue is simple, the covariance sector remains

stable, and the coupling derivative at the zero is nonzero.

At zero actual temperature, use the right derivative of the gapped branch.

The coupling bracket is finite, increasing, nonnegative and encloses the specified

crossing; an endpoint root is valid within $64\epsilon\max(1,\|H\|_2)$ in curvature.

The root is resolved to absolute coupling tolerance $5\times10^{-12}$;

the returned thermal slope is required within absolute error $10^{-8}$.

Any converged scientifically equivalent evaluation is admissible.

Returns
-------
A Python float giving the total thermal derivative of the critical cubic coupling.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def critical_temperature_slope(
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_temperature: float,
    temperature: float,
    bracket: np.ndarray,
) -> float:
    r"""Compute the thermal slope of the positional-instability coupling with the reference potential fixed.

    Parameters
    ----------
    bare : np.ndarray, shape (d, d)
        Finite real symmetric positive-definite fixed $D_0$, with $1\le d\le5$.
    centroid : np.ndarray, shape (d,)
        Finite fixed centroid $b$.
    directions : np.ndarray, shape (R, d)
        Finite direction rows, with $1\le R\le32$.
    cubic : np.ndarray, shape (R,)
        Finite signed base cubic coefficients $g_r$, before multiplication by the trial coupling.
    quartic : np.ndarray, shape (R,)
        Finite nonnegative quartic coefficients.
    sextic : np.ndarray, shape (R,)
        Finite nonnegative sextic coefficients.
    reference_temperature : float
        Finite nonnegative temperature used once to define $c_r$.
    temperature : float
        Finite nonnegative actual temperature, which may differ from the reference temperature.
    bracket : np.ndarray, shape (2,)
        Finite nonnegative increasing coupling endpoints enclosing one stable-to-unstable crossing.

    Returns
    -------
    thermal_slope : float
        Derivative of the critical dimensionless coupling per reduced thermal energy.

    Raises
    ------
    ValueError
        If model data, temperatures or bracket violate their declared domain, or
        endpoint curvatures do not enclose a stable-to-unstable crossing.
    RuntimeError
        If covariance optimization or response fails, the root is unresolved,
        the critical positional eigenvalue is not numerically simple, its coupling
        derivative is unresolved, or the final thermal slope is not finite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _critical_state(
    coupling,
    bare,
    centroid,
    directions,
    cubic,
    quartic,
    sextic,
    reference_variances,
    temperature,
):
    scaled = coupling * cubic
    auxiliary = _oracle_relax_auxiliary_matrix(
        bare,
        centroid,
        directions,
        scaled,
        quartic,
        sextic,
        reference_variances,
        temperature,
        tolerance=2e-13,
    )
    covariance, pair_response = _oracle_quantum_matrix_response(auxiliary, temperature)
    ensemble = _oracle_build_correlated_ensemble(
        auxiliary,
        temperature,
        bare,
        centroid,
        directions,
        scaled,
        quartic,
        sextic,
        reference_variances,
    )
    hessian = _oracle_relaxed_positional_hessian(auxiliary, pair_response, *ensemble)
    return auxiliary, covariance, hessian


def _locate_instability(
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_temperature: float,
    temperature: float,
    bracket: np.ndarray,
) -> float:
    bare = _symmetric(bare, "bare", positive=True)
    reference_covariance, _ = _oracle_quantum_matrix_response(
        bare, reference_temperature
    )
    directions = _real(directions, "directions")
    if directions.ndim != 2 or directions.shape[1] != len(bare):
        raise ValueError("directions must have shape (R,d)")
    fixed = np.einsum("ri,ij,rj->r", directions, reference_covariance, directions)
    data = _model_data(bare, centroid, directions, cubic, quartic, sextic, fixed)
    temperature = _temperature(temperature)
    bracket = _vector(bracket, 2, "bracket")
    if bracket[0] < 0 or bracket[1] <= bracket[0]:
        raise ValueError("bracket must be increasing and nonnegative")

    def _curvature(coupling):
        hessian = _critical_state(coupling, *data, temperature)[2]
        return float(np.linalg.eigvalsh(hessian)[0])

    endpoint_hessians = [
        _critical_state(value, *data, temperature)[2] for value in bracket
    ]
    left, right = [float(np.linalg.eigvalsh(matrix)[0]) for matrix in endpoint_hessians]
    endpoint_tolerances = [
        64 * np.finfo(float).eps * max(1.0, np.linalg.norm(matrix, 2))
        for matrix in endpoint_hessians
    ]
    if left < -endpoint_tolerances[0] or right > endpoint_tolerances[1]:
        raise ValueError("bracket must enclose a stable-to-unstable crossing")
    if abs(left) <= endpoint_tolerances[0]:
        return float(bracket[0])
    if abs(right) <= endpoint_tolerances[1]:
        return float(bracket[1])
    try:
        result = brentq(
            _curvature, bracket[0], bracket[1], xtol=5e-12, rtol=2e-14, maxiter=100
        )
    except RuntimeError as error:
        raise RuntimeError("critical coupling did not converge") from error
    return float(result)


def _oracle_critical_temperature_slope(
    bare: np.ndarray,
    centroid: np.ndarray,
    directions: np.ndarray,
    cubic: np.ndarray,
    quartic: np.ndarray,
    sextic: np.ndarray,
    reference_temperature: float,
    temperature: float,
    bracket: np.ndarray,
) -> float:
    critical = _locate_instability(
        bare,
        centroid,
        directions,
        cubic,
        quartic,
        sextic,
        reference_temperature,
        temperature,
        bracket,
    )
    reference_covariance, _ = _oracle_quantum_matrix_response(
        bare, reference_temperature
    )
    fixed = np.einsum("ri,ij,rj->r", directions, reference_covariance, directions)
    data = _model_data(bare, centroid, directions, cubic, quartic, sextic, fixed)
    auxiliary, _, hessian = _critical_state(critical, *data, temperature)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    if len(eigenvalues) > 1 and eigenvalues[1] - eigenvalues[0] <= 1e-10 * max(
        1.0, np.linalg.norm(hessian, 2)
    ):
        raise RuntimeError("critical positional eigenvalue is not numerically simple")
    args = (
        auxiliary,
        data[1],
        data[2],
        critical * data[3],
        data[4],
        data[5],
        fixed,
        temperature,
    )
    coupling_response = _oracle_stationary_hessian_tangent(*args, data[3], 0.0)[2]
    thermal_response = _oracle_stationary_hessian_tangent(
        *args, np.zeros_like(data[3]), 1.0
    )[2]
    mode = eigenvectors[:, 0]
    coupling_partial = float(mode @ coupling_response @ mode)
    thermal_partial = float(mode @ thermal_response @ mode)
    if abs(coupling_partial) <= 1e-12 * max(1.0, np.linalg.norm(coupling_response, 2)):
        raise RuntimeError("critical coupling derivative cannot be resolved")
    result = -thermal_partial / coupling_partial
    if not np.isfinite(result):
        raise RuntimeError("critical thermal slope is not finite")
    return float(result)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return deterministic normal, boundary and declared invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\n",
            "call": "critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "gold_call": "_oracle_critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ntemperature=.35\n",
            "call": "critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "gold_call": "_oracle_critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nsextic[:]=0.0\n",
            "call": "critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "gold_call": "_oracle_critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbare = bare[:3,:3].copy()\ncentroid = centroid[:3].copy()\ndirections = directions[:,:3].copy()\ntemperature = 0.5\n",
            "call": "critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "gold_call": "_oracle_critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\ntemperature=0.0\n",
            "call": "critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "gold_call": "_oracle_critical_temperature_slope(bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nfrom scipy.optimize import brentq\nbare = np.diag(np.array([0.8, 0.8, 1.3, 2.1])**2)\ncentroid = np.array([0.16, -0.12, 0.09, 0.07])\ndirections = np.array([[1.,.2,-.3,.4],[.1,1.,.5,-.2],[-.4,.3,1.,.1],[.2,-.1,.4,1.],[.7,.7,-.2,.1],[.3,-.6,.6,.4],[-.5,.2,.4,.7],[.6,-.4,-.1,.5]])\ncubic = np.array([.8,-.55,.7,.4,-.6,.5,.35,-.45])\nquartic = np.array([.9,.7,1.1,.8,.6,1.2,.5,.75])\nsextic = np.array([.7,.4,.9,.5,.6,.8,.3,.65])\ntemperature = 0.7\nreference_temperature = 0.7\nbracket = np.array([0.5,2.0])\nbracket=np.array([.1,.2])\ndef _raises(function,*args):\n    try:\n        function(*args)\n    except ValueError:\n        return 1.0\n    return 0.0\n",
            "call": "_raises(critical_temperature_slope,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "gold_call": "_raises(_oracle_critical_temperature_slope,bare.copy(),centroid.copy(),directions.copy(),cubic.copy(),quartic.copy(),sextic.copy(),reference_temperature,temperature,bracket.copy())",
            "tol": 1e-08,
        },
    ]
