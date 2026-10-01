"""
Compute noncommuting first and second responses of the exact harmonic propagator.

An anchored basis need not diagonalize the retained harmonic stiffness. For canonical state $y=(q,\pi)$, the reference equations are

$$\dot y=L(\lambda)y,\qquad

L(\lambda)=\begin{bmatrix}0\&I\\-K(\lambda)&0\end{bmatrix},\qquad R_h(\lambda)=e^{hL(\lambda)}.$$

At a fixed parameter, this is exactly the paper's harmonic modal rotation written in a different orthonormal frame. Parameter derivatives of a matrix exponential are Fréchet derivatives. In particular,

$$R_h'=\int_0^h e^{(h-t)L_0}L_1e^{tL_0}\,dt.$$

The second derivative includes one insertion of $L_2$ and both time orderings of two insertions of $L_1$. Because $K_0,K_1,K_2$ need not commute, differentiating eigenvalues while holding eigenvectors fixed loses terms. Repeated eigenvalues of $K_0$ are allowed and must have a continuous response.



With the canonical matrix $\mathbb S=\begin{bmatrix}0\&I\\-I&0\end{bmatrix}$, the exact propagator and its raw first two derivatives satisfy

$$R_0^T\mathbb S R_0=\mathbb S,$$

$$R_1^T\mathbb S R_0+R_0^T\mathbb S R_1=0,$$

$$R_2^T\mathbb S R_0+2R_1^T\mathbb S R_1+R_0^T\mathbb S R_2=0.$$

These relations provide structural checks; they do not determine the derivatives by themselves. Signed time steps use the same analytic family. The requested result is the full exact propagator jet, not a finite-difference approximation at displaced parameter values.

Returns
-------
np.ndarray, shape (3,2*k,2*k), exact harmonic propagator and its noncommuting first two responses.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def harmonic_frechet_response(stiffness: "np.ndarray", h: float) -> "np.ndarray":
    r"""Compute noncommuting first and second responses of the exact harmonic propagator.

    Parameters
    ----------
    stiffness : np.ndarray
        Shape (3,k,k), symmetric stiffness jet with positive-definite layer 0.
        Derivative matrices can be noncommuting; repeated nominal eigenvalues allowed.
    h : float
        Finite signed step duration, independent of lambda.

    Returns
    -------
    result, np.ndarray
        Shape (3,2*k,2*k), exact exp(h*L(lambda)) and its first two raw parameter
        derivatives for L=[[0,I],[-K,0]], with all q coordinates before momenta.
        Compute analytic Fréchet responses, without parameter finite differences.

    All derivative arrays have leading axis (value, first derivative, second derivative) at lambda=0, with no factorial scaling. All data are real, finite, and conforming; inputs must be preserved.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm


def _oracle_harmonic_frechet_response(
    stiffness: "np.ndarray", h: float
) -> "np.ndarray":
    k = stiffness.shape[1]
    L = np.zeros((3, 2 * k, 2 * k))
    L[0, :k, k:] = np.eye(k)
    L[:, k:, :k] = -stiffness
    size = 2 * k
    lift = np.zeros((3 * size, 3 * size))
    for i in range(3):
        lift[i * size : (i + 1) * size, i * size : (i + 1) * size] = L[0]
    lift[:size, size : 2 * size] = L[1]
    lift[size : 2 * size, 2 * size :] = L[1]
    lift[:size, 2 * size :] = L[2] / 2
    E = expm(h * lift)
    result = np.stack(
        (E[:size, :size], E[:size, size : 2 * size], 2 * E[:size, 2 * size :])
    )
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return coupled, boundary and response-stress cases."""
    return [
        {
            "setup": """import numpy as np
stiffness = np.array([[[0.6431065189180685, -0.04656968603231221, 0.04610333065506644], [-0.04656968603231225, 1.114970323984326, -0.04542657389493296], [0.04610333065506661, -0.045426573894933, 2.1011799147616497]], [[0.014263223451211095, 0.02488934569796713, -0.06885644454607474], [0.024889345697967122, 0.022872623203695784, -0.013302502239380518], [-0.06885644454607479, -0.013302502239380518, -0.09136623825215225]], [[-0.024397337543928525, 0.0011448881493686608, -0.009847887657903141], [0.00114488814936866, -0.00987308981391529, -0.005516994393790621], [-0.009847887657903148, -0.0055169943937906215, -0.03723278304260587]]], dtype=float)
h = 0.16

def preserved(fn, *args):
    snapshots = [(x, x.copy()) for x in args if isinstance(x, np.ndarray)]
    result = fn(*args)
    for value, original in snapshots:
        assert np.array_equal(value, original), "inputs must be preserved"
    return result
""",
            "call": "preserved(harmonic_frechet_response, stiffness.copy(), h)",
            "gold_call": "preserved(_oracle_harmonic_frechet_response, stiffness.copy(), h)",
            "tol": 2e-09,
        },
        {
            "setup": """import numpy as np
stiffness = np.array([[[1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 2.5]], [[0.1, 0.2, -0.1], [0.2, -0.15, 0.08], [-0.1, 0.08, 0.2]], [[0.03, -0.02, 0.04], [-0.02, 0.01, 0.05], [0.04, 0.05, -0.01]]], dtype=float)
h = 0.27
""",
            "call": "harmonic_frechet_response(stiffness.copy(), h)",
            "gold_call": "_oracle_harmonic_frechet_response(stiffness.copy(), h)",
            "tol": 2e-09,
        },
        {
            "setup": """import numpy as np
stiffness = np.array([[[1.2, 0.0, 0.0], [0.0, 1.2, 0.0], [0.0, 0.0, 2.5]], [[0.1, 0.2, -0.1], [0.2, -0.15, 0.08], [-0.1, 0.08, 0.2]], [[0.03, -0.02, 0.04], [-0.02, 0.01, 0.05], [0.04, 0.05, -0.01]]], dtype=float)
h = -0.19
""",
            "call": "harmonic_frechet_response(stiffness.copy(), h)",
            "gold_call": "_oracle_harmonic_frechet_response(stiffness.copy(), h)",
            "tol": 2e-09,
        },
    ]
