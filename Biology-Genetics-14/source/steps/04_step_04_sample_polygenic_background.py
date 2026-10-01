"""
Sample the dense polygenic background in the GRM eigenbasis.

The mixed model separates sparse cis effects from a dense component

g ~ N(0, sigma2 * eta_g * K), where K is the genetic relationship matrix.

Rotating into the eigenbasis of K diagonalizes the Gaussian conditional. For

eigenvalue lambda_i and cis residual e_i = y_rot_i - (X_rot beta)_i, define

s_i = eta_g * lambda_i / (1 + eta_g * lambda_i). The conditional mean is

s_i e_i, the conditional variance is sigma2 * s_i, and a supplied standard

normal variate gives g_rot_i = s_i e_i + sqrt(sigma2 * s_i) z_i. A zero

eigenvalue has zero shrinkage and variance, so its dense-effect component is

deterministically zero regardless of z_i.

Inputs

------

y_rot, X_rot : expression and genotype data in the GRM eigenbasis

beta : current sparse cis-effect vector

eigenvalues : nonnegative eigenvalues lambda_i of the GRM

sigma2, eta_g : residual variance and polygenic-to-residual variance ratio

z_normal : fixed standard-normal variates for the rotated components

Returns

-------

g_rot : sampled dense effect in the eigenbasis

shrinkage : componentwise coefficients s_i

conditional_variance : componentwise Gaussian variances sigma2 * s_i

A dense genetic background can be sampled efficiently after rotation into the genetic-relationship-matrix eigenbasis, where its components decouple. This step updates that rotated background while respecting rank-deficient directions.

Returns
-------
float64 array of shape (3, n): rows contain g_rot, shrinkage, and conditional_variance, respectively
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def sample_polygenic_background(
    y_rot: np.ndarray,
    X_rot: np.ndarray,
    beta: np.ndarray,
    eigenvalues: np.ndarray,
    sigma2: float,
    eta_g: float,
    z_normal: np.ndarray,
) -> np.ndarray:
    """Sample the dense polygenic effect componentwise in the rotated basis.

    Parameters
    ----------
    y_rot : np.ndarray
        Rotated expression vector with shape (n,).
    X_rot : np.ndarray
        Rotated genotype matrix with shape (n, p).
    beta : np.ndarray
        Sparse-effect vector with shape (p,).
    eigenvalues : np.ndarray
        Nonnegative GRM eigenvalues with shape (n,).
    sigma2 : float
        Positive residual variance.
    eta_g : float
        Positive polygenic-to-residual variance ratio.
    z_normal : np.ndarray
        Standard-normal variates with shape (n,).

    Raises
    ------
    ValueError
        If dimensions, eigenvalues, variances, or supplied variates are invalid.

    Returns
    -------
    result : np.ndarray
        Float64 array of shape (3, n). Rows contain the dense sample, shrinkage
        coefficients, and conditional variances, respectively.
    """
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sample_polygenic_background(
    y_rot: np.ndarray,
    X_rot: np.ndarray,
    beta: np.ndarray,
    eigenvalues: np.ndarray,
    sigma2: float,
    eta_g: float,
    z_normal: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    expression = np.asarray(y_rot, dtype=np.float64)
    X = np.asarray(X_rot, dtype=np.float64)
    effects = np.asarray(beta, dtype=np.float64)
    eigvals = np.asarray(eigenvalues, dtype=np.float64)
    normals = np.asarray(z_normal, dtype=np.float64)

    if (
        X.ndim != 2
        or X.shape[0] == 0
        or X.shape[1] == 0
        or not np.all(np.isfinite(X))
    ):
        raise ValueError("X_rot must be a nonempty finite matrix")

    if expression.shape != (X.shape[0],) or not np.all(
        np.isfinite(expression)
    ):
        raise ValueError("y_rot must be a finite vector of length n")

    if effects.shape != (X.shape[1],) or not np.all(np.isfinite(effects)):
        raise ValueError("beta must be a finite vector of length p")

    if (
        eigvals.shape != (X.shape[0],)
        or not np.all(np.isfinite(eigvals))
        or np.any(eigvals < 0.0)
    ):
        raise ValueError(
            "eigenvalues must be a finite nonnegative vector of length n"
        )

    if normals.shape != (X.shape[0],) or not np.all(np.isfinite(normals)):
        raise ValueError("z_normal must be a finite vector of length n")

    try:
        residual_variance = float(sigma2)
        polygenic_ratio = float(eta_g)
    except (TypeError, ValueError) as exc:
        raise ValueError("sigma2 and eta_g must be scalar") from exc

    if not np.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("sigma2 must be positive and finite")

    if not np.isfinite(polygenic_ratio) or polygenic_ratio <= 0.0:
        raise ValueError("eta_g must be positive and finite")

    rotated_cis_residual = expression - X @ effects
    effective_eigenvalues = np.where(eigvals < 1e-12, 0.0, eigvals)

    shrinkage = (
        polygenic_ratio
        * effective_eigenvalues
        / (1.0 + polygenic_ratio * effective_eigenvalues)
    )
    conditional_variance = residual_variance * shrinkage
    g_rot = (
        shrinkage * rotated_cis_residual
        + np.sqrt(conditional_variance) * normals
    )

    return np.vstack((g_rot, shrinkage, conditional_variance))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
X = np.array([[1.0, 0.5], [-0.5, 1.0], [0.0, 0.0]])
y = np.array([1.2, -0.3, 0.7])
beta = np.array([0.4, 0.0])
eigenvalues = np.array([1.8, 0.7, 0.0])
z = np.array([0.2, -0.7, 1.1])
""",
            "call": "sample_polygenic_background(y, X, beta, eigenvalues, 0.55, 0.65, z)",
            "gold_call": "_oracle_sample_polygenic_background(y, X, beta, eigenvalues, 0.55, 0.65, z)",
        },
        {
            "setup": """import numpy as np
X = np.zeros((1, 1))
y = np.array([3.0])
beta = np.array([0.0])
eigenvalues = np.array([1e-14])
z = np.array([99.0])
""",
            "call": "sample_polygenic_background(y, X, beta, eigenvalues, 1.0, 2.0, z)",
            "gold_call": "_oracle_sample_polygenic_background(y, X, beta, eigenvalues, 1.0, 2.0, z)",
        },
        {
            "setup": """import numpy as np
X = np.zeros((1, 1))
y = np.array([1.0])
beta = np.array([0.0])
eigenvalues = np.array([-0.1])
z = np.array([0.0])
def run_model():
    try:
        sample_polygenic_background(y, X, beta, eigenvalues, 1.0, 1.0, z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_sample_polygenic_background(y, X, beta, eigenvalues, 1.0, 1.0, z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
