"""
Sample SNP effects sequentially within an active LD block.

Conditional on zeta_b = 1, the SNP indicators and effects are updated by a

coordinate-wise Gibbs scan in block order. The running residual is

r = y_rot - X_rot beta - g_rot. Before visiting SNP j, its old contribution

x_j beta_j is restored to r; the conditional slab variance v_j, mean mu_j, and

log Bayes factor are then computed from that restored residual. The posterior

inclusion log-odds equal logit(pi_j) + logBF_j. If the supplied uniform passes

this gate, beta_j = mu_j + sqrt(v_j) z_j and the new contribution is immediately

subtracted from r; otherwise beta_j and gamma_j are set to zero. Because later

SNPs see the effects sampled earlier in the same pass, this update preserves the

local dependence induced by linkage disequilibrium and is not equivalent to a

simultaneous diagonal scan.

Inputs

------

X_rot : rotated genotype matrix

block_indices : ordered indices of the active block

residual, beta, gamma : current residual and sparse state

sigma2, eta_beta : residual variance and slab variance ratio

inclusion_probabilities : annotation-informed probabilities pi_j

u_gate, z_normal : fixed uniform and standard-normal variates in block order

Returns

-------

residual : residual after the complete block scan

beta, gamma : updated sparse effects and inclusion indicators

scan_probabilities : conditional inclusion probabilities in visitation order

Within an active linkage-disequilibrium block, correlated variants must be updated conditionally so that earlier draws affect later ones through the working residual. This step performs that local sparse-effect scan.

Returns
-------
float64 vector of length n + 2p + len(block_indices), concatenating the updated residual, beta, numeric gamma indicators, and scan probabilities in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def scan_active_block(
    X_rot: np.ndarray,
    block_indices: np.ndarray,
    residual: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma2: float,
    eta_beta: float,
    inclusion_probabilities: np.ndarray,
    u_gate: np.ndarray,
    z_normal: np.ndarray,
) -> np.ndarray:
    """Perform one sequential spike-and-slab scan of an active LD block.

    Parameters
    ----------
    X_rot : np.ndarray
        Rotated genotype matrix with shape (n, p).
    block_indices : np.ndarray
        Ordered unique SNP indices for the active block.
    residual : np.ndarray
        Current residual y_rot - X_rot @ beta - g_rot.
    beta : np.ndarray
        Current sparse-effect vector with shape (p,).
    gamma : np.ndarray
        Current binary inclusion vector with shape (p,).
    sigma2 : float
        Positive residual variance.
    eta_beta : float
        Positive slab-to-residual variance ratio.
    inclusion_probabilities : np.ndarray
        Per-SNP prior probabilities with shape (p,).
    u_gate : np.ndarray
        Uniform variates in block order.
    z_normal : np.ndarray
        Standard-normal variates in block order.

    Raises
    ------
    ValueError
        If dimensions, parameters, or supplied variates are invalid, or if
        beta[j] is nonzero for any index at which gamma[j] is false.

    Returns
    -------
    result : np.ndarray
        Float64 vector containing `residual` (first n entries), `beta` (next p),
        numeric `gamma` indicators (next p), and `scan_probabilities` (remaining
        entries).
    """
    return result  # noqa: F821

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np  # noqa: E402, F811


def _oracle_scan_active_block(
    X_rot: np.ndarray,
    block_indices: np.ndarray,
    residual: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    sigma2: float,
    eta_beta: float,
    inclusion_probabilities: np.ndarray,
    u_gate: np.ndarray,
    z_normal: np.ndarray,
) -> np.ndarray:
    """Reference implementation."""
    X = np.asarray(X_rot, dtype=np.float64)
    indices = np.asarray(block_indices)
    running_residual = np.asarray(residual, dtype=np.float64).copy()
    effects = np.asarray(beta, dtype=np.float64).copy()
    indicators_raw = np.asarray(gamma)
    probabilities = np.asarray(inclusion_probabilities, dtype=np.float64)
    uniforms = np.asarray(u_gate, dtype=np.float64)
    normals = np.asarray(z_normal, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.all(np.isfinite(X)):
        raise ValueError("X_rot must be a nonempty finite matrix")
    if indices.ndim != 1 or indices.size == 0 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("block_indices must be a nonempty integer vector")
    indices = indices.astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= X.shape[1]) or np.unique(indices).size != indices.size:
        raise ValueError("block_indices must be unique valid column indices")
    if running_residual.shape != (X.shape[0],) or not np.all(np.isfinite(running_residual)):
        raise ValueError("residual must be a finite vector of length n")
    if effects.shape != (X.shape[1],) or not np.all(np.isfinite(effects)):
        raise ValueError("beta must be a finite vector of length p")
    if indicators_raw.shape != (X.shape[1],) or not np.all(
        (indicators_raw == 0) | (indicators_raw == 1)
    ):
        raise ValueError("gamma must be a binary vector of length p")
    indicators = indicators_raw.astype(bool).copy()
    if np.any(effects[~indicators] != 0.0):
        raise ValueError("beta must be zero wherever gamma is zero")
    if probabilities.shape != (X.shape[1],) or np.any(probabilities <= 0.0) or np.any(probabilities >= 1.0):
        raise ValueError("inclusion_probabilities must have shape (p,) and lie in (0, 1)")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("inclusion_probabilities must be finite")
    if uniforms.shape != (indices.size,) or np.any(uniforms < 0.0) or np.any(uniforms >= 1.0):
        raise ValueError("u_gate must contain one variate in [0, 1) per block SNP")
    if normals.shape != (indices.size,) or not np.all(np.isfinite(normals)):
        raise ValueError("z_normal must contain one finite variate per block SNP")
    try:
        residual_variance = float(sigma2)
        slab_ratio = float(eta_beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("sigma2 and eta_beta must be scalar") from exc
    if not np.isfinite(residual_variance) or residual_variance <= 0.0:
        raise ValueError("sigma2 must be positive and finite")
    if not np.isfinite(slab_ratio) or slab_ratio <= 0.0:
        raise ValueError("eta_beta must be positive and finite")

    scan_probabilities = np.empty(indices.size, dtype=np.float64)
    for offset, j in enumerate(indices):
        if indicators[j]:
            running_residual += X[:, j] * effects[j]
            indicators[j] = False
            effects[j] = 0.0

        x_j = X[:, j]
        xtx = float(x_j @ x_j)
        xty = float(x_j @ running_residual)
        variance = 1.0 / (
            xtx / residual_variance + 1.0 / (residual_variance * slab_ratio)
        )
        mean = variance * xty / residual_variance
        log_bayes_factor = (
            0.5 * np.log(variance)
            - 0.5 * np.log(residual_variance * slab_ratio)
            + 0.5 * mean * mean / variance
        )
        log_odds = (
            np.log(probabilities[j])
            - np.log1p(-probabilities[j])
            + log_bayes_factor
        )
        inclusion_probability = np.exp(-np.logaddexp(0.0, -log_odds))
        scan_probabilities[offset] = inclusion_probability
        if uniforms[offset] < inclusion_probability:
            indicators[j] = True
            effects[j] = mean + np.sqrt(variance) * normals[offset]
            running_residual -= X[:, j] * effects[j]

    return np.concatenate(
        (
            running_residual,
            effects,
            indicators.astype(np.float64),
            scan_probabilities,
        )
    )

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np  # noqa: E402, F811


def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
X = np.array([[1.3416407864998738] * 3, [1.0488088481701516, -1.0488088481701516, 1.0488088481701516], [1.02469507659596, 1.02469507659596, -1.02469507659596], [0.6123724356957945, -0.6123724356957945, -0.6123724356957945], [0.0, 0.0, 0.0]])
indices = np.array([0, 1, 2])
residual = np.array([1.3, 0.4, -0.9, 1.0, -0.5])
beta = np.zeros(3)
gamma = np.zeros(3, dtype=bool)
pi = np.array([0.17495631821290467, 0.1750862681640398, 0.16868159439781952])
u = np.array([0.12, 0.62, 0.18])
z = np.array([0.35, -0.8, 1.1])
""",
            "call": "scan_active_block(X, indices, residual, beta, gamma, 0.55, 0.85, pi, u, z)",
            "gold_call": "_oracle_scan_active_block(X, indices, residual, beta, gamma, 0.55, 0.85, pi, u, z)",
        },
        {
            "setup": """import numpy as np
X = np.zeros((1, 1))
indices = np.array([0])
residual = np.array([2.0])
beta = np.array([0.0])
gamma = np.array([False])
pi = np.array([0.5])
u = np.array([0.0])
z = np.array([0.0])
""",
            "call": "scan_active_block(X, indices, residual, beta, gamma, 1.0, 1.0, pi, u, z)",
            "gold_call": "_oracle_scan_active_block(X, indices, residual, beta, gamma, 1.0, 1.0, pi, u, z)",
        },
        {
            "setup": """import numpy as np
X = np.array([[1.0, 1.0], [1.0, -1.0], [0.5, 0.5]])
indices = np.array([0, 1])
residual = np.array([0.2, -0.6, 0.1])
beta = np.array([0.3, 0.0])
gamma = np.array([True, False])
pi = np.array([0.25, 0.75])
u = np.array([0.1, 0.8])
z = np.array([0.2, -0.4])
""",
            "call": "scan_active_block(X, indices, residual, beta, gamma, 1.0, 0.8, pi, u, z)",
            "gold_call": "_oracle_scan_active_block(X, indices, residual, beta, gamma, 1.0, 0.8, pi, u, z)",
        },
        {
            "setup": """import numpy as np
X = np.ones((2, 1))
indices = np.array([0])
residual = np.ones(2)
beta = np.array([1.0])
gamma = np.array([False])
pi = np.array([0.5])
u = np.array([0.2])
z = np.array([0.0])
def run_model():
    try:
        scan_active_block(X, indices, residual, beta, gamma, 1.0, 1.0, pi, u, z)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_scan_active_block(X, indices, residual, beta, gamma, 1.0, 1.0, pi, u, z)
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
