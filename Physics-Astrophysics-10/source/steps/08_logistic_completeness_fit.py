"""
Fit the unbinned logistic recovery efficiency curve.

Fit the unbinned Bernoulli likelihood p(m) = expit[beta_0 + beta_1(m - 28.7)]. Start at (beta_0, beta_1) = (0, -2.2) and perform at most 80 damped Newton steps using X = [1, m - 28.7], with linear predictors clipped to [-40, 40], Hessian X^T diag(p(1-p)) X + 1e-8 I, and backtracking halves until the negative log likelihood does not increase; stop when the scaled step norm is below 1e-11. Require at least six finite rows, both binary outcomes, and a decreasing fit beta_1 < -1e-6, then compute m_50 = 28.7 - beta_0 / beta_1.

Returns
-------
One finite native Python `float`, the fitted 50-percent-completeness magnitude.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Fit the unbinned logistic recovery efficiency curve."""

import numpy as np


def logistic_completeness_fit(recoveries: np.ndarray) -> float:
    """Return the fitted 50-percent-completeness magnitude from binary recoveries.

    Stop when the accepted scaled Newton step has norm below 1e-11.
    """
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_logistic_completeness_fit(recoveries: np.ndarray) -> float:
    """Use damped Newton iterations for p(m)=expit(beta0+beta1*m)."""
    import numpy as np

    records = np.asarray(recoveries, dtype=float)
    if records.ndim != 2 or records.shape[1] != 2 or len(records) < 6 or not np.all(np.isfinite(records)):
        raise ValueError("invalid recovery table")
    magnitude, flag = records[:, 0], records[:, 1]
    if np.any((flag != 0.0) & (flag != 1.0)) or np.all(flag == flag[0]):
        raise ValueError("recovery flags must contain both binary outcomes")
    design = np.column_stack([np.ones_like(magnitude), magnitude - 28.7])
    beta = np.array([0.0, -2.2], dtype=float)
    for _ in range(80):
        eta = np.clip(design @ beta, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-eta))
        weight = np.maximum(probability * (1.0 - probability), 1e-8)
        hessian = design.T @ (weight[:, None] * design) + 1e-8 * np.eye(2)
        step = np.linalg.solve(hessian, design.T @ (flag - probability))
        scale = 1.0
        base = float(np.sum(np.logaddexp(0.0, eta) - flag * eta))
        while scale > 2.0**-20:
            trial = beta + scale * step
            trial_eta = np.clip(design @ trial, -40.0, 40.0)
            trial_loss = float(np.sum(np.logaddexp(0.0, trial_eta) - flag * trial_eta))
            if trial_loss <= base:
                beta = trial
                break
            scale *= 0.5
        if np.linalg.norm(scale * step) < 1e-11:
            break
    if beta[1] >= -1e-6:
        raise ValueError("efficiency fit is not decreasing with magnitude")
    return float(28.7 - beta[0] / beta[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, noisy, and broad-transition fit cases."""
    return [
        {"setup": "import numpy as np\nr=np.array([[27.,1.],[27.4,1.],[27.9,1.],[28.3,0.],[28.8,1.],[29.1,0.],[29.5,0.],[30.,0.]])", "call": "logistic_completeness_fit(r)", "gold_call": "_oracle_logistic_completeness_fit(r)"},
        {"setup": "import numpy as np\nr=np.array([[26.8,1.],[27.5,0.],[28.0,1.],[28.5,1.],[29.,0.],[29.4,0.],[30.,0.]])", "call": "logistic_completeness_fit(r)", "gold_call": "_oracle_logistic_completeness_fit(r)"},
        {"setup": "import numpy as np\nr=np.array([[27.,1.],[27.6,1.],[28.2,0.],[28.8,1.],[29.4,0.],[30.,0.],[30.4,0.]])", "call": "logistic_completeness_fit(r)", "gold_call": "_oracle_logistic_completeness_fit(r)"},
    ]
