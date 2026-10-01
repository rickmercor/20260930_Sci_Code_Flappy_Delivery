"""
Compute this iteration's adaptive step size from the paper's auxiliary sequence and advance that sequence to the next iteration, per the paper's optimal step-size rule.

The paper's method uses a dynamically shrinking step size eta_k obtained by minimizing a one-step error recursion; the result is an explicit rule that maps the current value beta_k of a deterministic auxiliary sequence to eta_k, using the relaxation parameter alpha, the error-bound constant gamma and the spectral quantity sigma_max(T) of Step 2, and then updates beta_k to beta_{k+1} in a way that telescopes the recursion so that beta_k decreases strictly to zero (Theorem 2.1, eq. (7)). Early on this rule behaves like a constant step tied to sigma_max(T); late on it decays like 1/k (Remark 2.5). Consult eq. (7) of the paper for the exact step-size expression and the exact auxiliary update.

Returns
-------
tuple (float, float) — (eta_k, beta_{k+1}), the step size for this iteration and the next auxiliary value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def aabk_adaptive_step(beta_k: float, alpha: float, gamma: float,
                       sigma_max_T: float) -> tuple:
    '''Compute the adaptive step size eta_k and the next auxiliary value
    beta_{k+1} per eq. (7) of the paper.

    Parameters
    ----------
    beta_k : float
        Current value of the paper's auxiliary sequence, >= 0.
    alpha : float
        Relaxation parameter of the paper's coupling identity, > 0.
    gamma : float
        The paper's error-bound constant (Assumption 3.2), > 0.
    sigma_max_T : float
        Largest eigenvalue of the paper's matrix T (from
        aabk_spectral_bound), > 0.

    Returns
    -------
    result : tuple of (float, float)
        (eta_k, beta_next): the step size to use at this iteration and the
        auxiliary value for the next iteration, both as defined in eq. (7)
        of the paper (not restated here), as native Python floats.

    Raises
    ------
    ValueError
        If beta_k is negative, or if alpha, gamma or sigma_max_T is not
        strictly positive.
    '''
    return eta_k, beta_next  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_aabk_adaptive_step(beta_k: float, alpha: float, gamma: float,
                               sigma_max_T: float) -> tuple:
    if beta_k < 0:
        raise ValueError("beta_k must be nonnegative")
    if not (alpha > 0):
        raise ValueError("alpha must be strictly positive")
    if not (gamma > 0):
        raise ValueError("gamma must be strictly positive")
    if not (sigma_max_T > 0):
        raise ValueError("sigma_max_T must be strictly positive")
    # eq. (7): eta_k = alpha gamma beta_k / (1 + 2 alpha gamma sigma_max(T) beta_k),
    #          beta_{k+1} = beta_k (1 - alpha gamma eta_k / 2).
    eta_k = alpha * gamma * beta_k / (1.0 + 2.0 * alpha * gamma * sigma_max_T * beta_k)
    beta_next = beta_k * (1.0 - alpha * gamma * eta_k / 2.0)
    return float(eta_k), float(beta_next)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: iteration 0 of the exact problem instance ---
        {
            "setup": """import numpy as np
beta_k = 51.73863084908991
alpha = 1.0
gamma = 0.02593873031184615
sigma_max_T = 0.6178735365193508
""",
            "call": "np.array(aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
            "gold_call": "np.array(_oracle_aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
        },
        # --- Normal: iteration 3 of the exact problem instance. ---
        {
            "setup": """import numpy as np
beta_k = 50.73151073805644
alpha = 1.0
gamma = 0.02593873031184615
sigma_max_T = 0.6178735365193508
""",
            "call": "np.array(aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
            "gold_call": "np.array(_oracle_aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
        },
        # --- Edge: very large beta_k with a large gamma (early regime),
        #     where the step should approach the largest admissible constant
        #     step tied to sigma_max(T). ---
        {
            "setup": """import numpy as np
beta_k = 1.0e9
alpha = 0.5
gamma = 2.0
sigma_max_T = 0.25
""",
            "call": "np.array(aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
            "gold_call": "np.array(_oracle_aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
        },
        # --- Edge: a mid-sized beta_k with a large gamma, where both the
        #     step and the auxiliary update move substantially in one
        #     iteration. ---
        {
            "setup": """import numpy as np
beta_k = 0.7088309493721018
alpha = 1.0
gamma = 2.0
sigma_max_T = 0.6178735365193508
""",
            "call": "np.array(aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
            "gold_call": "np.array(_oracle_aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
        },
        # --- Edge: very small beta_k (late regime), where the step becomes
        #     proportional to beta_k. ---
        {
            "setup": """import numpy as np
beta_k = 1.0e-6
alpha = 2.0
gamma = 0.5
sigma_max_T = 0.8
""",
            "call": "np.array(aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
            "gold_call": "np.array(_oracle_aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
        },
        # --- Boundary: beta_k = 0 gives a zero step and keeps beta at 0. ---
        {
            "setup": """import numpy as np
beta_k = 0.0
alpha = 1.0
gamma = 1.0
sigma_max_T = 0.5
""",
            "call": "np.array(aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
            "gold_call": "np.array(_oracle_aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T))",
        },
        # --- Invalid: negative beta_k -> ValueError ---
        {
            "setup": """import numpy as np
beta_k = -1.0
alpha = 1.0
gamma = 1.0
sigma_max_T = 0.5
def run_model():
    try:
        aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: non-positive gamma -> ValueError ---
        {
            "setup": """import numpy as np
beta_k = 1.0
alpha = 1.0
gamma = 0.0
sigma_max_T = 0.5
def run_model():
    try:
        aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_aabk_adaptive_step(beta_k, alpha, gamma, sigma_max_T)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
