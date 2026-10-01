"""
Compute the auxiliary log mass, the exactly marginalized evaluation log mass, and the first two exponent-response fields of the retained stay-or-displace mixture without forming exponentially large or small masses.

The auxiliary chain mass combines each retained row mass with an amplitude power. Marginalizing the evaluation kernel redistributes that mass through a direct stay term and incoming retained connections without changing its normalization. Differentiating this positive mixture with respect to the amplitude exponent gives a score and a second-derivative ratio that can be evaluated as log-sum-exp weighted moments of log amplitudes.

Returns
-------
tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], containing log_auxiliary, log_evaluation, exponent_score, and exponent_second_ratio, each with shape (n_states,) in binary64.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def _logsumexp(values: np.ndarray) -> float:
    """Return a stable log-sum-exp of a nonempty finite one-dimensional array."""
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum), dtype=float)))

def compute_ir_log_fields(
    retained_magnitudes: np.ndarray,
    log_abs_psi: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute log-domain auxiliary and evaluation fields.

    Parameters
    ----------
    retained_magnitudes : np.ndarray
        Symmetric nonnegative retained magnitude matrix ``W`` with zero diagonal and
        strictly positive row sums.
    log_abs_psi : np.ndarray
        Finite logarithms of nonzero wave-function magnitudes.
    alpha : float
        Finite auxiliary amplitude exponent in ``[0, 2]``.
    beta : float
        Finite evaluation displacement probability in ``[0, 1)``.

    Returns
    -------
    log_auxiliary : np.ndarray
        Unnormalized auxiliary log masses.
    log_evaluation : np.ndarray
        Exactly marginalized unnormalized evaluation log masses.
    exponent_score : np.ndarray
        First derivative of ``log_evaluation`` with respect to ``alpha``.
    exponent_second_ratio : np.ndarray
        Ratio of the second derivative of the unnormalized evaluation mass to that mass.

    Raises
    ------
    ValueError
        If any input is invalid or incompatible.
    """
    return (
        np.empty_like(np.asarray(log_abs_psi), dtype=float),
        np.empty_like(np.asarray(log_abs_psi), dtype=float),
        np.empty_like(np.asarray(log_abs_psi), dtype=float),
        np.empty_like(np.asarray(log_abs_psi), dtype=float),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_ir_log_fields(
    retained_magnitudes: np.ndarray,
    log_abs_psi: np.ndarray,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference implementation."""
    import numpy as np

    def lse(values: np.ndarray) -> float:
        maximum = float(np.max(values))
        return maximum + float(np.log(np.sum(np.exp(values - maximum), dtype=float)))

    W = np.asarray(retained_magnitudes, dtype=float)
    logs = np.asarray(log_abs_psi, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] < 2:
        raise ValueError("retained_magnitudes must be square with at least two states")
    if not np.all(np.isfinite(W)) or np.any(W < 0.0):
        raise ValueError("retained_magnitudes must be finite and nonnegative")
    if not np.allclose(W, W.T, rtol=0.0, atol=1e-12):
        raise ValueError("retained_magnitudes must be symmetric")
    if not np.allclose(np.diag(W), 0.0, rtol=0.0, atol=0.0):
        raise ValueError("retained_magnitudes must have zero diagonal")
    row_mass = np.sum(W, axis=1, dtype=float)
    if np.any(row_mass <= 0.0):
        raise ValueError("each retained row mass must be positive")
    if logs.ndim != 1 or logs.shape != (W.shape[0],) or not np.all(np.isfinite(logs)):
        raise ValueError("log_abs_psi must be a finite vector matching the state count")
    try:
        exponent = float(alpha)
        displacement = float(beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha and beta must be finite scalars") from exc
    if not np.isfinite(exponent) or exponent < 0.0 or exponent > 2.0:
        raise ValueError("alpha must lie in [0, 2]")
    if not np.isfinite(displacement) or displacement < 0.0 or displacement >= 1.0:
        raise ValueError("beta must lie in [0, 1)")

    log_auxiliary = np.log(row_mass) + exponent * logs
    n_states = W.shape[0]
    log_evaluation = np.empty(n_states, dtype=float)
    score = np.empty(n_states, dtype=float)
    second_ratio = np.empty(n_states, dtype=float)
    for y in range(n_states):
        term_logs: list[float] = []
        term_amplitude_logs: list[float] = []
        term_logs.append(float(np.log1p(-displacement) + np.log(row_mass[y]) + exponent * logs[y]))
        term_amplitude_logs.append(float(logs[y]))
        if displacement > 0.0:
            for x in np.flatnonzero(W[:, y] > 0.0):
                term_logs.append(float(np.log(displacement) + np.log(W[x, y]) + exponent * logs[x]))
                term_amplitude_logs.append(float(logs[x]))
        values = np.asarray(term_logs, dtype=float)
        amplitudes = np.asarray(term_amplitude_logs, dtype=float)
        log_total = lse(values)
        mixture_weights = np.exp(values - log_total)
        log_evaluation[y] = log_total
        score[y] = float(np.dot(mixture_weights, amplitudes))
        second_ratio[y] = float(np.dot(mixture_weights, amplitudes * amplitudes))
    return (
        log_auxiliary.astype(float),
        log_evaluation.astype(float),
        score.astype(float),
        second_ratio.astype(float),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return deterministic unit-test specifications."""
    return [
        {
            "setup": """import numpy as np
W = np.array([[0.0,0.8,0.35],[0.8,0.0,0.55],[0.35,0.55,0.0]], dtype=float)
log_abs_psi = np.log(np.array([0.82,0.37,0.145], dtype=float))
alpha = 1.35
beta = 0.47
""",
            "call": "np.concatenate(compute_ir_log_fields(W, log_abs_psi, alpha, beta))",
            "gold_call": "np.concatenate(_oracle_compute_ir_log_fields(W, log_abs_psi, alpha, beta))",
        },
        {
            "setup": """import numpy as np
W = np.array([[0.0,2.0],[2.0,0.0]], dtype=float)
log_abs_psi = np.array([-2.0,3.0], dtype=float)
alpha = 1.7
beta = 0.0
""",
            "call": "np.concatenate(compute_ir_log_fields(W, log_abs_psi, alpha, beta))",
            "gold_call": "np.concatenate(_oracle_compute_ir_log_fields(W, log_abs_psi, alpha, beta))",
        },
        {
            "setup": """import numpy as np
W = np.array([[0.0,1.0,2.0],[1.0,0.0,3.0],[2.0,3.0,0.0]], dtype=float)
log_abs_psi = np.array([-4.0,0.5,2.5], dtype=float)
alpha = 0.0
beta = 0.6
""",
            "call": "np.concatenate(compute_ir_log_fields(W, log_abs_psi, alpha, beta))",
            "gold_call": "np.concatenate(_oracle_compute_ir_log_fields(W, log_abs_psi, alpha, beta))",
        },
        {
            "setup": """import numpy as np
W = np.array([[0.0,1.0,0.5],[1.0,0.0,0.25],[0.5,0.25,0.0]], dtype=float)
log_abs_psi = np.array([-1200.0,0.0,1200.0], dtype=float)
alpha = 1.9
beta = 0.4
""",
            "call": "np.concatenate(compute_ir_log_fields(W, log_abs_psi, alpha, beta))",
            "gold_call": "np.concatenate(_oracle_compute_ir_log_fields(W, log_abs_psi, alpha, beta))",
        },
        {
            "setup": """import numpy as np
W = np.array([[0.0,1.0],[1.0,0.0]], dtype=float)
log_abs_psi = np.array([0.0,1.0], dtype=float)
def run_model():
    try:
        compute_ir_log_fields(W, log_abs_psi, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_ir_log_fields(W, log_abs_psi, 1.0, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
W = np.array([[0.0,1.0],[0.5,0.0]], dtype=float)
log_abs_psi = np.array([0.0,1.0], dtype=float)
def run_model():
    try:
        compute_ir_log_fields(W, log_abs_psi, 1.0, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_ir_log_fields(W, log_abs_psi, 1.0, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
W = np.array([[0.0,1.0,0.0],[1.0,0.0,0.0],[0.0,0.0,0.0]], dtype=float)
log_abs_psi = np.array([0.0,1.0,2.0], dtype=float)
def run_model():
    try:
        compute_ir_log_fields(W, log_abs_psi, 1.0, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_ir_log_fields(W, log_abs_psi, 1.0, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
W = np.array([[0.0,1.0],[1.0,0.0]], dtype=float)
log_abs_psi = np.array([0.0,np.nan], dtype=float)
def run_model():
    try:
        compute_ir_log_fields(W, log_abs_psi, 1.0, 0.4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_ir_log_fields(W, log_abs_psi, 1.0, 0.4)
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
