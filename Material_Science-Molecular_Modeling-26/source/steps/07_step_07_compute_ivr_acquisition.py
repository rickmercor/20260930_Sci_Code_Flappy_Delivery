"""
Score one candidate or a batch of candidate window centres by their individual reductions of the posterior variance of the free-energy integral, allowing heteroscedastic observation noise.

The integral-variance-reduction score is the amount by which the posterior variance of the free-energy integral over the whole range would fall if one further umbrella window were run at a candidate location. It is evaluated marginally: every candidate is scored as a single additional window conditioned on the same existing design, never as a joint batch, so scoring a menu of candidates is not the same as scoring the set of them together. Recomputing the entire posterior separately for each candidate would reproduce the definition but is not necessary; the drop admits a closed-form update built from quantities the existing design already fixes, which is what makes a hundred-node menu affordable inside an acquisition loop.




What a candidate offers is a restrained simulation, not a noiseless reading of the latent gradient. The prospective window would report a mean force carrying the candidate observation-noise variance declared for it, and the reduction to be scored is the one that such a measurement buys; the windows already run enter through their own observation-noise variances in the same way.




Nothing in the score depends on the measured force values, only on where the windows sit and how precisely each is or would be measured. With a stationary covariance this criterion therefore produces a purely geometric, value-blind design: it fills the widest gaps in the collective-variable range first and cannot by itself be steered towards any particular region of the landscape, which is the limitation the combined acquisition later exists to remove.

Returns
-------
float or np.ndarray: one IVR score for a scalar candidate, otherwise one score per candidate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_ivr_acquisition(centers: np.ndarray, candidate: np.ndarray,
                            lower: float = 0.0, upper: float = 288.0,
                            variance: float = 0.25, lengthscale: float = 20.0,
                            noise: np.ndarray = 1.0e-3,
                            candidate_noise: np.ndarray = None) -> np.ndarray:
    """Score one candidate or a batch by integral-variance reduction.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n,) holding the collective-variable values of the
        windows run so far, all inside [lower, upper], with n >= 1.
    candidate : float or np.ndarray
        Scalar candidate or one-dimensional array of candidates inside
        [lower, upper]. Each candidate is scored as one additional observation
        conditioned on the same existing design; the result is not a joint
        batch-observation score.
    lower : float
        Lower end of the collective-variable range.
    upper : float
        Upper end of the collective-variable range (upper > lower).
    variance : float
        Process variance of the covariance function (variance > 0).
    lengthscale : float
        Lengthscale of the covariance function (lengthscale > 0).
    noise : float or np.ndarray
        Non-negative scalar noise variance shared by existing observations or
        an array of shape (n,) giving one variance per existing window.
    candidate_noise : float or np.ndarray, optional
        Non-negative scalar or one-dimensional array broadcast-compatible with
        candidate. If omitted, the scalar value of noise is reused; it must be
        supplied explicitly when noise is heteroscedastic.

    Raises
    ------
    ValueError
        If an input is non-numeric, has an invalid shape, is non-finite, or
        cannot be broadcast as documented; if upper is not greater than lower,
        variance or lengthscale is non-positive, any noise variance is
        negative, a centre or candidate lies outside the range, or the
        observation covariance is singular.

    Returns
    -------
    score : float or np.ndarray
        A native Python float for a scalar candidate, otherwise an array with
        one marginal variance-reduction score per candidate.
    """
    return score  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _is_real_scalar(value) -> bool:
    """Return True for a real numeric scalar, excluding booleans."""
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return False
    return isinstance(value, (int, float, np.integer, np.floating))


def _oracle_compute_ivr_acquisition(centers: np.ndarray, candidate: np.ndarray,
                                    lower: float = 0.0, upper: float = 288.0,
                                    variance: float = 0.25, lengthscale: float = 20.0,
                                    noise: np.ndarray = 1.0e-3,
                                    candidate_noise: np.ndarray = None) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    for name, value in (("lower", lower), ("upper", upper),
                        ("variance", variance), ("lengthscale", lengthscale)):
        if not _is_real_scalar(value):
            raise ValueError(f"{name} must be a real number")
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(upper) <= float(lower):
        raise ValueError("upper must be greater than lower")
    if float(variance) <= 0.0:
        raise ValueError("variance must be > 0")
    if float(lengthscale) <= 0.0:
        raise ValueError("lengthscale must be > 0")
    try:
        centers = np.asarray(centers, dtype=float)
        candidates = np.asarray(candidate, dtype=float)
        noise_array = np.asarray(noise, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("centers, candidate, and noise must contain real numbers")
    if centers.ndim != 1 or centers.size < 1:
        raise ValueError("at least one window is required")
    if candidates.ndim > 1 or candidates.size < 1:
        raise ValueError("candidate must be a scalar or non-empty one-dimensional array")
    if noise_array.ndim > 1:
        raise ValueError("noise must be scalar or one-dimensional")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must contain only finite entries")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("candidate must contain only finite entries")
    if np.any(centers < float(lower)) or np.any(centers > float(upper)):
        raise ValueError("every window centre must lie inside the integration range")
    if np.any(candidates < float(lower)) or np.any(candidates > float(upper)):
        raise ValueError("every candidate must lie inside the integration range")

    noise_was_scalar = noise_array.ndim == 0
    if noise_was_scalar:
        noise_vector = np.full(centers.size, float(noise_array))
    elif noise_array.shape == centers.shape:
        noise_vector = noise_array.astype(float, copy=False)
    else:
        raise ValueError("heteroscedastic noise must have shape (n,)")
    if np.any(~np.isfinite(noise_vector)) or np.any(noise_vector < 0.0):
        raise ValueError("noise variances must be finite and non-negative")

    if candidate_noise is None:
        if not noise_was_scalar:
            raise ValueError("candidate_noise is required with heteroscedastic noise")
        candidate_noise_array = np.asarray(float(noise_array))
    else:
        try:
            candidate_noise_array = np.asarray(candidate_noise, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("candidate_noise must contain real numbers")
        if candidate_noise_array.ndim > 1:
            raise ValueError("candidate_noise must be scalar or one-dimensional")
    try:
        candidate_noise_vector = np.broadcast_to(candidate_noise_array, candidates.shape)
    except ValueError:
        raise ValueError("candidate_noise must be broadcast-compatible with candidate")
    if (np.any(~np.isfinite(candidate_noise_vector)) or
            np.any(candidate_noise_vector < 0.0)):
        raise ValueError("candidate noise variances must be finite and non-negative")

    def _embed(points):
        return float(variance) * float(lengthscale) * (
            2.0 - np.exp(-(points - float(lower)) / float(lengthscale))
            - np.exp(-(float(upper) - points) / float(lengthscale)))

    embedding = _embed(centers)
    flat_candidates = candidates.reshape(-1)
    cross = float(variance) * np.exp(
        -np.abs(flat_candidates[:, None] - centers[None, :]) / float(lengthscale))
    gram = float(variance) * np.exp(-np.abs(centers[:, None] - centers[None, :]) / float(lengthscale))
    gram = gram + np.diag(noise_vector)

    try:
        solved_embedding = np.linalg.solve(gram, embedding)
        solved_cross = np.linalg.solve(gram, cross.T)
    except np.linalg.LinAlgError:
        raise ValueError("the covariance matrix of the observations is singular")

    numerator = _embed(flat_candidates) - cross @ solved_embedding
    candidate_noise_flat = np.broadcast_to(
        candidate_noise_vector, candidates.shape).reshape(-1)
    denominator = (float(variance) + candidate_noise_flat
                   - np.einsum("ij,ji->i", cross, solved_cross))
    scores = np.zeros_like(numerator)
    safe = denominator > 0.0
    scores[safe] = numerator[safe] ** 2 / denominator[safe]
    scores = scores.reshape(candidates.shape)
    if scores.ndim == 0:
        return float(scores)
    return scores

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: a candidate in the middle of the unsampled gap (normal scenario) ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
candidate = 142.545454545454547
""",
            "call": "compute_ivr_acquisition(centers, candidate)",
            "gold_call": "_oracle_compute_ivr_acquisition(centers, candidate)",
        },
        # --- Valid: a candidate close to an existing window, where the score collapses ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
candidate = 2.9090909090909092
""",
            "call": "compute_ivr_acquisition(centers, candidate)",
            "gold_call": "_oracle_compute_ivr_acquisition(centers, candidate)",
        },
        # --- Valid: the same candidate scored against a denser design ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 72.727272727272734, 142.545454545454547,
                    212.363636363636374, 284.16, 285.0])
candidate = 107.636363636363640
""",
            "call": "compute_ivr_acquisition(centers, candidate)",
            "gold_call": "_oracle_compute_ivr_acquisition(centers, candidate)",
        },
        # --- Boundary: the candidate placed exactly on an existing window ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16, 285.0])
candidate = 3.0
""",
            "call": "compute_ivr_acquisition(centers, candidate)",
            "gold_call": "_oracle_compute_ivr_acquisition(centers, candidate)",
        },
        # --- Edge: a single existing window and a candidate at the far end of the range ---
        {
            "setup": """import numpy as np
centers = np.array([144.0])
candidate = 288.0
""",
            "call": "compute_ivr_acquisition(centers, candidate)",
            "gold_call": "_oracle_compute_ivr_acquisition(centers, candidate)",
        },
        # --- Valid batch: score a nonuniform menu in one conditioned solve ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 142.5, 284.16, 285.0])
candidates = np.array([0.0, 20.0, 72.0, 142.5, 216.0, 288.0])
""",
            "call": "compute_ivr_acquisition(centers, candidates)",
            "gold_call": "_oracle_compute_ivr_acquisition(centers, candidates)",
        },
        # --- Valid batch: heteroscedastic existing and candidate noise ---
        {
            "setup": """import numpy as np
centers = np.array([3.0, 72.0, 144.0, 216.0, 285.0])
candidates = np.array([36.0, 108.0, 180.0, 252.0])
noise = np.array([1.0e-5, 4.0e-4, 2.0e-3, 7.0e-4, 3.0e-5])
candidate_noise = np.array([2.0e-4, 8.0e-4, 3.0e-3, 5.0e-5])
""",
            "call": "compute_ivr_acquisition(centers, candidates, 0.0, 288.0, 0.25, 20.0, noise, candidate_noise)",
            "gold_call": "_oracle_compute_ivr_acquisition(centers, candidates, 0.0, 288.0, 0.25, 20.0, noise, candidate_noise)",
        },
        # --- Boundary batch: scalar candidate noise broadcast over the menu ---
        {
            "setup": """import numpy as np
centers = np.array([0.0, 96.0, 192.0, 288.0])
candidates = np.array([0.0, 48.0, 144.0, 240.0, 288.0])
noise = np.array([0.0, 1.0e-3, 2.0e-3, 4.0e-3])
""",
            "call": "compute_ivr_acquisition(centers, candidates, 0.0, 288.0, 0.25, 20.0, noise, 5.0e-4)",
            "gold_call": "_oracle_compute_ivr_acquisition(centers, candidates, 0.0, 288.0, 0.25, 20.0, noise, 5.0e-4)",
        },
        # --- Invalid: a candidate outside the integration range ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
def run_model():
    try:
        compute_ivr_acquisition(centers, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_ivr_acquisition(centers, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative white-noise variance ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0])
def run_model():
    try:
        compute_ivr_acquisition(centers, 100.0, 0.0, 288.0, 0.25, 20.0, -1.0e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_ivr_acquisition(centers, 100.0, 0.0, 288.0, 0.25, 20.0, -1.0e-6)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: heteroscedastic training noise without candidate noise ---
        {
            "setup": """import numpy as np
centers = np.array([1.6, 3.0, 284.16])
candidates = np.array([72.0, 144.0])
noise = np.array([1.0e-4, 2.0e-4, 3.0e-4])
def run_model():
    try:
        compute_ivr_acquisition(centers, candidates, 0.0, 288.0, 0.25, 20.0, noise)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_ivr_acquisition(centers, candidates, 0.0, 288.0, 0.25, 20.0, noise)
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
