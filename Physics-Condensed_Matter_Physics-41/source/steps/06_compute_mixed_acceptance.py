"""
Compute the mixed-proposal Metropolis acceptance probability using the proposal correction and the anharmonic residual potential.

Harmonic Trotter splitting places the quadratic reference action in the proposal, so the Metropolis exponent contains the change of the shifted cubic-quartic residual together with the generalized Hastings correction. Derive and evaluate the residual change as one paired, scale-aware quantity: the two residual values, or the unweighted cubic and quartic powers, may individually exceed binary64 range even when their weighted difference and the final acceptance probability are finite. Combine the result with the Hastings factor in log space and cap the probability at one before exponentiation.

Returns
-------
float, the mixed-proposal acceptance probability in the closed interval [0, 1]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import math

from numbers import Real

import numpy as np


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def compute_mixed_acceptance(
    old: float,
    trial: float,
    tau: float,
    minimum: float,
    curvature: float,
    alpha_3: float,
    alpha_4: float,
    hastings_factor: float,
) -> float:
    """Compute the harmonic-splitting mixed-proposal acceptance probability.

    Parameters
    ----------
    old, trial : float
        Current and proposed bead positions.
    tau : float
        Positive imaginary-time step.
    minimum : float
        Position of the local minimum.
    curvature : float
        Positive harmonic curvature.
    alpha_3, alpha_4 : float
        Finite cubic and quartic coefficients of the shifted residual.
    hastings_factor : float
        Positive finite generalized proposal correction.

    Returns
    -------
    acceptance : float
        Acceptance probability in the closed interval ``[0, 1]``. The
        residual-potential change must be evaluated directly in a
        scale-aware paired form whenever separate residuals or separate
        unweighted powers are not representable although the exact
        weighted difference is finite. The untruncated ratio must be
        handled in log space.

    Raises
    ------
    ValueError
        If any scalar input is not a finite real value; if ``tau``,
        ``curvature``, or ``hastings_factor`` is not strictly positive; if
        the finite residual-action difference cannot be represented; or
        if the resulting probability is non-finite.
    """
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_mixed_acceptance(
    old: float,
    trial: float,
    tau: float,
    minimum: float,
    curvature: float,
    alpha_3: float,
    alpha_4: float,
    hastings_factor: float,
) -> float:
    """Reference implementation with a scale-safe residual difference."""
    import math
    import numbers
    from decimal import Decimal, InvalidOperation, Overflow, localcontext

    import numpy as np

    named_values = (
        ("old", old),
        ("trial", trial),
        ("tau", tau),
        ("minimum", minimum),
        ("curvature", curvature),
        ("alpha_3", alpha_3),
        ("alpha_4", alpha_4),
        ("hastings_factor", hastings_factor),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    old_f = converted["old"]
    trial_f = converted["trial"]
    tau_f = converted["tau"]
    minimum_f = converted["minimum"]
    curvature_f = converted["curvature"]
    alpha_3_f = converted["alpha_3"]
    alpha_4_f = converted["alpha_4"]
    factor_f = converted["hastings_factor"]
    if tau_f <= 0.0 or curvature_f <= 0.0 or factor_f <= 0.0:
        raise ValueError("tau, curvature, and hastings_factor must be > 0")

    q_trial = trial_f - minimum_f
    q_old = old_f - minimum_f

    # Preserve the ordinary binary64 route for the task instance and other
    # moderate regimes, then fall back only when separate powers or
    # residuals overflow or cancel to a non-finite value.
    try:
        residual_trial = 0.5 * curvature_f * (
            alpha_3_f * q_trial**3 + alpha_4_f * q_trial**4
        )
        residual_old = 0.5 * curvature_f * (
            alpha_3_f * q_old**3 + alpha_4_f * q_old**4
        )
        delta_residual = residual_trial - residual_old
        if not math.isfinite(delta_residual):
            raise ArithmeticError("non-finite separately evaluated residual change")
    except (ArithmeticError, OverflowError):
        try:
            with localcontext() as context:
                context.prec = 120
                q_trial_d = Decimal.from_float(trial_f) - Decimal.from_float(
                    minimum_f
                )
                q_old_d = Decimal.from_float(old_f) - Decimal.from_float(minimum_f)
                delta_d = (
                    Decimal("0.5")
                    * Decimal.from_float(curvature_f)
                    * (
                        Decimal.from_float(alpha_3_f)
                        * (q_trial_d**3 - q_old_d**3)
                        + Decimal.from_float(alpha_4_f)
                        * (q_trial_d**4 - q_old_d**4)
                    )
                )
                delta_residual = float(delta_d)
        except (InvalidOperation, Overflow, OverflowError) as exc:
            raise ValueError("residual-potential change must be finite") from exc
        if not math.isfinite(delta_residual):
            raise ValueError("residual-potential change must be finite")

    log_ratio = math.log(factor_f) - tau_f * delta_residual
    if math.isnan(log_ratio):
        raise ValueError("acceptance log-ratio must not be NaN")
    if log_ratio >= 0.0:
        return 1.0
    try:
        acceptance = math.exp(log_ratio)
    except OverflowError as exc:
        raise ValueError("acceptance probability must be finite") from exc
    if not math.isfinite(acceptance):
        raise ValueError("acceptance probability must be finite")
    return float(acceptance)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, saturation, cancellation, and invalid specifications."""
    return [
        {
            "setup": """old = 0.26\ntrial = 0.2948678088618854\ntau = 0.6\nminimum = -0.15\ncurvature = 3.2\nalpha_3 = 0.3125\nalpha_4 = 12.5\nhastings_factor = 1.0275693891240696""",
            "call": "compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
            "gold_call": "_oracle_compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
        },
        {
            "setup": """old = 0.03\ntrial = 0.187003474188357\ntau = 0.6\nminimum = -0.15\ncurvature = 3.2\nalpha_3 = 0.3125\nalpha_4 = 12.5\nhastings_factor = 1.0""",
            "call": "compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
            "gold_call": "_oracle_compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
        },
        {
            "setup": """old = -0.6\ntrial = 0.28489603528975127\ntau = 0.6\nminimum = -0.15\ncurvature = 3.2\nalpha_3 = 0.3125\nalpha_4 = 12.5\nhastings_factor = 0.9374551596534505""",
            "call": "compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
            "gold_call": "_oracle_compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
        },
        {
            "setup": """old = 2.0\ntrial = 0.0\ntau = 1.0\nminimum = 0.0\ncurvature = 1.0\nalpha_3 = 0.0\nalpha_4 = 1.0\nhastings_factor = 1.0""",
            "call": "compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
            "gold_call": "_oracle_compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
        },
        {
            "setup": """old = 0.0\ntrial = 0.1\ntau = 1.0\nminimum = 0.0\ncurvature = 1.0\nalpha_3 = 0.0\nalpha_4 = 1.0\nhastings_factor = 0.0\ndef run_model():\n    try:\n        compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """old = 1.0e80\ntrial = -1.0e80\ntau = 1.0\nminimum = 0.0\ncurvature = 2.0\nalpha_3 = 1.0e-240\nalpha_4 = 1.0\nhastings_factor = 0.049787068367863944""",
            "call": "compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
            "gold_call": "_oracle_compute_mixed_acceptance(old, trial, tau, minimum, curvature, alpha_3, alpha_4, hastings_factor)",
        },
    ]
