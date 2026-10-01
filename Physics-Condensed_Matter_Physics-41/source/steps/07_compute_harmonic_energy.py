"""
Evaluate the finite-bead harmonic-reference total-energy estimator on a periodic one-dimensional worldline.

Evaluate the thermodynamic finite-$P$ total-energy estimator associated with harmonic Trotter splitting, not the primitive free-particle estimator. Every periodic bond contributes an exact harmonic-link term and every bead contributes the shifted cubic-quartic residual. Use a stable hyperbolic representation at both small and large imaginary-time arguments. Also evaluate the weighted residual polynomial in a scale-aware order: unweighted $q^3$ or $q^4$ and the separate cubic and quartic contributions may exceed binary64 range even when their mass-weighted sum, including a near-cancellation, is finite.

Returns
-------
float, the finite-P harmonic-reference total-energy estimate as a native Python float
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


def compute_harmonic_energy(
    path: np.ndarray,
    beta: float,
    mass: float,
    omega: float,
    hbar: float,
    minimum: float,
    alpha_3: float,
    alpha_4: float,
) -> float:
    """Evaluate the finite-P harmonic-reference total-energy estimator.

    Parameters
    ----------
    path : numpy.ndarray
        One-dimensional periodic path with at least two finite bead
        positions.
    beta : float
        Positive inverse temperature.
    mass : float
        Positive particle mass.
    omega : float
        Positive harmonic reference frequency.
    hbar : float
        Positive reduced Planck constant.
    minimum : float
        Position of the local minimum.
    alpha_3, alpha_4 : float
        Finite coefficients of the shifted anharmonic residual.

    Returns
    -------
    energy : float
        Finite harmonic-reference total-energy estimate as a native
        Python float. Periodic closure, stable small- and large-argument
        hyperbolic limits, and scale-aware evaluation of the mass-weighted
        cubic-quartic residual are required. The result must remain finite
        whenever the exact weighted link and residual terms are finite,
        even if unweighted powers or separate polynomial contributions are
        not representable in binary64.

    Raises
    ------
    ValueError
        If ``path`` cannot be represented as a finite one-dimensional
        float array with at least two beads; if any scalar input is not a
        finite real value; if ``beta``, ``mass``, ``omega``, or ``hbar``
        is not strictly positive; or if the resulting energy is non-finite.
    """
    return float("nan")

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_harmonic_energy(
    path: np.ndarray,
    beta: float,
    mass: float,
    omega: float,
    hbar: float,
    minimum: float,
    alpha_3: float,
    alpha_4: float,
) -> float:
    """Reference implementation with stable link and residual evaluation."""
    import math
    import numbers
    from decimal import Decimal, InvalidOperation, Overflow, localcontext

    import numpy as np

    try:
        path_array = np.asarray(path, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("path must be convertible to a finite float array") from exc
    if path_array.ndim != 1 or path_array.size < 2:
        raise ValueError("path must be one-dimensional with at least two beads")
    if not np.all(np.isfinite(path_array)):
        raise ValueError("path must contain only finite values")

    named_values = (
        ("beta", beta),
        ("mass", mass),
        ("omega", omega),
        ("hbar", hbar),
        ("minimum", minimum),
        ("alpha_3", alpha_3),
        ("alpha_4", alpha_4),
    )
    converted: dict[str, float] = {}
    for name, value in named_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be a finite real scalar")
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"{name} must be finite")
        converted[name] = value_f

    beta_f = converted["beta"]
    mass_f = converted["mass"]
    omega_f = converted["omega"]
    hbar_f = converted["hbar"]
    minimum_f = converted["minimum"]
    alpha_3_f = converted["alpha_3"]
    alpha_4_f = converted["alpha_4"]
    if beta_f <= 0.0 or mass_f <= 0.0 or omega_f <= 0.0 or hbar_f <= 0.0:
        raise ValueError("beta, mass, omega, and hbar must be > 0")

    bead_count = int(path_array.size)
    a = beta_f * hbar_f * omega_f / bead_count
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError("beta*hbar*omega/P must be finite and > 0")

    tanh_a = math.tanh(a)
    if not math.isfinite(tanh_a) or tanh_a <= 0.0:
        raise ValueError("tanh(beta*hbar*omega/P) must be finite and > 0")

    if a < 350.0:
        sinh_a = math.sinh(a)
        cosh_a = math.cosh(a)
        inv_sinh = 1.0 / sinh_a
        inv_cosh_plus_one = 1.0 / (cosh_a + 1.0)
    else:
        # Both inverse factors are naturally zero once exp(-a) underflows.
        exp_minus_a = math.exp(-a)
        if exp_minus_a == 0.0:
            inv_sinh = 0.0
            inv_cosh_plus_one = 0.0
        else:
            inv_sinh = 2.0 * exp_minus_a / (1.0 - exp_minus_a * exp_minus_a)
            inv_cosh_plus_one = 2.0 * exp_minus_a / (
                (1.0 + exp_minus_a) * (1.0 + exp_minus_a)
            )

    q = path_array - minimum_f
    q_next = np.roll(q, -1)
    constant = hbar_f * omega_f / (2.0 * bead_count * tanh_a)
    link_scale = mass_f * omega_f * omega_f / (2.0 * bead_count)
    with np.errstate(over="ignore", invalid="ignore"):
        link = link_scale * (
            -(q - q_next) ** 2 * (inv_sinh * inv_sinh)
            + 2.0 * q * q_next * inv_cosh_plus_one
        )
    if not np.all(np.isfinite(link)):
        raise ValueError("harmonic-link contributions must be finite")

    # Preserve the original binary64 route in ordinary regimes. If the
    # unweighted powers overflow or separately weighted contributions
    # cancel through non-finite intermediates, recompute only the residual
    # in extended decimal range and convert the final weighted bead terms.
    with np.errstate(over="ignore", invalid="ignore"):
        residual = (
            0.5
            * mass_f
            * omega_f
            * omega_f
            * (alpha_3_f * q**3 + alpha_4_f * q**4)
            / bead_count
        )

    if not np.all(np.isfinite(residual)):
        residual_values: list[float] = []
        try:
            with localcontext() as context:
                context.prec = 150
                coefficient = (
                    Decimal("0.5")
                    * Decimal.from_float(mass_f)
                    * Decimal.from_float(omega_f)
                    * Decimal.from_float(omega_f)
                    / Decimal(bead_count)
                )
                alpha_3_d = Decimal.from_float(alpha_3_f)
                alpha_4_d = Decimal.from_float(alpha_4_f)
                minimum_d = Decimal.from_float(minimum_f)
                for position in path_array.tolist():
                    q_d = Decimal.from_float(float(position)) - minimum_d
                    term_d = coefficient * (
                        alpha_3_d * q_d**3 + alpha_4_d * q_d**4
                    )
                    residual_values.append(float(term_d))
        except (InvalidOperation, Overflow, OverflowError) as exc:
            raise ValueError("weighted residual contributions must be finite") from exc
        residual = np.asarray(residual_values, dtype=float)
        if not np.all(np.isfinite(residual)):
            raise ValueError("weighted residual contributions must be finite")

    energy = float(np.sum(constant + link + residual, dtype=float))
    if not math.isfinite(energy):
        raise ValueError("energy estimate must be finite")
    return energy

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, hyperbolic, scale-separated, cancellation, and invalid cases."""
    return [
        {
            "setup": """import numpy as np\npath = np.array([0.08, 0.24, 0.22, 0.26, 0.187003474188357, -0.12, -0.60, -0.05], dtype=float)\nbeta = 4.8\nmass = 1.3\nomega = (3.2 / 1.3) ** 0.5\nhbar = 0.7\nminimum = -0.15\nalpha_3 = 0.3125\nalpha_4 = 12.5""",
            "call": "compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
            "gold_call": "_oracle_compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
        },
        {
            "setup": """import numpy as np\npath = np.ones(4, dtype=float)\nbeta = 4.0e-8\nmass = 1.0\nomega = 1.0\nhbar = 1.0\nminimum = 0.0\nalpha_3 = 0.0\nalpha_4 = 0.0""",
            "call": "compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
            "gold_call": "_oracle_compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
        },
        {
            "setup": """import numpy as np\npath = np.zeros(4, dtype=float)\nbeta = 3200.0\nmass = 1.0\nomega = 1.0\nhbar = 1.0\nminimum = 0.0\nalpha_3 = 0.0\nalpha_4 = 0.0""",
            "call": "compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
            "gold_call": "_oracle_compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
        },
        {
            "setup": """import numpy as np\npath = np.array([0.0], dtype=float)\nbeta = 1.0\nmass = 1.0\nomega = 1.0\nhbar = 1.0\nminimum = 0.0\nalpha_3 = 0.0\nalpha_4 = 0.0\ndef run_model():\n    try:\n        compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np\npath = np.array([0.0, 0.1], dtype=float)\nbeta = -1.0\nmass = 1.0\nomega = 1.0\nhbar = 1.0\nminimum = 0.0\nalpha_3 = 0.0\nalpha_4 = 0.0\ndef run_model():\n    try:\n        compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_gold():\n    try:\n        _oracle_compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np\npath = np.full(4, 1.0e80, dtype=float)\nbeta = 4.0\nmass = 1.0e-160\nomega = 1.0\nhbar = 1.0\nminimum = 0.0\nalpha_3 = 0.0\nalpha_4 = 1.0e-160""",
            "call": "compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
            "gold_call": "_oracle_compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
        },
        {
            "setup": """import numpy as np\npath = np.full(4, 1.0e80, dtype=float)\nbeta = 3200.0\nmass = 1.0e-144\nomega = 1.0\nhbar = 1.0\nminimum = 0.0\nalpha_3 = float(np.nextafter(-2.0e-80, 0.0))\nalpha_4 = 2.0e-160""",
            "call": "compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
            "gold_call": "_oracle_compute_harmonic_energy(path, beta, mass, omega, hbar, minimum, alpha_3, alpha_4)",
        },
    ]
