"""
Return the coefficients of the conventional three-pole Pade approximants of the kinetic response, the ones obtained by matching its asymptotic power series rather than the plasma's own roots.

The conventional three-moment closures fix their Pade coefficients once and for all by matching the adiabatic and fluid expansions of the kinetic response, trading orders in one limit against orders in the other. The members of that family differ only in how the available matching conditions are shared between the two limits.

Returns
-------
np.ndarray of shape (3,), complex: the numerator, linear denominator and quadratic denominator coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def build_asymptotic_pade_coefficients(label: str) -> np.ndarray:
    """Return the Pade coefficients of one conventional asymptotic closure.

    Three members are supported. ``"R30"`` matches the adiabatic expansion of
    the kinetic response through third order and the fluid expansion at leading
    order only. ``"R31"`` gives up one adiabatic order to gain one fluid order.
    ``"HP"`` gives up a second adiabatic order to gain a second fluid order and
    is the Hammett-Perkins closure.

    Parameters
    ----------
    label : str
        One of ``"R30"``, ``"R31"`` or ``"HP"``.

    Returns
    -------
    pade_coefficients : np.ndarray
        Complex array of shape (3,) holding, in order, the numerator
        coefficient, the linear denominator coefficient and the quadratic
        denominator coefficient, in the same convention as sub-problem 03.

    Raises
    ------
    ValueError
        If ``label`` is not one of the three supported strings, including when
        it is not a string at all. A lookup that propagates a ``KeyError`` or a
        ``TypeError`` instead does not satisfy this contract.
    """
    return pade_coefficients  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_asymptotic_pade_coefficients(label: str) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    if not isinstance(label, str):
        raise ValueError("label must be one of 'R30', 'R31' or 'HP'")

    root_pi = np.sqrt(np.pi)
    if label == "R30":
        # Three adiabatic orders matched; the quadratic coefficient is free and
        # is not minus two, so this member alone keeps an electrostatic term in
        # the heat flux.
        numerator = -1j * root_pi * (np.pi - 3.0) / (4.0 - np.pi)
        linear = -1j * root_pi / (4.0 - np.pi)
        quadratic = -(3.0 * np.pi - 8.0) / (4.0 - np.pi) + 0.0j
    elif label == "R31":
        # One adiabatic order traded for one fluid order, which forces the
        # quadratic coefficient to minus two.
        numerator = -1j * (4.0 - np.pi) / root_pi
        linear = -4.0j / root_pi
        quadratic = -2.0 + 0.0j
    elif label == "HP":
        # A second adiabatic order traded for a second fluid order; this is the
        # Hammett-Perkins member.
        numerator = -1j * root_pi / 2.0
        linear = -3.0j * root_pi / 2.0
        quadratic = -2.0 + 0.0j
    else:
        raise ValueError("label must be one of 'R30', 'R31' or 'HP'")

    return np.array([numerator, linear, quadratic], dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the Hammett-Perkins member, the benchmark of this study ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
label = "HP"
""",
            "call": "sig(build_asymptotic_pade_coefficients(label), 1.0)",
            "gold_call": "sig(_oracle_build_asymptotic_pade_coefficients(label), 1.0)",
        },
        # --- Valid: the member that trades one adiabatic order for one fluid
        #     order, which is the family the wave-number-dependent closure uses ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
label = "R31"
""",
            "call": "sig(build_asymptotic_pade_coefficients(label), 1.0)",
            "gold_call": "sig(_oracle_build_asymptotic_pade_coefficients(label), 1.0)",
        },
        # --- Boundary: the member whose quadratic coefficient is not minus two,
        #     the only one that keeps an electrostatic term in the heat flux ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
label = "R30"
""",
            "call": "sig(build_asymptotic_pade_coefficients(label), 1.0)",
            "gold_call": "sig(_oracle_build_asymptotic_pade_coefficients(label), 1.0)",
        },
        # --- Edge: all three members read out together in a fixed label order, so
        #     that a table which is individually plausible but assigned to the
        #     wrong label, or which perturbs a coefficient the single-member cases
        #     happen not to separate, still moves the returned value ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a).ravel()
    v = np.concatenate([a.real, a.imag]) if np.iscomplexobj(a) else np.asarray(a, dtype=float)
    k = np.arange(v.size, dtype=float) + 1.0
    return round(float((np.abs(v).sum() + v @ np.cos(k)) / s), 9)
def table():
    return np.concatenate([np.asarray(build_asymptotic_pade_coefficients(name))
                           for name in ("R30", "R31", "HP")])
def gold_table():
    return np.concatenate([np.asarray(_oracle_build_asymptotic_pade_coefficients(name))
                           for name in ("R30", "R31", "HP")])
""",
            "call": "sig(table(), 1.0)",
            "gold_call": "sig(gold_table(), 1.0)",
        },
        # --- Invalid: an unrecognised member ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_asymptotic_pade_coefficients("R32")
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_asymptotic_pade_coefficients("R32")
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a non-string label ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        build_asymptotic_pade_coefficients(31)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_build_asymptotic_pade_coefficients(31)
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
