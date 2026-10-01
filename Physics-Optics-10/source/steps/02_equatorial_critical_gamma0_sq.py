"""
Compute the critical squared-chirality value at which the medium's equatorial (kz = 0) exceptional-point ring is born, for axial permittivities in the regime -12 < eps_z < -4.

This critical value marks a genuine non-Hermitian spectral transition of the generalized Maxwell eigenproblem, distinct from the ordinary elliptic/hyperbolic (optical Lifshitz) transition of the equal-frequency surface. The construction is physically meaningful only within the -12 < eps_z < -4 window; outside it (including eps_z <= -12) the medium's finite-frequency classification is governed by a different regime entirely and this particular equatorial construction does not apply. Consult the paper's own derivation for the precise algebraic form; do not assume it is a simple rescaling of the eps_z = 4 or eps_z = 0 special points.

Returns
-------
float: the critical value of gamma0^2, always strictly positive on its domain of validity (-12 < eps_z < -4).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def equatorial_critical_gamma0_sq(eps_z: float) -> float:
    """Critical gamma0^2 for the equatorial finite-frequency EP ring.

    Args:
        eps_z (float): axial relative permittivity, must satisfy
            -12 < eps_z < -4 for this critical condition to be physically
            meaningful.

    Raises:
        ValueError: if eps_z <= -12 or eps_z >= -4.

    Expected return:
        float: critical value of gamma0^2, strictly positive.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: equatorial_critical_gamma0_sq
def _oracle_equatorial_critical_gamma0_sq(eps_z: float) -> float:
    if not (-12.0 < eps_z < -4.0):
        raise ValueError("equatorial critical ring only defined for -12 < eps_z < -4")
    return (eps_z - 4.0) ** 2 / (-8.0 * (eps_z + 4.0))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = equatorial_critical_gamma0_sq(-6.0)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_equatorial_critical_gamma0_sq(-6.0), 8)",
            "tol": 1e-6,
        },
        {
            "setup": "v = equatorial_critical_gamma0_sq(-11.0)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_equatorial_critical_gamma0_sq(-11.0), 8)",
            "tol": 1e-6,
        },
        {
            "setup": "v = equatorial_critical_gamma0_sq(-4.5)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_equatorial_critical_gamma0_sq(-4.5), 8)",
            "tol": 1e-6,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        equatorial_critical_gamma0_sq(-4.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_equatorial_critical_gamma0_sq(-4.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        equatorial_critical_gamma0_sq(-12.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_equatorial_critical_gamma0_sq(-12.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        equatorial_critical_gamma0_sq(-15.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_equatorial_critical_gamma0_sq(-15.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        equatorial_critical_gamma0_sq(9.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_equatorial_critical_gamma0_sq(9.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
