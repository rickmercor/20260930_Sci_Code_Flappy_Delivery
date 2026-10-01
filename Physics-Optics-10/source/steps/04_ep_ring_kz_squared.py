"""
Compute the squared axial wavevector component kz^2 on the finite-frequency exceptional-point ring of the gyroelectric-chiral medium, at a prescribed frequency omega, for parameter pairs (eps_z, gamma0) that admit such a ring (see the sector classifier).

This is the paper's own closed-form solution to a generalized eigenvalue coalescence problem; it is not the result of a direct/naive extremization or a textbook exceptional-point formula for an isotropic or two-band system. The exact algebraic combination of eps_z, gamma0, and the square root of gamma0^2 * (gamma0^2 - 4) that appears here is specific to this paper's derivation consult the paper's own equations rather than approximating it.

Returns
-------
float: kz^2 at the finite-frequency EP ring, non-negative when the parameters lie in the finite-frequency EP sector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ep_ring_kz_squared(eps_z: float, gamma0: float, omega: float) -> float:
    """Squared axial wavevector on the finite-frequency EP ring.

    Args:
        eps_z (float): axial relative permittivity of the uniaxial medium.
        gamma0 (float): isotropic chirality parameter, gamma0 >= 0.
        omega (float): frequency, omega > 0.

    Raises:
        ValueError: if omega <= 0, or if gamma0^2 * (gamma0^2 - 4) < 0 (no real
            exceptional locus at this gamma0), or if the construction is
            degenerate for this (eps_z, gamma0) pair.

    Expected return:
        float: kz^2 evaluated at the finite-frequency exceptional-point ring,
        scaling as omega^2.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: ep_ring_kz_squared
def _oracle_ep_ring_kz_squared(eps_z: float, gamma0: float, omega: float) -> float:
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    g0sq = gamma0 ** 2
    disc = g0sq * (g0sq - 4.0)
    if disc < 0.0:
        raise ValueError("no real exceptional locus: gamma0^2 * (gamma0^2 - 4) < 0")
    s = np.sqrt(disc)
    denom = (eps_z - 4.0) * (3.0 * g0sq + 4.0)
    if denom == 0.0:
        raise ValueError("degenerate denominator for this (eps_z, gamma0) pair")
    return (
        omega ** 2
        * (g0sq - 4.0)
        / denom
        * ((g0sq - 4.0) * (eps_z - 4.0 * g0sq - 4.0) + 2.0 * (eps_z + 2.0 * g0sq + 4.0) * s)
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = ep_ring_kz_squared(9.0, 2.5, 1.0)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_ep_ring_kz_squared(9.0, 2.5, 1.0), 8)",
            "tol": 1e-6,
        },
        {
            "setup": "v = ep_ring_kz_squared(9.0, 2.5, 2.0)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_ep_ring_kz_squared(9.0, 2.5, 2.0), 8)",
            "tol": 1e-6,
        },
        {
            "setup": "v = ep_ring_kz_squared(-15.0, 3.0, 1.0)",
            "call": "round(v, 6)",
            "gold_call": "round(_oracle_ep_ring_kz_squared(-15.0, 3.0, 1.0), 6)",
            "tol": 1e-5,
        },
        {
            "setup": "v = ep_ring_kz_squared(-6.0, 2.5, 1.0)",
            "call": "round(v, 6)",
            "gold_call": "round(_oracle_ep_ring_kz_squared(-6.0, 2.5, 1.0), 6)",
            "tol": 1e-5,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        ep_ring_kz_squared(9.0, 2.5, -1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_ep_ring_kz_squared(9.0, 2.5, -1.0)\n"
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
                "        ep_ring_kz_squared(9.0, 1.0, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_ep_ring_kz_squared(9.0, 1.0, 1.0)\n"
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
