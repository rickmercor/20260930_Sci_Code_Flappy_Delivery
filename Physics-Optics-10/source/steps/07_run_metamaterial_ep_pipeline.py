"""
Chain the six earlier steps into the full pipeline: reduce the constitutive tensors to their dispersion coefficients, use those coefficients to gate physical admissibility, classify whether a finite-frequency exceptional-point ring exists, compute its axial and transverse squared wavevector components at the prescribed frequency, and combine them into the total wavevector magnitude. The reference implementation calls the earlier public functions by name rather than reproducing their contents.

This step reproduces the paper's own worked configuration (eps_z=9.0, gamma0=2.5, omega=1.0) end to end. Every earlier convention comes together here: the constitutive-tensor reduction and its physical-admissibility consequence, the three-regime classification, and the distinct sign conventions of the axial and transverse closed forms.

Returns
-------
float: the total wavevector magnitude k_total on the finite-frequency exceptional-point ring.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_metamaterial_ep_pipeline(eps_z: float, gamma0: float, omega: float) -> float:
    """Chain the earlier steps and report the EP-ring wavevector magnitude.

    Args:
        eps_z (float): axial relative permittivity of the uniaxial medium.
        gamma0 (float): isotropic chirality parameter, gamma0 >= 0.
        omega (float): frequency, omega > 0.

    Raises:
        ValueError: if omega <= 0, if gamma0 < 0, if the reduced dispersion
            coefficients rule out physical admissibility of a finite-frequency
            ring, or if this (eps_z, gamma0) pair does not admit a
            finite-frequency exceptional-point ring.

    Expected return:
        float: k_total, the total wavevector magnitude on the finite-frequency
        exceptional-point ring.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: run_metamaterial_ep_pipeline
def _oracle_run_metamaterial_ep_pipeline(eps_z: float, gamma0: float, omega: float) -> float:
    if gamma0 < 0.0:
        raise ValueError("gamma0 must be non-negative")
    if omega <= 0.0:
        raise ValueError("omega must be positive")
    coeffs = _oracle_reduced_dispersion_coefficients(eps_z, gamma0)
    a_perp = float(coeffs[0])
    if not (a_perp < 0.0):
        raise ValueError(
            "physical admissibility check failed: the transverse reduced "
            "coefficient A_perp must be negative (gamma0^2 > 4) for any "
            "finite-frequency EP ring to exist"
        )
    sector = _oracle_classify_ep_sector(eps_z, gamma0)
    if sector != 2:
        raise ValueError("this (eps_z, gamma0) pair does not admit a finite-frequency EP ring")
    kz2 = _oracle_ep_ring_kz_squared(eps_z, gamma0, omega)
    rho = _oracle_ep_ring_transverse_squared(eps_z, gamma0, omega)
    parts = _oracle_combine_ep_wavevector(kz2, rho)
    return float(parts[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "round(run_metamaterial_ep_pipeline(9.0, 2.5, 1.0), 6)",
            "gold_call": "round(_oracle_run_metamaterial_ep_pipeline(9.0, 2.5, 1.0), 6)",
            "tol": 1e-6,
        },
        {
            "setup": "",
            "call": "round(run_metamaterial_ep_pipeline(9.0, 2.5, 2.0), 6)",
            "gold_call": "round(_oracle_run_metamaterial_ep_pipeline(9.0, 2.5, 2.0), 6)",
            "tol": 1e-6,
        },
        {
            "setup": "",
            "call": "round(run_metamaterial_ep_pipeline(-15.0, 3.0, 1.0), 6)",
            "gold_call": "round(_oracle_run_metamaterial_ep_pipeline(-15.0, 3.0, 1.0), 6)",
            "tol": 1e-6,
        },
        {
            "setup": "import numpy as np\ndef positive_finite(fn):\n    value = fn(9.0, 2.5, 1.0)\n    return bool(np.isfinite(value) and value > 0.0)",
            "call": "positive_finite(run_metamaterial_ep_pipeline)",
            "gold_call": "positive_finite(_oracle_run_metamaterial_ep_pipeline)",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        run_metamaterial_ep_pipeline(9.0, 1.0, 1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_run_metamaterial_ep_pipeline(9.0, 1.0, 1.0)\n"
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
