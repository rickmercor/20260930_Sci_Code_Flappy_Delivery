"""
Compute the effective topological-insulator-side permittivity eps_2,eff seen by the evanescent surface-wave field when the topological insulator is not a semi-infinite bulk but a thin film of thickness d on a dielectric substrate.

The paper replaces the bulk TI permittivity by a simple effective-medium estimate for thin films, built with a fixed effective evanescent sampling length that the paper itself chooses. Consult the paper's own thin-film estimate (its construction and its fixed sampling length) rather than a generic mixing rule; the bulk value is recovered only when the film fills the whole sampled region.

Returns
-------
float: effective permittivity eps_2,eff of the TI side (film + substrate).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def thin_film_effective_permittivity(d_nm: float, eps_ti: float, eps_sub: float) -> float:
    """Effective TI-side permittivity for a thin TI film on a substrate.

    Args:
        d_nm (float): TI film thickness in nm; must be strictly positive and
            strictly smaller than the paper's own effective sampling length.
        eps_ti (float): relative permittivity of the TI material, > 0.
        eps_sub (float): relative permittivity of the substrate, > 0.

    Raises:
        ValueError: if d_nm <= 0, if d_nm is not smaller than the paper's
            sampling length, or if eps_ti <= 0 or eps_sub <= 0.

    Expected return:
        float: eps_2,eff, lying strictly between eps_sub and eps_ti and moving
        monotonically toward eps_ti as the film thickens.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: thin_film_effective_permittivity
def _oracle_thin_film_effective_permittivity(d_nm: float, eps_ti: float, eps_sub: float) -> float:
    D_nm = 100.0  # paper's effective evanescent sampling length, Eq. (55)
    if d_nm <= 0.0 or d_nm >= D_nm:
        raise ValueError("film thickness must satisfy 0 < d < D")
    if eps_ti <= 0.0 or eps_sub <= 0.0:
        raise ValueError("permittivities must be positive")
    return (d_nm / D_nm) * eps_ti + (1.0 - d_nm / D_nm) * eps_sub

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "v = thin_film_effective_permittivity(12.0, 41.0, 3.9)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_thin_film_effective_permittivity(12.0, 41.0, 3.9), 8)",
            "tol": 1e-06,
        },
        {
            "setup": "v = thin_film_effective_permittivity(5.0, 41.0, 6.7)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_thin_film_effective_permittivity(5.0, 41.0, 6.7), 8)",
            "tol": 1e-06,
        },
        {
            "setup": "v = thin_film_effective_permittivity(40.0, 41.0, 9.4)",
            "call": "round(v, 8)",
            "gold_call": "round(_oracle_thin_film_effective_permittivity(40.0, 41.0, 9.4), 8)",
            "tol": 1e-06,
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        thin_film_effective_permittivity(0.0, 41.0, 3.9)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_thin_film_effective_permittivity(0.0, 41.0, 3.9)\n"
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
                "        thin_film_effective_permittivity(150.0, 41.0, 3.9)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_thin_film_effective_permittivity(150.0, 41.0, 3.9)\n"
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
                "        thin_film_effective_permittivity(10.0, 41.0, -1.0)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_thin_film_effective_permittivity(10.0, 41.0, -1.0)\n"
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
