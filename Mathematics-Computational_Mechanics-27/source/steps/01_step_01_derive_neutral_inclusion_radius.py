"""
This step fixes the geometry of the benchmark before any discretisation is considered. From the three bulk moduli, the shared Poisson ratio and the outer coating radius it returns the inclusion volume fraction that renders the coated sphere neutral, the inclusion radius that fraction implies, and the equivalent bulk modulus evaluated at that fraction.

The third quantity is not consumed downstream. It is returned because neutrality requires it to reproduce the matrix bulk modulus to machine precision, so it is the cheapest available evidence that the closed form and the neutrality condition have been paired correctly rather than merely being plausible.

A sphere of radius $r_i$ is surrounded by a concentric coating out to radius $r_c$ and the pair is embedded in a matrix. All three phases are isotropic and share one Poisson ratio, so each phase is fixed by its bulk modulus alone. The coated sphere is neutral when its equivalent bulk modulus equals the matrix bulk modulus, in which case a macroscopic hydrostatic strain is transmitted through the matrix without any disturbance, and the exact local field outside the coating is the undisturbed macroscopic one.

The equivalent bulk modulus of a coated sphere whose inclusion occupies the volume fraction $f = (r_i / r_c)^3$ of the coated sphere is $K_{eq}(f) = K_c + f (K_i - K_c) / [1 + (1 - f) (K_i - K_c) / (K_c + 4 \mu_c / 3)]$, and neutrality is the scalar equation $K_{eq}(f) = K_m$. Because $K_{eq}$ is a ratio of two functions that are affine in $f$, the equation is linear in $f$ once cleared and admits the closed form $f = (K_m - K_c) (1 + t) / [(K_i - K_c) + (K_m - K_c) t]$ with $t = (K_i - K_c) / (K_c + 4 \mu_c / 3)$, so no iteration is required.

Returns
-------
dict, the neutral-inclusion radius with the volume fraction and the equivalent bulk modulus check.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def derive_neutral_inclusion_radius(
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_coating: float,
) -> dict:
    """Solve the neutrality condition for the inclusion radius of a coated sphere.

    Parameters
    ----------
    bulk_matrix : float
        Bulk modulus of the matrix, strictly positive.
    bulk_coating : float
        Bulk modulus of the coating, strictly positive.
    bulk_inclusion : float
        Bulk modulus of the inclusion, strictly positive.
    poisson_ratio : float
        Poisson ratio shared by all three phases.
    radius_coating : float
        Outer radius of the coating, strictly positive.

    Returns
    -------
    dict
        Keys volume_fraction, radius_inclusion and effective_bulk.

    Raises
    ------
    ValueError
        If any bulk modulus is not strictly positive, if poisson_ratio lies outside the open interval (-1, 1/2), if radius_coating is not strictly positive, if the coating and inclusion moduli coincide so that no neutral radius exists, or if the neutrality condition has no admissible root in (0, 1).
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _shear_from_bulk(bulk, poisson_ratio):
    return 3.0 * bulk * (1.0 - 2.0 * poisson_ratio) / (2.0 * (1.0 + poisson_ratio))


def _equivalent_bulk(bulk_coating, bulk_inclusion, shear_coating, fraction):
    contrast = bulk_inclusion - bulk_coating
    denominator = 1.0 + (1.0 - fraction) * contrast / (bulk_coating + 4.0 * shear_coating / 3.0)
    return bulk_coating + fraction * contrast / denominator


def _oracle_derive_neutral_inclusion_radius(
    bulk_matrix: float,
    bulk_coating: float,
    bulk_inclusion: float,
    poisson_ratio: float,
    radius_coating: float,
) -> dict:
    """Reference implementation."""
    bulk_matrix = float(bulk_matrix)
    bulk_coating = float(bulk_coating)
    bulk_inclusion = float(bulk_inclusion)
    poisson_ratio = float(poisson_ratio)
    radius_coating = float(radius_coating)
    if min(bulk_matrix, bulk_coating, bulk_inclusion) <= 0.0:
        raise ValueError("bulk moduli must be strictly positive")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie in the open interval (-1, 1/2)")
    if radius_coating <= 0.0:
        raise ValueError("radius_coating must be strictly positive")
    if bulk_inclusion == bulk_coating:
        raise ValueError("a neutral radius does not exist when the coating and the "
                         "inclusion share a bulk modulus")
    shear_coating = _shear_from_bulk(bulk_coating, poisson_ratio)
    contrast_ratio = (bulk_inclusion - bulk_coating) / (bulk_coating + 4.0 * shear_coating / 3.0)
    numerator = (bulk_matrix - bulk_coating) * (1.0 + contrast_ratio)
    denominator = (bulk_inclusion - bulk_coating) + (bulk_matrix - bulk_coating) * contrast_ratio
    fraction = numerator / denominator
    if not 0.0 < fraction < 1.0:
        raise ValueError("the neutrality condition has no admissible root in (0, 1)")
    return {
        "volume_fraction": fraction,
        "radius_inclusion": radius_coating * fraction ** (1.0 / 3.0),
        "effective_bulk": _equivalent_bulk(bulk_coating, bulk_inclusion, shear_coating, fraction),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """def summarize(data):
    return (round(data["volume_fraction"], 13),
            round(data["radius_inclusion"], 11),
            round(data["effective_bulk"], 13))
import numpy as np
""",
            "call": "summarize(derive_neutral_inclusion_radius(1.0, 0.808024, 8.080240, 0.25, 2*np.pi))",
            "gold_call": "summarize(_oracle_derive_neutral_inclusion_radius(1.0, 0.808024, 8.080240, 0.25, 2*np.pi))",
        },
        {
            "setup": """def consistency(fn, Km, Kc, Ki, nu, rc):
    data = fn(Km, Kc, Ki, nu, rc)
    ratio = (data["radius_inclusion"] / rc) ** 3
    return (round(abs(data["effective_bulk"] - Km), 14),
            round(abs(ratio - data["volume_fraction"]), 14))
""",
            "call": "consistency(derive_neutral_inclusion_radius, 2.5, 1.4, 11.0, 0.3, 4.0)",
            "gold_call": "consistency(_oracle_derive_neutral_inclusion_radius, 2.5, 1.4, 11.0, 0.3, 4.0)",
        },
        {
            "setup": """def run(fn):
    codes = []
    for args in [(1.0, 0.8, 0.8, 0.25, 1.0), (1.0, 0.8, 8.0, 0.6, 1.0), (-1.0, 0.8, 8.0, 0.25, 1.0)]:
        try:
            fn(*args)
            codes.append(0)
        except ValueError:
            codes.append(1)
        except Exception:
            codes.append(2)
    return tuple(codes)
""",
            "call": "run(derive_neutral_inclusion_radius)",
            "gold_call": "run(_oracle_derive_neutral_inclusion_radius)",
        },
    ]
