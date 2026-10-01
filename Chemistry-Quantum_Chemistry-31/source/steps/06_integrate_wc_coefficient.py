"""
Evaluate the weak-coupling Airy transmission coefficient.

Use the source's weak-coupling treatment over the semi-infinite energy domain
beginning at ``energy_floor`` and the requested Gauss-Laguerre order.  When
``cap_probability`` is true, enforce the source probability cap before the
thermal integration.  Physical scalars must be finite and positive except
for nonnegative coupling and an unrestricted finite energy floor.  Require
``-beta*energy_floor < 600`` and an integer quadrature order from eight through
100.  Invalid inputs raise ``ValueError``.

Returns
-------
float, finite nonnegative reduced weak-coupling rate coefficient in E_h/hbar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_wc_coefficient(
    coupling: float,
    gradient_gap: float,
    gradient_product_root: float,
    reduced_mass: float,
    beta: float,
    energy_floor: float,
    quadrature_order: int,
    cap_probability: bool = True,
) -> float:
    """Return the energy-integrated weak-coupling transmission coefficient.

    Parameters
    ----------
    coupling
        Finite nonnegative diabatic coupling in ``E_h``.
    gradient_gap, gradient_product_root
        Finite strictly positive gradient quantities in ``E_h bohr^-1``.
    reduced_mass
        Finite strictly positive mass in ``m_e``.
    beta
        Finite strictly positive inverse temperature in ``E_h^-1``.
    energy_floor
        Finite lower integration energy in ``E_h`` satisfying
        ``-beta * energy_floor < 600``.
    quadrature_order
        Integer Gauss-Laguerre order from eight through 100.
    cap_probability
        Boolean selecting whether Airy probabilities above one are capped.

    Returns
    -------
    float
        Finite nonnegative reduced weak-coupling rate coefficient in
        ``E_h/hbar``.

    Raises
    ------
    ValueError
        If an input is non-finite, has an invalid type or physical domain, or
        violates the transformed-energy bound, or if the transformed
        probability or quadrature is not finite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import airy


def _step6_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _oracle_integrate_wc_coefficient(
    coupling: float,
    gradient_gap: float,
    gradient_product_root: float,
    reduced_mass: float,
    beta: float,
    energy_floor: float,
    quadrature_order: int,
    cap_probability: bool = True,
) -> float:
    coupling_value = _step6_real_scalar(coupling, "coupling")
    gradient_gap_value = _step6_real_scalar(gradient_gap, "gradient_gap")
    gradient_product_root_value = _step6_real_scalar(
        gradient_product_root, "gradient_product_root"
    )
    reduced_mass_value = _step6_real_scalar(reduced_mass, "reduced_mass")
    beta_value = _step6_real_scalar(beta, "beta")
    energy_floor_value = _step6_real_scalar(energy_floor, "energy_floor")
    if (
        coupling_value < 0.0
        or gradient_gap_value <= 0.0
        or gradient_product_root_value <= 0.0
        or reduced_mass_value <= 0.0
        or beta_value <= 0.0
        or -beta_value * energy_floor_value >= 600.0
        or not isinstance(quadrature_order, (int, np.integer))
        or isinstance(quadrature_order, (bool, np.bool_))
        or int(quadrature_order) < 8
        or int(quadrature_order) > 100
        or not isinstance(cap_probability, (bool, np.bool_))
    ):
        raise ValueError("physical scalars or control arguments are outside the domain")
    if coupling_value == 0.0:
        return 0.0
    nodes, weights = np.polynomial.laguerre.laggauss(int(quadrature_order))
    if (
        np.any(~np.isfinite(nodes))
        or np.any(~np.isfinite(weights))
        or np.any(nodes <= 0.0)
        or np.any(weights <= 0.0)
    ):
        raise ValueError("Gauss-Laguerre rule is non-finite")
    energies = energy_floor_value + nodes / beta_value
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        airy_scale = np.cbrt(
            16.0
            * reduced_mass_value
            / (gradient_product_root_value * gradient_gap_value)
        )
        airy_argument = -(
            energies
            * gradient_gap_value
            * airy_scale
            / (2.0 * gradient_product_root_value)
        )
        airy_value = airy(airy_argument)[0]
        probability = (
            np.pi**2
            * airy_scale**2
            * np.square(np.float64(coupling_value))
            * airy_value**2
        )
    if np.any(np.isnan(probability)) or np.any(probability < 0.0):
        raise ValueError("the weak-coupling probability is non-finite")
    if bool(cap_probability):
        probability = np.minimum(probability, 1.0)
    elif np.any(~np.isfinite(probability)):
        raise ValueError("the uncapped weak-coupling probability is non-finite")
    coefficient = (
        np.exp(-beta_value * energy_floor_value)
        * np.dot(weights, probability)
        / beta_value
    )
    if not np.isfinite(coefficient):
        raise ValueError("the weak-coupling integral is non-finite")
    return float(coefficient)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nfrom scipy.special import airy",
            "call": "integrate_wc_coefficient(4e-4,.09,.046,23000.,3800.,-.003,64,True)",
            "gold_call": "_oracle_integrate_wc_coefficient(4e-4,.09,.046,23000.,3800.,-.003,64,True)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import airy",
            "call": "integrate_wc_coefficient(0.0,.2,.1,12000.,2500.,0.0,24,True)",
            "gold_call": "_oracle_integrate_wc_coefficient(0.0,.2,.1,12000.,2500.,0.0,24,True)",
            "tol": 0.0,
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import airy",
            "call": "integrate_wc_coefficient(.02,.03,.025,9000.,1500.,-.001,40,True)",
            "gold_call": "_oracle_integrate_wc_coefficient(.02,.03,.025,9000.,1500.,-.001,40,True)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import airy\ndef check(fn):\n try: fn(1e-4,.1,.05,10000.,5000.,-.2,32,True)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(integrate_wc_coefficient)",
            "gold_call": "check(_oracle_integrate_wc_coefficient)",
            "tol": 0.0,
        },
    ]
