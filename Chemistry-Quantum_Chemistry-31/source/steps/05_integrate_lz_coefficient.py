"""
Evaluate the declared finite-order benchmark for the positive-energy thermal
transmission average over crossing velocity.

Use exactly the requested Gauss-Laguerre order on the transformed
semi-infinite thermal domain; do not adapt the order or extrapolate it to a
converged continuum value, and do not introduce a finite velocity cutoff.
All scalar inputs must be finite,
``coupling`` nonnegative, ``gradient_gap``, ``reduced_mass``, and ``beta``
positive, and ``quadrature_order`` an integer from eight through 100.  Return
the reduced coefficient in the units declared by the signature.  Invalid
inputs raise ``ValueError``.

Returns
-------
float, finite nonnegative reduced LZ/Holstein rate coefficient in E_h/hbar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_lz_coefficient(
    coupling: float,
    gradient_gap: float,
    reduced_mass: float,
    beta: float,
    quadrature_order: int,
) -> float:
    """Return the thermally averaged LZ/Holstein coefficient.

    Parameters
    ----------
    coupling
        Finite nonnegative diabatic coupling in ``E_h``.
    gradient_gap
        Finite strictly positive gradient gap in ``E_h bohr^-1``.
    reduced_mass
        Finite strictly positive mass in ``m_e``.
    beta
        Finite strictly positive inverse temperature in ``E_h^-1``.
    quadrature_order
        Integer Gauss-Laguerre order from eight through 100. It defines the
        finite deterministic benchmark rather than an adaptive convergence
        target.

    Returns
    -------
    float
        Finite nonnegative reduced rate coefficient in ``E_h/hbar``.

    Raises
    ------
    ValueError
        If a scalar is non-finite or outside its physical domain, or if the
        quadrature order is not an integer from eight through 100, or if the
        transformed quadrature is not finite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _step5_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _oracle_integrate_lz_coefficient(
    coupling: float,
    gradient_gap: float,
    reduced_mass: float,
    beta: float,
    quadrature_order: int,
) -> float:
    coupling_value = _step5_real_scalar(coupling, "coupling")
    gradient_gap_value = _step5_real_scalar(gradient_gap, "gradient_gap")
    reduced_mass_value = _step5_real_scalar(reduced_mass, "reduced_mass")
    beta_value = _step5_real_scalar(beta, "beta")
    if (
        coupling_value < 0.0
        or gradient_gap_value <= 0.0
        or reduced_mass_value <= 0.0
        or beta_value <= 0.0
        or not isinstance(quadrature_order, (int, np.integer))
        or isinstance(quadrature_order, (bool, np.bool_))
        or int(quadrature_order) < 8
        or int(quadrature_order) > 100
    ):
        raise ValueError("physical scalars or quadrature_order are outside the domain")
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
    velocities = np.sqrt(2.0 * nodes / (beta_value * reduced_mass_value))
    transmission = _oracle_compute_holstein_transmission(
        velocities,
        coupling_value,
        gradient_gap_value,
    )
    coefficient = float(np.dot(weights, transmission) / beta_value)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("the LZ/Holstein integral is non-finite")
    return coefficient

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nc=4.2e-4; dg=.091; mu=23000.; beta=3800.",
            "call": "integrate_lz_coefficient(c,dg,mu,beta,48)",
            "gold_call": "_oracle_integrate_lz_coefficient(c,dg,mu,beta,48)",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np",
            "call": "integrate_lz_coefficient(0.0,0.2,18000.0,4200.0,24)",
            "gold_call": "_oracle_integrate_lz_coefficient(0.0,0.2,18000.0,4200.0,24)",
            "tol": 0.0,
        },
        {
            "setup": "import numpy as np\nc=.02; dg=.04; mu=8000.; beta=1200.",
            "call": "integrate_lz_coefficient(c,dg,mu,beta,32)",
            "gold_call": "_oracle_integrate_lz_coefficient(c,dg,mu,beta,32)",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np\ndef check(fn):\n try: fn(1e-4,.1,10000.,3000.,7)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(integrate_lz_coefficient)",
            "gold_call": "check(_oracle_integrate_lz_coefficient)",
            "tol": 0.0,
        },
    ]
