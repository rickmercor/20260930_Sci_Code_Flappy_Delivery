"""
Compute the entrance ray curvature and its two thermal partial derivatives.

The pumped cylinder has radius $a$ and length $L$. Its heat-source profile $S(z)=S_0e^{-\alpha z}$ is normalized by the specified total deposited heat.



$$Q=\pi a^2\int_0^L S(z)\,dz.$$



The returned entrance curvature belongs to the position-slope ray equation $r''+g e^{-\alpha z}r=0$. The absorption derivative holds $Q$ fixed. At $\alpha=0$, the source is the continuous uniform-deposition limit and remains nonzero when $Q>0$.

Returns
-------
A real array of shape $(3,)$ contains $g$, $\partial_Qg$, and $\partial_\alpha g$ in that order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_thermal_curvature(
    heat: float,
    alpha: float,
    length: float,
    pump_radius: float,
    conductivity: float,
    index: float,
    thermooptic: float,
) -> "np.ndarray":
    r"""Return the curvature and its heat-load and absorption derivatives.

    Parameters
    ----------
    heat : float
        Total deposited heat $Q\geq0$, in $\mathrm{W}$.
    alpha : float
        Absorption coefficient $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    length : float
        Crystal length $L>0$, in $\mathrm{m}$.
    pump_radius : float
        Pump radius $a>0$, in $\mathrm{m}$.
    conductivity : float
        Thermal conductivity $\kappa>0$, in $\mathrm{W\,m^{-1}\,K^{-1}}$.
    index : float
        On-axis index $n_0>0$.
    thermooptic : float
        Thermo-optic coefficient $\beta>0$, in $\mathrm{K^{-1}}$.

    Returns
    -------
    curvature : np.ndarray
        Shape $(3,)$ in the order $(g,\partial_Qg,\partial_\alpha g)$,
        with units $\mathrm{m^{-2}}$, $\mathrm{m^{-2}\,W^{-1}}$, and
        $\mathrm{m^{-1}}$, respectively.

    Raises
    ------
    ValueError
        If an input is nonfinite, a nonnegative input is negative, or a
        strictly positive input is nonpositive.

    Notes
    -----
    The zero-absorption limit holds total deposited heat fixed.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_scalar(value: float, name: str, positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        raise ValueError(f"{name} violates its finite nonnegative/positive contract")
    return result


def _oracle_compute_thermal_curvature(
    heat: float,
    alpha: float,
    length: float,
    pump_radius: float,
    conductivity: float,
    index: float,
    thermooptic: float,
) -> "np.ndarray":
    heat = _finite_scalar(heat, "heat")
    alpha = _finite_scalar(alpha, "alpha")
    length = _finite_scalar(length, "length", True)
    pump_radius = _finite_scalar(pump_radius, "pump_radius", True)
    conductivity = _finite_scalar(conductivity, "conductivity", True)
    index = _finite_scalar(index, "index", True)
    thermooptic = _finite_scalar(thermooptic, "thermooptic", True)
    x = alpha * length
    if x < 1e-4:
        factor = (1.0 + x / 2.0 + x**2 / 12.0 - x**4 / 720.0) / length
        derivative = 0.5 + x / 6.0 - x**3 / 180.0
    else:
        denominator = -np.expm1(-x)
        factor = alpha / denominator
        derivative = (denominator - x * np.exp(-x)) / denominator**2
    scale = thermooptic / (2.0 * conductivity * index * np.pi * pump_radius**2)
    heat_derivative = scale * factor
    return np.array(
        [heat * heat_derivative, heat_derivative, heat * scale * derivative]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, and edge cases."""
    return [
        {
            "setup": """import numpy as np
""",
            "call": "compute_thermal_curvature(260.0, 180.0, .01, .0002, 14.0, 1.82, 7.3e-6)",
            "gold_call": "_oracle_compute_thermal_curvature(260.0, 180.0, .01, .0002, 14.0, 1.82, "
            "7.3e-6)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "compute_thermal_curvature(60.0, 0.0, .01, .0002, 14.0, 1.82, 7.3e-6)",
            "gold_call": "_oracle_compute_thermal_curvature(60.0, 0.0, .01, .0002, 14.0, 1.82, 7.3e-6)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
""",
            "call": "compute_thermal_curvature(0.0, 2e-4, .02, .0003, 10.0, 1.6, 8e-6)",
            "gold_call": "_oracle_compute_thermal_curvature(0.0, 2e-4, .02, .0003, 10.0, 1.6, 8e-6)",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np

def _raises(function):
    try:
        function(-1.0, 180.0, .01, .0002, 14.0, 1.82, 7.3e-6)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_raises(compute_thermal_curvature)",
            "gold_call": "_raises(_oracle_compute_thermal_curvature)",
        },
    ]
