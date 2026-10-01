"""
Find the heat load on a specified branch of the aperture-capture equation.

Changing $Q$ changes the entire thermally induced transfer matrix. The operating heat load is defined on the supplied interval by



$$F(Q_*,\alpha)=f_*,\qquad Q_*\in[Q_{\mathrm{lo}},Q_{\mathrm{hi}}].$$



The interval selects one simple root. Optical refocusing can produce multiple roots over a larger range, so branch selection is part of the physical specification. An exact endpoint root is retained; an interval with same-sign nonzero endpoint residuals is rejected.

Returns
-------
A float gives the deposited heat load in watts on the specified capture branch.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_capture_heat(
    thermal: "np.ndarray",
    beam: "np.ndarray",
    alpha: float,
    aperture: float,
    target: float,
    heat_interval: "np.ndarray",
    order: int = 96,
) -> float:
    r"""Return the heat load meeting the aperture target on the specified branch.

    Parameters
    ----------
    thermal : np.ndarray
        Finite positive shape $(5,)$ vector $(L,a,\kappa,n_0,\beta)$ in
        $\mathrm{m}$, $\mathrm{m}$, $\mathrm{W\,m^{-1}\,K^{-1}}$,
        dimensionless units, and $\mathrm{K^{-1}}$, respectively.
    beam : np.ndarray
        Finite real shape $(3,)$ vector $(\lambda,w_0,\theta)$: positive
        vacuum wavelength and envelope radius in $\mathrm{m}$, and
        internal cone angle in $[0,\pi/2)$ radians.
    alpha : float
        Finite absorption $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    aperture : float
        Positive finite exit aperture radius, in $\mathrm{m}$.
    target : float
        Finite desired captured fraction strictly between zero and one.
    heat_interval : np.ndarray
        Finite real shape $(2,)$ vector of strictly increasing positive
        heat loads, in $\mathrm{W}$, enclosing one simple root of the
        capture equation; an endpoint root is allowed.
    order : int, optional
        Integer quadrature order at least $16$, default $96$.

    Returns
    -------
    heat : float
        Heat load $Q_*$ in $\mathrm{W}$ with capture residual at most
        $10^{-9}$ for a resolved quadrature.

    Raises
    ------
    ValueError
        If input arrays have invalid shape, reality, finiteness, or range;
        alpha, aperture, target, or order is invalid; the interval does not
        bracket a root; or a called thermal, transfer, or field computation
        rejects its data.
    RuntimeError
        If the bracketed scalar solve fails to converge.

    Notes
    -----
    Uniqueness inside the supplied interval is a caller precondition.
    Inputs are not modified. There is no random state.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _capture_inputs(thermal, beam, alpha, aperture, target, heat_interval, order):
    thermal = _real_array(thermal, (5,), "thermal")
    beam = _real_array(beam, (3,), "beam")
    heat_interval = _real_array(heat_interval, (2,), "heat_interval")
    if np.any(thermal <= 0.0) or np.any(beam[:2] <= 0.0):
        raise ValueError("thermal parameters, wavelength, and waist must be positive")
    alpha = _finite_scalar(alpha, "alpha")
    aperture = _finite_scalar(aperture, "aperture", True)
    if not np.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between zero and one")
    if heat_interval[0] <= 0.0 or heat_interval[1] <= heat_interval[0]:
        raise ValueError("heat_interval must have increasing positive endpoints")
    order = _quadrature_order(order)
    wave_number = 2.0 * np.pi * thermal[3] / beam[0]
    _beam_parameters(wave_number, beam[1], beam[2])
    return (
        thermal,
        beam,
        alpha,
        aperture,
        float(target),
        heat_interval,
        order,
        wave_number,
    )


def _oracle_solve_capture_heat(
    thermal: "np.ndarray",
    beam: "np.ndarray",
    alpha: float,
    aperture: float,
    target: float,
    heat_interval: "np.ndarray",
    order: int = 96,
) -> float:
    thermal, beam, alpha, aperture, target, heat_interval, order, wave_number = (
        _capture_inputs(thermal, beam, alpha, aperture, target, heat_interval, order)
    )

    def _residual(heat):
        g = _oracle_compute_thermal_curvature(heat, alpha, *thermal)[0]
        matrix = _oracle_compute_ray_matrix(g, alpha, thermal[0])
        fraction = _oracle_compute_aperture_response(
            matrix, np.zeros((2, 2)), wave_number, beam[1], beam[2], aperture, order
        )[0]
        return fraction - target

    return float(brentq(_residual, *heat_interval, xtol=1e-10, rtol=2e-13))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and input-preservation cases."""
    return [
        {
            "setup": """import numpy as np
thermal = np.array([.01, .0002, 14., 1.82, 7.3e-6])
beam = np.array([1.064e-6, 7e-5, .003])

def _preserved(function, *arguments):
    copied = [argument.copy() if isinstance(argument, np.ndarray) else argument for argument in arguments]
    before = [argument.copy() if isinstance(argument, np.ndarray) else argument for argument in copied]
    result = function(*copied)
    for previous, current in zip(before, copied):
        if isinstance(previous, np.ndarray):
            assert np.array_equal(previous, current), "input array was modified"
    return result
""",
            "call": "_preserved(solve_capture_heat, thermal.copy(), beam.copy(), 180.0, 2.5e-5, .7, "
            "np.array([250.,300.]))",
            "gold_call": "_preserved(_oracle_solve_capture_heat, thermal.copy(), beam.copy(), 180.0, "
            "2.5e-5, .7, np.array([250.,300.]))",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
thermal = np.array([.01, .0002, 14., 1.82, 7.3e-6])
beam = np.array([1.064e-6, 7e-5, .003])
""",
            "call": "solve_capture_heat(thermal.copy(), beam.copy(), 0.0, 2.5e-5, .65, "
            "np.array([150.,190.]))",
            "gold_call": "_oracle_solve_capture_heat(thermal.copy(), beam.copy(), 0.0, 2.5e-5, .65, "
            "np.array([150.,190.]))",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
thermal = np.array([.01, .0002, 14., 1.82, 7.3e-6])
beam = np.array([1.064e-6, 7e-5, .003])
""",
            "call": "solve_capture_heat(thermal.copy(), beam.copy(), 180.0, 2.5e-5, .65, "
            "np.array([150.,190.]))",
            "gold_call": "_oracle_solve_capture_heat(thermal.copy(), beam.copy(), 180.0, 2.5e-5, .65, "
            "np.array([150.,190.]))",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
thermal = np.array([.01, .0002, 14., 1.82, 7.3e-6])
beam = np.array([1.064e-6, 7e-5, .003])
thermal[2] *= 1.01
""",
            "call": "solve_capture_heat(thermal.copy(), beam.copy(), 180.0, 2.5e-5, .7, "
            "np.array([250.,300.]))",
            "gold_call": "_oracle_solve_capture_heat(thermal.copy(), beam.copy(), 180.0, 2.5e-5, .7, "
            "np.array([250.,300.]))",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
thermal = np.array([.01, .0002, 14., 1.82, 7.3e-6])
beam = np.array([1.064e-6, 7e-5, .003])

def _raises(function):
    try:
        function(thermal.copy(), beam.copy(), 180.0, 2.5e-5, .1, np.array([250.,300.]))
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": "_raises(solve_capture_heat)",
            "gold_call": "_raises(_oracle_solve_capture_heat)",
        },
    ]
