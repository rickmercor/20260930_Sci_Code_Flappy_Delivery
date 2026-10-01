"""
Determine the leading operating-heat bias caused by midpoint slicing.

At each resolution $N$, use the midpoint-slice optical transfer to find the deposited heat $Q_{*,N}$ that gives the specified exit-aperture fraction on the supplied branch. Let $Q_*$ denote the same root for the continuously varying lens. The dimensionless requested coefficient is



$$C=\lim_{N\to\infty}N^2\frac{Q_{*,N}-Q_*}{Q_*}.$$



The detector aperture and entrance-beam normalization are identical in the continuous and sliced problems. A simple heat root is required.

Returns
-------
A float is the dimensionless leading relative heat-bias coefficient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_midpoint_heat_bias(
    thermal: "np.ndarray",
    beam: "np.ndarray",
    alpha: float,
    aperture: float,
    target: float,
    heat_interval: "np.ndarray",
    order: int = 96,
) -> float:
    r"""Return the dimensionless leading sliced-root heat bias.

    Parameters
    ----------
    thermal : np.ndarray
        Finite positive shape $(5,)$ vector $(L,a,\kappa,n_0,\beta)$,
        in $\mathrm m$, $\mathrm m$, $\mathrm{W\,m^{-1}\,K^{-1}}$,
        dimensionless units, and $\mathrm{K^{-1}}$.
    beam : np.ndarray
        Finite real shape $(3,)$ vector $(\lambda,w_0,\theta)$, with
        positive wavelength and waist in $\mathrm m$ and internal angle
        in $[0,\pi/2)$ radians.
    alpha : float
        Finite absorption $\alpha\geq0$, in $\mathrm{m^{-1}}$.
    aperture : float
        Positive finite exit aperture radius in $\mathrm m$.
    target : float
        Desired finite captured fraction strictly between zero and one.
    heat_interval : np.ndarray
        Finite shape $(2,)$ positive increasing heat bracket in $\mathrm W$
        containing one simple continuous root.
    order : int, optional
        Integer quadrature order at least $16$, default $96$.

    Returns
    -------
    bias : float
        Dimensionless leading coefficient $C$ of the relative heat bias.

    Raises
    ------
    ValueError
        If an input has invalid shape, finiteness, reality, or range;
        the interval fails to bracket a root; or the heat response vanishes.
    RuntimeError
        If the bracketed operating-heat solve does not converge.

    Notes
    -----
    Inputs are not modified. The interval selects the desired root branch.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_midpoint_heat_bias(
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
    heat = _oracle_solve_capture_heat(
        thermal, beam, alpha, aperture, target, heat_interval, order
    )
    g, g_heat, _ = _oracle_compute_thermal_curvature(heat, alpha, *thermal)
    matrix = _oracle_compute_ray_matrix(g, alpha, thermal[0])
    sliced_direction = _oracle_compute_midpoint_transfer_error(
        g, alpha, thermal[0], order
    )
    heat_direction = _oracle_compute_ray_response(
        g, alpha, thermal[0], g_heat, 0.0, order
    )
    sliced_capture = _oracle_compute_aperture_response(
        matrix, sliced_direction, wave_number, beam[1], beam[2], aperture, order
    )[1]
    heat_capture = _oracle_compute_aperture_response(
        matrix, heat_direction, wave_number, beam[1], beam[2], aperture, order
    )[1]
    if not np.isfinite(heat_capture) or abs(heat_capture) < 1e-14:
        raise ValueError("the operating heat root is not simple")
    bias = -sliced_capture / (heat * heat_capture)
    if not np.isfinite(bias):
        raise ValueError("midpoint heat bias is nonfinite")
    return float(bias)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Check benchmark, scaling, alternate aperture, uniform lens, and invalid shape."""
    return [{'setup': 'import numpy as np\n'
               'thermal = np.array([.01, .0002, 14., 1.82, 7.3e-6])\n'
               'beam = np.array([1.064e-6, 7e-5, .003])\n'
               '\n'
               'def _preserved(function, *arguments):\n'
               '    copied = [argument.copy() if isinstance(argument, np.ndarray) else argument '
               'for argument in arguments]\n'
               '    before = [argument.copy() if isinstance(argument, np.ndarray) else argument '
               'for argument in copied]\n'
               '    result = function(*copied)\n'
               '    for previous, current in zip(before, copied):\n'
               '        if isinstance(previous, np.ndarray):\n'
               '            assert np.array_equal(previous, current), "input array was modified"\n'
               '    return result\n',
      'call': '_preserved(compute_midpoint_heat_bias, thermal.copy(), beam.copy(), 180., 2.5e-5, '
              '.7, np.array([250., 300.]))',
      'gold_call': '_preserved(_oracle_compute_midpoint_heat_bias, thermal.copy(), beam.copy(), '
                   '180., 2.5e-5, .7, np.array([250., 300.]))',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'thermal = np.array([.01, .0002, 28., 1.82, 7.3e-6])\n'
               'beam = np.array([1.064e-6, 7e-5, .003])\n',
      'call': 'compute_midpoint_heat_bias(thermal.copy(), beam.copy(), 180., 2.5e-5, .7, '
              'np.array([500., 600.]))',
      'gold_call': '_oracle_compute_midpoint_heat_bias(thermal.copy(), beam.copy(), 180., 2.5e-5, '
                   '.7, np.array([500., 600.]))',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'thermal = np.array([.01, .0002, 14., 1.82, 7.3e-6])\n'
               'beam = np.array([1.064e-6, 7e-5, .003])\n',
      'call': 'compute_midpoint_heat_bias(thermal.copy(), beam.copy(), 180., 2.6e-5, .7, '
              'np.array([220., 330.]))',
      'gold_call': '_oracle_compute_midpoint_heat_bias(thermal.copy(), beam.copy(), 180., 2.6e-5, '
                   '.7, np.array([220., 330.]))',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'thermal = np.array([.01, .0002, 14., 1.82, 7.3e-6])\n'
               'beam = np.array([1.064e-6, 7e-5, .003])\n',
      'call': 'compute_midpoint_heat_bias(thermal.copy(), beam.copy(), 0., 2.5e-5, .65, '
              'np.array([150., 190.]))',
      'gold_call': '_oracle_compute_midpoint_heat_bias(thermal.copy(), beam.copy(), 0., 2.5e-5, '
                   '.65, np.array([150., 190.]))',
      'tol': 1e-06},
     {'setup': 'import numpy as np\n'
               'thermal = np.array([.01, .0002, 14., 1.82, 7.3e-6])\n'
               'beam = np.array([1.064e-6, 7e-5, .003])\n'
               '\n'
               'def _raises(function):\n'
               '    try:\n'
               '        function(thermal[:-1], beam, 180., 2.5e-5, .7, np.array([250., 300.]))\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises(compute_midpoint_heat_bias)',
      'gold_call': '_raises(_oracle_compute_midpoint_heat_bias)'}]
