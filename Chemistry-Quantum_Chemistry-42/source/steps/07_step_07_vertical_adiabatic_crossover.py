"""
Step 07: Displacement where vertical and adiabatic reference Hessians are equally accurate (orchestrator).

Smallest excited-state displacement at which the vertical reference Hessian matches the adiabatic one (orchestrator).

A thawed Gaussian with a constant reference Hessian needs that Hessian at a single geometry. The adiabatic choice, the
Hessian at the excited-state minimum, is exact for harmonic surfaces and best when the wavepacket stays near the minimum.
The vertical choice, the Hessian at the Franck-Condon point, needs no excited-state optimization and describes the width
dynamics right after the transition, which dominates a low-resolution spectrum when the displacement is large. As the
minimum moves away from the Franck-Condon point, the spectral accuracy of the vertical choice therefore catches up with
that of the adiabatic choice.

The search evaluates g(d) = cos_vertical(d) - cos_adiabatic(d), from the previous step, on an evenly spaced grid of
displacements from d_min to d_max inclusive. The crossover is located in the first scan interval [d_(k-1), d_k] with
g(d_(k-1)) < 0 and g(d_k) >= 0, returning d_k if g(d_k) = 0 and otherwise refined by Brent's method to an absolute
tolerance of 1e-8.

Returns
-------
float, displacement d_x in the first scan bracket at which the vertical and adiabatic contrast cosines are equal
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def vertical_adiabatic_crossover(surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float, d_min: float, d_max: float, n_scan: int) -> float:
    '''Displacement, refined in the first scan bracket, where the vertical and adiabatic spectral contrast cosines are equal.

    Parameters
    ----------
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] of the excited-state surface; the stretch coordinate of its minimum is
        the variable being searched.
    ground_params : np.ndarray
        Array [omega_1, omega_2, theta] of the initial state.
    q1_axis : np.ndarray
        Evenly spaced grid points of q1 for the exact propagation.
    q2_axis : np.ndarray
        Evenly spaced grid points of q2 for the exact propagation.
    time_step : float
        Common time step dt > 0.
    broadening_time : float
        Broadening time tau > 0.
    d_min : float
        First scan displacement.
    d_max : float
        Last scan displacement, d_max > d_min.
    n_scan : int
        Number of evenly spaced scan displacements, n_scan >= 2.

    Returns
    -------
    result : float
        Crossover displacement d_x.

    Raises
    ------
    ValueError
        If d_max <= d_min or n_scan < 2, if g(d_min) >= 0 so that the scan starts at or past the crossover, or if g stays
        negative over the whole scan.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import brentq


def _crossover_gap(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float) -> float:
    """Vertical minus adiabatic spectral contrast cosine at one displacement."""
    diag = _oracle_method_diagnostics(displacement, surface_params, ground_params, q1_axis, q2_axis, time_step, broadening_time)
    return float(diag[2] - diag[1])


def _oracle_vertical_adiabatic_crossover(surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float, d_min: float, d_max: float, n_scan: int) -> float:
    """Reference implementation."""
    import numpy as np
    from scipy.optimize import brentq
    if not d_max > d_min or int(n_scan) < 2:
        raise ValueError("require d_max > d_min and n_scan >= 2")
    grid = np.linspace(d_min, d_max, int(n_scan))
    args = (surface_params, ground_params, q1_axis, q2_axis, time_step, broadening_time)
    previous = _crossover_gap(grid[0], *args)
    if previous >= 0.0:
        raise ValueError("the scan starts at or beyond the crossover")
    for k in range(1, grid.size):
        current = _crossover_gap(grid[k], *args)
        if current >= 0.0:
            return float(brentq(lambda d: _crossover_gap(d, *args), grid[k - 1], grid[k], xtol=1e-8))
    raise ValueError("no crossover within the scanned displacements")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: stretch-dependent bend with a negative Duschinsky angle, coarse grid and short broadening ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.85, 0.02, 0.45, 0.5, 0.4])\n"
                     "gp = np.array([1.0, 0.5, np.deg2rad(-35.0)])\n"
                     "a1 = np.linspace(-8.0, 16.0, 64, endpoint=False)\n"
                     "a2 = np.linspace(-8.0, 8.0, 32, endpoint=False)\n",
            "call": "vertical_adiabatic_crossover(par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 2.5, 1.0, 3.0, 3)",
            "gold_call": "_oracle_vertical_adiabatic_crossover(par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 2.5, 1.0, 3.0, 3)",
            "tol": 1e-6,
        },
        # --- Normal: stiffer, more anharmonic stretch with a stiffer bend and a larger broadening time ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([1.1, 0.03, 0.6, 0.3, 0.8])\n"
                     "gp = np.array([1.0, 0.7, np.deg2rad(20.0)])\n"
                     "a1 = np.linspace(-8.0, 14.0, 64, endpoint=False)\n"
                     "a2 = np.linspace(-8.0, 8.0, 32, endpoint=False)\n",
            "call": "vertical_adiabatic_crossover(par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 3.0, 1.0, 2.5, 4)",
            "gold_call": "_oracle_vertical_adiabatic_crossover(par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 3.0, 1.0, 2.5, 4)",
            "tol": 1e-6,
        },
        # --- Boundary: the smallest scan, two displacements, so the whole scanned range is the bracket ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.85, 0.02, 0.45, 0.5, 0.4])\n"
                     "gp = np.array([1.0, 0.5, np.deg2rad(-35.0)])\n"
                     "a1 = np.linspace(-8.0, 16.0, 64, endpoint=False)\n"
                     "a2 = np.linspace(-8.0, 8.0, 32, endpoint=False)\n",
            "call": "vertical_adiabatic_crossover(par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 2.5, 1.8, 2.6, 2)",
            "gold_call": "_oracle_vertical_adiabatic_crossover(par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 2.5, 1.8, 2.6, 2)",
            "tol": 1e-6,
        },
        # --- Error: a scan that starts where the vertical choice is already at least as accurate must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([0.85, 0.02, 0.45, 0.3, 0.8]), np.array([1.0, 0.5, np.deg2rad(20.0)]),\n"
                     "           np.linspace(-8.0, 16.0, 64, endpoint=False), np.linspace(-8.0, 8.0, 32, endpoint=False),\n"
                     "           0.1, 2.0, 2.0, 3.0, 2)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(vertical_adiabatic_crossover)",
            "gold_call": "_probe(_oracle_vertical_adiabatic_crossover)",
        },
        # --- Error: a scan that ends before the crossover, so the gap stays negative, must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(np.array([0.85, 0.02, 0.45, 0.5, 0.4]), np.array([1.0, 0.5, np.deg2rad(-35.0)]),\n"
                     "           np.linspace(-8.0, 16.0, 64, endpoint=False), np.linspace(-8.0, 8.0, 32, endpoint=False),\n"
                     "           0.1, 2.5, 0.6, 1.6, 3)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(vertical_adiabatic_crossover)",
            "gold_call": "_probe(_oracle_vertical_adiabatic_crossover)",
        },
    ]
