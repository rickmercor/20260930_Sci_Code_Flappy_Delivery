"""
Step 06: Spectral accuracy and energy excursions of four Gaussian variants.

Accuracy of four thawed Gaussian propagations against exact quantum dynamics for one excited-state displacement.

For a given displacement of the excited-state stretch minimum, the exact autocorrelation is obtained by grid propagation
and the same absorption event is simulated with a thawed Gaussian whose width matrix feels the local Hessian along the
trajectory, the Hessian at the excited-state minimum (adiabatic reference), the Hessian of the excited-state surface at
the initial-state minimum (vertical reference), or the initial-state Hessian. Each approximate autocorrelation is compared
with the exact one through the unshifted spectral contrast cosine of the Gaussian-broadened lineshapes, and the true energy
of the adiabatic and local harmonic wavepackets is monitored along the propagation, which exposes the contrast between a
bounded width and an unbounded one.

All propagations start from the vibrational ground state of the rotated harmonic initial state, use one common time step and
run to the time 6 tau, beyond which the Gaussian damping exp(-t^2 / (2 tau^2)) has removed the autocorrelation.
Coordinates are mass-weighted with unit masses and hbar = 1.

Returns
-------
numpy.ndarray of shape (6,), contrast cosines of the local, adiabatic, vertical and initial variants and the energy excursions of the adiabatic and local variants
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def method_diagnostics(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float) -> "np.ndarray":
    '''Spectral contrast cosines of four thawed Gaussian variants and the energy excursions of two of them.

    The number of steps is n = round(6 broadening_time / time_step). The exact autocorrelation comes from the split-operator
    propagation on the given grid, each thawed Gaussian autocorrelation and energy series from the fourth-order composition propagation
    with the same time step and number of steps, and each cosine from the unshifted spectral contrast cosine with the exact
    autocorrelation as the reference. The energy excursion of a variant is the largest |E_k - E_0| over the n + 1 sampled
    energy expectation values E_k.

    Parameters
    ----------
    displacement : float
        Stretch coordinate d of the excited-state minimum.
    surface_params : np.ndarray
        Array [omega_e, chi, omega_b, gamma, delta] of the excited-state surface.
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

    Returns
    -------
    result : np.ndarray
        Array [cos_local, cos_adiabatic, cos_vertical, cos_initial, excursion_adiabatic, excursion_local] of shape (6,).

    Raises
    ------
    ValueError
        If time_step or broadening_time is not strictly positive, or if round(6 broadening_time / time_step) < 1.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_method_diagnostics(displacement: float, surface_params: "np.ndarray", ground_params: "np.ndarray", q1_axis: "np.ndarray", q2_axis: "np.ndarray", time_step: float, broadening_time: float) -> "np.ndarray":
    """Reference implementation."""
    import numpy as np
    if not time_step > 0.0 or not broadening_time > 0.0:
        raise ValueError("time_step and broadening_time must be strictly positive")
    n = int(round(6.0 * broadening_time / time_step))
    if n < 1:
        raise ValueError("the propagation must contain at least one step")
    reference = _oracle_exact_autocorrelation(displacement, surface_params, ground_params, q1_axis, q2_axis, time_step, n)
    cosines = []
    excursions = {}
    for mode in (0, 1, 2, 3):
        auto = _oracle_thawed_gaussian_autocorrelation(displacement, surface_params, ground_params, mode, time_step, n)
        cosines.append(float(_oracle_spectral_contrast_cosine(auto, reference, time_step, broadening_time)[0]))
        if mode in (0, 1):
            energy = _oracle_thawed_gaussian_energy(displacement, surface_params, ground_params, mode, time_step, n)
            excursions[mode] = float(np.max(np.abs(energy - energy[0])))
    return np.array(cosines + [excursions[1], excursions[0]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: benchmark-like model on a coarse grid with a short broadening time ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.85, 0.02, 0.45, 0.3, 0.8])\n"
                     "gp = np.array([1.0, 0.5, np.deg2rad(20.0)])\n"
                     "a1 = np.linspace(-8.0, 16.0, 72, endpoint=False)\n"
                     "a2 = np.linspace(-8.0, 8.0, 32, endpoint=False)\n",
            "call": "method_diagnostics(2.2, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 2.0)",
            "gold_call": "_oracle_method_diagnostics(2.2, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.1, 2.0)",
            "tol": 1e-7,
        },
        # --- Edge: nearly harmonic surface without coupling, where the adiabatic reference Hessian is almost exact ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.9, 1e-4, 0.5, 0.0, 0.6])\n"
                     "gp = np.array([1.2, 0.4, 0.5])\n"
                     "a1 = np.linspace(-9.0, 11.0, 64, endpoint=False)\n"
                     "a2 = np.linspace(-9.0, 9.0, 48, endpoint=False)\n",
            "call": "method_diagnostics(1.5, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.05, 2.5)",
            "gold_call": "_oracle_method_diagnostics(1.5, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.05, 2.5)",
            "tol": 1e-7,
        },
        # --- Normal: stiffer, more anharmonic stretch with a negative Duschinsky angle and negative coupling ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([1.1, 0.03, 0.6, -0.35, 0.4])\n"
                     "gp = np.array([0.8, 0.7, -0.5])\n"
                     "a1 = np.linspace(-7.0, 13.0, 64, endpoint=False)\n"
                     "a2 = np.linspace(-7.0, 7.0, 40, endpoint=False)\n",
            "call": "method_diagnostics(1.8, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.08, 1.6)",
            "gold_call": "_oracle_method_diagnostics(1.8, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.08, 1.6)",
            "tol": 1e-7,
        },
        # --- Boundary: the shortest allowed propagation, a single step, where each cosine uses one trapezoid interval ---
        {
            "setup": "import numpy as np\n"
                     "par = np.array([0.85, 0.02, 0.45, 0.3, 0.8])\n"
                     "gp = np.array([1.0, 0.5, 0.35])\n"
                     "a1 = np.linspace(-8.0, 16.0, 48, endpoint=False)\n"
                     "a2 = np.linspace(-8.0, 8.0, 24, endpoint=False)\n",
            "call": "method_diagnostics(1.2, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.4, 0.09)",
            "gold_call": "_oracle_method_diagnostics(1.2, par.copy(), gp.copy(), a1.copy(), a2.copy(), 0.4, 0.09)",
            "tol": 1e-9,
        },
        # --- Error: a broadening time too short for a single step must raise ValueError ---
        {
            "setup": "import numpy as np\n"
                     "def _probe(fn):\n"
                     "    try:\n"
                     "        fn(1.0, np.array([0.85, 0.02, 0.45, 0.3, 0.8]), np.array([1.0, 0.5, 0.2]),\n"
                     "           np.linspace(-6.0, 6.0, 16, endpoint=False), np.linspace(-6.0, 6.0, 16, endpoint=False), 1.0, 0.05)\n"
                     "    except ValueError:\n"
                     "        return 1\n"
                     "    except Exception:\n"
                     "        return 2\n"
                     "    return 0\n",
            "call": "_probe(method_diagnostics)",
            "gold_call": "_probe(_oracle_method_diagnostics)",
        },
    ]
