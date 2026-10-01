"""
Implement compute_offset_trajectory which reconstructs the instantaneous shielding frequency seen by a spin during sample rotation.

The rotor-modulated Fourier coefficients of a shielding interaction determine a periodic
frequency that a spin experiences as the sample turns. Evaluating that series at the sampling
instants of a piecewise-constant pulse sequence converts the modulated interaction into one
effective resonance offset per interval, which is what a piecewise-constant propagation needs.

The reconstructed frequency is real because the coefficients of opposite order are complex
conjugates of one another. Its average over a complete rotor period vanishes for a purely
anisotropic interaction whenever the rotor is inclined at the magic angle, which is the
statement that sample rotation averages the anisotropy away in the absence of irradiation.

Returns
-------
np.ndarray, real array of shape components.shape[:-1] + (n_steps,) holding the instantaneous shielding frequency in rad/s at each sampling instant
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_offset_trajectory(components: "np.ndarray", omega_r: float,
                              times: "np.ndarray") -> "np.ndarray":
    '''Evaluate the rotor-modulated shielding frequency at a set of instants.

    Parameters
    ----------
    components : np.ndarray
        Complex array whose trailing axis has length 5 and holds the Fourier
        coefficients for m = -2, -1, 0, 1, 2 in that order. Leading axes are
        treated as independent crystallites.
    omega_r : float
        Sample spinning frequency in rad/s.
    times : np.ndarray
        Real array of shape (n_steps,) holding the sampling instants in
        seconds.

    Returns
    -------
    trajectory : np.ndarray
        Real array of shape components.shape[:-1] + (n_steps,) holding the
        shielding frequency in rad/s at each instant, obtained as the real
        part of the reconstructed series.

    Raises
    ------
    ValueError
        If the trailing axis of components does not have length 5, if times
        is not one dimensional, if omega_r is not finite, or if times
        contains values that are not finite.
    '''
    return trajectory  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_offset_trajectory(components: "np.ndarray", omega_r: float,
                                      times: "np.ndarray") -> "np.ndarray":
    components = np.asarray(components, dtype=complex)
    times = np.asarray(times, dtype=float)
    if components.ndim < 1 or components.shape[-1] != 5:
        raise ValueError("components must have a trailing axis of length 5")
    if times.ndim != 1:
        raise ValueError("times must be a one-dimensional array")
    if not np.isfinite(float(omega_r)):
        raise ValueError("omega_r must be finite")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")

    orders = np.arange(-2, 3)
    phases = np.exp(1j * float(omega_r) * np.outer(times, orders))
    return np.real(np.tensordot(components, phases, axes=([-1], [1])))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    magic = ("import numpy as np\n"
             "beta_rl = float(np.arccos(1.0 / np.sqrt(3.0)))\n"
             "omega_r = 2.0 * np.pi * 25.0e3\n"
             "times = np.arange(1, 41) * 2.0e-6\n"
             "comp = _oracle_compute_shielding_fourier_components(0.0, 2.0 * np.pi * 80.0e3, 0.5, 0.3, 0.9, 1.7, beta_rl)\n")
    return [
        # Normal: the graded crystallite sampled on the 40 pulse intervals of the
        # modulation period.
        {
            "setup": magic,
            "call": "[round(float(v), 6) for v in compute_offset_trajectory(comp, omega_r, times)]",
            "gold_call": "[round(float(v), 6) for v in _oracle_compute_offset_trajectory(comp, omega_r, times)]",
        },
        # Boundary: coefficients that carry only a stationary term reproduce a
        # constant frequency at every instant.
        {
            "setup": "import numpy as np\ncomp0 = np.array([0.0, 0.0, 2.0 * np.pi * 7.5e3, 0.0, 0.0], dtype=complex)\ntimes = np.linspace(0.0, 5.0e-5, 9)\n",
            "call": "[round(float(v), 8) for v in compute_offset_trajectory(comp0, 2.0 * np.pi * 25.0e3, times)]",
            "gold_call": "[round(float(v), 8) for v in _oracle_compute_offset_trajectory(comp0, 2.0 * np.pi * 25.0e3, times)]",
        },
        # Edge: a purely anisotropic interaction at the magic angle averages to
        # zero over a complete rotor period.
        {
            "setup": magic + "cycle = np.arange(20) * (1.0 / 25.0e3) / 20.0\n",
            "call": "round(float(np.mean(compute_offset_trajectory(comp, omega_r, cycle))), 6)",
            "gold_call": "round(float(np.mean(_oracle_compute_offset_trajectory(comp, omega_r, cycle))), 6)",
        },
        # Edge: several crystallites evaluated at once, which pins the broadcast
        # shape as well as the values.
        {
            "setup": ("import numpy as np\n"
                      "beta_rl = float(np.arccos(1.0 / np.sqrt(3.0)))\n"
                      "stack = np.array([_oracle_compute_shielding_fourier_components(2.0 * np.pi * 1.0e3, 2.0 * np.pi * 80.0e3, 0.5, a, b, g, beta_rl) for a, b, g in [(0.0, 0.5, 0.0), (1.0, 1.5, 2.0), (2.5, 2.5, 4.0)]])\n"
                      "times = np.arange(1, 7) * 4.0e-6\n"),
            "call": "[round(float(v), 6) for v in compute_offset_trajectory(stack, 2.0 * np.pi * 25.0e3, times).ravel()]",
            "gold_call": "[round(float(v), 6) for v in _oracle_compute_offset_trajectory(stack, 2.0 * np.pi * 25.0e3, times).ravel()]",
        },
        # Boundary: at zero time the reconstruction reduces to the sum of all
        # five coefficients.
        {
            "setup": magic,
            "call": "round(float(compute_offset_trajectory(comp, omega_r, np.array([0.0]))[0]), 6)",
            "gold_call": "round(float(_oracle_compute_offset_trajectory(comp, omega_r, np.array([0.0]))[0]), 6)",
        },
        # Edge: the trajectory is periodic in the rotor period, so instants
        # separated by one full turn must agree.
        {
            "setup": magic + "pair = np.array([3.0e-6, 3.0e-6 + 1.0 / 25.0e3, 17.0e-6, 17.0e-6 + 2.0 / 25.0e3])\n",
            "call": "[round(float(v), 6) for v in compute_offset_trajectory(comp, omega_r, pair)]",
            "gold_call": "[round(float(v), 6) for v in _oracle_compute_offset_trajectory(comp, omega_r, pair)]",
        },
        # Edge: a static sample, where the rotor phase never advances and the
        # anisotropy survives as a constant shift.
        {
            "setup": magic,
            "call": "[round(float(v), 6) for v in compute_offset_trajectory(comp, 0.0, times[:5])]",
            "gold_call": "[round(float(v), 6) for v in _oracle_compute_offset_trajectory(comp, 0.0, times[:5])]",
        },
        # Invalid input must raise ValueError rather than return a trajectory.
        {
            "setup": "\n\nimport numpy as np\ndef _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "_exception_code(compute_offset_trajectory, np.zeros(4, dtype=complex), 1.0, np.zeros(3))",
            "gold_call": "_exception_code(_oracle_compute_offset_trajectory, np.zeros(4, dtype=complex), 1.0, np.zeros(3))",
        },
    ]
