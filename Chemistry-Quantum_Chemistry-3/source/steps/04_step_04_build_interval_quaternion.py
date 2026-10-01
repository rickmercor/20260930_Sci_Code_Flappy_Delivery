"""
Implement build_interval_quaternion which represents one piecewise-constant irradiation interval as a rotation quaternion.

Over an interval in which the radio-frequency amplitude, the radio-frequency phase and the
resonance offset are all held constant, the single-spin evolution is a rigid rotation about a
fixed axis. The transverse part of that axis is set by the irradiation amplitude and phase and
the longitudinal part by the instantaneous offset, which here already contains the rotor-modulated
shielding of the crystallite.

Representing each interval by a quaternion rather than by a propagator matrix keeps the
single-spin bookkeeping to four real numbers per interval and makes the composition of many
intervals a sequence of cheap products, which is what allows an effective-field trajectory to be
optimized directly.

A rotation through an angle theta about a unit axis (n_x, n_y, n_z) is represented in
scalar-first form as (cos(theta/2), -sin(theta/2)*n_x, -sin(theta/2)*n_y, -sin(theta/2)*n_z).

Returns
-------
np.ndarray, real array of shape broadcast_shape + (4,) holding the scalar-first rotation quaternion of the interval
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_interval_quaternion(omega_rf: "np.ndarray", phi_rf: "np.ndarray",
                              delta_omega: "np.ndarray", dt: float) -> "np.ndarray":
    '''Build the rotation quaternion of one constant-amplitude irradiation interval.

    Parameters
    ----------
    omega_rf : np.ndarray
        Radio-frequency amplitude in rad/s. Scalars and arrays are accepted
        and are broadcast against the other two spin parameters.
    phi_rf : np.ndarray
        Radio-frequency phase in radians, measured from the x axis of the
        rotating frame.
    delta_omega : np.ndarray
        Instantaneous resonance offset in rad/s, including any rotor-modulated
        shielding contribution.
    dt : float
        Interval duration in seconds. Must be non-negative and finite.

    Returns
    -------
    quaternion : np.ndarray
        Real array of shape broadcast_shape + (4,) holding the scalar-first
        quaternion of the interval. An interval with vanishing amplitude and
        vanishing offset returns the identity quaternion (1, 0, 0, 0).

    Raises
    ------
    ValueError
        If dt is negative or not finite, if any spin parameter is not finite,
        or if the three spin parameters cannot be broadcast together.
    '''
    return quaternion  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_build_interval_quaternion(omega_rf: "np.ndarray", phi_rf: "np.ndarray",
                                      delta_omega: "np.ndarray", dt: float) -> "np.ndarray":
    dt = float(dt)
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    try:
        omega_rf, phi_rf, delta_omega = np.broadcast_arrays(
            np.asarray(omega_rf, dtype=float),
            np.asarray(phi_rf, dtype=float),
            np.asarray(delta_omega, dtype=float))
    except ValueError:
        raise ValueError("omega_rf, phi_rf and delta_omega must be broadcast compatible")
    if not (np.all(np.isfinite(omega_rf)) and np.all(np.isfinite(phi_rf))
            and np.all(np.isfinite(delta_omega))):
        raise ValueError("omega_rf, phi_rf and delta_omega must be finite")

    magnitude = np.sqrt(omega_rf ** 2 + delta_omega ** 2)
    half_angle = magnitude * dt / 2.0
    safe = np.where(magnitude == 0.0, 1.0, magnitude)
    sine = np.sin(half_angle)
    transverse = sine * omega_rf / safe

    quaternion = np.stack([np.cos(half_angle),
                           -transverse * np.cos(phi_rf),
                           -transverse * np.sin(phi_rf),
                           -sine * delta_omega / safe], axis=-1)
    return quaternion

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    common = "import numpy as np\n"
    return [
        # Normal: the first interval of the graded pulse element with the
        # rotor-modulated shielding of a representative crystallite.
        {
            "setup": common,
            "call": "[round(float(v), 10) for v in build_interval_quaternion(2.0 * np.pi * 99882.6002, np.deg2rad(308.58), 2.0 * np.pi * 20.1e3, 2.0e-6)]",
            "gold_call": "[round(float(v), 10) for v in _oracle_build_interval_quaternion(2.0 * np.pi * 99882.6002, np.deg2rad(308.58), 2.0 * np.pi * 20.1e3, 2.0e-6)]",
        },
        # Boundary: an on-resonance interval rotates purely in the transverse
        # plane, so the last component must vanish exactly.
        {
            "setup": common,
            "call": "[round(float(v), 12) for v in build_interval_quaternion(2.0 * np.pi * 100.0e3, 0.9, 0.0, 2.0e-6)]",
            "gold_call": "[round(float(v), 12) for v in _oracle_build_interval_quaternion(2.0 * np.pi * 100.0e3, 0.9, 0.0, 2.0e-6)]",
        },
        # Edge: a free-evolution interval with no irradiation is a rotation about
        # the field axis alone.
        {
            "setup": common,
            "call": "[round(float(v), 12) for v in build_interval_quaternion(0.0, 1.3, 2.0 * np.pi * 45.0e3, 2.0e-6)]",
            "gold_call": "[round(float(v), 12) for v in _oracle_build_interval_quaternion(0.0, 1.3, 2.0 * np.pi * 45.0e3, 2.0e-6)]",
        },
        # Boundary: with neither irradiation nor offset the interval leaves the
        # spin untouched.
        {
            "setup": common,
            "call": "[round(float(v), 12) for v in build_interval_quaternion(0.0, 0.7, 0.0, 2.0e-6)]",
            "gold_call": "[round(float(v), 12) for v in _oracle_build_interval_quaternion(0.0, 0.7, 0.0, 2.0e-6)]",
        },
        # Edge: the quaternion of any interval is a unit quaternion.
        {
            "setup": common + "rows = [(2.0 * np.pi * 8.0e4, 0.3, 2.0 * np.pi * 1.2e5, 2.0e-6), (2.0 * np.pi * 1.0e3, 2.9, -2.0 * np.pi * 9.0e4, 1.0e-5), (0.0, 0.0, 0.0, 3.0e-6)]\n",
            "call": "[round(float(np.linalg.norm(build_interval_quaternion(a, p, d, t))), 12) for a, p, d, t in rows]",
            "gold_call": "[round(float(np.linalg.norm(_oracle_build_interval_quaternion(a, p, d, t))), 12) for a, p, d, t in rows]",
        },
        # Edge: an on-resonance x-phase interval whose flip angle is the target
        # rotation of the optimization.
        {
            "setup": common + "amp = (2.0 * np.pi / 3.0) / 80.0e-6\n",
            "call": "[round(float(v), 12) for v in build_interval_quaternion(amp, 0.0, 0.0, 80.0e-6)]",
            "gold_call": "[round(float(v), 12) for v in _oracle_build_interval_quaternion(amp, 0.0, 0.0, 80.0e-6)]",
        },
        # Normal: a batch of intervals built at once, which pins the broadcast
        # behaviour used by the powder-averaged optimization.
        {
            "setup": common + "amps = 2.0 * np.pi * np.array([[9.9e4, 8.0e4], [1.0e4, 0.0]])\nphs = np.array([[0.1, 2.0], [4.0, 5.5]])\noff = 2.0 * np.pi * np.array([[1.0e4, -6.0e4], [3.0e4, 2.0e4]])\n",
            "call": "[round(float(v), 10) for v in build_interval_quaternion(amps, phs, off, 2.0e-6).ravel()]",
            "gold_call": "[round(float(v), 10) for v in _oracle_build_interval_quaternion(amps, phs, off, 2.0e-6).ravel()]",
        },
        # Invalid input must raise ValueError rather than return a quaternion.
        {
            "setup": "\n\nimport numpy as np\ndef _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "_exception_code(build_interval_quaternion, 1.0, 0.0, 1.0, -2.0e-6)",
            "gold_call": "_exception_code(_oracle_build_interval_quaternion, 1.0, 0.0, 1.0, -2.0e-6)",
        },
    ]
