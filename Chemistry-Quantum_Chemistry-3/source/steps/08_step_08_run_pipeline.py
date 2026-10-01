"""
Implement run_pipeline which refines a piecewise-constant irradiation element against a target rotation and reports the effective field it generates.

An irradiation element intended to survive a large anisotropic shielding is judged by the field
it generates once it is repeated: the field must stay close to the design target across every
crystallite in a powder and across the band of resonance offsets the sample presents. Refining
the element means moving its amplitudes and phases uphill on the score that measures exactly that
agreement, subject to the amplitude ceiling the probe can deliver.

The refinement used here is a fixed-length steepest ascent in which the amplitude increments and
the phase increments are each rescaled by the largest magnitude in their own derivative block, so
that one iteration moves the largest amplitude by the amplitude step and the largest phase by the
phase step. Amplitudes are held inside the permitted range after every iteration. The reported
quantity is the transverse component of the effective field along the design axis, averaged over
the same crystallites and offsets that entered the score.

Returns
-------
float, the crystallite- and offset-averaged x component of the refined element's effective field in Hz, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_pipeline(amplitudes_hz: "np.ndarray", phases_deg: "np.ndarray", dt: float,
                 omega_r: float, omega_aniso: float, eta: float, offsets: "np.ndarray",
                 grid_shape: tuple, flip_angle: float, n_iter: int, amp_step: float,
                 phase_step: float, amp_max: float) -> float:
    '''Refine an irradiation element and report its mean effective-field x component.

    The rotor axis is inclined at the magic angle. The target rotation is a
    rotation through flip_angle about the x axis of the rotating frame, taken
    over the whole modulation period, which is the interval duration times the
    number of intervals.

    Each iteration evaluates the score and its derivatives for the current
    element, divides the amplitude derivatives by the largest absolute
    amplitude derivative and the phase derivatives by the largest absolute
    phase derivative, adds amp_step times the rescaled amplitude derivatives
    to the amplitudes and phase_step times the rescaled phase derivatives to
    the phases, and then clips the amplitudes to the interval from zero to
    amp_max. A derivative block whose largest magnitude is zero leaves its
    parameters unchanged.

    Parameters
    ----------
    amplitudes_hz : np.ndarray
        Real array of shape (n_steps,) holding the starting radio-frequency
        amplitude of each interval in Hz.
    phases_deg : np.ndarray
        Real array of shape (n_steps,) holding the starting radio-frequency
        phase of each interval in degrees.
    dt : float
        Duration of one interval in seconds.
    omega_r : float
        Sample spinning frequency in rad/s.
    omega_aniso : float
        Anisotropic shielding frequency in rad/s.
    eta : float
        Shielding asymmetry parameter, 0 <= eta <= 1.
    offsets : np.ndarray
        Real array of shape (n_offsets,) holding the isotropic resonance
        offsets in rad/s over which the element is scored and averaged.
    grid_shape : tuple
        Three positive integers giving the number of polar, first azimuthal
        and second azimuthal crystallite nodes.
    flip_angle : float
        Target rotation angle in radians about the x axis.
    n_iter : int
        Number of steepest-ascent iterations. Must be non-negative. Zero
        returns the field of the unrefined element.
    amp_step : float
        Largest amplitude increment applied in one iteration, in rad/s.
    phase_step : float
        Largest phase increment applied in one iteration, in radians.
    amp_max : float
        Upper bound on the radio-frequency amplitude in rad/s.

    Returns
    -------
    mean_field_x : float
        The crystallite- and offset-averaged x component of the effective
        field of the refined element, converted to Hz, as a native Python
        float.

    Raises
    ------
    ValueError
        If n_iter is negative or not an integer, if amp_max is not positive
        and finite, if amp_step or phase_step is negative, if grid_shape is
        not a sequence of three integers, or if any argument forwarded to an
        earlier stage fails its own validation.
    '''
    return mean_field_x  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_pipeline(amplitudes_hz: "np.ndarray", phases_deg: "np.ndarray", dt: float,
                         omega_r: float, omega_aniso: float, eta: float, offsets: "np.ndarray",
                         grid_shape: tuple, flip_angle: float, n_iter: int, amp_step: float,
                         phase_step: float, amp_max: float) -> float:
    if isinstance(n_iter, bool) or not isinstance(n_iter, (int, np.integer)):
        raise ValueError("n_iter must be an integer")
    if int(n_iter) < 0:
        raise ValueError("n_iter must be non-negative")
    amp_max = float(amp_max)
    if not np.isfinite(amp_max) or amp_max <= 0.0:
        raise ValueError("amp_max must be positive and finite")
    amp_step = float(amp_step)
    phase_step = float(phase_step)
    if not (np.isfinite(amp_step) and np.isfinite(phase_step)) or amp_step < 0.0 or phase_step < 0.0:
        raise ValueError("amp_step and phase_step must be finite and non-negative")
    try:
        shape = tuple(grid_shape)
    except TypeError:
        raise ValueError("grid_shape must hold three integers")
    if len(shape) != 3:
        raise ValueError("grid_shape must hold three integers")

    beta_rl = float(np.arccos(1.0 / np.sqrt(3.0)))
    offsets = np.asarray(offsets, dtype=float)
    amplitudes = 2.0 * np.pi * np.asarray(amplitudes_hz, dtype=float)
    phases = np.deg2rad(np.asarray(phases_deg, dtype=float))
    n_steps = amplitudes.size
    tau_m = float(dt) * n_steps

    half = 0.5 * float(flip_angle)
    q_target = np.array([np.cos(half), -np.sin(half), 0.0, 0.0])

    crystallites = _oracle_build_powder_grid(int(shape[0]), int(shape[1]), int(shape[2]))

    for _ in range(int(n_iter)):
        result = _oracle_compute_fidelity_and_gradient(amplitudes, phases, dt, offsets,
                                                       crystallites, omega_aniso, eta,
                                                       beta_rl, omega_r, q_target)
        gradient_amplitude = result[1:n_steps + 1]
        gradient_phase = result[n_steps + 1:]
        scale_amplitude = np.max(np.abs(gradient_amplitude))
        scale_phase = np.max(np.abs(gradient_phase))
        if scale_amplitude > 0.0:
            amplitudes = np.clip(amplitudes + amp_step * gradient_amplitude / scale_amplitude,
                                 0.0, amp_max)
        if scale_phase > 0.0:
            phases = phases + phase_step * gradient_phase / scale_phase

    times = np.arange(1, n_steps + 1) * float(dt)
    components = np.array([
        _oracle_compute_shielding_fourier_components(0.0, omega_aniso, eta,
                                                     row[0], row[1], row[2], beta_rl)
        for row in crystallites])
    base = _oracle_compute_offset_trajectory(components, omega_r, times)
    delta = (base[:, None, :] + offsets[None, :, None]).reshape(-1, n_steps)
    weights = np.repeat(crystallites[:, 3], offsets.size) / offsets.size

    quaternions = _oracle_build_interval_quaternion(amplitudes[None, :], phases[None, :],
                                                    delta, dt)
    q_total = _oracle_compose_quaternion_sequence(quaternions)
    fields = _oracle_extract_effective_field(q_total, tau_m)
    return float(weights @ fields[:, 0] / (2.0 * np.pi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    shape_rows = [
        (99882.6002, 308.58), (97593.2747, 91.90), (99938.7784, 322.74), (99788.5816, 235.95),
        (98277.6433, 158.68), (99577.6807, 107.58), (98881.0528, 55.28), (94404.7994, 30.57),
        (99306.0922, 8.96), (99774.6973, 6.07), (98040.4008, 358.64), (99810.2557, 19.22),
        (91823.9307, 35.18), (99879.0133, 64.16), (99432.6134, 115.58), (99913.2100, 171.43),
        (99930.7510, 249.82), (99937.6518, 334.75), (99206.1576, 84.39), (99416.0881, 205.60),
        (99192.2675, 18.55), (75896.7021, 262.57), (99512.2651, 169.77), (99014.8033, 79.16),
        (99886.4254, 9.71), (99490.0204, 311.73), (99904.9940, 264.74), (99625.5643, 231.86),
        (99500.3110, 212.56), (99651.1342, 208.43), (99852.4588, 204.51), (99754.5915, 222.47),
        (99205.2418, 242.32), (99857.1682, 268.75), (99945.0147, 319.18), (99554.7653, 13.80),
        (99728.1894, 94.72), (99434.0476, 193.81), (71090.5744, 327.26), (98881.1991, 38.66)]
    element = ("import numpy as np\n"
               "amp_hz = np.array(" + repr([row[0] for row in shape_rows]) + ")\n"
               "ph_deg = np.array(" + repr([row[1] for row in shape_rows]) + ")\n"
               "omega_r = 2.0 * np.pi * 25.0e3\n"
               "flip = 2.0 * np.pi / 3.0\n"
               "amp_max = 2.0 * np.pi * 100.0e3\n")
    return [
        # Normal: the graded configuration, a 80 kHz anisotropy scored over a
        # 70 kHz offset band and refined for ten iterations.
        {
            "setup": element + "off = 2.0 * np.pi * np.linspace(-35.0e3, 35.0e3, 7)\n",
            "call": "round(run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 80.0e3, 0.5, off, (6, 6, 3), flip, 10, 2.0 * np.pi * 200.0, 0.01, amp_max), 9)",
            "gold_call": "round(_oracle_run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 80.0e3, 0.5, off, (6, 6, 3), flip, 10, 2.0 * np.pi * 200.0, 0.01, amp_max), 9)",
        },
        # Boundary: no refinement at all, which reports the field of the starting
        # element and isolates the analysis from the ascent.
        {
            "setup": element + "off = 2.0 * np.pi * np.linspace(-35.0e3, 35.0e3, 7)\n",
            "call": "round(run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 80.0e3, 0.5, off, (6, 6, 3), flip, 0, 2.0 * np.pi * 200.0, 0.01, amp_max), 9)",
            "gold_call": "round(_oracle_run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 80.0e3, 0.5, off, (6, 6, 3), flip, 0, 2.0 * np.pi * 200.0, 0.01, amp_max), 9)",
        },
        # Boundary: without any anisotropy the element recovers the field it was
        # designed to produce.
        {
            "setup": element + "off = np.array([0.0])\n",
            "call": "round(run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 0.0, 0.5, off, (2, 2, 1), flip, 0, 2.0 * np.pi * 200.0, 0.01, amp_max), 9)",
            "gold_call": "round(_oracle_run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 0.0, 0.5, off, (2, 2, 1), flip, 0, 2.0 * np.pi * 200.0, 0.01, amp_max), 9)",
        },
        # Edge: an axially symmetric shielding tensor, which changes the rotor
        # modulation seen by every crystallite.
        {
            "setup": element + "off = 2.0 * np.pi * np.linspace(-35.0e3, 35.0e3, 5)\n",
            "call": "round(run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 80.0e3, 0.0, off, (4, 4, 2), flip, 4, 2.0 * np.pi * 200.0, 0.01, amp_max), 9)",
            "gold_call": "round(_oracle_run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 80.0e3, 0.0, off, (4, 4, 2), flip, 4, 2.0 * np.pi * 200.0, 0.01, amp_max), 9)",
        },
        # Edge: a much larger anisotropy than the element was designed for, which
        # drives the effective field away from the target.
        {
            "setup": element + "off = 2.0 * np.pi * np.array([-50.0e3, 0.0, 50.0e3])\n",
            "call": "round(run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 200.0e3, 0.5, off, (4, 3, 2), flip, 3, 2.0 * np.pi * 500.0, 0.02, amp_max), 9)",
            "gold_call": "round(_oracle_run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 200.0e3, 0.5, off, (4, 3, 2), flip, 3, 2.0 * np.pi * 500.0, 0.02, amp_max), 9)",
        },
        # Edge: a tighter amplitude ceiling, so the clipping in the ascent becomes
        # the binding constraint.
        {
            "setup": element + "off = 2.0 * np.pi * np.linspace(-20.0e3, 20.0e3, 3)\n",
            "call": "round(run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 80.0e3, 0.5, off, (3, 3, 2), flip, 5, 2.0 * np.pi * 1000.0, 0.05, 2.0 * np.pi * 90.0e3), 9)",
            "gold_call": "round(_oracle_run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 80.0e3, 0.5, off, (3, 3, 2), flip, 5, 2.0 * np.pi * 1000.0, 0.05, 2.0 * np.pi * 90.0e3), 9)",
        },
        # Edge: a different design rotation, which moves the target quaternion and
        # therefore the whole ascent trajectory.
        {
            "setup": element + "off = 2.0 * np.pi * np.array([-15.0e3, 15.0e3])\n",
            "call": "round(run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 60.0e3, 0.3, off, (3, 2, 2), np.pi / 2.0, 6, 2.0 * np.pi * 300.0, 0.02, amp_max), 9)",
            "gold_call": "round(_oracle_run_pipeline(amp_hz, ph_deg, 2.0e-6, omega_r, 2.0 * np.pi * 60.0e3, 0.3, off, (3, 2, 2), np.pi / 2.0, 6, 2.0 * np.pi * 300.0, 0.02, amp_max), 9)",
        },
        # Invalid input must raise ValueError rather than return a field.
        {
            "setup": "\n\nimport numpy as np\ndef _exception_code(function, *args, **kwargs):\n    try:\n        function(*args, **kwargs)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
            "call": "_exception_code(run_pipeline, np.ones(4) * 1.0e5, np.zeros(4), 2.0e-6, 1.0e5, 1.0e5, 0.5, np.zeros(1), (2, 2, 1), 1.0, -1, 1.0, 0.01, 1.0e6)",
            "gold_call": "_exception_code(_oracle_run_pipeline, np.ones(4) * 1.0e5, np.zeros(4), 2.0e-6, 1.0e5, 1.0e5, 0.5, np.zeros(1), (2, 2, 1), 1.0, -1, 1.0, 0.01, 1.0e6)",
        },
    ]
