"""
Chain every preceding step into the graded number by integrating the driven gap dynamics and returning the deepest suppression the pulse produces. The routine determines the critical temperature from the pairing parameters, places the simulation and reference temperatures at the requested fractions of it, builds the equilibrium gap and the inertia coefficient there, forms the drive coefficient, then integrates the equation of motion from equilibrium at rest and reports the minimum gap magnitude divided by its equilibrium value. Invalid input raises ValueError: every scalar argument must be finite, the cutoff, coupling, damping and t_max must be strictly positive, the drive strength must be non-negative, and the two temperature fractions must lie strictly between zero and one.

For a uniform sample the order parameter carries no phase winding and stays real and positive, so the gauge-covariant gradient term collapses to a contribution proportional to the squared vector potential and the whole problem becomes a single driven nonlinear ordinary differential equation. The restoring force is the difference between the pairing term and the gap kernel, and these two cancel exactly at equilibrium, which is why an undriven run stays at the equilibrium gap forever and why the pulse is what sets the amplitude in motion. Because the equation carries a second time derivative the amplitude oscillates about equilibrium rather than relaxing toward it, and the coefficient of that derivative therefore decides how far the pulse drives the gap down. Energies are millielectronvolts and times are picoseconds, which are independent units rather than a natural-unit pair, so the inertial term needs the squared reduced Planck constant to carry the same units as the energies it is balanced against.

Returns
-------
float, the minimum gap magnitude during the pulse divided by the equilibrium gap
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def run_pipeline(cutoff, coupling, sim_fraction, ref_fraction, drive_strength,
                 damping, frequency, sigma_before, sigma_after, phase,
                 t_center, t_max):
    """Return the minimum gap magnitude during the pulse divided by its equilibrium value.

    Parameters
    ----------
    cutoff : float
        Symmetric band-energy cutoff in millielectronvolts.
    coupling : float
        Dimensionless product of the pairing constant with the density of states.
    sim_fraction : float
        Simulation temperature as a fraction of the critical temperature.
    ref_fraction : float
        Drive-normalization reference temperature as a fraction of the critical temperature.
    drive_strength : float
        Dimensionless pump strength fixing the squared pulse amplitude.
    damping : float
        Damping coefficient of the first time derivative, in picoseconds.
    frequency : float
        Pulse carrier frequency in terahertz.
    sigma_before, sigma_after : float
        Gaussian widths in picoseconds before and after the pulse centre.
    phase : float
        Carrier phase in radians.
    t_center : float
        Pulse centre in picoseconds.
    t_max : float
        End of the integration window in picoseconds.

    Returns
    -------
    float
        Minimum gap magnitude reached during the window divided by the
        equilibrium gap magnitude at the simulation temperature.

    Raises
    ------
    ValueError
        If any argument is not a finite scalar, if ``cutoff``, ``coupling``,
        ``damping`` or ``t_max`` is not strictly positive, if ``drive_strength``
        is negative, or if either temperature fraction is outside the open
        interval from zero to one.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp


def _oracle_run_pipeline(cutoff, coupling, sim_fraction, ref_fraction, drive_strength,
                       damping, frequency, sigma_before, sigma_after, phase,
                       t_center, t_max):
    scalars = (cutoff, coupling, sim_fraction, ref_fraction, drive_strength,
               damping, frequency, sigma_before, sigma_after, phase, t_center, t_max)
    converted = []
    for value in scalars:
        # A string is a scalar to numpy but has no float value, and asking
        # numpy whether it is finite raises TypeError rather than the ValueError
        # this function documents. Convert first and turn every failure into the
        # documented error.
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError("every argument must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("every argument must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError("every argument must be a finite scalar")
        converted.append(numeric)
    (cutoff, coupling, sim_fraction, ref_fraction, drive_strength,
     damping, frequency, sigma_before, sigma_after, phase,
     t_center, t_max) = converted
    if cutoff <= 0.0 or coupling <= 0.0:
        raise ValueError("cutoff and coupling must be strictly positive")
    if damping <= 0.0 or t_max <= 0.0:
        raise ValueError("damping and t_max must be strictly positive")
    if drive_strength < 0.0:
        raise ValueError("drive_strength must be non-negative")
    for fraction in (sim_fraction, ref_fraction):
        if not 0.0 < fraction < 1.0:
            raise ValueError("temperature fractions must lie strictly between zero and one")

    t_c = float(_oracle_critical_temperature(cutoff, coupling))
    t_sim = sim_fraction * t_c
    t_ref = ref_fraction * t_c

    gap_0 = float(_oracle_equilibrium_gap(t_sim, cutoff, coupling))
    if gap_0 <= 0.0:
        raise ValueError("no superconducting solution at the simulation temperature")

    inertia = float(_oracle_inertia_coefficient(gap_0, t_sim, cutoff))
    coefficient = float(_oracle_drive_coefficient(drive_strength, t_sim, t_ref, cutoff, coupling))
    inverse_coupling = 1.0 / coupling

    # The reduced inertia coefficient carries units of inverse energy squared,
    # because its outer weight contributes one inverse energy, the derivative with
    # respect to quasiparticle energy contributes two more and the band measure
    # returns one. Multiplying the second time derivative by it therefore lands in
    # inverse energy per squared time while every other term is an energy, and the
    # two are reconciled by the squared reduced Planck constant, which carries
    # squared energy times squared time. The damping value is already a time, so
    # its term is an energy as it stands and takes no further factor.
    hbar_mev_ps = 0.6582119569
    inertial = hbar_mev_ps * hbar_mev_ps * inertia

    def derivative(t, state):
        gap, rate = state
        envelope = float(np.asarray(_oracle_pulse_envelope(
            np.array([t]), t_center, sigma_before, sigma_after, frequency, phase))[0])
        safe_gap = max(gap, 1e-12)
        kernel = float(_oracle_gap_kernel(safe_gap, t_sim, cutoff))
        force = (-coefficient * envelope ** 2 * gap
                 - inverse_coupling * gap
                 + gap * kernel)
        return [rate, (force - damping * rate) / inertial]

    solution = solve_ivp(derivative, (0.0, t_max), [gap_0, 0.0], dense_output=True,
                         rtol=1e-11, atol=1e-14, max_step=0.01)
    if not solution.success:
        raise ValueError("the gap dynamics failed to integrate")

    # The target is the smallest magnitude anywhere in the window, so a sampled
    # grid is not enough: the extremum falls between samples, and refining one
    # grid family by powers of two keeps landing on the same points and looks
    # convergent while missing it. Scan coarsely for the bracket, then descend
    # on the dense output by golden-section, which needs no derivative and
    # cannot step outside the bracket.
    scan = np.linspace(0.0, t_max, 4001)
    magnitude = np.abs(solution.sol(scan)[0])
    index = int(np.argmin(magnitude))
    left = scan[max(index - 1, 0)]
    right = scan[min(index + 1, scan.size - 1)]

    def magnitude_at(t):
        return abs(float(np.asarray(solution.sol(t))[0]))

    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = float(left), float(right)
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = magnitude_at(c), magnitude_at(d)
    for _ in range(200):
        if b - a < 1e-13:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = magnitude_at(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = magnitude_at(d)
    minimum = min(magnitude_at(0.5 * (a + b)), float(magnitude[index]))
    return float(minimum / gap_0)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "round(float(run_pipeline(2.6, 1.11, 0.5, 0.8, 0.4, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)), 9)",
            "gold_call": "round(float(_oracle_run_pipeline(2.6, 1.11, 0.5, 0.8, 0.4, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)), 9)",
        },
        {
            "setup": "",
            "call": "round(float(run_pipeline(2.6, 1.11, 0.5, 0.8, 0.0, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)), 9)",
            "gold_call": "round(float(_oracle_run_pipeline(2.6, 1.11, 0.5, 0.8, 0.0, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)), 9)",
        },
        {
            "setup": "",
            "call": "int(bool(0.0 < run_pipeline(2.6, 1.11, 0.5, 0.8, 0.2, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0) <= 1.0))",
            "gold_call": "int(bool(0.0 < _oracle_run_pipeline(2.6, 1.11, 0.5, 0.8, 0.2, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0) <= 1.0))",
        },
        {
            "setup": "",
            "call": "int(bool(run_pipeline(2.6, 1.11, 0.5, 0.8, 0.6, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0) < run_pipeline(2.6, 1.11, 0.5, 0.8, 0.2, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)))",
            "gold_call": "int(bool(_oracle_run_pipeline(2.6, 1.11, 0.5, 0.8, 0.6, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0) < _oracle_run_pipeline(2.6, 1.11, 0.5, 0.8, 0.2, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)))",
        },
        {
            "setup": "def run_model():\n    try:\n        run_pipeline(2.6, 1.11, 1.5, 0.8, 0.4, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_run_pipeline(2.6, 1.11, 1.5, 0.8, 0.4, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        run_pipeline(2.6, 1.11, 0.5, 0.8, -0.1, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_run_pipeline(2.6, 1.11, 0.5, 0.8, -0.1, 0.03, 0.6, 1.341, 5.065, -6.33, 4.76, 40.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
