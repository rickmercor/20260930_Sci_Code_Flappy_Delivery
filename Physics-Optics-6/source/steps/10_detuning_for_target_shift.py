"""
Find the operating point at which the resonator actually delivers a prescribed frequency shift, by driving the full delayed-response model rather than the slowly varying estimate of it.

The delayed-response theory that keeps the finite duration of the vibrational ringing names a pulse duration, and hence a detuning, for any requested shift. It is a better estimate than the slowly varying one, but it is still a perturbative statement about an isolated pulse, so the detuning it names is not the detuning at which the driven resonator does what was asked.

A driven resonator holds a stationary pulse only above a detuning that rises with the drive amplitude; below that limit the circulating field breathes rather than settling. Each trial costs a complete settling run.

Returns
-------
float, the dimensionless detuning at which the full delayed-response model delivers the requested shift.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def detuning_for_target_shift(lambda0: float, D1: float, D2: float, Q_int: float, Q_ext: float,
                              f_R: float, tau1: float, tau2: float, omega_target: float,
                              pump_power: float, zeta_floor: float, n_modes: int, dtau: float,
                              n_steps: int) -> float:
    '''Return the dimensionless detuning at which the full model gives omega_target.

    At a given dimensionless detuning the resonator is simulated exactly as the rest of this
    pipeline simulates it: the periodic fast-time window is the round-trip time expressed in
    units of the characteristic dispersive duration, the field starts from the conservative
    pulse at that detuning, the drive amplitude is the square root of pump_power, the delayed
    response is kept in full, the field is advanced for n_steps steps of dtau, and the shift is
    read off the comb of the field that results.

    The value returned is the detuning at which that simulated shift equals omega_target. It is
    sought only on the closed interval whose lower end is zeta_floor and whose upper end is the
    detuning the delayed-response theory assigns to omega_target, that is the square of the ratio
    of the characteristic dispersive duration to the pulse duration that theory names. The
    simulated shift is strictly decreasing on that interval, so the value is unique there.
    Return it accurate to a relative precision of 1e-4 or better.

    Parameters
    ----------
    lambda0 : float
        Vacuum wavelength of the resonance in metres.
    D1 : float
        Free spectral range as an angular frequency in radians per second.
    D2 : float
        Second-order dispersion of the mode family in radians per second.
    Q_int : float
        Intrinsic quality factor.
    Q_ext : float
        External coupling quality factor.
    f_R : float
        Fraction of the nonlinear response carried by the delayed channel.
    tau1 : float
        Vibrational period in seconds.
    tau2 : float
        Vibrational lifetime in seconds.
    omega_target : float
        Required angular frequency shift in radians per second, strictly negative.
    pump_power : float
        Dimensionless drive power, the square of the drive amplitude.
    zeta_floor : float
        Lower end of the search interval, a dimensionless detuning at or above which the driven
        field settles to a stationary pulse at this drive power.
    n_modes : int
        Number of resonator modes retained, equal to the number of fast-time samples.
    dtau : float
        Integration step in dimensionless slow time, used for every trial.
    n_steps : int
        Number of integration steps in every trial.

    Returns
    -------
    zeta : float
        The dimensionless detuning at which the full model delivers omega_target, as a native
        Python float.

    Raises
    ------
    ValueError
        If f_R is not strictly positive, or omega_target is not strictly negative; both leave
        the delayed-response estimate that closes the search interval undefined. Also if
        omega_target does not lie between the shifts the full model produces at the two ends of
        that interval, so that the interval contains no such detuning: the shift at zeta_floor
        must be above omega_target and the shift at the upper end must be below it.
    '''
    return zeta  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_detuning_for_target_shift(lambda0: float, D1: float, D2: float, Q_int: float, Q_ext: float,
                                      f_R: float, tau1: float, tau2: float, omega_target: float,
                                      pump_power: float, zeta_floor: float, n_modes: int, dtau: float,
                                      n_steps: int) -> float:
    if not (float(f_R) > 0.0):
        raise ValueError("f_R must be strictly positive")
    if not (float(omega_target) < 0.0):
        raise ValueError("omega_target must be strictly negative")

    scales = _oracle_cavity_scales(lambda0, D1, D2, Q_int, Q_ext)
    tau0 = scales[2]
    window = scales[3] / tau0
    kernel = _oracle_raman_kernel_spectrum(n_modes, window, tau0, tau1, tau2)
    amplitude = np.sqrt(pump_power)
    target = float(omega_target)

    def _shift(zeta):
        seed = _oracle_soliton_seed(zeta, n_modes, window)
        props = _oracle_lle_propagators(zeta, n_modes, window, dtau)
        spectrum = _oracle_propagate_soliton(seed, kernel, props, f_R, amplitude, dtau, n_steps)
        return _oracle_soliton_observables(spectrum, window, tau0)[1]

    # the delayed-response estimate over-assigns the detuning, so it closes the interval from above
    lower = float(zeta_floor)
    upper = (tau0 / _oracle_nonadiabatic_soliton_duration(target, tau0, f_R, tau1, tau2)) ** 2
    if not (upper > lower):
        raise ValueError("the delayed-response estimate does not lie above zeta_floor")
    # the shift is monotone on this interval, so bracketing it here is what makes the root unique
    if not (_shift(lower) > target):
        raise ValueError("the shift at zeta_floor is already below omega_target")
    if not (_shift(upper) < target):
        raise ValueError("the shift at the delayed-response estimate does not reach omega_target")

    for _ in range(60):
        mid = 0.5 * (lower + upper)
        if _shift(mid) > target:
            lower = mid
        else:
            upper = mid
        if upper - lower < 1e-5 * upper:
            break
    return float(0.5 * (lower + upper))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    device = """import numpy as np
lam = 1546e-9
D1 = 2.0 * np.pi * 1.02e12
D2 = 2.0 * np.pi * 41.2e6
Qi = 6.2e6
Qe = 2.6e6
tau1 = 2.0 * np.pi * 12.2e-15
tau2 = 32.0e-15
"""
    # The answer is a root of a simulated quantity, so it carries two spreads: the search's own
    # stopping tolerance, 1e-4 relative by contract, which is 0.0014 on a detuning of order ten,
    # and the envelope fit's optimiser route. A tolerance of 5e-3 absolute covers both. It stays
    # far below any error that matters: reading the comb with an intensity-weighted mean instead
    # of the envelope fit moves the root by about 0.8, and a coarser grid by 0.05.
    return [
        # Normal: the benchmark target, on a grid coarse enough to keep the search affordable.
        # Root 13.5748 on [10, 17.7959]; the shift there falls from -295.0 GHz to -792.9 GHz
        # without reversing, so the bracket holds one detuning and the search cannot land
        # elsewhere.
        {
            "setup": device,
            "call": ("detuning_for_target_shift(lam, D1, D2, Qi, Qe, 0.0217, tau1, tau2, "
                     "-2.0*np.pi*500e9, 20.0, 10.0, 128, 8.0e-4, 20000)"),
            "gold_call": ("_oracle_detuning_for_target_shift(lam, D1, D2, Qi, Qe, 0.0217, tau1, tau2, "
                          "-2.0*np.pi*500e9, 20.0, 10.0, 128, 8.0e-4, 20000)"),
            "tol": 5e-3,
        },
        # Boundary: a weaker drive, whose stationary branch begins lower, with a proportionately
        # smaller target. Root 9.5960 on [7.5, 12.5836]. The floor moves with the drive, so a
        # search that hard-codes the benchmark's floor is bracketing the wrong interval here.
        {
            "setup": device,
            "call": ("detuning_for_target_shift(lam, D1, D2, Qi, Qe, 0.0217, tau1, tau2, "
                     "-2.0*np.pi*250e9, 12.0, 7.5, 128, 8.0e-4, 20000)"),
            "gold_call": ("_oracle_detuning_for_target_shift(lam, D1, D2, Qi, Qe, 0.0217, tau1, tau2, "
                          "-2.0*np.pi*250e9, 12.0, 7.5, 128, 8.0e-4, 20000)"),
            "tol": 5e-3,
        },
        # Edge: half the delayed fraction, which weakens the shift at every detuning and so
        # moves the root far up the branch, to 17.4940 on [10, 22.3562]. The answer is well
        # above the benchmark's, testing that the search is not anchored to it.
        {
            "setup": device,
            "call": ("detuning_for_target_shift(lam, D1, D2, Qi, Qe, 0.011, tau1, tau2, "
                     "-2.0*np.pi*400e9, 20.0, 10.0, 128, 8.0e-4, 20000)"),
            "gold_call": ("_oracle_detuning_for_target_shift(lam, D1, D2, Qi, Qe, 0.011, tau1, tau2, "
                          "-2.0*np.pi*400e9, 20.0, 10.0, 128, 8.0e-4, 20000)"),
            "tol": 5e-3,
        },
        # Invalid: a non-negative target leaves the delayed-response estimate undefined, so the
        # interval cannot be closed at all.
        {
            "setup": device + """
def run(fn):
    try:
        fn(lam, D1, D2, Qi, Qe, 0.0217, tau1, tau2, 0.0, 20.0, 10.0, 64, 4.0e-4, 10)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run(detuning_for_target_shift)",
            "gold_call": "run(_oracle_detuning_for_target_shift)",
        },
        # Invalid: a target the interval does not contain. A shift of only -50 GHz needs a pulse
        # so long that the delayed-response theory places it below zeta_floor, so the interval is
        # empty before a single trial is run. An implementation that does not check the interval
        # before searching returns a number here, or bisects on a reversed bracket.
        {
            "setup": device + """
def run_unbracketed(fn):
    try:
        fn(lam, D1, D2, Qi, Qe, 0.0217, tau1, tau2, -2.0*np.pi*50e9, 20.0, 10.0, 64, 4.0e-4, 3000)
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_unbracketed(detuning_for_target_shift)",
            "gold_call": "run_unbracketed(_oracle_detuning_for_target_shift)",
        },
    ]
