"""
Compute the certified asymptotic secret-key rate per emitted signal of a decoy-state protocol whose transmitter prepares its nominal intensities only to within a bounded relative deviation, with the deviations correlated across a finite span of rounds.

The rate has two competing pieces. Privacy amplification keeps only the share of key-basis detections that can be attributed to single-photon emissions, discounted by the leakage implied by the estimated phase error, while error correction consumes a share of every key-basis detection set by the bit-error rate the protocol is designed to tolerate. Whatever cannot be attributed to single photons is conceded entirely to the adversary, so the correlated intensity drift acts on the rate only through how much it loosens the single-photon estimates. A non-positive result means the configuration supports no key.

The first entry of the nominal settings is the signal setting, and the rate is reported per signal emitted rather than per detection, so the basis- and intensity-selection probabilities multiply both pieces. The estimated phase error is the ratio of the single-photon error parameter to the single-photon detection parameter in the check basis, the common single-photon emission probability cancelling between them; once that ratio reaches one half the privacy-amplification piece vanishes. The channel transmittance follows from the link length through the supplied attenuation coefficient, and the detector efficiency multiplies it.

Returns
-------
key_rate : float — Certified asymptotic secret-key rate per emitted signal. May be non-positive, in which case the configuration supports no key.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_certified_key_rate(
    intensities: 'np.ndarray',
    probabilities: 'np.ndarray',
    key_basis_probability: float,
    distance_km: float,
    attenuation_db_per_km: float,
    detector_efficiency: float,
    dark_count: float,
    misalignment: float,
    delta_max: float,
    correlation_range: int,
    n_cut: int,
    error_correction_efficiency: float,
    tolerated_error_rate: float,
    max_iterations: int = 40,
    objective_rtol: float = 1e-12,
) -> float:
    '''Compute the certified asymptotic secret-key rate per emitted signal.

    Parameters
    ----------
    intensities : np.ndarray
        Shape (A,). Nominal mean photon numbers, all strictly positive. Entry
        zero is the signal setting.
    probabilities : np.ndarray
        Shape (A,). Selection probability of each nominal setting.
        Non-negative and summing to one.
    key_basis_probability : float
        Probability with which each party selects the key basis, in [0, 1].
    distance_km : float
        Link length in kilometres. Non-negative.
    attenuation_db_per_km : float
        Fibre attenuation coefficient in decibels per kilometre. Non-negative.
    detector_efficiency : float
        Efficiency of each detector, in [0, 1].
    dark_count : float
        Dark-count probability of each detector per gate, in [0, 1).
    misalignment : float
        Polarisation misalignment angle in radians, in [0, pi / 4].
    delta_max : float
        Maximum relative deviation of every actual mean photon number from its
        nominal setting, in [0, 1).
    correlation_range : int
        Memory span of the intensity correlations, in rounds. At least one.
    n_cut : int
        Photon-number cut-off. At least one.
    error_correction_efficiency : float
        Error-correction efficiency factor. At least one.
    tolerated_error_rate : float
        Bit-error rate the protocol is designed to tolerate in the raw
        key-basis key, in [0, 0.5].
    max_iterations : int
        Positive refinement budget passed to Step 08. One deliberately
        uses the unrefined channel-reference bounds. For larger budgets,
        the final single-anchor certification LP and exact-feasibility
        polishing are additional to the refinement solves.
    objective_rtol : float
        Strictly positive relative refinement tolerance, as in Step 08.
        Its original-feasibility and final-gap checks must also pass.

    Returns
    -------
    key_rate : float
        Certified asymptotic secret-key rate per emitted signal. May be
        non-positive, in which case the configuration supports no key.

    Raises
    ------
    ValueError
        If any argument lies outside its stated range or any array argument has
        the wrong shape, any LP fails, or Step 08 exhausts a budget above
        one without satisfying its convergence and certification checks.
    '''
    return key_rate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _binary_entropy(fraction: float) -> float:
    """Shannon entropy of a biased bit, in bits."""
    value = float(fraction)
    if value <= 0.0 or value >= 1.0:
        return 0.0
    return float(-value * np.log2(value) - (1.0 - value) * np.log2(1.0 - value))


def _oracle_compute_certified_key_rate(intensities: "np.ndarray", probabilities: "np.ndarray", key_basis_probability: float, distance_km: float, attenuation_db_per_km: float, detector_efficiency: float, dark_count: float, misalignment: float, delta_max: float, correlation_range: int, n_cut: int, error_correction_efficiency: float, tolerated_error_rate: float, max_iterations: int = 40, objective_rtol: float = 1e-12) -> float:
    scalars = (key_basis_probability, distance_km, attenuation_db_per_km,
               detector_efficiency, error_correction_efficiency, tolerated_error_rate)
    for value in scalars:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError("scalar arguments must be real")
    key_basis_probability = float(key_basis_probability)
    distance_km = float(distance_km)
    attenuation_db_per_km = float(attenuation_db_per_km)
    detector_efficiency = float(detector_efficiency)
    error_correction_efficiency = float(error_correction_efficiency)
    tolerated_error_rate = float(tolerated_error_rate)
    if not np.isfinite(key_basis_probability) or key_basis_probability < 0.0 or key_basis_probability > 1.0:
        raise ValueError("key_basis_probability must lie in [0, 1]")
    if not np.isfinite(distance_km) or distance_km < 0.0:
        raise ValueError("distance_km must be non-negative")
    if not np.isfinite(attenuation_db_per_km) or attenuation_db_per_km < 0.0:
        raise ValueError("attenuation_db_per_km must be non-negative")
    if not np.isfinite(detector_efficiency) or detector_efficiency < 0.0 or detector_efficiency > 1.0:
        raise ValueError("detector_efficiency must lie in [0, 1]")
    if not np.isfinite(error_correction_efficiency) or error_correction_efficiency < 1.0:
        raise ValueError("error_correction_efficiency must be at least one")
    if not np.isfinite(tolerated_error_rate) or tolerated_error_rate < 0.0 or tolerated_error_rate > 0.5:
        raise ValueError("tolerated_error_rate must lie in [0, 0.5]")

    settings, weights = _check_intensity_family(intensities, probabilities)
    transmittance = 10.0 ** (-attenuation_db_per_km * distance_km / 10.0) * detector_efficiency

    key_rates, check_rates, error_rates = _oracle_compute_channel_observables(
        settings, transmittance, misalignment, dark_count)
    yield_reference, error_reference = _oracle_compute_fock_reference_points(
        n_cut, transmittance, misalignment, dark_count)
    key_yield, check_yield, error_bound, _, _, _ = _oracle_compute_certified_parameter_bounds(
        key_rates, check_rates, error_rates, yield_reference, error_reference,
        settings, weights, delta_max, correlation_range, n_cut,
        max_iterations, objective_rtol)

    emission_lower, _, _ = _oracle_compute_photon_number_bounds(
        float(settings[0]), delta_max, n_cut)
    sifting = key_basis_probability ** 2 * float(weights[0])

    if key_yield <= 0.0 or check_yield <= 0.0:
        privacy = 0.0
    else:
        phase_error = min(error_bound / check_yield, 0.5)
        privacy = (sifting * float(emission_lower[1]) * key_yield
                   * (1.0 - _binary_entropy(phase_error)))

    reconciliation = (error_correction_efficiency * sifting * float(key_rates[0])
                      * _binary_entropy(tolerated_error_rate))
    return float(privacy - reconciliation)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Retain cases with isolated inputs and numerical checks."""
    from textwrap import dedent

    common = dedent(
        """\
        from copy import deepcopy

        import numpy as np


        def _isolated(function, *arguments, **keywords):
            arguments, keywords = deepcopy((arguments, keywords))
            return function(*arguments, **keywords)


        intensities = np.array([0.48, 0.1, 0.0001])
        probabilities = np.array([0.7, 0.18, 0.12])
        base = (
            dict(key_basis_probability=0.9, distance_km=35.0,
            attenuation_db_per_km=0.2, detector_efficiency=0.65,
            dark_count=7.2e-08, misalignment=0.08, delta_max=0.001,
            correlation_range=3, n_cut=10,
            error_correction_efficiency=1.16,
            tolerated_error_rate=0.0065)
        )


        def rate(fn, **overrides):
            settings = dict(base)
            settings.update(overrides)
            return (
                _isolated(fn, intensities, probabilities, **settings)
            )
        """
    )
    return [
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": 'rate(compute_certified_key_rate)',
            "gold_call": 'rate(_oracle_compute_certified_key_rate)',
            "tol": 2e-08,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, distance_km=12.0)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, distance_km=12.0)'
            ),
            "tol": 2e-08,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": 'rate(compute_certified_key_rate, delta_max=0.0)',
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, delta_max=0.0)'
            ),
            "tol": 1e-09,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, distance_km=0.0)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, distance_km=0.0)'
            ),
            "tol": 2e-08,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, tolerated_error_rate=0.0)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, tolerated_error_ra'
                'te=0.0)'
            ),
            "tol": 2e-08,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, delta_max=0.03, correlatio'
                'n_range=4)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, delta_max=0.03, co'
                'rrelation_range=4)'
            ),
            "tol": 1e-09,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, distance_km=120.0)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, distance_km=120.0)'
            ),
            "tol": 1e-09,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, max_iterations=1)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, max_iterations=1)'
            ),
            "tol": 1e-09,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, correlation_range=1)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, correlation_range='
                '1)'
            ),
            "tol": 2e-08,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, correlation_range=2)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, correlation_range='
                '2)'
            ),
            "tol": 2e-08,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, correlation_range=3)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, correlation_range='
                '3)'
            ),
            "tol": 2e-08,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, correlation_range=4)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, correlation_range='
                '4)'
            ),
            "tol": 2e-08,
        },
        {
            "setup": common + dedent(
                """\

                """
            ),
            "call": (
                'rate(compute_certified_key_rate, dark_count=0.0, misalignme'
                'nt=0.0, tolerated_error_rate=0.0)'
            ),
            "gold_call": (
                'rate(_oracle_compute_certified_key_rate, dark_count=0.0, mi'
                'salignment=0.0, tolerated_error_rate=0.0)'
            ),
            "tol": 2e-08,
        },
        {
            "setup": common + dedent(
                """\

                def run_model(fn, *args, **kwargs):
                    try:
                        _isolated(fn, *args, **kwargs)
                        return 0.0
                    except ValueError:
                        return 1.0
                """
            ),
            "call": (
                'run_model(rate, compute_certified_key_rate, tolerated_error'
                '_rate=0.6)'
            ),
            "gold_call": (
                'run_model(rate, _oracle_compute_certified_key_rate, tolerat'
                'ed_error_rate=0.6)'
            ),
        },
        {
            "setup": common + dedent(
                """\

                def run_model(fn, *args, **kwargs):
                    try:
                        _isolated(fn, *args, **kwargs)
                        return 0.0
                    except ValueError:
                        return 1.0
                """
            ),
            "call": (
                'run_model(rate, compute_certified_key_rate, error_correctio'
                'n_efficiency=0.8)'
            ),
            "gold_call": (
                'run_model(rate, _oracle_compute_certified_key_rate, error_c'
                'orrection_efficiency=0.8)'
            ),
        },
        {
            "setup": common + dedent(
                """\

                def run_model(fn, *args, **kwargs):
                    try:
                        _isolated(fn, *args, **kwargs)
                        return 0.0
                    except ValueError:
                        return 1.0
                """
            ),
            "call": (
                'run_model(rate, compute_certified_key_rate, distance_km=-1.'
                '0)'
            ),
            "gold_call": (
                'run_model(rate, _oracle_compute_certified_key_rate, distanc'
                'e_km=-1.0)'
            ),
        },
        {
            "setup": common + dedent(
                """\

                def run_model(fn, *args, **kwargs):
                    try:
                        _isolated(fn, *args, **kwargs)
                        return 0.0
                    except ValueError:
                        return 1.0
                """
            ),
            "call": (
                'run_model(rate, compute_certified_key_rate, key_basis_proba'
                'bility=1.4)'
            ),
            "gold_call": (
                'run_model(rate, _oracle_compute_certified_key_rate, key_bas'
                'is_probability=1.4)'
            ),
        },
    ]
