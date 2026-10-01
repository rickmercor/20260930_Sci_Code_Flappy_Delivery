"""
Run the complete finite-order two-isotope benchmark and compare rate-model
KIEs.

Build the deterministic H and D snapshots from the supplied seed, reduce both
diabatic extended Hessians, compute the peaked-crossing descriptor, evaluate
the distance profiles, and obtain each isotope's LZ and WC thermal rates using
exactly the requested Gauss-Laguerre orders without adaptive extrapolation.
The returned value is therefore the deterministic finite-rule benchmark at
those orders, not a quadrature-order-extrapolated continuum value.  Let
``KIE_LZ = k_H_LZ/k_D_LZ`` and ``KIE_WC = k_H_WC/k_D_WC``; return the signed
percentage ``100*(KIE_WC/KIE_LZ - 1)``.

``temperature`` must be positive, ``n_distance`` an odd integer of at least 9,
and both quadrature orders integers from 8 through 100.  Invalid inputs raise
``ValueError``.

Returns
-------
float, finite signed relative percentage change 100 * (KIE_WC / KIE_LZ - 1), where each KIE is the corresponding H/D reduced-rate ratio.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_neogrt_kie_shift(
    seed: int,
    temperature: float,
    n_distance: int = 61,
    lz_order: int = 48,
    wc_order: int = 64,
) -> float:
    """Return the signed WC-versus-LZ percentage shift in the H/D KIE.

    Parameters
    ----------
    seed
        Integer seed for the deterministic H and D snapshots.
    temperature
        Finite strictly positive temperature in kelvin.
    n_distance
        Odd integer number of distance nodes, at least nine.
    lz_order, wc_order
        Integer Gauss-Laguerre orders from eight through 100 defining the
        finite deterministic benchmark.

    Returns
    -------
    float
        Finite relative percentage ``100 * (KIE_WC / KIE_LZ - 1)``.

    Raises
    ------
    ValueError
        If the seed or grid controls are not valid integers, the temperature
        is non-finite or nonpositive, or either isotope rate is nonphysical.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _step8_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _step8_make_snapshot(
    seed: int,
    isotope_index: int,
    n_distance: int,
) -> tuple:
    rng = np.random.default_rng(int(seed) + 7919 * int(isotope_index))
    classical_dim = 6
    quantum_dim = 3
    matrices = []
    for surface in range(2):
        factor = rng.normal(
            scale=0.055 + 0.004 * surface,
            size=(classical_dim, classical_dim),
        )
        h_cc = factor.T @ factor + np.diag(
            np.linspace(0.075, 0.145, classical_dim)
        )
        h_qc = rng.normal(scale=0.0055, size=(quantum_dim, classical_dim))
        quantum_factor = rng.normal(scale=0.035, size=(quantum_dim, quantum_dim))
        h_qq = (
            quantum_factor.T @ quantum_factor
            + (0.27 + 0.02 * surface) * np.eye(quantum_dim)
        )
        matrices.append(np.block([[h_cc, h_qc.T], [h_qc, h_qq]]))
    base_a = np.array([0.031, -0.025, 0.019, -0.014, 0.012, -0.009])
    base_b = np.array([-0.027, 0.021, -0.016, 0.017, -0.010, 0.011])
    gradient_a = (
        (1.0 + 0.035 * isotope_index) * base_a
        + rng.normal(scale=0.0018, size=classical_dim)
    )
    gradient_b = (
        (1.0 - 0.025 * isotope_index) * base_b
        + rng.normal(scale=0.0018, size=classical_dim)
    )
    masses = 1822.888486217313 * np.array(
        [12.0, 12.0, 15.99491462, 15.99491462, 14.003074, 14.003074]
    )
    reference_distance = (
        4.94 + 0.022 * isotope_index + float(rng.normal(scale=0.004))
    )
    distance = np.linspace(4.42, 5.50, int(n_distance))
    reactant = np.array(
        [
            0.0,
            0.00015 * (-1.0 if isotope_index == 0 else 1.0),
            0.0125 + 0.0012 * isotope_index,
            -0.0018 + 0.0002 * isotope_index,
            0.0021,
        ]
    )
    mecp = np.array(
        [
            0.00485 + 0.00058 * isotope_index,
            -0.0016 + 0.00018 * isotope_index,
            0.0088 + 0.0007 * isotope_index,
            0.0011,
            0.0015,
        ]
    )
    product = np.array(
        [
            0.00055 + 0.00016 * isotope_index,
            -0.00035,
            0.0112 + 0.0008 * isotope_index,
            0.0014,
            0.0018,
        ]
    )
    coupling = np.array(
        [
            (0.00048 if isotope_index == 0 else 0.000215)
            * (1.0 + float(rng.normal(scale=0.025))),
            -2.20 - 0.16 * isotope_index,
            -0.55 - 0.06 * isotope_index,
        ]
    )
    critical_distance = 4.505 + 0.018 * isotope_index
    return (
        matrices,
        gradient_a,
        gradient_b,
        masses,
        distance,
        reactant,
        mecp,
        product,
        coupling,
        reference_distance,
        critical_distance,
    )


def _oracle_compute_neogrt_kie_shift(
    seed: int,
    temperature: float,
    n_distance: int = 61,
    lz_order: int = 48,
    wc_order: int = 64,
) -> float:
    integer_inputs = [seed, n_distance, lz_order, wc_order]
    if any(
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        for value in integer_inputs
    ):
        raise ValueError("seed, n_distance, and quadrature orders must be integers")
    temperature_value = _step8_real_scalar(temperature, "temperature")
    if (
        temperature_value <= 0.0
        or int(n_distance) < 9
        or int(n_distance) % 2 == 0
        or int(lz_order) < 8
        or int(wc_order) < 8
        or int(lz_order) > 100
        or int(wc_order) > 100
    ):
        raise ValueError("temperature, distance count, or quadrature order is invalid")
    beta = 1.0 / (3.1668114e-6 * temperature_value)
    isotope_rates = []
    for isotope_index in (0, 1):
        snapshot = _step8_make_snapshot(int(seed), isotope_index, int(n_distance))
        (
            matrices,
            gradient_a,
            gradient_b,
            masses,
            distance,
            reactant,
            mecp,
            product,
            coupling,
            reference_distance,
            critical_distance,
        ) = snapshot
        hessian_a = _oracle_reduce_neo_hessian(matrices[0], 3)
        hessian_b = _oracle_reduce_neo_hessian(matrices[1], 3)
        descriptor = _oracle_compute_crossing_descriptor(
            hessian_a,
            hessian_b,
            gradient_a,
            gradient_b,
            masses,
            True,
        )
        profile_values = _oracle_evaluate_distance_profiles(
            distance,
            reactant,
            mecp,
            product,
            coupling,
            reference_distance,
            critical_distance,
        )
        rates = _oracle_compute_thermal_isotope_rates(
            distance,
            profile_values,
            beta,
            descriptor,
            int(lz_order),
            int(wc_order),
        )
        isotope_rates.append(rates)
    if any(
        not np.isfinite(rates[0])
        or not np.isfinite(rates[1])
        or rates[0] <= 0.0
        or rates[1] <= 0.0
        for rates in isotope_rates
    ):
        raise ValueError("both isotope rates must be finite and strictly positive")
    kie_lz = float(isotope_rates[0][0] / isotope_rates[1][0])
    kie_wc = float(isotope_rates[0][1] / isotope_rates[1][1])
    answer = 100.0 * (kie_wc / kie_lz - 1.0)
    if not np.isfinite(answer):
        raise ValueError("the KIE comparison must be finite")
    return float(answer)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nfrom scipy.special import airy",
            "call": "compute_neogrt_kie_shift(26091312,82.0,61,48,64)",
            "gold_call": "_oracle_compute_neogrt_kie_shift(26091312,82.0,61,48,64)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import airy",
            "call": "compute_neogrt_kie_shift(173,64.0,9,16,16)",
            "gold_call": "_oracle_compute_neogrt_kie_shift(173,64.0,9,16,16)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import airy",
            "call": "compute_neogrt_kie_shift(42,50.0,31,24,32)",
            "gold_call": "_oracle_compute_neogrt_kie_shift(42,50.0,31,24,32)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import airy\ndef check(fn):\n try: fn(7,80.0,8,16,16)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(compute_neogrt_kie_shift)",
            "gold_call": "check(_oracle_compute_neogrt_kie_shift)",
            "tol": 0.0,
        },
    ]
