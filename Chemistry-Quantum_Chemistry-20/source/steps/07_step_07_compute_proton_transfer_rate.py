"""
Compose every earlier step to obtain the thermal proton-transfer rate constant, in units of 1e12 per second, of a symmetric hydrogen bond described by a fitted trigonometric double well. Orchestrator: it fits the double well to the scan (fit_trigonometric_well), converts the temperature to the reduced inverse temperature of the fitted model and evaluates the reduced rate constant (compute_reduced_rate_constant, which uses compute_well_energy_levels, compute_right_moving_flux and compute_transmission_probability, the last two built on evaluate_well_eigenfunction), then converts that rate to per-second units, consuming each output rather than reimplementing any step.

The reduced rate constant of the model is turned into a dimensional rate through the frequency scale set by the proton mass and the confining half-width, so the fitted width enters both the thermal weights and the prefactor.

Returns
-------
float: thermal proton-transfer rate constant Gamma in units of 1e12 per second.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_proton_transfer_rate(
    barrier_kcal_per_mol: float = 5.0,
    minimum_offset_angstrom: float = 0.55,
    donor_acceptor_distance_angstrom: float = 2.78,
    temperature_kelvin: float = 400.0,
) -> float:
    """Return the thermal proton-transfer rate constant in units of 1e12 per second.

    Fit the trigonometric double well to the scan with
    ``fit_trigonometric_well`` to obtain ``[L, m, p]``, form the reduced
    inverse temperature ``beta = hbar**2 pi**2 / (8 M L**2 k_B T)``, evaluate
    ``k(beta)`` with ``compute_reduced_rate_constant`` and return
    ``Gamma = k(beta) hbar / (M L**2)`` divided by ``1e12`` per second. Here
    ``M`` is the proton mass, ``L`` is converted from angstrom to metres, and
    the constants are ``hbar = 1.054571817e-34`` J s,
    ``M = 1.67262192369e-27`` kg and ``k_B = 1.380649e-23`` J/K. The defaults
    reproduce the problem statement.

    Parameters
    ----------
    barrier_kcal_per_mol : float
        Barrier height of the scan at the midpoint, in kcal/mol.
    minimum_offset_angstrom : float
        Distance of each minimum from the midpoint, in angstrom.
    donor_acceptor_distance_angstrom : float
        Approximate heavy-atom distance, in angstrom.
    temperature_kelvin : float
        Temperature in kelvin.

    Returns
    -------
    float
        Rate constant ``Gamma`` in units of ``1e12`` per second.

    Raises
    ------
    ValueError
        If ``temperature_kelvin`` is not a finite positive real number
        (booleans are rejected), or if the scan inputs are invalid as in
        ``fit_trigonometric_well``.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_proton_transfer_rate(
    barrier_kcal_per_mol: float = 5.0,
    minimum_offset_angstrom: float = 0.55,
    donor_acceptor_distance_angstrom: float = 2.78,
    temperature_kelvin: float = 400.0,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import math
    import numpy as np

    if isinstance(temperature_kelvin, bool) or not isinstance(
            temperature_kelvin, (int, float, np.integer, np.floating)):
        raise ValueError("temperature must be a real number")
    if not (math.isfinite(temperature_kelvin) and temperature_kelvin > 0.0):
        raise ValueError("temperature must be finite and positive")
    width, order, p = _oracle_fit_trigonometric_well(
        barrier_kcal_per_mol, minimum_offset_angstrom, donor_acceptor_distance_angstrom)
    hbar, proton_mass, boltzmann, _, _ = _well_constants()
    width_m = float(width) * 1.0e-10
    beta = hbar ** 2 * math.pi ** 2 / (8.0 * proton_mass * width_m ** 2 * boltzmann * temperature_kelvin)
    reduced_rate = _oracle_compute_reduced_rate_constant(int(round(order)), float(p), beta)
    return float(reduced_rate * hbar / (proton_mass * width_m ** 2) / 1.0e12)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "compute_proton_transfer_rate()",
            "gold_call": "_oracle_compute_proton_transfer_rate()",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_proton_transfer_rate(5.0, 0.55, 2.78, 60.0)",
            "gold_call": "_oracle_compute_proton_transfer_rate(5.0, 0.55, 2.78, 60.0)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_proton_transfer_rate(8.0, 0.42, 2.80, 300.0)",
            "gold_call": "_oracle_compute_proton_transfer_rate(8.0, 0.42, 2.80, 300.0)",
        },
        {
            "setup": "import numpy as np\n",
            "call": "compute_proton_transfer_rate(1.3, 1.0, 3.15, 1500.0)",
            "gold_call": "_oracle_compute_proton_transfer_rate(1.3, 1.0, 3.15, 1500.0)",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_proton_transfer_rate(5.0, 0.55, 2.78, -10.0))",
            "gold_call": "_status(lambda: _oracle_compute_proton_transfer_rate(5.0, 0.55, 2.78, -10.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_proton_transfer_rate(5.0, 1.5, 2.78, 400.0))",
            "gold_call": "_status(lambda: _oracle_compute_proton_transfer_rate(5.0, 1.5, 2.78, 400.0))",
        },
    ]
