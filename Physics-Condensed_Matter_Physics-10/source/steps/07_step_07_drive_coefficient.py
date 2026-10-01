"""
Evaluate the dimensionless coefficient multiplying the squared pulse in the gap equation of motion. The routine takes a finite non-negative drive strength, positive simulation and reference temperatures in kelvin, a positive cutoff in millielectronvolts and a positive dimensionless coupling, then returns the coefficient as a single float. Invalid input raises ValueError: drive strength must be a finite non-negative scalar, both temperatures, cutoff and coupling must be finite scalars strictly greater than zero and the equilibrium gap must not vanish at either temperature.

The drive is normalized against the stiffness at a reference temperature. The constant of proportionality between stiffness and inertia cancels between the two evaluations, leaving only their ratio, and consequently choosing the reference temperature equal to the simulation temperature would make the normalization carry no information at all.

Returns
-------
float giving the dimensionless coefficient multiplying the squared normalized pulse
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def drive_coefficient(drive_strength, t_sim, t_ref, cutoff, coupling):
    """Return the dimensionless coefficient multiplying the squared pulse.

    Parameters
    ----------
    drive_strength : float
        Finite non-negative squared-pulse strength.
    t_sim : float
        Positive simulation temperature in kelvin.
    t_ref : float
        Positive reference temperature in kelvin.
    cutoff : float
        Positive symmetric energy cutoff in millielectronvolts.
    coupling : float
        Positive dimensionless product of pairing strength and density of states.

    Returns
    -------
    float
        Dimensionless coefficient multiplying the squared normalized pulse.

    Raises
    ------
    ValueError
        If any input is not a finite scalar, if ``drive_strength`` is negative,
        if another input is not strictly positive, or if either equilibrium gap
        vanishes.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_drive_coefficient(drive_strength, t_sim, t_ref, cutoff, coupling):
    values = (
        (drive_strength, "drive_strength"),
        (t_sim, "t_sim"),
        (t_ref, "t_ref"),
        (cutoff, "cutoff"),
        (coupling, "coupling"),
    )
    converted = []
    for value, name in values:
        if not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite scalar")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(numeric):
            raise ValueError(name + " must be a finite scalar")
        converted.append(numeric)
    drive_strength, t_sim, t_ref, cutoff, coupling = converted
    if drive_strength < 0.0:
        raise ValueError("drive_strength must be non-negative")
    if t_sim <= 0.0:
        raise ValueError("t_sim must be strictly greater than zero")
    if t_ref <= 0.0:
        raise ValueError("t_ref must be strictly greater than zero")
    if cutoff <= 0.0:
        raise ValueError("cutoff must be strictly greater than zero")
    if coupling <= 0.0:
        raise ValueError("coupling must be strictly greater than zero")

    # Magnitudes far outside the millielectronvolt range overflow inside the
    # upstream gap and inertia integrals, or leave a product of two huge
    # numbers that is no longer finite. Either is a refusal of the input
    # rather than a result, so it is reported as the documented ValueError.
    try:
        gap_sim = float(_oracle_equilibrium_gap(t_sim, cutoff, coupling))
        gap_ref = float(_oracle_equilibrium_gap(t_ref, cutoff, coupling))
        if gap_sim <= 0.0 or gap_ref <= 0.0:
            raise ValueError("equilibrium gap must not vanish at either temperature")
        inertia_sim = float(_oracle_inertia_coefficient(gap_sim, t_sim, cutoff))
        inertia_ref = float(_oracle_inertia_coefficient(gap_ref, t_ref, cutoff))
        result = float(drive_strength * inertia_sim / inertia_ref)
    except (OverflowError, ZeroDivisionError, FloatingPointError) as exc:
        raise ValueError("no finite drive coefficient at these magnitudes") from exc
    if not np.isfinite(result):
        raise ValueError("no finite drive coefficient at these magnitudes")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "round(float(drive_coefficient(0.4, 7.6306183138, 12.2089893021, 2.6, 1.11)), 12)",
            "gold_call": "round(float(_oracle_drive_coefficient(0.4, 7.6306183138, 12.2089893021, 2.6, 1.11)), 12)",
        },
        {
            "setup": "",
            "call": "round(float(drive_coefficient(0.4, 7.6306183138, 7.6306183138, 2.6, 1.11)), 12)",
            "gold_call": "round(float(_oracle_drive_coefficient(0.4, 7.6306183138, 7.6306183138, 2.6, 1.11)), 12)",
        },
        {
            "setup": "",
            "call": "round(float(drive_coefficient(0.0, 7.6306183138, 12.2089893021, 2.6, 1.11)), 12)",
            "gold_call": "round(float(_oracle_drive_coefficient(0.0, 7.6306183138, 12.2089893021, 2.6, 1.11)), 12)",
        },
        {
            "setup": "",
            "call": "int(drive_coefficient(0.4, 5.0, 10.0, 2.6, 1.11) > 0.4)",
            "gold_call": "int(_oracle_drive_coefficient(0.4, 5.0, 10.0, 2.6, 1.11) > 0.4)",
        },
        {
            "setup": "def run_model():\n    try:\n        drive_coefficient(-0.1, 7.6306183138, 12.2089893021, 2.6, 1.11)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_drive_coefficient(-0.1, 7.6306183138, 12.2089893021, 2.6, 1.11)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
