"""
Classify particles in a completed one-dimensional host well with the physical-frame energy rule. Position, canonical velocity and boosted potential are finite one-dimensional arrays of equal length with at least three entries. Well IDs are a nonempty one-dimensional sequence of distinct valid native or NumPy integers, and minimum and saddle IDs are valid integral members of the well. Recenter with `$q_i = position_i - position_minimum$` and `$u_i = canonical_velocity_i - mean(canonical_velocity[j] for j in well_ids)$`. The physical potential is ``Phi_i = boosted_potential_i + 0.25 * hubble_rate**2 * (omega_m * (1.0 + turnaround_overdensity) - 2.0 * omega_lambda) * (scale_factor * q_i)**2``. The energy is ``E_i = Phi_i + 0.5 * (scale_factor * hubble_rate * q_i + u_i / scale_factor)**2`` and the escape energy is `$Phi_saddle$`. Return a zero-one mask that marks only well members satisfying strict `$E_i < E_escape$`, plus all energies and physical potentials as native floats and the native-float escape energy. Preserve finite representable physical potentials and energies despite overflow in an intermediate power or product; invalid input or an unrepresentable returned potential or energy raises ValueError.

The boosted turn-around field orders graph contours, but particle binding is tested in the translated physical frame. The completed well, including its saddle, supplies the particle-count mean canonical velocity; the saddle reference corresponds to a physically stationary particle and therefore contains no particle kinetic term. The fixed coefficients are 0.25 and 0.5, equality is unbound and the scale factor must be positive while the Hubble rate and both density parameters and the turn-around overdensity must be nonnegative and finite.

Returns
-------
tuple of list[int], float, list[float] and list[float], the mask, escape, physical potential and energies
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def host_energy_mask(position, canonical_velocity, boosted_potential, well_ids,
                     saddle_id, minimum_id, scale_factor, hubble_rate, omega_m,
                     omega_lambda, turnaround_overdensity):
    """Return host binding flags, escape energy and physical potentials.

    Parameters
    ----------
    position : array_like
        Finite one-dimensional positions.
    canonical_velocity : array_like
        Finite one-dimensional canonical velocities.
    boosted_potential : array_like
        Finite one-dimensional boosted turn-around potentials.
    well_ids : array_like of int
        Nonempty distinct valid native or NumPy integer IDs in the completed well.
    saddle_id : int
        Valid integral saddle ID belonging to the well.
    minimum_id : int
        Valid integral minimum ID belonging to the well.
    scale_factor : float
        Positive finite scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.
    omega_m : float
        Nonnegative finite matter density parameter.
    omega_lambda : float
        Nonnegative finite dark-energy density parameter.
    turnaround_overdensity : float
        Nonnegative finite turn-around overdensity.

    Returns
    -------
    tuple
        Native-int zero-one mask, native-float escape energy, native-float
        physical potentials and native-float particle energies.

    Raises
    ------
    ValueError
        If an input violates the contract or a returned potential or energy is not
        representable as a finite native float; avoidable intermediate overflow
        does not invalidate a representable result.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_host_energy_mask(position, canonical_velocity, boosted_potential,
                           well_ids, saddle_id, minimum_id, scale_factor,
                           hubble_rate, omega_m, omega_lambda,
                           turnaround_overdensity):
    def numeric_vector(value, name):
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(name + " must be a numeric one-dimensional array") from exc
        if array.ndim != 1 or array.size < 3:
            raise ValueError(name + " must have at least three entries")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(name + " must contain real numeric values")
        try:
            array = np.asarray(array, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must contain finite real values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(name + " must contain finite real values")
        return array

    position_array = numeric_vector(position, "position")
    velocity_array = numeric_vector(canonical_velocity, "canonical_velocity")
    boosted_array = numeric_vector(boosted_potential, "boosted_potential")
    if position_array.shape != velocity_array.shape or position_array.shape != boosted_array.shape:
        raise ValueError("particle arrays must have equal length")
    node_count = int(position_array.size)

    try:
        well_array = np.asarray(well_ids, dtype=object)
    except Exception as exc:
        raise ValueError("well_ids must be a one-dimensional integer sequence") from exc
    if well_array.ndim != 1 or well_array.size == 0:
        raise ValueError("well_ids must be nonempty and one-dimensional")
    well = []
    for item in well_array.tolist():
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError("well_ids must contain integral node IDs")
        node = int(item)
        if node < 0 or node >= node_count:
            raise ValueError("well_ids contains an invalid node ID")
        well.append(node)
    if len(set(well)) != len(well):
        raise ValueError("well_ids must be distinct")
    well_set = set(well)

    converted_ids = []
    for name, item in (("saddle_id", saddle_id), ("minimum_id", minimum_id)):
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
            raise ValueError(name + " must be integral")
        node = int(item)
        if node < 0 or node >= node_count or node not in well_set:
            raise ValueError(name + " must be a valid member of the well")
        converted_ids.append(node)
    saddle_id, minimum_id = converted_ids

    scalars = []
    for name, value, positive in (
        ("scale_factor", scale_factor, True),
        ("hubble_rate", hubble_rate, False),
        ("omega_m", omega_m, False),
        ("omega_lambda", omega_lambda, False),
        ("turnaround_overdensity", turnaround_overdensity, False),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(name + " must be a finite scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(scalar) or (scalar <= 0.0 if positive else scalar < 0.0):
            raise ValueError(name + " lies outside its valid range")
        scalars.append(scalar)
    scale_factor, hubble_rate, omega_m, omega_lambda, turnaround_overdensity = scalars

    def wide_result():
        from fractions import Fraction as F
        x, velocity, boosted = [[F(float(v)) for v in array]
                                for array in (position_array, velocity_array, boosted_array)]
        scale, hubble, matter, vacuum, density = map(F, scalars)
        mean = sum(velocity[j] for j in well) / len(well)
        coefficient = hubble * hubble * (matter * (1 + density) - 2 * vacuum) * scale * scale / 4
        displacement = [v - x[minimum_id] for v in x]
        potential = [boosted[i] + coefficient * q * q for i, q in enumerate(displacement)]
        total = [potential[i] + (scale * hubble * q + (velocity[i] - mean) / scale)**2 / 2
                 for i, q in enumerate(displacement)]
        try:
            converted = ([float(v) for v in potential], [float(v) for v in total])
        except OverflowError as exc:
            raise ValueError("physical potentials and energies cannot be represented as finite floats") from exc
        if not all(np.all(np.isfinite(v)) for v in converted):
            raise ValueError("physical potentials and energies cannot be represented as finite floats")
        return converted

    physical_potential, energy = wide_result()
    escape = float(physical_potential[saddle_id])
    mask = [0] * node_count
    for node in well:
        mask[node] = int(energy[node] < escape)
    return (mask, escape, [float(value) for value in physical_potential],
            [float(value) for value in energy])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "host_energy_mask([0, 1e-100, 2e-100], [0, 0, 0], [0, 0, 0], [0, 1, 2], 2, 0, 1e-200, 1e150, 1.0, 0.0, 3.0)[0]",
            "gold_call": "[1, 1, 0]",
        },
        {
            "setup": "",
            "call": "host_energy_mask([0, 0, 0], [0, 0, 0], [0, 1, 2], [0, 1, 2], 2, 0, 1.0, 1e200, 0.0, 0.0, 4.55)",
            "gold_call": "([1, 1, 0], 2.0, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])",
        },
        {
            "setup": "",
            "call": "tuple(np.round(v, 12).tolist() if isinstance(v, list) else round(v, 12) for v in host_energy_mask([0, 1, 2], [0, 0, 0], [0, 1, 2], [0, 1, 2], 2, 0, 1e-200, 1e200, 0.0, 0.0, 4.55))",
            "gold_call": "([1.0, 1.0, 0.0], 2.0, [0.0, 1.0, 2.0], [0.0, 1.5, 4.0])",
        },
        {
            "setup": "def overflow_status(function):\n    try:\n        function([0, 1, 2], [0, 0, 0], [0, 1, 2], [0, 1, 2], 2, 0, 1.0, 1e200, 1.0, 0.0, 4.55)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "overflow_status(host_energy_mask)",
            "gold_call": "overflow_status(_oracle_host_energy_mask)",
        },
        {
            "setup": "position = [10.0, 9.7, 10.4, 9.4, 8.5, 10.8, 11.0, 11.1, 10.9, 11.5, 12.5, 12.7, 13.0, 9.1, 8.9]\nvelocity = [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 2.7, 2.7, 2.7, 0.0, -2.0, 0.0, 0.0, -2.0, 0.0]\nboosted = [0.0, 0.6, 0.9, 1.5, 3.3, 3.2, 0.7, 1.0, 1.1, 6.0, 4.0, 2.0, -10.0, 5.0, -2.0]\nwell = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]\n",
            "call": "(host_energy_mask(position, velocity, boosted, well, np.int64(10), np.int64(0), 1.0, 1.0, 1.0, 0.0, 4.55)[0], round(host_energy_mask(position, velocity, boosted, well, 10, 0, 1.0, 1.0, 1.0, 0.0, 4.55)[1], 8))",
            "gold_call": "(_oracle_host_energy_mask(position, velocity, boosted, well, np.int64(10), np.int64(0), 1.0, 1.0, 1.0, 0.0, 4.55)[0], round(_oracle_host_energy_mask(position, velocity, boosted, well, 10, 0, 1.0, 1.0, 1.0, 0.0, 4.55)[1], 8))",
        },
        {
            "setup": "position = [0.0, 1.0, 2.0]\nvelocity = [0.0, 0.0, 0.0]\nboosted = [0.0, 1.0, 2.0]\n",
            "call": "host_energy_mask(position, velocity, boosted, np.array([0, 1, 2], dtype=np.int64), 2, 0, 1.0, 0.0, 1.0, 0.0, 0.0)[0]",
            "gold_call": "_oracle_host_energy_mask(position, velocity, boosted, np.array([0, 1, 2], dtype=np.int64), 2, 0, 1.0, 0.0, 1.0, 0.0, 0.0)[0]",
        },
        {
            "setup": "position = [-1.0, 0.0, 2.0, 3.0]\nvelocity = [2.0, -1.0, 0.5, 4.0]\nboosted = [0.0, 1.0, 3.0, -2.0]\n",
            "call": "tuple(np.round(host_energy_mask(position, velocity, boosted, [0, 1, 2], 2, 0, 0.5, 2.0, 0.3, 0.7, 1.2)[2], 8))",
            "gold_call": "tuple(np.round(_oracle_host_energy_mask(position, velocity, boosted, [0, 1, 2], 2, 0, 0.5, 2.0, 0.3, 0.7, 1.2)[2], 8))",
        },
        {
            "setup": "position = [0.0, 1.0, 2.0]\nvelocity = [0.0, 0.0, 0.0]\nboosted = [0.0, 1.0, 2.0]\n",
            "call": "(host_energy_mask(position, velocity, boosted, [0, 1, 2], 2, 0, 1.0, 0.0, 1.0, 0.0, 0.0)[0][2], type(host_energy_mask(position, velocity, boosted, [0, 1, 2], 2, 0, 1.0, 0.0, 1.0, 0.0, 0.0)[0][0]) is int)",
            "gold_call": "(_oracle_host_energy_mask(position, velocity, boosted, [0, 1, 2], 2, 0, 1.0, 0.0, 1.0, 0.0, 0.0)[0][2], type(_oracle_host_energy_mask(position, velocity, boosted, [0, 1, 2], 2, 0, 1.0, 0.0, 1.0, 0.0, 0.0)[0][0]) is int)",
        },
        {
            "setup": "def run_model():\n    try:\n        host_energy_mask([0, 1, 2], [0, 1], [0, 1, 2], [0, 1], 1, 0, 1.0, 1.0, 1.0, 0.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_host_energy_mask([0, 1, 2], [0, 1], [0, 1, 2], [0, 1], 1, 0, 1.0, 1.0, 1.0, 0.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def invalid_status(function):\n    try:\n        function([0, 1, 2], [0, 0, 0], [0, 1, 2], [0, np.int64(1)], 2, 0, 0.0, 1.0, 1.0, 0.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "invalid_status(host_energy_mask)",
            "gold_call": "invalid_status(_oracle_host_energy_mask)",
        },
    ]
