"""
Compute the boosted turn-around potential for every node in a one-dimensional particle system. Position, acceleration and raw potential must be finite one-dimensional numeric arrays of equal length with at least three entries. Seed IDs must form a nonempty one-dimensional sequence of distinct native or NumPy integers in range. First set the reference position and potential to the particle-count means over the seed IDs and evaluate the displayed boosted expression for each seed. Choose the seed minimizing `$(boosted_value[id], id)$` as the reference center, then evaluate every node relative to that center. With `$q_i = position_i - position_c$` and `$mean_grad = -mean(acceleration[j] for j in seed_ids)$`, return ``raw_potential_i - raw_potential_c - mean_grad * q_i - 0.25 * hubble_rate**2 * omega_m * (scale_factor * q_i)**2 * turnaround_overdensity`` as native Python floats. Preserve a finite representable result even when a scalar power overflows before multiplication by zero or compensating factors. Invalid input or a boosted result outside the finite native-float range raises ValueError.

An accelerated frame removes the nearly uniform gravitational gradient estimated from the seed particles, while the negative quadratic turn-around term accounts for the matter-driven cosmological acceleration across the candidate. Subtracting the raw potential at the chosen center fixes the otherwise arbitrary additive zero. The scale factor must be positive and finite, while the Hubble rate, matter density parameter and turn-around overdensity must be nonnegative and finite; the fixed quadratic coefficient is 0.25.

Returns
-------
list of native Python floats, one boosted turn-around potential per node
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def boosted_turnaround_potential(position, acceleration, raw_potential, seed_ids,
                                 scale_factor, hubble_rate, omega_m,
                                 turnaround_overdensity):
    """Return boosted turn-around potential values for all nodes.

    Parameters
    ----------
    position : array_like
        Finite one-dimensional positions with at least three entries.
    acceleration : array_like
        Finite one-dimensional accelerations, one per position.
    raw_potential : array_like
        Finite one-dimensional raw potentials, one per position.
    seed_ids : array_like of int
        Nonempty distinct valid native or NumPy integer node IDs.
    scale_factor : float
        Positive finite cosmological scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.
    omega_m : float
        Nonnegative finite matter density parameter.
    turnaround_overdensity : float
        Nonnegative finite turn-around overdensity.

    Returns
    -------
    list of float
        Boosted turn-around potential for every node.

    Raises
    ------
    ValueError
        If an input violates the contract or a boosted result is not representable
        as a finite native float; intermediate overflow alone is not a refusal.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_boosted_turnaround_potential(position, acceleration, raw_potential,
                                        seed_ids, scale_factor, hubble_rate,
                                        omega_m, turnaround_overdensity):
    def numeric_vector(value, name):
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(name + " must be a numeric one-dimensional array") from exc
        if array.ndim != 1 or array.size < 3:
            raise ValueError(name + " must be one-dimensional with at least three entries")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(name + " must contain real numeric values")
        try:
            array = np.asarray(array, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must contain finite real values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(name + " must contain finite real values")
        return array

    def integral_ids(value, name, size, nonempty=True):
        try:
            array = np.asarray(value, dtype=object)
        except Exception as exc:
            raise ValueError(name + " must be a one-dimensional integer sequence") from exc
        if array.ndim != 1 or (nonempty and array.size == 0):
            raise ValueError(name + " must be a nonempty one-dimensional integer sequence")
        ids = []
        for item in array.tolist():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise ValueError(name + " must contain integral node IDs")
            node = int(item)
            if node < 0 or node >= size:
                raise ValueError(name + " contains an invalid node ID")
            ids.append(node)
        if len(set(ids)) != len(ids):
            raise ValueError(name + " must contain distinct node IDs")
        return ids

    position_array = numeric_vector(position, "position")
    acceleration_array = numeric_vector(acceleration, "acceleration")
    potential_array = numeric_vector(raw_potential, "raw_potential")
    if position_array.shape != acceleration_array.shape or position_array.shape != potential_array.shape:
        raise ValueError("particle arrays must have equal length")
    seeds = integral_ids(seed_ids, "seed_ids", position_array.size)
    for name, value, positive in (
        ("scale_factor", scale_factor, True),
        ("hubble_rate", hubble_rate, False),
        ("omega_m", omega_m, False),
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
        if name == "scale_factor":
            scale_factor = scalar
        elif name == "hubble_rate":
            hubble_rate = scalar
        elif name == "omega_m":
            omega_m = scalar
        else:
            turnaround_overdensity = scalar

    def wide_result():
        # Exact products avoid overflow and underflow before compensating factors.
        from fractions import Fraction as F
        x, acceleration, potential = [[F(float(v)) for v in array]
                                      for array in (position_array, acceleration_array, potential_array)]
        scale, hubble, matter, density = map(F, (scale_factor, hubble_rate, omega_m, turnaround_overdensity))
        gradient = -sum(acceleration[j] for j in seeds) / len(seeds)
        coefficient = hubble * hubble * matter * scale * scale * density / 4
        def field(center_x, center_potential):
            return [potential[i] - center_potential - gradient * (x[i] - center_x)
                    - coefficient * (x[i] - center_x)**2 for i in range(len(x))]
        preliminary = field(sum(x[j] for j in seeds) / len(seeds),
                            sum(potential[j] for j in seeds) / len(seeds))
        center = min(seeds, key=lambda node: (preliminary[node], node))
        try:
            result = [float(v) for v in field(x[center], potential[center])]
        except OverflowError as exc:
            raise ValueError("boosted potential cannot be represented as finite floats") from exc
        if not all(np.isfinite(result)):
            raise ValueError("boosted potential cannot be represented as finite floats")
        return result

    return wide_result()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "np.round(np.asarray(boosted_turnaround_potential([0, 1e-100, 2e-100], [0, 0, 0], [0, 0, 0], [0], 1e-200, 1e150, 1.0, 4.0)) / 1e-300, 12).tolist()",
            "gold_call": "[0.0, -1.0, -4.0]",
        },
        {
            "setup": "",
            "call": "boosted_turnaround_potential([0, 1, 2], [0, 0, 0], [0, 1, 2], [0], 1.0, 1e200, 0.0, 4.55)",
            "gold_call": "[0.0, 1.0, 2.0]",
        },
        {
            "setup": "",
            "call": "np.round(boosted_turnaround_potential([0, 1, 2], [0, 0, 0], [0, 1, 2], [0], 1e-200, 1e200, 1.0, 4.0), 12).tolist()",
            "gold_call": "[0.0, 0.0, -2.0]",
        },
        {
            "setup": "def overflow_status(function):\n    try:\n        function([0, 1, 2], [0, 0, 0], [0, 1, 2], [0], 1.0, 1e200, 1.0, 4.55)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "overflow_status(boosted_turnaround_potential)",
            "gold_call": "overflow_status(_oracle_boosted_turnaround_potential)",
        },
        {
            "setup": "position = [10.0, 9.7, 10.4, 9.4, 8.5]\nacceleration = [-0.3, -0.5, -0.4, -0.2, -0.6]\nraw = [-20.0, -19.417625, -18.758, -18.3305, -14.740625]\n",
            "call": "np.round(boosted_turnaround_potential(position, acceleration, raw, [0, 1, 2, 3, 4], 1.0, 1.0, 1.0, 4.55), 12).tolist()",
            "gold_call": "np.round(_oracle_boosted_turnaround_potential(position, acceleration, raw, [0, 1, 2, 3, 4], 1.0, 1.0, 1.0, 4.55), 12).tolist()",
        },
        {
            "setup": "position = np.array([0.0, 1.0, 2.0])\nacceleration = np.array([3.0, 3.0, 3.0])\nraw = np.array([0.0, -2.5, 5.0])\n",
            "call": "boosted_turnaround_potential(position, acceleration, raw, np.array([0, 1], dtype=np.int64), 1.0, 0.0, 0.0, 0.0)",
            "gold_call": "_oracle_boosted_turnaround_potential(position, acceleration, raw, np.array([0, 1], dtype=np.int64), 1.0, 0.0, 0.0, 0.0)",
        },
        {
            "setup": "position = [0.0, 1.0, 2.0, 3.0]\nacceleration = [0.0, 0.0, 0.0, 0.0]\nraw = [2.0, -1.0, -1.0, 4.0]\n",
            "call": "boosted_turnaround_potential(position, acceleration, raw, [2, 1], 1.0, 0.0, 1.0, 0.0)",
            "gold_call": "_oracle_boosted_turnaround_potential(position, acceleration, raw, [2, 1], 1.0, 0.0, 1.0, 0.0)",
        },
        {
            "setup": "position = [-1.0e6, 0.0, 1.0e6]\nacceleration = [1.0e-6, 2.0e-6, 3.0e-6]\nraw = [3.0, 0.0, 7.0]\n",
            "call": "np.round(boosted_turnaround_potential(position, acceleration, raw, [0, 1], 0.5, 1.0e-4, 0.3, 2.5), 8).tolist()",
            "gold_call": "np.round(_oracle_boosted_turnaround_potential(position, acceleration, raw, [0, 1], 0.5, 1.0e-4, 0.3, 2.5), 8).tolist()",
        },
        {
            "setup": "def run_model():\n    try:\n        boosted_turnaround_potential([0, 1, 2], [0, 1], [0, 1, 2], [0], 1.0, 1.0, 1.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_boosted_turnaround_potential([0, 1, 2], [0, 1], [0, 1, 2], [0], 1.0, 1.0, 1.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        boosted_turnaround_potential([0, 1, 2], [0, 0, 0], [0, 1, 2], [np.int64(0), np.int64(0)], 1.0, 1.0, 1.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_boosted_turnaround_potential([0, 1, 2], [0, 0, 0], [0, 1, 2], [np.int64(0), np.int64(0)], 1.0, 1.0, 1.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def run_model():\n    try:\n        boosted_turnaround_potential([0, 1, 2], [0, 0, 0], [0, np.nan, 2], [0], 1.0, 1.0, 1.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_boosted_turnaround_potential([0, 1, 2], [0, 0, 0], [0, np.nan, 2], [0], 1.0, 1.0, 1.0, 1.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
