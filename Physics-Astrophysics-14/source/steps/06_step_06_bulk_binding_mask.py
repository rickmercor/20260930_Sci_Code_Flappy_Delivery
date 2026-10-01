"""
Apply the all-or-nothing host binding decision to a locally rebound subgroup. Position and canonical velocity must be finite, nonempty one-dimensional arrays of equal length, host physical potential must be a finite one-dimensional array of that length and the host individual mask must contain exactly zero or one. Locally bound subgroup IDs and surface IDs are distinct one-dimensional native or NumPy integral sequences in range. Copy the input mask. An empty subgroup is a no-op. Otherwise let `$H$` be IDs marked one in the original mask, `$v_host = mean(v_i for i in H)$`, `$q_mean = mean(position_i - position_minimum for i in S)$`, `$u_mean = mean(canonical_velocity_i for i in S) - v_host$` and `$v_bulk = scale_factor * hubble_rate * q_mean + u_mean / scale_factor$`. Form ``E_proxy = max(host_physical_potential[j] for j on the local surface) + 0.5 * v_bulk**2``. Set every locally bound subgroup ID to zero only when strict `$E_proxy > host_escape$`; equality remains bound. Return native Python ints. Preserve a finite representable proxy despite overflow in an intermediate power or product. Invalid input or a proxy outside the finite native-float range raises ValueError.

A candidate that is self-bound in its own accelerated frame can still escape the primary host as one coherent object. Its host-relative physical bulk velocity supplies a single kinetic term, while the maximum host-frame physical potential on the rebound object's graph surface supplies the potential proxy. The fixed kinetic coefficient is 0.5. A passing subgroup never rescues a particle rejected by the individual mask, because the decision modifies a copy of that mask only on strict bulk rejection.

Returns
-------
list of native Python ints, the updated zero-one host binding mask
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def bulk_binding_mask(position, canonical_velocity, host_physical_potential,
                      host_individual_mask, subgroup_bound_ids,
                      subgroup_surface_ids, host_escape, minimum_id,
                      scale_factor, hubble_rate):
    """Return the host mask after one subgroup bulk decision.

    Parameters
    ----------
    position : array_like
        Finite nonempty one-dimensional positions.
    canonical_velocity : array_like
        Finite one-dimensional canonical velocities of the same length.
    host_physical_potential : array_like
        Finite one-dimensional host-frame physical potentials.
    host_individual_mask : array_like of int
        One-dimensional zero-one host binding mask.
    subgroup_bound_ids : array_like of int
        Distinct valid locally bound subgroup IDs.
    subgroup_surface_ids : array_like of int
        Distinct valid graph-surface IDs disjoint from a nonempty subgroup.
    host_escape : float
        Finite host escape energy.
    minimum_id : int
        Valid native or NumPy integral host minimum ID.
    scale_factor : float
        Positive finite scale factor.
    hubble_rate : float
        Nonnegative finite Hubble rate.

    Returns
    -------
    list of int
        Copied zero-one host mask after the strict bulk rejection decision.

    Raises
    ------
    ValueError
        If an input violates the contract or a nonempty subgroup's proxy is not
        representable as a finite native float; preserve representable proxies.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bulk_binding_mask(position, canonical_velocity,
                            host_physical_potential, host_individual_mask,
                            subgroup_bound_ids, subgroup_surface_ids,
                            host_escape, minimum_id, scale_factor, hubble_rate):
    arrays = []
    for value, name in ((position, "position"),
                        (canonical_velocity, "canonical_velocity"),
                        (host_physical_potential, "host_physical_potential")):
        try:
            array = np.asarray(value)
        except Exception as exc:
            raise ValueError(name + " must be a numeric one-dimensional array") from exc
        if array.ndim != 1 or array.size == 0:
            raise ValueError(name + " must be nonempty and one-dimensional")
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
            raise ValueError(name + " must contain real numeric values")
        try:
            array = np.asarray(array, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must contain finite real values") from exc
        if not np.all(np.isfinite(array)):
            raise ValueError(name + " must contain finite real values")
        arrays.append(array)
    position_array, velocity_array, physical_array = arrays
    if position_array.shape != velocity_array.shape or position_array.shape != physical_array.shape:
        raise ValueError("particle arrays must have equal length")
    node_count = int(position_array.size)

    mask_array = np.asarray(host_individual_mask, dtype=object)
    if mask_array.ndim != 1 or mask_array.size != node_count:
        raise ValueError("host_individual_mask must have one entry per node")
    mask = []
    for value in mask_array.tolist():
        if isinstance(value, (bool, np.bool_)):
            mask.append(int(value))
        elif isinstance(value, (int, np.integer)) and int(value) in (0, 1):
            mask.append(int(value))
        else:
            raise ValueError("host_individual_mask must contain only zero and one")

    def ids(value, name):
        array = np.asarray(value, dtype=object)
        if array.ndim != 1:
            raise ValueError(name + " must be one-dimensional")
        converted = []
        for item in array.tolist():
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, (int, np.integer)):
                raise ValueError(name + " must contain integral node IDs")
            node = int(item)
            if node < 0 or node >= node_count:
                raise ValueError(name + " contains an invalid node ID")
            converted.append(node)
        if len(set(converted)) != len(converted):
            raise ValueError(name + " must contain distinct node IDs")
        return converted

    subgroup = ids(subgroup_bound_ids, "subgroup_bound_ids")
    surface = ids(subgroup_surface_ids, "subgroup_surface_ids")
    if isinstance(minimum_id, (bool, np.bool_)) or not isinstance(minimum_id, (int, np.integer)):
        raise ValueError("minimum_id must be integral")
    minimum_id = int(minimum_id)
    if minimum_id < 0 or minimum_id >= node_count:
        raise ValueError("minimum_id is out of range")
    scalars = []
    for name, value, positive in (("host_escape", host_escape, None),
                                  ("scale_factor", scale_factor, True),
                                  ("hubble_rate", hubble_rate, False)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(name + " must be a finite scalar")
        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite scalar") from exc
        if not np.isfinite(scalar):
            raise ValueError(name + " must be finite")
        if positive is True and scalar <= 0.0:
            raise ValueError(name + " must be positive")
        if positive is False and scalar < 0.0:
            raise ValueError(name + " must be nonnegative")
        scalars.append(scalar)
    host_escape, scale_factor, hubble_rate = scalars

    output = list(mask)
    if not subgroup:
        return [int(value) for value in output]
    if not surface:
        raise ValueError("a nonempty subgroup must have a nonempty surface")
    if set(subgroup) & set(surface):
        raise ValueError("subgroup and surface IDs must be disjoint")
    host_ids = [node for node, value in enumerate(mask) if value == 1]
    if not host_ids:
        raise ValueError("a nonempty subgroup requires a host velocity sample")

    def wide_proxy():
        from fractions import Fraction as F
        x = [F(float(v)) for v in position_array]
        velocity = [F(float(v)) for v in velocity_array]
        scale, hubble = F(scale_factor), F(hubble_rate)
        host_velocity = sum(velocity[j] for j in host_ids) / len(host_ids)
        q = sum(x[j] - x[minimum_id] for j in subgroup) / len(subgroup)
        residual = sum(velocity[j] for j in subgroup) / len(subgroup) - host_velocity
        bulk = scale * hubble * q + residual / scale
        exact = F(float(np.max(physical_array[surface]))) + bulk * bulk / 2
        try:
            value = float(exact)
        except OverflowError as exc:
            raise ValueError("bulk proxy energy cannot be represented as a finite float") from exc
        if not np.isfinite(value):
            raise ValueError("bulk proxy energy cannot be represented as a finite float")
        return value

    proxy = wide_proxy()
    if proxy > host_escape:
        for node in subgroup:
            output[node] = 0
    return [int(value) for value in output]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "",
            "call": "bulk_binding_mask([0, 1e250, 2e250], [0, 0, 0], [0, 0, 0], [1, 1, 1], [1], [2], 2e-301, 0, 1e-300, 1e-100)",
            "gold_call": "[1, 0, 1]",
        },
        {
            "setup": "",
            "call": "bulk_binding_mask([0, 0, 0], [0, 0, 0], [0, 1, 2], [1, 1, 1], [1], [2], 1.0, 0, 1.0, 1e200)",
            "gold_call": "[1, 0, 1]",
        },
        {
            "setup": "",
            "call": "bulk_binding_mask([0, 1, 2], [1e200, 1e200, 1e200], [0, 1, 2], [1, 1, 1], [1], [2], 2.0, 0, 1.0, 0.0)",
            "gold_call": "[1, 1, 1]",
        },
        {
            "setup": "def overflow_status(function):\n    try:\n        function([0, 1, 2], [0, 1e200, 0], [0, 1, 2], [1, 1, 1], [1], [2], 1.0, 0, 1.0, 0.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "overflow_status(bulk_binding_mask)",
            "gold_call": "overflow_status(_oracle_bulk_binding_mask)",
        },
        {
            "setup": "position = [0.0, 1.0, 2.0, 3.0, 4.0]\nvelocity = [0.0, 0.0, 4.0, 4.0, 0.0]\nphysical = [0.0, 1.0, 2.0, 5.0, 7.0]\nmask = [1, 1, 1, 1, 0]\n",
            "call": "bulk_binding_mask(position, velocity, physical, mask, np.array([2, 3], dtype=np.int64), np.array([1, 4], dtype=np.int64), 6.0, np.int64(0), 1.0, 0.0)",
            "gold_call": "_oracle_bulk_binding_mask(position, velocity, physical, mask, np.array([2, 3], dtype=np.int64), np.array([1, 4], dtype=np.int64), 6.0, np.int64(0), 1.0, 0.0)",
        },
        {
            "setup": "position = [0.0, 1.0, 2.0, 3.0]\nvelocity = [0.0, 0.0, 0.5, 0.5]\nphysical = [0.0, 1.0, 2.0, 3.0]\nmask = [1, 0, 1, 1]\n",
            "call": "bulk_binding_mask(position, velocity, physical, mask, [2, 3], [1], 5.0, 0, 1.0, 0.0)",
            "gold_call": "_oracle_bulk_binding_mask(position, velocity, physical, mask, [2, 3], [1], 5.0, 0, 1.0, 0.0)",
        },
        {
            "setup": "position = [0.0, 1.0, 2.0]\nvelocity = [0.0, 0.0, 2.0]\nphysical = [0.0, 1.0, 3.0]\nmask = [1, 1, 1]\n",
            "call": "bulk_binding_mask(position, velocity, physical, mask, [2], [1], 3.888888888888889, 0, 1.0, 0.0)",
            "gold_call": "_oracle_bulk_binding_mask(position, velocity, physical, mask, [2], [1], 3.888888888888889, 0, 1.0, 0.0)",
        },
        {
            "setup": "mask = [0, 0, 0]\n",
            "call": "bulk_binding_mask([0, 1, 2], [0, 0, 0], [0, 1, 2], mask, [], [], 1.0, 0, 1.0, 0.0)",
            "gold_call": "_oracle_bulk_binding_mask([0, 1, 2], [0, 0, 0], [0, 1, 2], mask, [], [], 1.0, 0, 1.0, 0.0)",
        },
        {
            "setup": "position = [-2.0, 0.0, 4.0, 7.0]\nvelocity = [1.0, 2.0, 5.0, 6.0]\nphysical = [0.0, 2.0, 4.0, 5.0]\nmask = [1, 1, 1, 1]\n",
            "call": "bulk_binding_mask(position, velocity, physical, mask, [2, 3], [1], 20.0, 1, 0.5, 2.0)",
            "gold_call": "_oracle_bulk_binding_mask(position, velocity, physical, mask, [2, 3], [1], 20.0, 1, 0.5, 2.0)",
        },
        {
            "setup": "def run_model():\n    try:\n        bulk_binding_mask([0, 1, 2], [0, 0, 0], [0, 1, 2], [1, 1, 1], [1], [], 2.0, 0, 1.0, 0.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n\ndef run_gold():\n    try:\n        _oracle_bulk_binding_mask([0, 1, 2], [0, 0, 0], [0, 1, 2], [1, 1, 1], [1], [], 2.0, 0, 1.0, 0.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": "def invalid_status(function):\n    try:\n        function([0, 1, 2], [0, 0, 0], [0, 1, 2], [1, 1, 1], [1], [1, 2], 2.0, 0, 1.0, 0.0)\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n    return 0\n",
            "call": "invalid_status(bulk_binding_mask)",
            "gold_call": "invalid_status(_oracle_bulk_binding_mask)",
        },
    ]
