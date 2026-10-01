"""
Step 03: physical tensor-Q2 states at native volume nodes.

Each cell supplies a tensor-Q2 mapped conserved polynomial q_h with term columns (coefficient, power_x, power_r, power_s2, power_s3). The first-velocity power uses that cell's local r coordinate.  Return the corresponding physical-distribution values on the native ascending GL3 tensor grid.



The output is one float64 array of shape (2, 3, 3, 3, 3) with axes (cell, x, r, s2, s3).

Returns
-------
numpy.ndarray: Float64 array of shape ``(2,3,3,3,3)`` with axes ``(cell,x,r,s2,s3)`` on ascending GL3 nodes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import json
import numpy as np

def physical_distribution_nodal_state(case_json):
    """Return the two physical-distribution states on native volume nodes.

    Parameters
    ----------
    case_json : str
        JSON string encoding a complete two-cell case as an object.
        A supplied string is evaluated using its own coefficient values.

        Required JSON members:
        - ``map_coefficients_u1_u2_u3``: three length-3 lists giving the
          coefficients of ``u1(s1)``, ``u2(s2)``, and ``u3(s3)`` in
          ascending powers ``[constant, linear, quadratic]``.
        - ``electric_normal_coefficients``: length-3 list of coefficients
          of ``E1(x)`` in ascending powers.
        - ``magnetic_s2_coefficients``: length-3 list of coefficients of
          ``B2(x)`` in ascending powers.
        - ``magnetic_s3_coefficients``: length-3 list of coefficients of
          ``B3(x)`` in ascending powers.
        - ``q_left_terms``: nonempty list of left-cell polynomial terms.
        - ``q_right_terms``: nonempty list of right-cell polynomial terms.

        Each term is ``[coefficient, power_x, power_r, power_s2, power_s3]``
        and represents the coefficient times the corresponding monomial.
        The polynomial is the sum of its terms. Powers are integers in
        ``{0, 1, 2}``; coefficients are finite real numbers. Booleans are
        not accepted as coefficients or powers.

        Here ``s2=b`` and ``s3=c``. Each term's ``r`` is local to its cell:
        ``r_L=2*s1+1`` for ``s1`` in ``[-1,0]`` and ``r_R=2*s1-1`` for
        ``s1`` in ``[0,1]``. Every velocity map must have a strictly positive
        derivative throughout ``[-1,1]``. All six JSON members are required
        for every supplied case. An optional ``id`` is descriptive only.

    Returns
    -------
    numpy.ndarray
        Float64 array of shape ``(2,3,3,3,3)`` with axes
        ``(cell,x,r,s2,s3)`` on ascending GL3 nodes.

    Raises
    ------
    ValueError
        If the JSON or required arrays are malformed, non-finite, any power
        lies outside tensor Q2, or the mapped geometry is inadmissible.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import json
import numpy as np

def _state_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
    )
def _state_decode_case(case_json):
    if not isinstance(case_json, str):
        raise ValueError("case_json must be a string")
    try:
        case = json.loads(case_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("case_json is not valid JSON") from exc
    if not isinstance(case, dict):
        raise ValueError("case_json must decode to an object")

    raw_maps = case.get("map_coefficients_u1_u2_u3")
    if not isinstance(raw_maps, list) or len(raw_maps) != 3:
        raise ValueError("map coefficients must have shape (3, 3)")
    if any(not isinstance(row, list) or len(row) != 3 for row in raw_maps):
        raise ValueError("map coefficients must have shape (3, 3)")
    if any(not _state_number(value) for row in raw_maps for value in row):
        raise ValueError("map coefficients must be finite numbers")

    for key in (
        "electric_normal_coefficients",
        "magnetic_s2_coefficients",
        "magnetic_s3_coefficients",
    ):
        values = case.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{key} must contain three coefficients")
        if any(not _state_number(value) for value in values):
            raise ValueError(f"{key} must contain finite numbers")

    decoded_terms = []
    for key in ("q_left_terms", "q_right_terms"):
        terms = case.get(key)
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"{key} must be a nonempty term table")
        clean = []
        for term in terms:
            if not isinstance(term, list) or len(term) != 5:
                raise ValueError(f"{key} rows must have five entries")
            if not _state_number(term[0]):
                raise ValueError(f"{key} coefficients must be finite")
            powers = []
            for power in term[1:]:
                if isinstance(power, bool) or not isinstance(power, int):
                    raise ValueError(f"{key} powers must be integers")
                if power < 0 or power > 2:
                    raise ValueError(f"{key} powers must lie in [0, 2]")
                powers.append(power)
            clean.append((float(term[0]), *powers))
        decoded_terms.append(clean)
    return case, decoded_terms
def _state_evaluate_terms(terms, nodes):
    grids = np.meshgrid(nodes, nodes, nodes, nodes, indexing="ij")
    values = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for coefficient, *powers in terms:
        contribution = np.full(values.shape, coefficient, dtype=np.float64)
        for grid, power in zip(grids, powers):
            contribution *= grid ** power
        values += contribution
    return values
def _oracle_physical_distribution_nodal_state(case_json):
    """Reference implementation of
    :func:`physical_distribution_nodal_state`."""
    _, (left_terms, right_terms) = _state_decode_case(case_json)
    geometry = _oracle_mapped_velocity_geometry(case_json)
    if not isinstance(geometry, np.ndarray) or geometry.shape != (4, 2, 3, 3, 3):
        raise ValueError("mapped geometry has the wrong shape")
    volume_jacobian = np.asarray(geometry[0], dtype=np.float64)
    if not np.all(np.isfinite(volume_jacobian)) or np.any(volume_jacobian <= 0.0):
        raise ValueError("volume Jacobians must be finite and positive")

    gl_nodes = np.array(
        [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)],
        dtype=np.float64,
    )
    conserved = np.stack(
        (
            _state_evaluate_terms(left_terms, gl_nodes),
            _state_evaluate_terms(right_terms, gl_nodes),
        )
    )
    physical = conserved / volume_jacobian[:, None, :, :, :]
    physical = np.asarray(physical, dtype=np.float64)
    if physical.shape != (2, 3, 3, 3, 3) or not np.all(np.isfinite(physical)):
        raise ValueError("physical distribution state is not finite")
    return physical

# =============================================================================
# TEST CASES
# =============================================================================

import json
import numpy as np

_STATE_REFERENCE_JSON = r'''{"id":"reference_mixed_force","map_coefficients_u1_u2_u3":[[0.0,1.0,0.25],[0.0,0.8,0.2],[0.0,1.1,0.25]],"electric_normal_coefficients":[0.1,0.24,-0.12],"magnetic_s2_coefficients":[0.45,-0.2,0.15],"magnetic_s3_coefficients":[0.35,0.25,-0.1],"q_left_terms":[[2.0,0,0,0,0],[0.18,1,0,0,0],[0.12,0,1,0,0],[-0.14,0,0,1,0],[0.1,0,0,0,1],[0.16,1,1,0,0],[0.11,0,1,1,0],[-0.09,0,1,0,1],[0.08,1,0,1,0],[0.07,1,0,0,1],[0.06,0,0,1,1],[0.1,0,2,2,0],[0.08,2,0,0,2],[0.05,1,1,0,1]],"q_right_terms":[[1.6,0,0,0,0],[-0.12,1,0,0,0],[-0.1,0,1,0,0],[0.16,0,0,1,0],[-0.08,0,0,0,1],[-0.14,1,1,0,0],[0.09,0,1,1,0],[0.12,0,1,0,1],[-0.07,1,0,1,0],[0.06,1,0,0,1],[-0.05,0,0,1,1],[0.07,0,2,0,2],[0.09,2,0,2,0],[-0.04,1,1,1,0]]}'''

_STATE_OFFSET_JSON = r'''{"id":"offset_narrow_maps","map_coefficients_u1_u2_u3":[[0.2,0.65,0.12],[-0.1,0.55,0.12],[0.15,0.75,0.18]],"electric_normal_coefficients":[-0.08,0.28,0.06],"magnetic_s2_coefficients":[0.25,0.14,0.04],"magnetic_s3_coefficients":[-0.12,0.31,-0.03],"q_left_terms":[[1.4,0,0,0,0],[0.15,1,0,0,0],[0.13,0,1,0,0],[-0.11,0,0,1,0],[0.09,0,0,0,1],[0.12,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[-0.05,1,0,0,1],[0.04,0,0,1,1],[0.08,0,2,2,0]],"q_right_terms":[[1.9,0,0,0,0],[-0.14,1,0,0,0],[0.11,0,1,0,0],[0.12,0,0,1,0],[-0.09,0,0,0,1],[-0.1,1,1,0,0],[0.08,0,1,1,0],[-0.06,0,1,0,1],[0.07,1,0,1,0],[0.05,1,0,0,1],[-0.04,0,0,1,1],[0.06,0,2,0,2]]}'''

_STATE_AFFINE_JSON = r'''{"id":"affine_map_edge_case","map_coefficients_u1_u2_u3":[[-0.15,1.05,0.0],[0.05,0.7,0.0],[-0.1,0.9,0.0]],"electric_normal_coefficients":[0.05,-0.1,0.15],"magnetic_s2_coefficients":[-0.3,0.2,0.1],"magnetic_s3_coefficients":[0.4,0.05,-0.12],"q_left_terms":[[2.2,0,0,0,0],[-0.12,1,0,0,0],[0.18,0,1,0,0],[0.14,0,0,1,0],[-0.1,0,0,0,1],[0.09,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[0.05,0,0,1,1]],"q_right_terms":[[1.1,0,0,0,0],[0.1,1,0,0,0],[-0.14,0,1,0,0],[-0.11,0,0,1,0],[0.13,0,0,0,1],[-0.08,1,1,0,0],[0.07,0,1,1,0],[-0.06,0,1,0,1],[0.05,1,0,0,1],[-0.04,0,0,1,1]]}'''

def test_cases():
    """Return nonlinear, offset, affine, and invalid state cases."""
    cases = []
    cases.append({
        "setup": "case_json = _STATE_REFERENCE_JSON\n",
        "call": "physical_distribution_nodal_state(case_json)",
        "gold_call": "_oracle_physical_distribution_nodal_state(case_json)",
    })
    cases.append({
        "setup": "case_json = _STATE_OFFSET_JSON\n",
        "call": "physical_distribution_nodal_state(case_json)",
        "gold_call": "_oracle_physical_distribution_nodal_state(case_json)",
    })
    cases.append({
        "setup": "case_json = _STATE_AFFINE_JSON\n",
        "call": "physical_distribution_nodal_state(case_json)",
        "gold_call": "_oracle_physical_distribution_nodal_state(case_json)",
    })
    cases.append({
        "setup": (
            "case_json = 'null'\n"
            "def run_model():\n"
            "    try:\n"
            "        physical_distribution_nodal_state(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_physical_distribution_nodal_state(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })
    cases.append({
        "setup": (
            "case = json.loads(_STATE_REFERENCE_JSON)\n"
            "case['q_left_terms'][0][1] = 3\n"
            "case_json = json.dumps(case, separators=(',', ':'))\n"
            "def run_model():\n"
            "    try:\n"
            "        physical_distribution_nodal_state(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_physical_distribution_nodal_state(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
        ),
        "call": "run_model()",
        "gold_call": "run_gold()",
    })
    return cases
