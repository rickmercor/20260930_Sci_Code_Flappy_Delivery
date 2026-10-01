"""
Step 07 (final orchestrator): mapped velocity-face damping.

For two adjacent velocity cells, run the preceding numerical components and return the requested paired internal-face damping scalar.  The default input is the reference mixed-force case.  A supplied case must be a compact JSON string in the same schema.

Returns
-------
float: The mapped velocity-face damping scalar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import json
import numpy as np

def mapped_velocity_face_damping(case_json=None):
    """Run the mapped two-cell face construction and return its damping.

    Parameters
    ----------
    case_json : str or None, optional
        JSON string encoding a complete two-cell case. ``None`` selects
        the reference maps, fields, and conserved polynomials in the
        Problem statement. A supplied string is evaluated using its own
        coefficient values.

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
    float
        The mapped velocity-face damping scalar.

    Raises
    ------
    ValueError
        If the JSON is invalid or any stage returns malformed, non-finite, or
        physically inadmissible data.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import json
import numpy as np

def _damping_case_json(case_json):
    if case_json is None:
        case_json = r'''{"id":"reference_mixed_force","map_coefficients_u1_u2_u3":[[0.0,1.0,0.25],[0.0,0.8,0.2],[0.0,1.1,0.25]],"electric_normal_coefficients":[0.1,0.24,-0.12],"magnetic_s2_coefficients":[0.45,-0.2,0.15],"magnetic_s3_coefficients":[0.35,0.25,-0.1],"q_left_terms":[[2.0,0,0,0,0],[0.18,1,0,0,0],[0.12,0,1,0,0],[-0.14,0,0,1,0],[0.1,0,0,0,1],[0.16,1,1,0,0],[0.11,0,1,1,0],[-0.09,0,1,0,1],[0.08,1,0,1,0],[0.07,1,0,0,1],[0.06,0,0,1,1],[0.1,0,2,2,0],[0.08,2,0,0,2],[0.05,1,1,0,1]],"q_right_terms":[[1.6,0,0,0,0],[-0.12,1,0,0,0],[-0.1,0,1,0,0],[0.16,0,0,1,0],[-0.08,0,0,0,1],[-0.14,1,1,0,0],[0.09,0,1,1,0],[0.12,0,1,0,1],[-0.07,1,0,1,0],[0.06,1,0,0,1],[-0.05,0,0,1,1],[0.07,0,2,0,2],[0.09,2,0,2,0],[-0.04,1,1,1,0]]}'''
    if not isinstance(case_json, str):
        raise ValueError("case_json must be a string or None")
    try:
        case = json.loads(case_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("case_json is not valid JSON") from exc
    if not isinstance(case, dict):
        raise ValueError("case_json must decode to an object")
    return case_json
def _damping_finite_array(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array
def _oracle_mapped_velocity_face_damping(case_json=None):
    """Reference implementation of :func:`mapped_velocity_face_damping`."""
    normalized = _damping_case_json(case_json)

    geometry = _damping_finite_array(
        _oracle_mapped_velocity_geometry(normalized),
        (4, 2, 3, 3, 3),
        "mapped geometry",
    )
    gradient = _damping_finite_array(
        _oracle_discrete_face_hamiltonian_gradient(normalized),
        (2, 3, 3),
        "Hamiltonian gradient",
    )
    state = _damping_finite_array(
        _oracle_physical_distribution_nodal_state(normalized),
        (2, 3, 3, 3, 3),
        "physical distribution state",
    )
    traces = _damping_finite_array(
        _oracle_distribution_face_traces(normalized),
        (2, 3, 3, 3),
        "face traces",
    )
    transport = _damping_finite_array(
        _oracle_mapped_normal_face_transport(normalized),
        (3, 3, 3),
        "mapped face transport",
    )
    loss = _oracle_paired_upwind_face_loss(normalized)
    if isinstance(loss, bool) or not np.isscalar(loss):
        raise ValueError("paired face loss must be a numeric scalar")
    loss = float(loss)

    if np.any(geometry <= 0.0):
        raise ValueError("mapped geometry must be strictly positive")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("Hamiltonian gradient must be finite")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(traces)):
        raise ValueError("distribution state and traces must be finite")
    if not np.all(np.isfinite(transport)):
        raise ValueError("mapped face transport must be finite")
    if not np.isfinite(loss) or loss < 0.0:
        raise ValueError("paired face loss must be finite and nonnegative")
    return loss

# =============================================================================
# TEST CASES
# =============================================================================

import json
import numpy as np

_GEOMETRY_REFERENCE_JSON = r'''{"id":"reference_mixed_force","map_coefficients_u1_u2_u3":[[0.0,1.0,0.25],[0.0,0.8,0.2],[0.0,1.1,0.25]],"electric_normal_coefficients":[0.1,0.24,-0.12],"magnetic_s2_coefficients":[0.45,-0.2,0.15],"magnetic_s3_coefficients":[0.35,0.25,-0.1],"q_left_terms":[[2.0,0,0,0,0],[0.18,1,0,0,0],[0.12,0,1,0,0],[-0.14,0,0,1,0],[0.1,0,0,0,1],[0.16,1,1,0,0],[0.11,0,1,1,0],[-0.09,0,1,0,1],[0.08,1,0,1,0],[0.07,1,0,0,1],[0.06,0,0,1,1],[0.1,0,2,2,0],[0.08,2,0,0,2],[0.05,1,1,0,1]],"q_right_terms":[[1.6,0,0,0,0],[-0.12,1,0,0,0],[-0.1,0,1,0,0],[0.16,0,0,1,0],[-0.08,0,0,0,1],[-0.14,1,1,0,0],[0.09,0,1,1,0],[0.12,0,1,0,1],[-0.07,1,0,1,0],[0.06,1,0,0,1],[-0.05,0,0,1,1],[0.07,0,2,0,2],[0.09,2,0,2,0],[-0.04,1,1,1,0]]}'''

_GEOMETRY_OFFSET_JSON = r'''{"id":"offset_narrow_maps","map_coefficients_u1_u2_u3":[[0.2,0.65,0.12],[-0.1,0.55,0.12],[0.15,0.75,0.18]],"electric_normal_coefficients":[-0.08,0.28,0.06],"magnetic_s2_coefficients":[0.25,0.14,0.04],"magnetic_s3_coefficients":[-0.12,0.31,-0.03],"q_left_terms":[[1.4,0,0,0,0],[0.15,1,0,0,0],[0.13,0,1,0,0],[-0.11,0,0,1,0],[0.09,0,0,0,1],[0.12,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[-0.05,1,0,0,1],[0.04,0,0,1,1],[0.08,0,2,2,0]],"q_right_terms":[[1.9,0,0,0,0],[-0.14,1,0,0,0],[0.11,0,1,0,0],[0.12,0,0,1,0],[-0.09,0,0,0,1],[-0.1,1,1,0,0],[0.08,0,1,1,0],[-0.06,0,1,0,1],[0.07,1,0,1,0],[0.05,1,0,0,1],[-0.04,0,0,1,1],[0.06,0,2,0,2]]}'''

_GEOMETRY_AFFINE_JSON = r'''{"id":"affine_map_edge_case","map_coefficients_u1_u2_u3":[[-0.15,1.05,0.0],[0.05,0.7,0.0],[-0.1,0.9,0.0]],"electric_normal_coefficients":[0.05,-0.1,0.15],"magnetic_s2_coefficients":[-0.3,0.2,0.1],"magnetic_s3_coefficients":[0.4,0.05,-0.12],"q_left_terms":[[2.2,0,0,0,0],[-0.12,1,0,0,0],[0.18,0,1,0,0],[0.14,0,0,1,0],[-0.1,0,0,0,1],[0.09,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[0.05,0,0,1,1]],"q_right_terms":[[1.1,0,0,0,0],[0.1,1,0,0,0],[-0.14,0,1,0,0],[-0.11,0,0,1,0],[0.13,0,0,0,1],[-0.08,1,1,0,0],[0.07,0,1,1,0],[-0.06,0,1,0,1],[0.05,1,0,0,1],[-0.04,0,0,1,1]]}'''

def test_cases():
    """Return three whole-pipeline configurations and invalid cases."""
    cases = []
    cases.append({
        "setup": "case_json = None\n",
        "call": "mapped_velocity_face_damping()",
        "gold_call": "_oracle_mapped_velocity_face_damping()",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_OFFSET_JSON\n",
        "call": "mapped_velocity_face_damping(case_json)",
        "gold_call": "_oracle_mapped_velocity_face_damping(case_json)",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_AFFINE_JSON\n",
        "call": "mapped_velocity_face_damping(case_json)",
        "gold_call": "_oracle_mapped_velocity_face_damping(case_json)",
    })
    cases.append({
        "setup": (
            "case_json = json.dumps(json.loads(_GEOMETRY_REFERENCE_JSON), "
            "separators=(',', ':'), allow_nan=False)\n"
        ),
        "call": "mapped_velocity_face_damping(case_json)",
        "gold_call": "_oracle_mapped_velocity_face_damping(case_json)",
    })
    cases.append({
        "setup": (
            "case_json = '{broken json'\n"
            "def run_model():\n"
            "    try:\n"
            "        mapped_velocity_face_damping(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_mapped_velocity_face_damping(case_json)\n"
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
            "case_json = {'id': 'not serialized'}\n"
            "def run_model():\n"
            "    try:\n"
            "        mapped_velocity_face_damping(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_mapped_velocity_face_damping(case_json)\n"
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
