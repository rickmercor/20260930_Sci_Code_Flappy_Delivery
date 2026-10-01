"""
Step 05: signed mapped transport on the shared velocity-space face.

Return the normal transport coefficient for the left-to-right face orientation specified by the serialized two-cell case.  Charge-to-mass ratio and light speed are both +1 in the supplied units.

The float64 output has shape ``(3,3,3)`` with axes ``(x,s2,s3)`` on the prescribed ascending tensor GL3 rule.

Returns
-------
numpy.ndarray: Float64 array of shape ``(3,3,3)`` containing the signed mapped normal face-transport coefficient.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _transport_poly

def mapped_normal_face_transport(case_json):
    """Return the signed mapped normal face-transport array.

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
        Float64 array of shape ``(3,3,3)`` containing the signed mapped normal
        face-transport coefficient.

    Raises
    ------
    ValueError
        If the JSON, serialized inputs, or requested result is malformed or
        non-finite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _transport_poly

def _transport_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
    )
def _transport_decode_case(case_json):
    if not isinstance(case_json, str):
        raise ValueError("case_json must be a string")
    try:
        case = json.loads(case_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("case_json is not valid JSON") from exc
    if not isinstance(case, dict):
        raise ValueError("case_json must decode to an object")

    fields = []
    for key in (
        "electric_normal_coefficients",
        "magnetic_s2_coefficients",
        "magnetic_s3_coefficients",
    ):
        values = case.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{key} must contain three coefficients")
        if any(not _transport_number(value) for value in values):
            raise ValueError(f"{key} must contain finite numbers")
        fields.append(np.asarray(values, dtype=np.float64))
    return case, fields
def _transport_shared_layer(geometry, layer, label):
    reference = geometry[layer, 0, 0]
    repeated = np.broadcast_to(reference[None, None, :, :], (2, 3, 3, 3))
    if not np.array_equal(geometry[layer], repeated):
        raise ValueError(f"{label} must be shared and broadcast over cells and r")
    return reference
def _oracle_mapped_normal_face_transport(case_json):
    """Reference implementation of :func:`mapped_normal_face_transport`."""
    _, fields = _transport_decode_case(case_json)
    geometry = np.asarray(
        _oracle_mapped_velocity_geometry(case_json), dtype=np.float64
    )
    gradient = np.asarray(
        _oracle_discrete_face_hamiltonian_gradient(case_json),
        dtype=np.float64,
    )
    if geometry.shape != (4, 2, 3, 3, 3):
        raise ValueError("mapped geometry must have shape (4,2,3,3,3)")
    if gradient.shape != (2, 3, 3):
        raise ValueError("Hamiltonian gradient must have shape (2,3,3)")
    if not np.all(np.isfinite(geometry)) or not np.all(np.isfinite(gradient)):
        raise ValueError("geometry and Hamiltonian gradient must be finite")

    surface = _transport_shared_layer(geometry, 1, "surface Jacobian")
    du2 = _transport_shared_layer(geometry, 2, "du2/ds2")
    du3 = _transport_shared_layer(geometry, 3, "du3/ds3")
    if np.any(surface <= 0.0) or np.any(du2 <= 0.0) or np.any(du3 <= 0.0):
        raise ValueError("face geometry must be strictly positive")
    expected_surface = du2 * du3
    if not np.allclose(
        surface,
        expected_surface,
        rtol=64.0 * np.finfo(np.float64).eps,
        atol=64.0 * np.finfo(np.float64).eps,
    ):
        raise ValueError("surface Jacobian is inconsistent with tangential maps")

    velocity2 = gradient[0] / du2
    velocity3 = gradient[1] / du3
    gl_nodes = np.array(
        [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)],
        dtype=np.float64,
    )
    electric1 = _transport_poly.polyval(gl_nodes, fields[0])
    magnetic2 = _transport_poly.polyval(gl_nodes, fields[1])
    magnetic3 = _transport_poly.polyval(gl_nodes, fields[2])
    magnetic = (
        velocity2[None, :, :] * magnetic3[:, None, None]
        - velocity3[None, :, :] * magnetic2[:, None, None]
    )
    characteristic = electric1[:, None, None] + magnetic
    result = surface[None, :, :] * characteristic
    result = result.astype(np.float64, copy=False)
    if result.shape != (3, 3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("mapped normal face transport must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _transport_poly

_GEOMETRY_REFERENCE_JSON = r'''{"id":"reference_mixed_force","map_coefficients_u1_u2_u3":[[0.0,1.0,0.25],[0.0,0.8,0.2],[0.0,1.1,0.25]],"electric_normal_coefficients":[0.1,0.24,-0.12],"magnetic_s2_coefficients":[0.45,-0.2,0.15],"magnetic_s3_coefficients":[0.35,0.25,-0.1],"q_left_terms":[[2.0,0,0,0,0],[0.18,1,0,0,0],[0.12,0,1,0,0],[-0.14,0,0,1,0],[0.1,0,0,0,1],[0.16,1,1,0,0],[0.11,0,1,1,0],[-0.09,0,1,0,1],[0.08,1,0,1,0],[0.07,1,0,0,1],[0.06,0,0,1,1],[0.1,0,2,2,0],[0.08,2,0,0,2],[0.05,1,1,0,1]],"q_right_terms":[[1.6,0,0,0,0],[-0.12,1,0,0,0],[-0.1,0,1,0,0],[0.16,0,0,1,0],[-0.08,0,0,0,1],[-0.14,1,1,0,0],[0.09,0,1,1,0],[0.12,0,1,0,1],[-0.07,1,0,1,0],[0.06,1,0,0,1],[-0.05,0,0,1,1],[0.07,0,2,0,2],[0.09,2,0,2,0],[-0.04,1,1,1,0]]}'''

_GEOMETRY_OFFSET_JSON = r'''{"id":"offset_narrow_maps","map_coefficients_u1_u2_u3":[[0.2,0.65,0.12],[-0.1,0.55,0.12],[0.15,0.75,0.18]],"electric_normal_coefficients":[-0.08,0.28,0.06],"magnetic_s2_coefficients":[0.25,0.14,0.04],"magnetic_s3_coefficients":[-0.12,0.31,-0.03],"q_left_terms":[[1.4,0,0,0,0],[0.15,1,0,0,0],[0.13,0,1,0,0],[-0.11,0,0,1,0],[0.09,0,0,0,1],[0.12,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[-0.05,1,0,0,1],[0.04,0,0,1,1],[0.08,0,2,2,0]],"q_right_terms":[[1.9,0,0,0,0],[-0.14,1,0,0,0],[0.11,0,1,0,0],[0.12,0,0,1,0],[-0.09,0,0,0,1],[-0.1,1,1,0,0],[0.08,0,1,1,0],[-0.06,0,1,0,1],[0.07,1,0,1,0],[0.05,1,0,0,1],[-0.04,0,0,1,1],[0.06,0,2,0,2]]}'''

_GEOMETRY_AFFINE_JSON = r'''{"id":"affine_map_edge_case","map_coefficients_u1_u2_u3":[[-0.15,1.05,0.0],[0.05,0.7,0.0],[-0.1,0.9,0.0]],"electric_normal_coefficients":[0.05,-0.1,0.15],"magnetic_s2_coefficients":[-0.3,0.2,0.1],"magnetic_s3_coefficients":[0.4,0.05,-0.12],"q_left_terms":[[2.2,0,0,0,0],[-0.12,1,0,0,0],[0.18,0,1,0,0],[0.14,0,0,1,0],[-0.1,0,0,0,1],[0.09,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[0.05,0,0,1,1]],"q_right_terms":[[1.1,0,0,0,0],[0.1,1,0,0,0],[-0.14,0,1,0,0],[-0.11,0,0,1,0],[0.13,0,0,0,1],[-0.08,1,1,0,0],[0.07,0,1,1,0],[-0.06,0,1,0,1],[0.05,1,0,0,1],[-0.04,0,0,1,1]]}'''

def test_cases():
    """Return mixed-force, offset, affine-edge, and invalid cases."""
    cases = []
    cases.append({
        "setup": "case_json = _GEOMETRY_REFERENCE_JSON\n",
        "call": "mapped_normal_face_transport(case_json)",
        "gold_call": "_oracle_mapped_normal_face_transport(case_json)",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_OFFSET_JSON\n",
        "call": "mapped_normal_face_transport(case_json)",
        "gold_call": "_oracle_mapped_normal_face_transport(case_json)",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_AFFINE_JSON\n",
        "call": "mapped_normal_face_transport(case_json)",
        "gold_call": "_oracle_mapped_normal_face_transport(case_json)",
    })
    cases.append({
        "setup": (
            "case_json = '[]'\n"
            "def run_model():\n"
            "    try:\n"
            "        mapped_normal_face_transport(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_mapped_normal_face_transport(case_json)\n"
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
            "case = json.loads(_GEOMETRY_REFERENCE_JSON)\n"
            "case['magnetic_s3_coefficients'] = [0.2, 0.1]\n"
            "case_json = json.dumps(case, separators=(',', ':'))\n"
            "def run_model():\n"
            "    try:\n"
            "        mapped_normal_face_transport(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_mapped_normal_face_transport(case_json)\n"
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
