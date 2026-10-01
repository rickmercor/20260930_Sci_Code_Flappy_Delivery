"""
Step 04: physical-distribution values on the shared cell face.

The serialized two-cell case defines two physical-distribution polynomials. The common face is the left cell's local r = +1 boundary and the right cell's local r = -1 boundary.  Return the two physical-distribution face arrays in ascending GL3 order.



The float64 output has shape (2, 3, 3, 3) with axes (side, x, s2, s3).

Returns
-------
numpy.ndarray: Float64 array of shape ``(2,3,3,3)``.  Row 0 is the left-cell trace at ``r=+1`` and row 1 is the right-cell trace at ``r=-1``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _trace_poly

def distribution_face_traces(case_json):
    """Return the left and right physical-distribution face traces.

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
        Float64 array of shape ``(2,3,3,3)``.  Row 0 is the left-cell trace at
        ``r=+1`` and row 1 is the right-cell trace at ``r=-1``.

    Raises
    ------
    ValueError
        If ``case_json`` is invalid or the requested trace result has the wrong
        shape or contains a non-finite value.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _trace_poly

def _trace_decode_case(case_json):
    if not isinstance(case_json, str):
        raise ValueError("case_json must be a string")
    try:
        case = json.loads(case_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("case_json is not valid JSON") from exc
    if not isinstance(case, dict):
        raise ValueError("case_json must decode to an object")
    return case
def _trace_endpoint_weights(endpoint):
    nodes = np.array(
        [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)],
        dtype=np.float64,
    )
    vandermonde = _trace_poly.polyvander(nodes, 2)
    coefficients = np.linalg.inv(vandermonde)
    return np.array(
        [
            _trace_poly.polyval(endpoint, coefficients[:, basis_index])
            for basis_index in range(3)
        ],
        dtype=np.float64,
    )
def _oracle_distribution_face_traces(case_json):
    """Reference implementation of :func:`distribution_face_traces`."""
    _trace_decode_case(case_json)
    state = np.asarray(
        _oracle_physical_distribution_nodal_state(case_json),
        dtype=np.float64,
    )
    if state.shape != (2, 3, 3, 3, 3):
        raise ValueError("physical distribution state must have shape (2,3,3,3,3)")
    if not np.all(np.isfinite(state)):
        raise ValueError("physical distribution state must be finite")

    left_weights = _trace_endpoint_weights(1.0)
    right_weights = _trace_endpoint_weights(-1.0)
    left = np.einsum("r,xrst->xst", left_weights, state[0])
    right = np.einsum("r,xrst->xst", right_weights, state[1])
    result = np.stack((left, right)).astype(np.float64, copy=False)
    if result.shape != (2, 3, 3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("physical face traces must be finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _trace_poly

_GEOMETRY_REFERENCE_JSON = r'''{"id":"reference_mixed_force","map_coefficients_u1_u2_u3":[[0.0,1.0,0.25],[0.0,0.8,0.2],[0.0,1.1,0.25]],"electric_normal_coefficients":[0.1,0.24,-0.12],"magnetic_s2_coefficients":[0.45,-0.2,0.15],"magnetic_s3_coefficients":[0.35,0.25,-0.1],"q_left_terms":[[2.0,0,0,0,0],[0.18,1,0,0,0],[0.12,0,1,0,0],[-0.14,0,0,1,0],[0.1,0,0,0,1],[0.16,1,1,0,0],[0.11,0,1,1,0],[-0.09,0,1,0,1],[0.08,1,0,1,0],[0.07,1,0,0,1],[0.06,0,0,1,1],[0.1,0,2,2,0],[0.08,2,0,0,2],[0.05,1,1,0,1]],"q_right_terms":[[1.6,0,0,0,0],[-0.12,1,0,0,0],[-0.1,0,1,0,0],[0.16,0,0,1,0],[-0.08,0,0,0,1],[-0.14,1,1,0,0],[0.09,0,1,1,0],[0.12,0,1,0,1],[-0.07,1,0,1,0],[0.06,1,0,0,1],[-0.05,0,0,1,1],[0.07,0,2,0,2],[0.09,2,0,2,0],[-0.04,1,1,1,0]]}'''

_GEOMETRY_OFFSET_JSON = r'''{"id":"offset_narrow_maps","map_coefficients_u1_u2_u3":[[0.2,0.65,0.12],[-0.1,0.55,0.12],[0.15,0.75,0.18]],"electric_normal_coefficients":[-0.08,0.28,0.06],"magnetic_s2_coefficients":[0.25,0.14,0.04],"magnetic_s3_coefficients":[-0.12,0.31,-0.03],"q_left_terms":[[1.4,0,0,0,0],[0.15,1,0,0,0],[0.13,0,1,0,0],[-0.11,0,0,1,0],[0.09,0,0,0,1],[0.12,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[-0.05,1,0,0,1],[0.04,0,0,1,1],[0.08,0,2,2,0]],"q_right_terms":[[1.9,0,0,0,0],[-0.14,1,0,0,0],[0.11,0,1,0,0],[0.12,0,0,1,0],[-0.09,0,0,0,1],[-0.1,1,1,0,0],[0.08,0,1,1,0],[-0.06,0,1,0,1],[0.07,1,0,1,0],[0.05,1,0,0,1],[-0.04,0,0,1,1],[0.06,0,2,0,2]]}'''

_GEOMETRY_AFFINE_JSON = r'''{"id":"affine_map_edge_case","map_coefficients_u1_u2_u3":[[-0.15,1.05,0.0],[0.05,0.7,0.0],[-0.1,0.9,0.0]],"electric_normal_coefficients":[0.05,-0.1,0.15],"magnetic_s2_coefficients":[-0.3,0.2,0.1],"magnetic_s3_coefficients":[0.4,0.05,-0.12],"q_left_terms":[[2.2,0,0,0,0],[-0.12,1,0,0,0],[0.18,0,1,0,0],[0.14,0,0,1,0],[-0.1,0,0,0,1],[0.09,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[0.05,0,0,1,1]],"q_right_terms":[[1.1,0,0,0,0],[0.1,1,0,0,0],[-0.14,0,1,0,0],[-0.11,0,0,1,0],[0.13,0,0,0,1],[-0.08,1,1,0,0],[0.07,0,1,1,0],[-0.06,0,1,0,1],[0.05,1,0,0,1],[-0.04,0,0,1,1]]}'''

def test_cases():
    """Return nonlinear, offset, affine-edge, and invalid trace cases."""
    cases = []
    cases.append({
        "setup": "case_json = _GEOMETRY_REFERENCE_JSON\n",
        "call": "distribution_face_traces(case_json)",
        "gold_call": "_oracle_distribution_face_traces(case_json)",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_OFFSET_JSON\n",
        "call": "distribution_face_traces(case_json)",
        "gold_call": "_oracle_distribution_face_traces(case_json)",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_AFFINE_JSON\n",
        "call": "distribution_face_traces(case_json)",
        "gold_call": "_oracle_distribution_face_traces(case_json)",
    })
    cases.append({
        "setup": (
            "case_json = '{broken json'\n"
            "def run_model():\n"
            "    try:\n"
            "        distribution_face_traces(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_distribution_face_traces(case_json)\n"
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
            "case_json = []\n"
            "def run_model():\n"
            "    try:\n"
            "        distribution_face_traces(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_distribution_face_traces(case_json)\n"
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
