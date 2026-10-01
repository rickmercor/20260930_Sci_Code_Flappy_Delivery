"""
Step 06: paired upwind loss across one internal velocity-space face.

For the serialized two-cell case, return the nonnegative paired-face loss on the explicitly prescribed tensor three-point Gauss-Legendre rule over ``(x,s2,s3)``.  This rule defines the requested discrete scalar.

Returns
-------
float: The nonnegative paired-face loss evaluated with the prescribed GL3 rule.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import json
import numpy as np

def paired_upwind_face_loss(case_json):
    """Return the nonnegative paired upwind face-loss scalar.

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
    float
        The nonnegative paired-face loss evaluated with the prescribed GL3
        rule.

    Raises
    ------
    ValueError
        If the JSON is invalid, an upstream array is malformed or non-finite,
        or the resulting loss is not finite and nonnegative.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import json
import numpy as np

def _loss_decode_case(case_json):
    if not isinstance(case_json, str):
        raise ValueError("case_json must be a string")
    try:
        case = json.loads(case_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("case_json is not valid JSON") from exc
    if not isinstance(case, dict):
        raise ValueError("case_json must decode to an object")
    return case
def _oracle_paired_upwind_face_loss(case_json):
    """Reference implementation of :func:`paired_upwind_face_loss`."""
    _loss_decode_case(case_json)
    traces = np.asarray(
        _oracle_distribution_face_traces(case_json), dtype=np.float64
    )
    transport = np.asarray(
        _oracle_mapped_normal_face_transport(case_json), dtype=np.float64
    )
    if traces.shape != (2, 3, 3, 3):
        raise ValueError("face traces must have shape (2,3,3,3)")
    if transport.shape != (3, 3, 3):
        raise ValueError("mapped face transport must have shape (3,3,3)")
    if not np.all(np.isfinite(traces)) or not np.all(np.isfinite(transport)):
        raise ValueError("face traces and mapped transport must be finite")

    jump = traces[0] - traces[1]
    density = 0.5 * np.abs(transport) * jump ** 2
    weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
    loss = float(
        np.einsum("i,j,k,ijk->", weights, weights, weights, density)
    )
    if not np.isfinite(loss) or loss < 0.0:
        raise ValueError("paired upwind face loss must be finite and nonnegative")
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
    """Return three mapped-face integrations and invalid input cases."""
    cases = []
    cases.append({
        "setup": "case_json = _GEOMETRY_REFERENCE_JSON\n",
        "call": "paired_upwind_face_loss(case_json)",
        "gold_call": "_oracle_paired_upwind_face_loss(case_json)",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_OFFSET_JSON\n",
        "call": "paired_upwind_face_loss(case_json)",
        "gold_call": "_oracle_paired_upwind_face_loss(case_json)",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_AFFINE_JSON\n",
        "call": "paired_upwind_face_loss(case_json)",
        "gold_call": "_oracle_paired_upwind_face_loss(case_json)",
    })
    cases.append({
        "setup": (
            "case_json = '{broken json'\n"
            "def run_model():\n"
            "    try:\n"
            "        paired_upwind_face_loss(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_paired_upwind_face_loss(case_json)\n"
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
            "case['q_left_terms'][0][1] = -1\n"
            "case_json = json.dumps(case, separators=(',', ':'))\n"
            "def run_model():\n"
            "    try:\n"
            "        paired_upwind_face_loss(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_paired_upwind_face_loss(case_json)\n"
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
