"""
Step 01: mapped geometry for two cells sharing a velocity-space face.

The input is one compact JSON object with the schema documented by the task. Velocity maps are quadratic polynomials with coefficients in ascending powers. The two fixed global s1 cells are [-1, 0] and [0, 1]; each uses a local coordinate r increasing from -1 to +1.  All arrays use the ascending three-point Gauss-Legendre grid.



The returned array has shape (4, 2, 3, 3, 3) and axes (quantity, cell, r, s2, s3).  Layer 0 is the full local volume Jacobian. Layer 1 is the induced shared-face Jacobian, repeated over cell and normal-node axes.  Layers 2 and 3 are the s2 and s3 map derivatives, respectively, repeated over unused axes.

Returns
-------
numpy.ndarray: Float64 array of shape ``(4, 2, 3, 3, 3)`` with axes ``(quantity,cell,r,s2,s3)``.  Layers are full local volume Jacobian, induced face Jacobian, ``du2/ds2``, and ``du3/ds3``.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _geometry_poly

def mapped_velocity_geometry(case_json):
    """Return mapped volume, face, and tangential geometry.

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
        Float64 array of shape ``(4, 2, 3, 3, 3)`` with axes
        ``(quantity,cell,r,s2,s3)``.  Layers are full local volume Jacobian,
        induced face Jacobian, ``du2/ds2``, and ``du3/ds3``.

    Raises
    ------
    ValueError
        If the JSON or required arrays are malformed, non-finite, or any map
        is not strictly increasing on its reference interval.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _geometry_poly

def _geometry_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
    )
def _geometry_decode_case(case_json):
    if not isinstance(case_json, str):
        raise ValueError("case_json must be a string")
    try:
        case = json.loads(case_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("case_json is not valid JSON") from exc
    if not isinstance(case, dict):
        raise ValueError("case_json must decode to an object")

    required_vectors = (
        "electric_normal_coefficients",
        "magnetic_s2_coefficients",
        "magnetic_s3_coefficients",
    )
    raw_maps = case.get("map_coefficients_u1_u2_u3")
    if not isinstance(raw_maps, list) or len(raw_maps) != 3:
        raise ValueError("map coefficients must have shape (3, 3)")
    if any(not isinstance(row, list) or len(row) != 3 for row in raw_maps):
        raise ValueError("map coefficients must have shape (3, 3)")
    if any(not _geometry_number(value) for row in raw_maps for value in row):
        raise ValueError("map coefficients must be finite numbers")
    maps = np.asarray(raw_maps, dtype=np.float64)

    for key in required_vectors:
        values = case.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{key} must contain three coefficients")
        if any(not _geometry_number(value) for value in values):
            raise ValueError(f"{key} must contain finite numbers")

    for key in ("q_left_terms", "q_right_terms"):
        terms = case.get(key)
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"{key} must be a nonempty term table")
        for term in terms:
            if not isinstance(term, list) or len(term) != 5:
                raise ValueError(f"{key} rows must have five entries")
            if not _geometry_number(term[0]):
                raise ValueError(f"{key} coefficients must be finite")
            for power in term[1:]:
                if isinstance(power, bool) or not isinstance(power, int):
                    raise ValueError(f"{key} powers must be integers")
                if power < 0 or power > 2:
                    raise ValueError(f"{key} powers must lie in [0, 2]")

    for axis, coefficients in enumerate(maps):
        derivative = _geometry_poly.polyder(coefficients)
        endpoints = np.array([-1.0, 1.0], dtype=np.float64)
        if np.any(_geometry_poly.polyval(endpoints, derivative) <= 0.0):
            raise ValueError(f"velocity map {axis + 1} is not increasing")
    return case, maps
def _oracle_mapped_velocity_geometry(case_json):
    """Reference implementation of :func:`mapped_velocity_geometry`."""
    _, maps = _geometry_decode_case(case_json)
    gl_nodes = np.array(
        [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)],
        dtype=np.float64,
    )
    derivatives = [_geometry_poly.polyder(row) for row in maps]
    du2 = _geometry_poly.polyval(gl_nodes, derivatives[1])
    du3 = _geometry_poly.polyval(gl_nodes, derivatives[2])
    surface = du2[:, None] * du3[None, :]

    cells = np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    volume = np.empty((2, 3, 3, 3), dtype=np.float64)
    for cell_index, (lower, upper) in enumerate(cells):
        half_width = 0.5 * (upper - lower)
        midpoint = 0.5 * (upper + lower)
        global_s1 = midpoint + half_width * gl_nodes
        normal = half_width * _geometry_poly.polyval(global_s1, derivatives[0])
        volume[cell_index] = normal[:, None, None] * surface[None, :, :]

    result = np.empty((4, 2, 3, 3, 3), dtype=np.float64)
    result[0] = volume
    result[1] = np.broadcast_to(surface[None, None, :, :], (2, 3, 3, 3))
    result[2] = np.broadcast_to(du2[None, None, :, None], (2, 3, 3, 3))
    result[3] = np.broadcast_to(du3[None, None, None, :], (2, 3, 3, 3))
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("mapped geometry must be finite and positive")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _geometry_poly

_GEOMETRY_REFERENCE_JSON = r'''{"id":"reference_mixed_force","map_coefficients_u1_u2_u3":[[0.0,1.0,0.25],[0.0,0.8,0.2],[0.0,1.1,0.25]],"electric_normal_coefficients":[0.1,0.24,-0.12],"magnetic_s2_coefficients":[0.45,-0.2,0.15],"magnetic_s3_coefficients":[0.35,0.25,-0.1],"q_left_terms":[[2.0,0,0,0,0],[0.18,1,0,0,0],[0.12,0,1,0,0],[-0.14,0,0,1,0],[0.1,0,0,0,1],[0.16,1,1,0,0],[0.11,0,1,1,0],[-0.09,0,1,0,1],[0.08,1,0,1,0],[0.07,1,0,0,1],[0.06,0,0,1,1],[0.1,0,2,2,0],[0.08,2,0,0,2],[0.05,1,1,0,1]],"q_right_terms":[[1.6,0,0,0,0],[-0.12,1,0,0,0],[-0.1,0,1,0,0],[0.16,0,0,1,0],[-0.08,0,0,0,1],[-0.14,1,1,0,0],[0.09,0,1,1,0],[0.12,0,1,0,1],[-0.07,1,0,1,0],[0.06,1,0,0,1],[-0.05,0,0,1,1],[0.07,0,2,0,2],[0.09,2,0,2,0],[-0.04,1,1,1,0]]}'''

_GEOMETRY_OFFSET_JSON = r'''{"id":"offset_narrow_maps","map_coefficients_u1_u2_u3":[[0.2,0.65,0.12],[-0.1,0.55,0.12],[0.15,0.75,0.18]],"electric_normal_coefficients":[-0.08,0.28,0.06],"magnetic_s2_coefficients":[0.25,0.14,0.04],"magnetic_s3_coefficients":[-0.12,0.31,-0.03],"q_left_terms":[[1.4,0,0,0,0],[0.15,1,0,0,0],[0.13,0,1,0,0],[-0.11,0,0,1,0],[0.09,0,0,0,1],[0.12,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[-0.05,1,0,0,1],[0.04,0,0,1,1],[0.08,0,2,2,0]],"q_right_terms":[[1.9,0,0,0,0],[-0.14,1,0,0,0],[0.11,0,1,0,0],[0.12,0,0,1,0],[-0.09,0,0,0,1],[-0.1,1,1,0,0],[0.08,0,1,1,0],[-0.06,0,1,0,1],[0.07,1,0,1,0],[0.05,1,0,0,1],[-0.04,0,0,1,1],[0.06,0,2,0,2]]}'''

_GEOMETRY_AFFINE_JSON = r'''{"id":"affine_map_edge_case","map_coefficients_u1_u2_u3":[[-0.15,1.05,0.0],[0.05,0.7,0.0],[-0.1,0.9,0.0]],"electric_normal_coefficients":[0.05,-0.1,0.15],"magnetic_s2_coefficients":[-0.3,0.2,0.1],"magnetic_s3_coefficients":[0.4,0.05,-0.12],"q_left_terms":[[2.2,0,0,0,0],[-0.12,1,0,0,0],[0.18,0,1,0,0],[0.14,0,0,1,0],[-0.1,0,0,0,1],[0.09,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[0.05,0,0,1,1]],"q_right_terms":[[1.1,0,0,0,0],[0.1,1,0,0,0],[-0.14,0,1,0,0],[-0.11,0,0,1,0],[0.13,0,0,0,1],[-0.08,1,1,0,0],[0.07,0,1,1,0],[-0.06,0,1,0,1],[0.05,1,0,0,1],[-0.04,0,0,1,1]]}'''

def test_cases():
    """Return normal, nonlinear-offset, affine-edge, and invalid cases."""
    cases = []
    cases.append({
        "setup": "case_json = _GEOMETRY_REFERENCE_JSON\n",
        "call": "mapped_velocity_geometry(case_json)",
        "gold_call": "_oracle_mapped_velocity_geometry(case_json)",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_OFFSET_JSON\n",
        "call": "mapped_velocity_geometry(case_json)",
        "gold_call": "_oracle_mapped_velocity_geometry(case_json)",
    })
    cases.append({
        "setup": "case_json = _GEOMETRY_AFFINE_JSON\n",
        "call": "mapped_velocity_geometry(case_json)",
        "gold_call": "_oracle_mapped_velocity_geometry(case_json)",
    })
    cases.append({
        "setup": (
            "case_json = '{broken json'\n"
            "def run_model():\n"
            "    try:\n"
            "        mapped_velocity_geometry(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_mapped_velocity_geometry(case_json)\n"
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
            "case['map_coefficients_u1_u2_u3'][1] = [0.0, 0.0, 0.0]\n"
            "case_json = json.dumps(case, separators=(',', ':'))\n"
            "def run_model():\n"
            "    try:\n"
            "        mapped_velocity_geometry(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_mapped_velocity_geometry(case_json)\n"
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
