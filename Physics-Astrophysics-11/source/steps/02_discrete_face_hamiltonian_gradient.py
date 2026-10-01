"""
Step 02: tangential gradient of the shared discrete Hamiltonian face.

For the fixed face at s1 = 0, the normalized Lorentz state is the unique tensor-Q2 polynomial matching sqrt(1 + u1^2 + u2^2 + u3^2) on the tensor Lobatto grid {-1, 0, 1}^3 in each cell.  Return its two tangential derivatives on the ascending tensor GL3 face grid.



The output has shape (2, 3, 3).  Layer 0 is d(gamma_h)/ds2 and layer 1 is d(gamma_h)/ds3; the final two axes are (s2, s3).

Returns
-------
numpy.ndarray: Float64 array of shape ``(2,3,3)``.  Layers contain the ``s2`` and ``s3`` derivatives at ascending GL3 ``(s2,s3)`` nodes.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _hamiltonian_poly

def discrete_face_hamiltonian_gradient(case_json):
    """Return the two tangential derivatives of the discrete face state.

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
        Float64 array of shape ``(2,3,3)``.  Layers contain the ``s2`` and
        ``s3`` derivatives at ascending GL3 ``(s2,s3)`` nodes.

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
from numpy.polynomial import polynomial as _hamiltonian_poly

def _hamiltonian_number(value):
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
    )
def _hamiltonian_decode_case(case_json):
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
    if any(not _hamiltonian_number(value) for row in raw_maps for value in row):
        raise ValueError("map coefficients must be finite numbers")
    maps = np.asarray(raw_maps, dtype=np.float64)

    for key in (
        "electric_normal_coefficients",
        "magnetic_s2_coefficients",
        "magnetic_s3_coefficients",
    ):
        values = case.get(key)
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{key} must contain three coefficients")
        if any(not _hamiltonian_number(value) for value in values):
            raise ValueError(f"{key} must contain finite numbers")

    for key in ("q_left_terms", "q_right_terms"):
        terms = case.get(key)
        if not isinstance(terms, list) or not terms:
            raise ValueError(f"{key} must be a nonempty term table")
        for term in terms:
            if not isinstance(term, list) or len(term) != 5:
                raise ValueError(f"{key} rows must have five entries")
            if not _hamiltonian_number(term[0]):
                raise ValueError(f"{key} coefficients must be finite")
            for power in term[1:]:
                if isinstance(power, bool) or not isinstance(power, int):
                    raise ValueError(f"{key} powers must be integers")
                if power < 0 or power > 2:
                    raise ValueError(f"{key} powers must lie in [0, 2]")

    endpoints = np.array([-1.0, 1.0], dtype=np.float64)
    for axis, coefficients in enumerate(maps):
        derivative = _hamiltonian_poly.polyder(coefficients)
        if np.any(_hamiltonian_poly.polyval(endpoints, derivative) <= 0.0):
            raise ValueError(f"velocity map {axis + 1} is not increasing")
    return case, maps
def _hamiltonian_lagrange(source, target, derivative=False):
    vandermonde = _hamiltonian_poly.polyvander(source, len(source) - 1)
    coefficients = np.linalg.inv(vandermonde)
    columns = []
    for basis_index in range(len(source)):
        polynomial = coefficients[:, basis_index]
        if derivative:
            polynomial = _hamiltonian_poly.polyder(polynomial)
        columns.append(_hamiltonian_poly.polyval(target, polynomial))
    return np.stack(columns, axis=1)
def _oracle_discrete_face_hamiltonian_gradient(case_json):
    """Reference implementation of
    :func:`discrete_face_hamiltonian_gradient`."""
    _, maps = _hamiltonian_decode_case(case_json)
    lobatto = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    gl_nodes = np.array(
        [-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)],
        dtype=np.float64,
    )

    u1_face = _hamiltonian_poly.polyval(0.0, maps[0])
    u2_lobatto = _hamiltonian_poly.polyval(lobatto, maps[1])
    u3_lobatto = _hamiltonian_poly.polyval(lobatto, maps[2])
    gamma_face = np.sqrt(
        1.0
        + u1_face * u1_face
        + u2_lobatto[:, None] ** 2
        + u3_lobatto[None, :] ** 2
    )

    interpolation = _hamiltonian_lagrange(lobatto, gl_nodes)
    derivative = _hamiltonian_lagrange(lobatto, gl_nodes, derivative=True)
    dgamma_ds2 = np.einsum(
        "ai,bj,ij->ab", derivative, interpolation, gamma_face
    )
    dgamma_ds3 = np.einsum(
        "ai,bj,ij->ab", interpolation, derivative, gamma_face
    )
    result = np.stack((dgamma_ds2, dgamma_ds3)).astype(np.float64, copy=False)
    if result.shape != (2, 3, 3) or not np.all(np.isfinite(result)):
        raise ValueError("discrete Hamiltonian gradient is not finite")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import json
import numpy as np
from numpy.polynomial import polynomial as _hamiltonian_poly

_HAMILTONIAN_REFERENCE_JSON = r'''{"id":"reference_mixed_force","map_coefficients_u1_u2_u3":[[0.0,1.0,0.25],[0.0,0.8,0.2],[0.0,1.1,0.25]],"electric_normal_coefficients":[0.1,0.24,-0.12],"magnetic_s2_coefficients":[0.45,-0.2,0.15],"magnetic_s3_coefficients":[0.35,0.25,-0.1],"q_left_terms":[[2.0,0,0,0,0],[0.18,1,0,0,0],[0.12,0,1,0,0],[-0.14,0,0,1,0],[0.1,0,0,0,1],[0.16,1,1,0,0],[0.11,0,1,1,0],[-0.09,0,1,0,1],[0.08,1,0,1,0],[0.07,1,0,0,1],[0.06,0,0,1,1],[0.1,0,2,2,0],[0.08,2,0,0,2],[0.05,1,1,0,1]],"q_right_terms":[[1.6,0,0,0,0],[-0.12,1,0,0,0],[-0.1,0,1,0,0],[0.16,0,0,1,0],[-0.08,0,0,0,1],[-0.14,1,1,0,0],[0.09,0,1,1,0],[0.12,0,1,0,1],[-0.07,1,0,1,0],[0.06,1,0,0,1],[-0.05,0,0,1,1],[0.07,0,2,0,2],[0.09,2,0,2,0],[-0.04,1,1,1,0]]}'''

_HAMILTONIAN_OFFSET_JSON = r'''{"id":"offset_narrow_maps","map_coefficients_u1_u2_u3":[[0.2,0.65,0.12],[-0.1,0.55,0.12],[0.15,0.75,0.18]],"electric_normal_coefficients":[-0.08,0.28,0.06],"magnetic_s2_coefficients":[0.25,0.14,0.04],"magnetic_s3_coefficients":[-0.12,0.31,-0.03],"q_left_terms":[[1.4,0,0,0,0],[0.15,1,0,0,0],[0.13,0,1,0,0],[-0.11,0,0,1,0],[0.09,0,0,0,1],[0.12,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[-0.05,1,0,0,1],[0.04,0,0,1,1],[0.08,0,2,2,0]],"q_right_terms":[[1.9,0,0,0,0],[-0.14,1,0,0,0],[0.11,0,1,0,0],[0.12,0,0,1,0],[-0.09,0,0,0,1],[-0.1,1,1,0,0],[0.08,0,1,1,0],[-0.06,0,1,0,1],[0.07,1,0,1,0],[0.05,1,0,0,1],[-0.04,0,0,1,1],[0.06,0,2,0,2]]}'''

_HAMILTONIAN_AFFINE_JSON = r'''{"id":"affine_map_edge_case","map_coefficients_u1_u2_u3":[[-0.15,1.05,0.0],[0.05,0.7,0.0],[-0.1,0.9,0.0]],"electric_normal_coefficients":[0.05,-0.1,0.15],"magnetic_s2_coefficients":[-0.3,0.2,0.1],"magnetic_s3_coefficients":[0.4,0.05,-0.12],"q_left_terms":[[2.2,0,0,0,0],[-0.12,1,0,0,0],[0.18,0,1,0,0],[0.14,0,0,1,0],[-0.1,0,0,0,1],[0.09,1,1,0,0],[-0.08,0,1,1,0],[0.07,0,1,0,1],[0.06,1,0,1,0],[0.05,0,0,1,1]],"q_right_terms":[[1.1,0,0,0,0],[0.1,1,0,0,0],[-0.14,0,1,0,0],[-0.11,0,0,1,0],[0.13,0,0,0,1],[-0.08,1,1,0,0],[0.07,0,1,1,0],[-0.06,0,1,0,1],[0.05,1,0,0,1],[-0.04,0,0,1,1]]}'''

def test_cases():
    """Return nonlinear, offset, affine, and invalid Hamiltonian cases."""
    cases = []
    cases.append({
        "setup": "case_json = _HAMILTONIAN_REFERENCE_JSON\n",
        "call": "discrete_face_hamiltonian_gradient(case_json)",
        "gold_call": "_oracle_discrete_face_hamiltonian_gradient(case_json)",
    })
    cases.append({
        "setup": "case_json = _HAMILTONIAN_OFFSET_JSON\n",
        "call": "discrete_face_hamiltonian_gradient(case_json)",
        "gold_call": "_oracle_discrete_face_hamiltonian_gradient(case_json)",
    })
    cases.append({
        "setup": "case_json = _HAMILTONIAN_AFFINE_JSON\n",
        "call": "discrete_face_hamiltonian_gradient(case_json)",
        "gold_call": "_oracle_discrete_face_hamiltonian_gradient(case_json)",
    })
    cases.append({
        "setup": (
            "case_json = '[]'\n"
            "def run_model():\n"
            "    try:\n"
            "        discrete_face_hamiltonian_gradient(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_discrete_face_hamiltonian_gradient(case_json)\n"
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
            "case = json.loads(_HAMILTONIAN_REFERENCE_JSON)\n"
            "case['map_coefficients_u1_u2_u3'][2] = [0.0, 0.1, 0.2]\n"
            "case_json = json.dumps(case, separators=(',', ':'))\n"
            "def run_model():\n"
            "    try:\n"
            "        discrete_face_hamiltonian_gradient(case_json)\n"
            "        return 0\n"
            "    except ValueError:\n"
            "        return 1\n"
            "    except Exception:\n"
            "        return 2\n"
            "def run_gold():\n"
            "    try:\n"
            "        _oracle_discrete_face_hamiltonian_gradient(case_json)\n"
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
