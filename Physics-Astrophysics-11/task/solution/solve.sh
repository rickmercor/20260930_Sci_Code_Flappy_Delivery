#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

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
def mapped_velocity_geometry(case_json):
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
def discrete_face_hamiltonian_gradient(case_json):
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
def physical_distribution_nodal_state(case_json):
    """Reference implementation of
    :func:`physical_distribution_nodal_state`."""
    _, (left_terms, right_terms) = _state_decode_case(case_json)
    geometry = mapped_velocity_geometry(case_json)
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
def distribution_face_traces(case_json):
    """Reference implementation of :func:`distribution_face_traces`."""
    _trace_decode_case(case_json)
    state = np.asarray(
        physical_distribution_nodal_state(case_json),
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
def mapped_normal_face_transport(case_json):
    """Reference implementation of :func:`mapped_normal_face_transport`."""
    _, fields = _transport_decode_case(case_json)
    geometry = np.asarray(
        mapped_velocity_geometry(case_json), dtype=np.float64
    )
    gradient = np.asarray(
        discrete_face_hamiltonian_gradient(case_json),
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
def paired_upwind_face_loss(case_json):
    """Reference implementation of :func:`paired_upwind_face_loss`."""
    _loss_decode_case(case_json)
    traces = np.asarray(
        distribution_face_traces(case_json), dtype=np.float64
    )
    transport = np.asarray(
        mapped_normal_face_transport(case_json), dtype=np.float64
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
def mapped_velocity_face_damping(case_json=None):
    """Reference implementation of :func:`mapped_velocity_face_damping`."""
    normalized = _damping_case_json(case_json)

    geometry = _damping_finite_array(
        mapped_velocity_geometry(normalized),
        (4, 2, 3, 3, 3),
        "mapped geometry",
    )
    gradient = _damping_finite_array(
        discrete_face_hamiltonian_gradient(normalized),
        (2, 3, 3),
        "Hamiltonian gradient",
    )
    state = _damping_finite_array(
        physical_distribution_nodal_state(normalized),
        (2, 3, 3, 3, 3),
        "physical distribution state",
    )
    traces = _damping_finite_array(
        distribution_face_traces(normalized),
        (2, 3, 3, 3),
        "face traces",
    )
    transport = _damping_finite_array(
        mapped_normal_face_transport(normalized),
        (3, 3, 3),
        "mapped face transport",
    )
    loss = paired_upwind_face_loss(normalized)
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
SCICODE_GOLD_EOF
