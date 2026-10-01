#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def assemble_transport_matched_correction(vertices: np.ndarray, cells: list[list[list[int]]], degree: int, epsilon: float, sigma_t: float, sigma_a: float, C_tr: float, theta: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    import numpy as np

    def _real_array(value):
        """Convert real numeric entries without silently discarding imaginary parts."""
        from numbers import Number
        try:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                if any(not isinstance(x, Number) or x.imag != 0 for x in array.flat):
                    raise ValueError("real numeric entries required")
                array = np.array([x.real for x in array.flat]).reshape(array.shape)
            elif array.dtype.kind not in "buifc" or np.any(array.imag != 0):
                raise ValueError("real numeric entries required")
            if array.dtype.kind == "c":
                array = array.real.copy(order="K")
            return np.asarray(array.real, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("real numeric entries required") from exc

    def _build_polyhedral_mesh(vertices, cells, degree):
        vertices = _real_array(vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
            raise ValueError("finite three-dimensional vertices required")
        if (isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer))
                or degree not in (1, 2, 3)):
            raise ValueError("degree must be 1, 2 or 3")
        try:
            cells = [[list(face) for face in cell] for cell in cells]
        except (TypeError, ValueError) as exc:
            raise ValueError("cell and face sequences required") from exc
        if not cells:
            raise ValueError("nonempty cells required")
        # Use a referenced vertex; unused coordinate rows cannot set the working origin.
        try:
            anchor = cells[0][0][0]
            if (isinstance(anchor, (bool, np.bool_)) or not isinstance(anchor, (int, np.integer))
                    or not 0 <= anchor < len(vertices)):
                raise ValueError("face vertex indices")
            vertices = vertices - vertices[anchor]
        except (IndexError, TypeError) as exc:
            raise ValueError("cell and face sequences required") from exc
        degree = int(degree)
        powers = [(t - b - c, b, c) for t in range(degree + 1)
                  for c in range(t + 1) for b in range(t - c + 1)]
        elements, incidence = [], {}
        for k, cell in enumerate(cells):
            all_ids, local_faces = set(), []
            if len(cell) < 4:
                raise ValueError("closed three-dimensional polyhedron required")
            for ids in cell:
                ids = np.asarray(ids, dtype=object)
                if (ids.ndim != 1 or len(ids) < 3
                        or any(isinstance(index, (bool, np.bool_))
                               or not isinstance(index, (int, np.integer))
                               or index < 0 or index >= len(vertices) for index in ids)
                        or len(set(ids)) != len(ids)):
                    raise ValueError("face vertex indices")
                ids = ids.astype(int)
                all_ids.update(ids.tolist())
                polygon = vertices[ids]
                relative = polygon - polygon[0]
                cross = np.sum(np.cross(relative, np.roll(relative, -1, axis=0)), axis=0)
                length = np.linalg.norm(cross)
                if length == 0:
                    raise ValueError("nondegenerate planar face required")
                normal = cross / length
                local_faces.append((polygon, normal))
                key = tuple(sorted(ids.tolist()))
                incidence.setdefault(key, []).append((k, polygon, normal))
            xyz = vertices[sorted(all_ids)]
            lower, upper = xyz.min(axis=0), xyz.max(axis=0)
            scale = (upper - lower) / 2
            centre = (upper + lower) / 2
            interior = xyz.mean(axis=0)
            diameter = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2).max()
            if np.min(scale) <= 0:
                raise ValueError("three-dimensional cells required")
            radius = min((polygon[0] - interior) @ normal for polygon, normal in local_faces)
            if radius <= 0:
                raise ValueError("outward faces of convex cells required")
            elements.append({"vertices": xyz, "centre": centre, "scale": scale,
                             "interior": interior, "diameter": diameter,
                             "radius": radius, "faces": local_faces})
        faces = []
        for rows in incidence.values():
            if len(rows) > 2:
                raise ValueError("nonmanifold face")
            kp, polygon, normal = rows[0]
            km = -1
            if len(rows) == 2:
                km, other, opposite = rows[1]
                if not np.allclose(normal, -opposite, atol=1e-10, rtol=0):
                    raise ValueError("shared outward normals must be opposite")
            faces.append((kp, km, polygon, normal))
        return elements, faces, powers

    def _evaluate_polynomial_basis(points, element, powers):
        z = (points - element["centre"]) / element["scale"]
        values = np.empty((len(z), len(powers)))
        gradient = np.zeros((len(z), len(powers), 3))
        for i, power in enumerate(powers):
            power = np.array(power)
            values[:, i] = np.prod(z ** power, axis=1)
            for d in range(3):
                if power[d]:
                    reduced = power.copy()
                    reduced[d] -= 1
                    gradient[:, i, d] = power[d] * np.prod(z ** reduced, axis=1) / element["scale"][d]
        return values, gradient

    def _gauss_unit_interval(degree):
        # p+2 Gauss points integrate the Duffy-transformed degree-2p products exactly.
        x, w = np.polynomial.legendre.leggauss(int(degree) + 2)
        return (x + 1) / 2, w / 2

    def _polygon_quadrature(polygon, degree):
        # A centroid fan partitions only this face integral, never the DG mesh.
        x, w = _gauss_unit_interval(degree)
        r, s = np.meshgrid(x, x, indexing="ij")
        # The collapsed triangle map contributes the Jacobian factor (1-r).
        wt = np.outer(w, w) * (1 - r)
        origin = polygon.mean(axis=0)
        points, weights = [], []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            points.append((origin + r[..., None] * (a - origin)
                           + ((1 - r) * s)[..., None] * (b - origin)).reshape(-1, 3))
            weights.append((wt * np.linalg.norm(np.cross(a - origin, b - origin))).ravel())
        return np.concatenate(points), np.concatenate(weights)

    def _polyhedron_quadrature(element, degree):
        # Oriented tetrahedral fans integrate one unbroken polyhedral polynomial space.
        x, w = _gauss_unit_interval(degree)
        r, s, t = np.meshgrid(x, x, x, indexing="ij")
        # The collapsed tetrahedron map contributes (1-r)^2*(1-s).
        wt = np.einsum("i,j,k->ijk", w, w, w) * (1 - r) ** 2 * (1 - s)
        origin = element["interior"]
        points, weights = [], []
        for polygon, normal in element["faces"]:
            face_centre = polygon.mean(axis=0)
            a = face_centre - origin
            for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
                b, c = first - origin, second - origin
                jac = np.dot(a, np.cross(b, c))
                if jac <= 0:
                    raise ValueError("positive oriented volume fan required")
                points.append((origin + r[..., None] * a + ((1 - r) * s)[..., None] * b
                               + ((1 - r) * (1 - s) * t)[..., None] * c).reshape(-1, 3))
                weights.append((wt * jac).ravel())
        return np.concatenate(points), np.concatenate(weights)

    def _assemble_volume_forms(elements, powers, degree):
        nb, size = len(powers), len(powers) * len(elements)
        mass, stiffness = np.zeros((size, size)), np.zeros((size, size))
        derivative = np.zeros((3, size, size))
        for k, element in enumerate(elements):
            points, wt = _polyhedron_quadrature(element, degree)
            values, gradients = _evaluate_polynomial_basis(points, element, powers)
            ix = slice(k * nb, (k + 1) * nb)
            mass[ix, ix] = values.T @ (wt[:, None] * values)
            stiffness[ix, ix] = np.einsum("qia,qja,q->ij", gradients, gradients, wt)
            for d in range(3):
                derivative[d, ix, ix] = values.T @ (wt[:, None] * gradients[:, :, d])
        return mass, stiffness, derivative

    def _assemble_face_traces(face, elements, powers, degree):
        # The missing minus cell on a vacuum face leaves the interior average at one half.
        kp, km, polygon, normal = face
        points, wt = _polygon_quadrature(polygon, degree)
        nb, size = len(powers), len(powers) * len(elements)
        jump = np.zeros((len(points), size))
        average = np.zeros_like(jump)
        grad_average = np.zeros_like(jump)
        for k, sign in ((kp, 1), (km, -1)):
            if k >= 0:
                values, gradient = _evaluate_polynomial_basis(points, elements[k], powers)
                ix = slice(k * nb, (k + 1) * nb)
                jump[:, ix], average[:, ix] = sign * values, values / 2
                grad_average[:, ix] = gradient @ normal / 2
        return jump, average, grad_average, wt

    def _require_positive(*values):
        normalized = []
        for value in values:
            scalar = _real_array(value)
            if (scalar.ndim != 0 or isinstance(np.asarray(value).item(), (bool, np.bool_))
                    or not np.isfinite(scalar) or scalar <= 0):
                raise ValueError("positive finite real scalars, excluding bool, required")
            # Preserve real scalar arithmetic; normalize real-valued complex/object storage.
            normalized.append(value if np.asarray(value).dtype.kind in "iuf" else float(scalar))
        return tuple(normalized)

    def _validate_angles(omegas, weights):
        om, wt = _real_array(omegas), _real_array(weights)
        if (om.ndim != 2 or om.shape[1] != 3 or len(om) == 0 or wt.shape != (len(om),)
                or not np.all(np.isfinite(om)) or not np.all(np.isfinite(wt)) or np.any(wt <= 0)):
            raise ValueError("three-dimensional positive angular quadrature required")
        close = lambda a, b: np.allclose(a, b, atol=1e-12, rtol=0)
        if (not close(np.sum(om * om, axis=1), 1) or not close(wt.sum(), 1)
                or not close(wt @ om, 0) or not close((om.T * wt) @ om, np.eye(3) / 3)):
            raise ValueError("unit directions and spherical moments required")
        todo, reverse = set(range(len(om))), np.empty(len(om), dtype=int)
        while todo:
            m = min(todo)
            todo.remove(m)
            matches = [j for j in sorted(todo) if close(om[j], -om[m]) and close(wt[j], wt[m])]
            if not matches:
                raise ValueError("equal-weight opposite pairs required")
            j = matches[0]
            todo.remove(j)
            reverse[m], reverse[j] = j, m
        return om, wt, reverse

    def _require_spd(matrix):
        a = _real_array(matrix)
        if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
                or not np.all(np.isfinite(a)) or not np.allclose(a, a.T, atol=1e-12, rtol=1e-10)):
            raise ValueError("finite symmetric positive-definite matrix required")
        try:
            np.linalg.cholesky(a)
        except np.linalg.LinAlgError as exc:
            raise ValueError("positive-definite matrix required") from exc
        return a

    def _local_correction(vertices, cells, degree, epsilon, sigma_t, sigma_a, C_tr, theta, omegas, weights):
        epsilon, sigma_t, sigma_a, C_tr, theta = _require_positive(epsilon, sigma_t, sigma_a, C_tr, theta)
        if theta >= 1 or epsilon >= np.sqrt(sigma_t / sigma_a):
            raise ValueError("positive scattering and theta below one required")
        om, wt, reverse = _validate_angles(omegas, weights)
        elements, faces, powers = _build_polyhedral_mesh(vertices, cells, degree)
        mass, stiffness, derivative = _assemble_volume_forms(elements, powers, degree)
        diffusion = 1 / (3 * sigma_t)
        result = epsilon * (diffusion * stiffness + sigma_a * mass)
        for face in faces:
            jump, average, grad_average, qw = _assemble_face_traces(face, elements, powers, degree)
            # beta is the angular transport floor on interior and vacuum faces.
            beta = np.dot(wt, abs(om @ face[3])) / 2
            inverse_h = max(1 / elements[k]["diameter"] for k in face[:2] if k >= 0)
            sip = C_tr * degree ** 2 * diffusion * inverse_h / (2 * theta ** 2)
            tau = max(epsilon * sip, beta)
            consistency = jump.T @ (qw[:, None] * grad_average)
            result += tau * (jump.T @ (qw[:, None] * jump))
            result -= epsilon * diffusion * (consistency + consistency.T)
        return _require_spd(result)

    return _local_correction(vertices, cells, degree, epsilon, sigma_t, sigma_a, C_tr, theta, omegas, weights)

import numpy as np

def assemble_directional_transport_data(vertices: np.ndarray, cells: list[list[list[int]]], degree: int, omega: np.ndarray, sigma_t: float, epsilon: float) -> np.ndarray:
    import numpy as np

    def _real_array(value):
        """Convert real numeric entries without silently discarding imaginary parts."""
        from numbers import Number
        try:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                if any(not isinstance(x, Number) or x.imag != 0 for x in array.flat):
                    raise ValueError("real numeric entries required")
                array = np.array([x.real for x in array.flat]).reshape(array.shape)
            elif array.dtype.kind not in "buifc" or np.any(array.imag != 0):
                raise ValueError("real numeric entries required")
            if array.dtype.kind == "c":
                array = array.real.copy(order="K")
            return np.asarray(array.real, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("real numeric entries required") from exc

    def _build_polyhedral_mesh(vertices, cells, degree):
        vertices = _real_array(vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
            raise ValueError("finite three-dimensional vertices required")
        if (isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer))
                or degree not in (1, 2, 3)):
            raise ValueError("degree must be 1, 2 or 3")
        try:
            cells = [[list(face) for face in cell] for cell in cells]
        except (TypeError, ValueError) as exc:
            raise ValueError("cell and face sequences required") from exc
        if not cells:
            raise ValueError("nonempty cells required")
        # Use a referenced vertex; unused coordinate rows cannot set the working origin.
        try:
            anchor = cells[0][0][0]
            if (isinstance(anchor, (bool, np.bool_)) or not isinstance(anchor, (int, np.integer))
                    or not 0 <= anchor < len(vertices)):
                raise ValueError("face vertex indices")
            vertices = vertices - vertices[anchor]
        except (IndexError, TypeError) as exc:
            raise ValueError("cell and face sequences required") from exc
        degree = int(degree)
        powers = [(t - b - c, b, c) for t in range(degree + 1)
                  for c in range(t + 1) for b in range(t - c + 1)]
        elements, incidence = [], {}
        for k, cell in enumerate(cells):
            all_ids, local_faces = set(), []
            if len(cell) < 4:
                raise ValueError("closed three-dimensional polyhedron required")
            for ids in cell:
                ids = np.asarray(ids, dtype=object)
                if (ids.ndim != 1 or len(ids) < 3
                        or any(isinstance(index, (bool, np.bool_))
                               or not isinstance(index, (int, np.integer))
                               or index < 0 or index >= len(vertices) for index in ids)
                        or len(set(ids)) != len(ids)):
                    raise ValueError("face vertex indices")
                ids = ids.astype(int)
                all_ids.update(ids.tolist())
                polygon = vertices[ids]
                relative = polygon - polygon[0]
                cross = np.sum(np.cross(relative, np.roll(relative, -1, axis=0)), axis=0)
                length = np.linalg.norm(cross)
                if length == 0:
                    raise ValueError("nondegenerate planar face required")
                normal = cross / length
                local_faces.append((polygon, normal))
                key = tuple(sorted(ids.tolist()))
                incidence.setdefault(key, []).append((k, polygon, normal))
            xyz = vertices[sorted(all_ids)]
            lower, upper = xyz.min(axis=0), xyz.max(axis=0)
            scale = (upper - lower) / 2
            centre = (upper + lower) / 2
            interior = xyz.mean(axis=0)
            diameter = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2).max()
            if np.min(scale) <= 0:
                raise ValueError("three-dimensional cells required")
            radius = min((polygon[0] - interior) @ normal for polygon, normal in local_faces)
            if radius <= 0:
                raise ValueError("outward faces of convex cells required")
            elements.append({"vertices": xyz, "centre": centre, "scale": scale,
                             "interior": interior, "diameter": diameter,
                             "radius": radius, "faces": local_faces})
        faces = []
        for rows in incidence.values():
            if len(rows) > 2:
                raise ValueError("nonmanifold face")
            kp, polygon, normal = rows[0]
            km = -1
            if len(rows) == 2:
                km, other, opposite = rows[1]
                if not np.allclose(normal, -opposite, atol=1e-10, rtol=0):
                    raise ValueError("shared outward normals must be opposite")
            faces.append((kp, km, polygon, normal))
        return elements, faces, powers

    def _evaluate_polynomial_basis(points, element, powers):
        z = (points - element["centre"]) / element["scale"]
        values = np.empty((len(z), len(powers)))
        gradient = np.zeros((len(z), len(powers), 3))
        for i, power in enumerate(powers):
            power = np.array(power)
            values[:, i] = np.prod(z ** power, axis=1)
            for d in range(3):
                if power[d]:
                    reduced = power.copy()
                    reduced[d] -= 1
                    gradient[:, i, d] = power[d] * np.prod(z ** reduced, axis=1) / element["scale"][d]
        return values, gradient

    def _gauss_unit_interval(degree):
        # p+2 Gauss points integrate the Duffy-transformed degree-2p products exactly.
        x, w = np.polynomial.legendre.leggauss(int(degree) + 2)
        return (x + 1) / 2, w / 2

    def _polygon_quadrature(polygon, degree):
        # A centroid fan partitions only this face integral, never the DG mesh.
        x, w = _gauss_unit_interval(degree)
        r, s = np.meshgrid(x, x, indexing="ij")
        # The collapsed triangle map contributes the Jacobian factor (1-r).
        wt = np.outer(w, w) * (1 - r)
        origin = polygon.mean(axis=0)
        points, weights = [], []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            points.append((origin + r[..., None] * (a - origin)
                           + ((1 - r) * s)[..., None] * (b - origin)).reshape(-1, 3))
            weights.append((wt * np.linalg.norm(np.cross(a - origin, b - origin))).ravel())
        return np.concatenate(points), np.concatenate(weights)

    def _polyhedron_quadrature(element, degree):
        # Oriented tetrahedral fans integrate one unbroken polyhedral polynomial space.
        x, w = _gauss_unit_interval(degree)
        r, s, t = np.meshgrid(x, x, x, indexing="ij")
        # The collapsed tetrahedron map contributes (1-r)^2*(1-s).
        wt = np.einsum("i,j,k->ijk", w, w, w) * (1 - r) ** 2 * (1 - s)
        origin = element["interior"]
        points, weights = [], []
        for polygon, normal in element["faces"]:
            face_centre = polygon.mean(axis=0)
            a = face_centre - origin
            for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
                b, c = first - origin, second - origin
                jac = np.dot(a, np.cross(b, c))
                if jac <= 0:
                    raise ValueError("positive oriented volume fan required")
                points.append((origin + r[..., None] * a + ((1 - r) * s)[..., None] * b
                               + ((1 - r) * (1 - s) * t)[..., None] * c).reshape(-1, 3))
                weights.append((wt * jac).ravel())
        return np.concatenate(points), np.concatenate(weights)

    def _assemble_volume_forms(elements, powers, degree):
        nb, size = len(powers), len(powers) * len(elements)
        mass, stiffness = np.zeros((size, size)), np.zeros((size, size))
        derivative = np.zeros((3, size, size))
        for k, element in enumerate(elements):
            points, wt = _polyhedron_quadrature(element, degree)
            values, gradients = _evaluate_polynomial_basis(points, element, powers)
            ix = slice(k * nb, (k + 1) * nb)
            mass[ix, ix] = values.T @ (wt[:, None] * values)
            stiffness[ix, ix] = np.einsum("qia,qja,q->ij", gradients, gradients, wt)
            for d in range(3):
                derivative[d, ix, ix] = values.T @ (wt[:, None] * gradients[:, :, d])
        return mass, stiffness, derivative

    def _assemble_face_traces(face, elements, powers, degree):
        # The missing minus cell on a vacuum face leaves the interior average at one half.
        kp, km, polygon, normal = face
        points, wt = _polygon_quadrature(polygon, degree)
        nb, size = len(powers), len(powers) * len(elements)
        jump = np.zeros((len(points), size))
        average = np.zeros_like(jump)
        grad_average = np.zeros_like(jump)
        for k, sign in ((kp, 1), (km, -1)):
            if k >= 0:
                values, gradient = _evaluate_polynomial_basis(points, elements[k], powers)
                ix = slice(k * nb, (k + 1) * nb)
                jump[:, ix], average[:, ix] = sign * values, values / 2
                grad_average[:, ix] = gradient @ normal / 2
        return jump, average, grad_average, wt

    def _require_positive(*values):
        normalized = []
        for value in values:
            scalar = _real_array(value)
            if (scalar.ndim != 0 or isinstance(np.asarray(value).item(), (bool, np.bool_))
                    or not np.isfinite(scalar) or scalar <= 0):
                raise ValueError("positive finite real scalars, excluding bool, required")
            # Preserve real scalar arithmetic; normalize real-valued complex/object storage.
            normalized.append(value if np.asarray(value).dtype.kind in "iuf" else float(scalar))
        return tuple(normalized)

    def _directional_data(vertices, cells, degree, omega, sigma_t, epsilon):
        sigma_t, epsilon = _require_positive(sigma_t, epsilon)
        omega = _real_array(omega)
        if (omega.shape != (3,) or not np.all(np.isfinite(omega))
                or not np.isclose(omega @ omega, 1, atol=1e-12, rtol=0)):
            raise ValueError("unit three-dimensional direction required")
        elements, faces, powers = _build_polyhedral_mesh(vertices, cells, degree)
        mass, stiffness, derivative = _assemble_volume_forms(elements, powers, degree)
        weak = np.einsum("a,aij->ij", omega, derivative)
        # Test rows and differentiated trial columns make the weak flux minus weak.T.
        full = (sigma_t / epsilon) * mass - weak.T
        for face in faces:
            jump, average, grad_average, qw = _assemble_face_traces(face, elements, powers, degree)
            speed = omega @ face[3]
            upwind = speed * average + abs(speed) * jump / 2
            full += jump.T @ (qw[:, None] * upwind)
        gradient = np.linalg.solve(sigma_t * mass, weak)
        return np.stack([mass, full, gradient])

    return _directional_data(vertices, cells, degree, omega, sigma_t, epsilon)

import numpy as np

def assemble_jump_lifting(vertices: np.ndarray, cells: list[list[list[int]]], degree: int, sigma_t: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    import numpy as np

    def _real_array(value):
        """Convert real numeric entries without silently discarding imaginary parts."""
        from numbers import Number
        try:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                if any(not isinstance(x, Number) or x.imag != 0 for x in array.flat):
                    raise ValueError("real numeric entries required")
                array = np.array([x.real for x in array.flat]).reshape(array.shape)
            elif array.dtype.kind not in "buifc" or np.any(array.imag != 0):
                raise ValueError("real numeric entries required")
            if array.dtype.kind == "c":
                array = array.real.copy(order="K")
            return np.asarray(array.real, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("real numeric entries required") from exc

    def _build_polyhedral_mesh(vertices, cells, degree):
        vertices = _real_array(vertices)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.all(np.isfinite(vertices)):
            raise ValueError("finite three-dimensional vertices required")
        if (isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer))
                or degree not in (1, 2, 3)):
            raise ValueError("degree must be 1, 2 or 3")
        try:
            cells = [[list(face) for face in cell] for cell in cells]
        except (TypeError, ValueError) as exc:
            raise ValueError("cell and face sequences required") from exc
        if not cells:
            raise ValueError("nonempty cells required")
        # Use a referenced vertex; unused coordinate rows cannot set the working origin.
        try:
            anchor = cells[0][0][0]
            if (isinstance(anchor, (bool, np.bool_)) or not isinstance(anchor, (int, np.integer))
                    or not 0 <= anchor < len(vertices)):
                raise ValueError("face vertex indices")
            vertices = vertices - vertices[anchor]
        except (IndexError, TypeError) as exc:
            raise ValueError("cell and face sequences required") from exc
        degree = int(degree)
        powers = [(t - b - c, b, c) for t in range(degree + 1)
                  for c in range(t + 1) for b in range(t - c + 1)]
        elements, incidence = [], {}
        for k, cell in enumerate(cells):
            all_ids, local_faces = set(), []
            if len(cell) < 4:
                raise ValueError("closed three-dimensional polyhedron required")
            for ids in cell:
                ids = np.asarray(ids, dtype=object)
                if (ids.ndim != 1 or len(ids) < 3
                        or any(isinstance(index, (bool, np.bool_))
                               or not isinstance(index, (int, np.integer))
                               or index < 0 or index >= len(vertices) for index in ids)
                        or len(set(ids)) != len(ids)):
                    raise ValueError("face vertex indices")
                ids = ids.astype(int)
                all_ids.update(ids.tolist())
                polygon = vertices[ids]
                relative = polygon - polygon[0]
                cross = np.sum(np.cross(relative, np.roll(relative, -1, axis=0)), axis=0)
                length = np.linalg.norm(cross)
                if length == 0:
                    raise ValueError("nondegenerate planar face required")
                normal = cross / length
                local_faces.append((polygon, normal))
                key = tuple(sorted(ids.tolist()))
                incidence.setdefault(key, []).append((k, polygon, normal))
            xyz = vertices[sorted(all_ids)]
            lower, upper = xyz.min(axis=0), xyz.max(axis=0)
            scale = (upper - lower) / 2
            centre = (upper + lower) / 2
            interior = xyz.mean(axis=0)
            diameter = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2).max()
            if np.min(scale) <= 0:
                raise ValueError("three-dimensional cells required")
            radius = min((polygon[0] - interior) @ normal for polygon, normal in local_faces)
            if radius <= 0:
                raise ValueError("outward faces of convex cells required")
            elements.append({"vertices": xyz, "centre": centre, "scale": scale,
                             "interior": interior, "diameter": diameter,
                             "radius": radius, "faces": local_faces})
        faces = []
        for rows in incidence.values():
            if len(rows) > 2:
                raise ValueError("nonmanifold face")
            kp, polygon, normal = rows[0]
            km = -1
            if len(rows) == 2:
                km, other, opposite = rows[1]
                if not np.allclose(normal, -opposite, atol=1e-10, rtol=0):
                    raise ValueError("shared outward normals must be opposite")
            faces.append((kp, km, polygon, normal))
        return elements, faces, powers

    def _evaluate_polynomial_basis(points, element, powers):
        z = (points - element["centre"]) / element["scale"]
        values = np.empty((len(z), len(powers)))
        gradient = np.zeros((len(z), len(powers), 3))
        for i, power in enumerate(powers):
            power = np.array(power)
            values[:, i] = np.prod(z ** power, axis=1)
            for d in range(3):
                if power[d]:
                    reduced = power.copy()
                    reduced[d] -= 1
                    gradient[:, i, d] = power[d] * np.prod(z ** reduced, axis=1) / element["scale"][d]
        return values, gradient

    def _gauss_unit_interval(degree):
        # p+2 Gauss points integrate the Duffy-transformed degree-2p products exactly.
        x, w = np.polynomial.legendre.leggauss(int(degree) + 2)
        return (x + 1) / 2, w / 2

    def _polygon_quadrature(polygon, degree):
        # A centroid fan partitions only this face integral, never the DG mesh.
        x, w = _gauss_unit_interval(degree)
        r, s = np.meshgrid(x, x, indexing="ij")
        # The collapsed triangle map contributes the Jacobian factor (1-r).
        wt = np.outer(w, w) * (1 - r)
        origin = polygon.mean(axis=0)
        points, weights = [], []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            points.append((origin + r[..., None] * (a - origin)
                           + ((1 - r) * s)[..., None] * (b - origin)).reshape(-1, 3))
            weights.append((wt * np.linalg.norm(np.cross(a - origin, b - origin))).ravel())
        return np.concatenate(points), np.concatenate(weights)

    def _polyhedron_quadrature(element, degree):
        # Oriented tetrahedral fans integrate one unbroken polyhedral polynomial space.
        x, w = _gauss_unit_interval(degree)
        r, s, t = np.meshgrid(x, x, x, indexing="ij")
        # The collapsed tetrahedron map contributes (1-r)^2*(1-s).
        wt = np.einsum("i,j,k->ijk", w, w, w) * (1 - r) ** 2 * (1 - s)
        origin = element["interior"]
        points, weights = [], []
        for polygon, normal in element["faces"]:
            face_centre = polygon.mean(axis=0)
            a = face_centre - origin
            for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
                b, c = first - origin, second - origin
                jac = np.dot(a, np.cross(b, c))
                if jac <= 0:
                    raise ValueError("positive oriented volume fan required")
                points.append((origin + r[..., None] * a + ((1 - r) * s)[..., None] * b
                               + ((1 - r) * (1 - s) * t)[..., None] * c).reshape(-1, 3))
                weights.append((wt * jac).ravel())
        return np.concatenate(points), np.concatenate(weights)

    def _assemble_volume_forms(elements, powers, degree):
        nb, size = len(powers), len(powers) * len(elements)
        mass, stiffness = np.zeros((size, size)), np.zeros((size, size))
        derivative = np.zeros((3, size, size))
        for k, element in enumerate(elements):
            points, wt = _polyhedron_quadrature(element, degree)
            values, gradients = _evaluate_polynomial_basis(points, element, powers)
            ix = slice(k * nb, (k + 1) * nb)
            mass[ix, ix] = values.T @ (wt[:, None] * values)
            stiffness[ix, ix] = np.einsum("qia,qja,q->ij", gradients, gradients, wt)
            for d in range(3):
                derivative[d, ix, ix] = values.T @ (wt[:, None] * gradients[:, :, d])
        return mass, stiffness, derivative

    def _assemble_face_traces(face, elements, powers, degree):
        # The missing minus cell on a vacuum face leaves the interior average at one half.
        kp, km, polygon, normal = face
        points, wt = _polygon_quadrature(polygon, degree)
        nb, size = len(powers), len(powers) * len(elements)
        jump = np.zeros((len(points), size))
        average = np.zeros_like(jump)
        grad_average = np.zeros_like(jump)
        for k, sign in ((kp, 1), (km, -1)):
            if k >= 0:
                values, gradient = _evaluate_polynomial_basis(points, elements[k], powers)
                ix = slice(k * nb, (k + 1) * nb)
                jump[:, ix], average[:, ix] = sign * values, values / 2
                grad_average[:, ix] = gradient @ normal / 2
        return jump, average, grad_average, wt

    def _require_positive(*values):
        normalized = []
        for value in values:
            scalar = _real_array(value)
            if (scalar.ndim != 0 or isinstance(np.asarray(value).item(), (bool, np.bool_))
                    or not np.isfinite(scalar) or scalar <= 0):
                raise ValueError("positive finite real scalars, excluding bool, required")
            # Preserve real scalar arithmetic; normalize real-valued complex/object storage.
            normalized.append(value if np.asarray(value).dtype.kind in "iuf" else float(scalar))
        return tuple(normalized)

    def _validate_angles(omegas, weights):
        om, wt = _real_array(omegas), _real_array(weights)
        if (om.ndim != 2 or om.shape[1] != 3 or len(om) == 0 or wt.shape != (len(om),)
                or not np.all(np.isfinite(om)) or not np.all(np.isfinite(wt)) or np.any(wt <= 0)):
            raise ValueError("three-dimensional positive angular quadrature required")
        close = lambda a, b: np.allclose(a, b, atol=1e-12, rtol=0)
        if (not close(np.sum(om * om, axis=1), 1) or not close(wt.sum(), 1)
                or not close(wt @ om, 0) or not close((om.T * wt) @ om, np.eye(3) / 3)):
            raise ValueError("unit directions and spherical moments required")
        todo, reverse = set(range(len(om))), np.empty(len(om), dtype=int)
        while todo:
            m = min(todo)
            todo.remove(m)
            matches = [j for j in sorted(todo) if close(om[j], -om[m]) and close(wt[j], wt[m])]
            if not matches:
                raise ValueError("equal-weight opposite pairs required")
            j = matches[0]
            todo.remove(j)
            reverse[m], reverse[j] = j, m
        return om, wt, reverse

    def _jump_lifting(vertices, cells, degree, sigma_t, omegas, weights):
        sigma_t, = _require_positive(sigma_t)
        om, wt, reverse = _validate_angles(omegas, weights)
        elements, faces, powers = _build_polyhedral_mesh(vertices, cells, degree)
        mass, stiffness, derivative = _assemble_volume_forms(elements, powers, degree)
        rhs = np.zeros((len(om),) + mass.shape)
        for face in faces:
            jump, average, grad_average, qw = _assemble_face_traces(face, elements, powers, degree)
            speeds = om @ face[3]
            beta = np.dot(wt, abs(speeds)) / 2
            for m, speed in enumerate(speeds):
                trace = speed * average + (beta - abs(speed) / 2) * jump
                # The trial jump is on the right; angular weight cancels in the Riesz equation.
                rhs[m] += trace.T @ (qw[:, None] * jump)
        return np.stack([np.linalg.solve(sigma_t * mass, row) for row in rhs])

    return _jump_lifting(vertices, cells, degree, sigma_t, omegas, weights)

import numpy as np

def solve_mean_zero_micro_action(data: np.ndarray, lifting: np.ndarray, epsilon: float, sigma_t: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    import numpy as np

    def _real_array(value):
        """Convert real numeric entries without silently discarding imaginary parts."""
        from numbers import Number
        try:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                if any(not isinstance(x, Number) or x.imag != 0 for x in array.flat):
                    raise ValueError("real numeric entries required")
                array = np.array([x.real for x in array.flat]).reshape(array.shape)
            elif array.dtype.kind not in "buifc" or np.any(array.imag != 0):
                raise ValueError("real numeric entries required")
            if array.dtype.kind == "c":
                array = array.real.copy(order="K")
            return np.asarray(array.real, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("real numeric entries required") from exc

    def _require_positive(*values):
        normalized = []
        for value in values:
            scalar = _real_array(value)
            if (scalar.ndim != 0 or isinstance(np.asarray(value).item(), (bool, np.bool_))
                    or not np.isfinite(scalar) or scalar <= 0):
                raise ValueError("positive finite real scalars, excluding bool, required")
            # Preserve real scalar arithmetic; normalize real-valued complex/object storage.
            normalized.append(value if np.asarray(value).dtype.kind in "iuf" else float(scalar))
        return tuple(normalized)

    def _validate_angles(omegas, weights):
        om, wt = _real_array(omegas), _real_array(weights)
        if (om.ndim != 2 or om.shape[1] != 3 or len(om) == 0 or wt.shape != (len(om),)
                or not np.all(np.isfinite(om)) or not np.all(np.isfinite(wt)) or np.any(wt <= 0)):
            raise ValueError("three-dimensional positive angular quadrature required")
        close = lambda a, b: np.allclose(a, b, atol=1e-12, rtol=0)
        if (not close(np.sum(om * om, axis=1), 1) or not close(wt.sum(), 1)
                or not close(wt @ om, 0) or not close((om.T * wt) @ om, np.eye(3) / 3)):
            raise ValueError("unit directions and spherical moments required")
        todo, reverse = set(range(len(om))), np.empty(len(om), dtype=int)
        while todo:
            m = min(todo)
            todo.remove(m)
            matches = [j for j in sorted(todo) if close(om[j], -om[m]) and close(wt[j], wt[m])]
            if not matches:
                raise ValueError("equal-weight opposite pairs required")
            j = matches[0]
            todo.remove(j)
            reverse[m], reverse[j] = j, m
        return om, wt, reverse

    def _require_spd(matrix):
        a = _real_array(matrix)
        if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
                or not np.all(np.isfinite(a)) or not np.allclose(a, a.T, atol=1e-12, rtol=1e-10)):
            raise ValueError("finite symmetric positive-definite matrix required")
        try:
            np.linalg.cholesky(a)
        except np.linalg.LinAlgError as exc:
            raise ValueError("positive-definite matrix required") from exc
        return a

    def _validate_directional_data(data, weights):
        data, wt = _real_array(data), _real_array(weights)
        if (data.ndim != 4 or data.shape[1] != 3 or data.shape[2] != data.shape[3]
                or data.shape[2] == 0 or len(data) == 0 or wt.shape != (len(data),)
                or not np.all(np.isfinite(data)) or not np.all(np.isfinite(wt))
                or np.any(wt <= 0) or not np.isclose(wt.sum(), 1, atol=1e-12, rtol=0)):
            raise ValueError("finite directional data and normalized positive weights required")
        mass = _require_spd(data[0, 0])
        if not np.allclose(data[:, 0], mass, atol=1e-12, rtol=0):
            raise ValueError("common mass matrix required")
        return data, wt

    def _micro_inputs(data, lifting, epsilon, sigma_t, omegas, weights):
        """Validate the common arrays and form T, S_m and K_m without solving for Y."""
        epsilon, sigma_t = _require_positive(epsilon, sigma_t)
        data, weights = _validate_directional_data(data, weights)
        omegas, weights, reverse = _validate_angles(omegas, weights)
        count, size = len(data), data.shape[2]
        if len(omegas) != count:
            raise ValueError("matching angular count required")
        lifting = _real_array(lifting)
        if lifting.shape != (count, size, size) or not np.all(np.isfinite(lifting)):
            raise ValueError("matching finite lifting maps required")
        mass, transports, gradients = data[0, 0], data[:, 1], data[:, 2]
        try:
            for transport in transports:
                np.linalg.solve(transport, np.eye(size))
        except np.linalg.LinAlgError as exc:
            raise ValueError("nonsingular directional forms required") from exc
        collision = sigma_t * mass
        streaming = np.stack([
            np.linalg.solve(collision, transport - collision / epsilon)
            for transport in transports
        ])
        coupling = gradients - lifting
        return mass, collision, streaming, coupling, weights, reverse

    def _solve_micro_action(data, lifting, epsilon, sigma_t, omegas, weights):
        epsilon, sigma_t = _require_positive(epsilon, sigma_t)
        mass, collision, streaming, coupling, weights, reverse = _micro_inputs(
            data, lifting, epsilon, sigma_t, omegas, weights
        )
        count, size = coupling.shape[:2]
        streaming_block = np.zeros((count * size, count * size))
        for m in range(count):
            block = slice(m * size, (m + 1) * size)
            streaming_block[block, block] = streaming[m]
        # Q projects out the weighted angular mean; it is never inverted.
        mean_zero = np.kron(
            np.eye(count) - np.ones((count, 1)) * weights, np.eye(size)
        )
        resolvent = np.eye(count * size) + epsilon * mean_zero @ streaming_block @ mean_zero
        try:
            solved = np.linalg.solve(resolvent, coupling.reshape(count * size, size))
        except np.linalg.LinAlgError as exc:
            raise ValueError("nonsingular micro block required") from exc
        return solved.reshape(count, size, size)

    return _solve_micro_action(data, lifting, epsilon, sigma_t, omegas, weights)

import numpy as np

def assemble_exact_scalar_correction(data: np.ndarray, lifting: np.ndarray, micro_action: np.ndarray, epsilon: float, sigma_t: float, sigma_a: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    import numpy as np

    def _real_array(value):
        """Convert real numeric entries without silently discarding imaginary parts."""
        from numbers import Number
        try:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                if any(not isinstance(x, Number) or x.imag != 0 for x in array.flat):
                    raise ValueError("real numeric entries required")
                array = np.array([x.real for x in array.flat]).reshape(array.shape)
            elif array.dtype.kind not in "buifc" or np.any(array.imag != 0):
                raise ValueError("real numeric entries required")
            if array.dtype.kind == "c":
                array = array.real.copy(order="K")
            return np.asarray(array.real, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("real numeric entries required") from exc

    def _require_positive(*values):
        normalized = []
        for value in values:
            scalar = _real_array(value)
            if (scalar.ndim != 0 or isinstance(np.asarray(value).item(), (bool, np.bool_))
                    or not np.isfinite(scalar) or scalar <= 0):
                raise ValueError("positive finite real scalars, excluding bool, required")
            # Preserve real scalar arithmetic; normalize real-valued complex/object storage.
            normalized.append(value if np.asarray(value).dtype.kind in "iuf" else float(scalar))
        return tuple(normalized)

    def _validate_angles(omegas, weights):
        om, wt = _real_array(omegas), _real_array(weights)
        if (om.ndim != 2 or om.shape[1] != 3 or len(om) == 0 or wt.shape != (len(om),)
                or not np.all(np.isfinite(om)) or not np.all(np.isfinite(wt)) or np.any(wt <= 0)):
            raise ValueError("three-dimensional positive angular quadrature required")
        close = lambda a, b: np.allclose(a, b, atol=1e-12, rtol=0)
        if (not close(np.sum(om * om, axis=1), 1) or not close(wt.sum(), 1)
                or not close(wt @ om, 0) or not close((om.T * wt) @ om, np.eye(3) / 3)):
            raise ValueError("unit directions and spherical moments required")
        todo, reverse = set(range(len(om))), np.empty(len(om), dtype=int)
        while todo:
            m = min(todo)
            todo.remove(m)
            matches = [j for j in sorted(todo) if close(om[j], -om[m]) and close(wt[j], wt[m])]
            if not matches:
                raise ValueError("equal-weight opposite pairs required")
            j = matches[0]
            todo.remove(j)
            reverse[m], reverse[j] = j, m
        return om, wt, reverse

    def _require_spd(matrix):
        a = _real_array(matrix)
        if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
                or not np.all(np.isfinite(a)) or not np.allclose(a, a.T, atol=1e-12, rtol=1e-10)):
            raise ValueError("finite symmetric positive-definite matrix required")
        try:
            np.linalg.cholesky(a)
        except np.linalg.LinAlgError as exc:
            raise ValueError("positive-definite matrix required") from exc
        return a

    def _validate_directional_data(data, weights):
        data, wt = _real_array(data), _real_array(weights)
        if (data.ndim != 4 or data.shape[1] != 3 or data.shape[2] != data.shape[3]
                or data.shape[2] == 0 or len(data) == 0 or wt.shape != (len(data),)
                or not np.all(np.isfinite(data)) or not np.all(np.isfinite(wt))
                or np.any(wt <= 0) or not np.isclose(wt.sum(), 1, atol=1e-12, rtol=0)):
            raise ValueError("finite directional data and normalized positive weights required")
        mass = _require_spd(data[0, 0])
        if not np.allclose(data[:, 0], mass, atol=1e-12, rtol=0):
            raise ValueError("common mass matrix required")
        return data, wt

    def _micro_inputs(data, lifting, epsilon, sigma_t, omegas, weights):
        """Validate the common arrays and form T, S_m and K_m without solving for Y."""
        epsilon, sigma_t = _require_positive(epsilon, sigma_t)
        data, weights = _validate_directional_data(data, weights)
        omegas, weights, reverse = _validate_angles(omegas, weights)
        count, size = len(data), data.shape[2]
        if len(omegas) != count:
            raise ValueError("matching angular count required")
        lifting = _real_array(lifting)
        if lifting.shape != (count, size, size) or not np.all(np.isfinite(lifting)):
            raise ValueError("matching finite lifting maps required")
        mass, transports, gradients = data[0, 0], data[:, 1], data[:, 2]
        try:
            for transport in transports:
                np.linalg.solve(transport, np.eye(size))
        except np.linalg.LinAlgError as exc:
            raise ValueError("nonsingular directional forms required") from exc
        collision = sigma_t * mass
        streaming = np.stack([
            np.linalg.solve(collision, transport - collision / epsilon)
            for transport in transports
        ])
        coupling = gradients - lifting
        return mass, collision, streaming, coupling, weights, reverse

    def _exact_correction(data, lifting, micro_action, epsilon, sigma_t, sigma_a, omegas, weights):
        epsilon, sigma_t, sigma_a = _require_positive(epsilon, sigma_t, sigma_a)
        if epsilon >= np.sqrt(sigma_t / sigma_a):
            raise ValueError("positive scattering required")
        mass, collision, streaming, coupling, weights, reverse = _micro_inputs(
            data, lifting, epsilon, sigma_t, omegas, weights
        )
        micro_action = _real_array(micro_action)
        if micro_action.shape != coupling.shape or not np.all(np.isfinite(micro_action)):
            raise ValueError("matching finite micro-action blocks required")
        exact = collision @ np.einsum("m,mij->ij", weights, streaming) + epsilon * sigma_a * mass
        # Reverse the solved action, not K before the resolvent. Supplied Y enters this product.
        exact -= epsilon * sum(
            weights[m] * coupling[m].T @ collision @ micro_action[reverse[m]]
            for m in range(len(weights))
        )
        return exact

    return _exact_correction(data, lifting, micro_action, epsilon, sigma_t, sigma_a, omegas, weights)

import numpy as np

def relative_correction_discrepancy(exact: np.ndarray, local: np.ndarray) -> float:
    import numpy as np

    def _real_array(value):
        """Convert real numeric entries without silently discarding imaginary parts."""
        from numbers import Number
        try:
            array = np.asarray(value)
            if array.dtype.kind == "O":
                if any(not isinstance(x, Number) or x.imag != 0 for x in array.flat):
                    raise ValueError("real numeric entries required")
                array = np.array([x.real for x in array.flat]).reshape(array.shape)
            elif array.dtype.kind not in "buifc" or np.any(array.imag != 0):
                raise ValueError("real numeric entries required")
            if array.dtype.kind == "c":
                array = array.real.copy(order="K")
            return np.asarray(array.real, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("real numeric entries required") from exc

    def _require_spd(matrix):
        a = _real_array(matrix)
        if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
                or not np.all(np.isfinite(a)) or not np.allclose(a, a.T, atol=1e-12, rtol=1e-10)):
            raise ValueError("finite symmetric positive-definite matrix required")
        try:
            np.linalg.cholesky(a)
        except np.linalg.LinAlgError as exc:
            raise ValueError("positive-definite matrix required") from exc
        return a

    def _relative_discrepancy(exact, local):
        exact, local = _require_spd(exact), _require_spd(local)
        if exact.shape != local.shape:
            raise ValueError("matching shapes required")
        chol = np.linalg.cholesky(local)
        whitened = np.linalg.solve(chol, exact - local)
        whitened = np.linalg.solve(chol, whitened.T).T
        return float(max(abs(np.linalg.eigvalsh((whitened + whitened.T) / 2))))

    return _relative_discrepancy(exact, local)

import numpy as np

def vacuum_matched_dsa_probe(vertices: np.ndarray | None = None, cells: list[list[list[int]]] | None = None, degree: int = 2, epsilon: float = 1/768, sigma_t: float = 1.0, sigma_a: float = 1.0, C_tr: float = 96.0, theta: float = 0.5, omegas: np.ndarray | None = None, weights: np.ndarray | None = None) -> float:
    import numpy as np
    if vertices is None and cells is None:
        vertices = np.asarray([[-0.0, 0.14790909090909096, 1.0], [-0.0, -0.0, 1.0], [0.36532786885245905, -0.0, 1.0], [0.36382520053475936, 0.09166276737967909, 1.0], [0.3480709342560553, 0.10993771626297587, 1.0], [-0.0, -0.0, -0.0], [-0.0, 0.8388181818181818, -0.0], [0.545655737704918, -0.0, -0.0], [0.5328575694237085, 0.7806882651537773, -0.0], [0.3754696582190189, 0.15557790177994812, 0.929615722441809], [1.0, 0.13061224489795908, 1.0], [1.0, -0.0, 1.0], [1.0, -0.0, -0.0], [1.0, 1.0, -0.0], [1.0, 1.0, 0.03181818181818184], [0.7481818181818182, 1.0, -0.0], [0.7556220607097051, 1.0, 0.015156049593843621], [-0.0, 1.0, -0.0], [-0.0, 1.0, 1.0], [0.4336538461538462, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=float)
        cells = [[[0, 1, 2, 3, 4], [5, 1, 0, 6], [1, 5, 7, 2], [5, 6, 8, 7], [9, 3, 2, 7, 8], [6, 0, 4, 9, 8], [4, 3, 9]], [[10, 11, 12, 13, 14], [15, 16, 14, 13], [3, 2, 11, 10], [2, 7, 12, 11], [7, 8, 15, 13, 12], [7, 2, 3, 9, 8], [16, 15, 8, 9], [9, 3, 10, 14, 16]], [[17, 18, 19, 16, 15], [18, 0, 4, 19], [0, 18, 17, 6], [6, 17, 15, 8], [0, 6, 8, 9, 4], [8, 15, 16, 9], [4, 9, 16, 19]], [[10, 14, 20], [16, 19, 20, 14], [19, 4, 3, 10, 20], [3, 4, 9], [3, 9, 16, 14, 10], [9, 4, 19, 16]]]
    elif vertices is None or cells is None:
        raise ValueError("supply both geometry arguments")
    if omegas is None and weights is None:
        omegas = np.concatenate([np.eye(3), -np.eye(3),
            np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)]) / np.sqrt(3)])
        weights = np.r_[np.full(6, 1/12), np.full(8, 1/16)]
    elif omegas is None or weights is None:
        raise ValueError("supply both angular arguments")
    local = assemble_transport_matched_correction(
        vertices, cells, degree, epsilon, sigma_t, sigma_a, C_tr, theta, omegas, weights)
    data = np.stack([assemble_directional_transport_data(
        vertices, cells, degree, direction, sigma_t, epsilon) for direction in omegas])
    lifting = assemble_jump_lifting(vertices, cells, degree, sigma_t, omegas, weights)
    micro_action = solve_mean_zero_micro_action(data, lifting, epsilon, sigma_t, omegas, weights)
    exact = assemble_exact_scalar_correction(
        data, lifting, micro_action, epsilon, sigma_t, sigma_a, omegas, weights)
    delta = relative_correction_discrepancy(exact, local)
    return round(float(delta), 6)
SCICODE_GOLD_EOF
