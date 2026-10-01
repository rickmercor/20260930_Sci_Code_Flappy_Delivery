"""
Transport-matched polynomial DG correction.

Section 3.6, Eqs. (24)-(28): set `D=1/(3*sigma_t)`, `beta_e=sum_m weights[m]*abs(omegas[m] dot n_e)/2`, `tau_SIP_e=C_tr*degree**2/(2*theta**2)*max_{K adjacent to e}(D/h_K)`, and `tau_e=max(epsilon*tau_SIP_e,beta_e)`. Assemble `a_local(u,v)=epsilon*sum_K(<D grad u,grad v>_K+<sigma_a u,v>_K) +sum_e tau_e<[u],[v]>_e-epsilon*sum_e<{D grad u dot n_e},[v]>_e-epsilon*sum_e<{D grad v dot n_e},[u]>_e`. On a physical boundary this is `tau_e<u,v>_e-(epsilon/2)<D grad u dot n_e,v>_e-(epsilon/2)<D grad v dot n_e,u>_e`. Use the total-degree polyhedral basis and exact-integral convention of the scientific background. The supported parameter domain requires an SPD result.

Returns
-------
A_local : np.ndarray, shape (N, N), float Symmetric local correction form; N=len(cells)*(degree+1)*(degree+2)*(degree+3)/6.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_transport_matched_correction(vertices: np.ndarray, cells: list[list[list[int]]], degree: int, epsilon: float, sigma_t: float, sigma_a: float, C_tr: float, theta: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    '''Transport-matched polynomial DG correction.

    Parameters
    ----------
    vertices : np.ndarray, shape (N_v, 3)
        Finite vertex coordinates of the guaranteed conforming convex polyhedral mesh.
    cells : list[list[list[int]]], nonempty
        Each cell is a list of its planar polygonal faces. A face is an outward
        counterclockwise vertex-index cycle with no repeated endpoint or index.
        Shared faces use identical indices in reverse cyclic order. Each cell
        has at least four faces and positive volume. Cells and faces are convex.
    degree : int
        Total polynomial degree, one of 1, 2, 3, common to every cell.
    epsilon : float
        Positive finite diffusive scale, less than sqrt(sigma_t/sigma_a).
    sigma_t, sigma_a : float
        Positive finite unscaled total and absorption coefficients.
    C_tr : float
        Positive finite penalty prefactor. The resulting local form must be SPD.
    theta : float
        Finite SIP split in the open interval (0, 1).
    omegas : np.ndarray, shape (N_omega, 3)
        Finite unit directions with the shared moments and equal-weight opposite pairs.
    weights : np.ndarray, shape (N_omega,)
        Positive finite angular weights, summing to one.

    Returns
    -------
    A_local : np.ndarray, shape (N, N), float
        Symmetric local correction form; N=len(cells)*(degree+1)*(degree+2)*(degree+3)/6.

    Raises
    ------
    ValueError
        If scalar ranges, geometry shapes or indices, angular moments or
        opposite pairing are invalid, or the local form is not positive definite.
    '''
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_transport_matched_correction(vertices: np.ndarray, cells: list[list[list[int]]], degree: int, epsilon: float, sigma_t: float, sigma_a: float, C_tr: float, theta: float, omegas: np.ndarray, weights: np.ndarray) -> np.ndarray:
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    imports = 'import numpy as np\n'
    mesh = 'vertices=np.array([[-0.0, 0.14790909090909096, 1.0], [-0.0, -0.0, 1.0], [0.36532786885245905, -0.0, 1.0], [0.36382520053475936, 0.09166276737967909, 1.0], [0.3480709342560553, 0.10993771626297587, 1.0], [-0.0, -0.0, -0.0], [-0.0, 0.8388181818181818, -0.0], [0.545655737704918, -0.0, -0.0], [0.5328575694237085, 0.7806882651537773, -0.0], [0.3754696582190189, 0.15557790177994812, 0.929615722441809], [1.0, 0.13061224489795908, 1.0], [1.0, -0.0, 1.0], [1.0, -0.0, -0.0], [1.0, 1.0, -0.0], [1.0, 1.0, 0.03181818181818184], [0.7481818181818182, 1.0, -0.0], [0.7556220607097051, 1.0, 0.015156049593843621], [-0.0, 1.0, -0.0], [-0.0, 1.0, 1.0], [0.4336538461538462, 1.0, 1.0], [1.0, 1.0, 1.0]]);cells=[[[0, 1, 2, 3, 4], [5, 1, 0, 6], [1, 5, 7, 2], [5, 6, 8, 7], [9, 3, 2, 7, 8], [6, 0, 4, 9, 8], [4, 3, 9]], [[10, 11, 12, 13, 14], [15, 16, 14, 13], [3, 2, 11, 10], [2, 7, 12, 11], [7, 8, 15, 13, 12], [7, 2, 3, 9, 8], [16, 15, 8, 9], [9, 3, 10, 14, 16]], [[17, 18, 19, 16, 15], [18, 0, 4, 19], [0, 18, 17, 6], [6, 17, 15, 8], [0, 6, 8, 9, 4], [8, 15, 16, 9], [4, 9, 16, 19]], [[10, 14, 20], [16, 19, 20, 14], [19, 4, 3, 10, 20], [3, 4, 9], [3, 9, 16, 14, 10], [9, 4, 19, 16]]]\n'
    angles = 'om=np.concatenate([np.eye(3),-np.eye(3),np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])/np.sqrt(3)]);wt=np.r_[np.full(6,1/12),np.full(8,1/16)]\n'
    parameters = 'p=2;eps=1/768;st=1.;sa=1.;ctr=96.;theta=.5\n'
    cube = 'vertices=np.array([[x,y,z] for x in (0.,1.) for y in (0.,1.) for z in (0.,1.)]);cells=[[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]]\n'
    domain_wrapper = 'def check(f):\n try:\n  f();return 0\n except ValueError:return 1\n except Exception:return 2\n'
    step_call = 'assemble_transport_matched_correction(vertices,cells,p,eps,st,sa,ctr,theta,om,wt)'
    call_1 = '#case:normal\n' + step_call
    call_2 = '_oracle_' + step_call
    return [
        {'setup': imports + mesh + angles + parameters, 'call': call_1, 'gold_call': call_2},
        {'setup': imports + mesh + angles + parameters + 'eps=1/8192\n', 'call': '#case:boundary\n' + step_call, 'gold_call': call_2},
        {'setup': imports + cube + angles + parameters + 'p=1;eps=1/128\n', 'call': '#case:edge\n' + step_call, 'gold_call': call_2},
        {'setup': imports + mesh + angles + parameters + 'p=3;eps=1/2048\n', 'call': call_1, 'gold_call': call_2},
        {'setup': imports + mesh + angles + parameters + 'eps=1/32;st=2.5;sa=.7\n', 'call': call_1, 'gold_call': call_2},
        {'setup': imports + mesh + angles + parameters + 'a=.23;b=.37;om=om@np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0.,0.,1.]])@np.array([[np.cos(b),0.,np.sin(b)],[0.,1.,0.],[-np.sin(b),0.,np.cos(b)]]);wt=np.full(14,1/14)\n', 'call': call_1, 'gold_call': call_2},
        {'setup': imports + cube + angles + parameters + 'p=0\n' + domain_wrapper, 'call': '#case:edge\ncheck(lambda:' + step_call + ')', 'gold_call': 'check(lambda:_oracle_' + step_call + ')'},
    ]
