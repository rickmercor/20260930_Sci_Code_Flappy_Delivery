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

def build_cloak_mesh(
    n_nodes: int,
    core_boundary: "np.ndarray",
    inner_boundary: "np.ndarray",
    outer_boundary: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Reference implementation (vectorised centroid classification)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _radius(coefficients, theta):
        order = np.arange(coefficients.shape[0], dtype=float)[:, None]
        return (coefficients[:, :1] * np.cos(order * theta)
                + coefficients[:, 1:] * np.sin(order * theta)).sum(axis=0)

    if not (_is_integer(n_nodes) and n_nodes >= 3):
        raise ValueError("n_nodes must be an integer of at least 3")
    curves = []
    for boundary in (core_boundary, inner_boundary, outer_boundary):
        coefficients = np.asarray(boundary, dtype=float)
        if (coefficients.ndim != 2 or coefficients.shape[1] != 2
                or coefficients.shape[0] == 0
                or not np.all(np.isfinite(coefficients))):
            raise ValueError("each boundary must be a non-empty finite (m, 2) array")
        curves.append(coefficients)

    n = int(n_nodes)
    axis = np.linspace(-1.0, 1.0, n)
    grid_x, grid_y = np.meshgrid(axis, axis)
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    col, row = np.meshgrid(np.arange(n - 1), np.arange(n - 1))
    corner = (row * n + col).ravel()
    triangles = np.empty((2 * corner.size, 3), dtype=int)
    triangles[0::2] = np.column_stack([corner, corner + 1, corner + n + 1])
    triangles[1::2] = np.column_stack([corner, corner + n + 1, corner + n])

    centroid = nodes[triangles].mean(axis=1)
    rho = np.hypot(centroid[:, 0], centroid[:, 1])
    theta = np.arctan2(centroid[:, 1], centroid[:, 0])
    r_core, r_inner, r_outer = (_radius(curve, theta) for curve in curves)
    if not (np.all(r_core > 0.0) and np.all(r_core < r_inner)
            and np.all(r_inner < r_outer)):
        raise ValueError("the boundaries must satisfy 0 < r_core < r_inner < r_outer")
    labels = np.zeros(triangles.shape[0], dtype=int)
    labels[rho < r_outer] = 1
    labels[rho < r_inner] = 2
    labels[rho < r_core] = 3
    return nodes, triangles, labels

import numpy as np

def solve_steady_conduction(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    conductivity: "np.ndarray",
    drive_angle: float,
) -> "np.ndarray":
    """Reference implementation (vectorised P1 assembly, sparse LU)."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(nodes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("nodes must have shape (n, 2) with n >= 3")
    if not np.all(np.isfinite(points)):
        raise ValueError("nodes must be finite")
    tri = np.asarray(triangles)
    if (tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] == 0
            or not np.issubdtype(tri.dtype, np.integer)):
        raise ValueError("triangles must be a non-empty integer (e, 3) array")
    if tri.min() < 0 or tri.max() >= points.shape[0]:
        raise ValueError("triangle node index out of range")
    kappa = np.asarray(conductivity, dtype=float)
    if kappa.shape != (tri.shape[0], 2, 2) or not np.all(np.isfinite(kappa)):
        raise ValueError("conductivity must be a finite (e, 2, 2) array")
    scale = np.maximum(1.0, np.abs(kappa).max(axis=(1, 2)))
    if np.any(np.abs(kappa[:, 0, 1] - kappa[:, 1, 0]) > 1e-12 * scale):
        raise ValueError("conductivity tensors must be symmetric")
    determinant = kappa[:, 0, 0] * kappa[:, 1, 1] - kappa[:, 0, 1] * kappa[:, 1, 0]
    if np.any(kappa[:, 0, 0] <= 0.0) or np.any(determinant <= 0.0):
        raise ValueError("conductivity tensors must be positive definite")
    if not _is_number(drive_angle):
        raise ValueError("drive_angle must be a finite number")

    corner = points[tri]
    edge1 = corner[:, 1] - corner[:, 0]
    edge2 = corner[:, 2] - corner[:, 0]
    jac = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    if np.any(jac == 0.0):
        raise ValueError("triangles must have non-zero area")
    grad = np.empty((tri.shape[0], 2, 3))
    grad[:, 0, 1] = edge2[:, 1] / jac
    grad[:, 1, 1] = -edge2[:, 0] / jac
    grad[:, 0, 2] = -edge1[:, 1] / jac
    grad[:, 1, 2] = edge1[:, 0] / jac
    grad[:, :, 0] = -grad[:, :, 1] - grad[:, :, 2]
    area = 0.5 * np.abs(jac)
    local = np.einsum("eai,eab,ebj->eij", grad, kappa, grad) * area[:, None, None]
    size = points.shape[0]
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, (1, 3)).ravel()
    stiffness = sp.csr_matrix((local.ravel(), (rows, cols)), shape=(size, size))

    on_boundary = ((np.abs(points[:, 0]) >= 1.0 - 1e-12)
                   | (np.abs(points[:, 1]) >= 1.0 - 1e-12))
    temperature = (np.cos(drive_angle) * points[:, 0]
                   + np.sin(drive_angle) * points[:, 1])
    free = ~on_boundary
    if np.any(free):
        block = stiffness[free][:, free].tocsc()
        rhs = -(stiffness[free][:, on_boundary] @ temperature[on_boundary])
        temperature[free] = spla.spsolve(block, rhs)
    return temperature

import numpy as np

def calibrate_compensation_shell(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    labels: "np.ndarray",
    inner_conductivity: float,
    background_conductivity: float,
    bracket: tuple[float, float],
    tolerance: float,
) -> float:
    """Reference implementation (bisection over the reference solver)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(nodes, dtype=float)
    tri = np.asarray(triangles)
    region = np.asarray(labels)
    if (region.ndim != 1 or tri.ndim != 2 or region.shape[0] != tri.shape[0]
            or not np.issubdtype(region.dtype, np.integer)
            or np.any((region < 0) | (region > 3))):
        raise ValueError("labels must be an (e,) integer array with values 0 to 3")
    k_inner, k_background = inner_conductivity, background_conductivity
    if not (_is_number(k_inner) and k_inner > 0.0
            and _is_number(k_background) and k_background > 0.0):
        raise ValueError("conductivities must be finite positive numbers")
    if len(bracket) != 2 or not all(_is_number(v) and v > 0.0 for v in bracket):
        raise ValueError("bracket must hold two finite positive numbers")
    lo, hi = float(bracket[0]), float(bracket[1])
    if not lo < hi:
        raise ValueError("bracket must satisfy lo < hi")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("nodes must have shape (n, 2)")

    on_boundary = ((np.abs(points[:, 0]) >= 1.0 - 1e-12)
                   | (np.abs(points[:, 1]) >= 1.0 - 1e-12))
    touched = np.zeros(points.shape[0], dtype=bool)
    touched[tri[region != 0].ravel()] = True
    exterior = ~touched & ~on_boundary
    if not np.any(exterior):
        raise ValueError("there is no exterior node")
    x_ext = points[exterior, 0]

    def _residual(k_outer):
        values = np.array([k_background, k_outer, k_inner, k_background])[region]
        tensors = values[:, None, None] * np.eye(2)
        temperature = solve_steady_conduction(points, tri, tensors, 0.0)
        return float(np.mean(x_ext * (temperature[exterior] - x_ext)))

    m_lo, m_hi = _residual(lo), _residual(hi)
    if not ((m_lo < 0.0 < m_hi) or (m_hi < 0.0 < m_lo)):
        raise ValueError("the residual does not change sign across the bracket")
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        m_mid = _residual(mid)
        if np.sign(m_mid) == np.sign(m_lo):
            lo, m_lo = mid, m_mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

import numpy as np

def compute_duality_jacobian(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    temperature: "np.ndarray",
    conductivity_values: "np.ndarray",
    background_conductivity: float,
    design_gradient: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation (closed form Lambda = lambda R S)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(nodes, dtype=float)
    tri = np.asarray(triangles)
    field = np.asarray(temperature, dtype=float)
    kappa_p = np.asarray(conductivity_values, dtype=float)
    g0 = np.asarray(design_gradient, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("nodes must be a finite (n, 2) array")
    if (tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] == 0
            or not np.issubdtype(tri.dtype, np.integer)
            or tri.min() < 0 or tri.max() >= points.shape[0]):
        raise ValueError("triangles must be a non-empty integer (e, 3) index array")
    if field.shape != (points.shape[0],) or not np.all(np.isfinite(field)):
        raise ValueError("temperature must be a finite (n,) array")
    if (kappa_p.shape != (tri.shape[0],) or not np.all(np.isfinite(kappa_p))
            or np.any(kappa_p <= 0.0)):
        raise ValueError("conductivity_values must be a positive finite (e,) array")
    if not (_is_number(background_conductivity) and background_conductivity > 0.0):
        raise ValueError("background_conductivity must be a finite positive number")
    if g0.shape != (2,) or not np.all(np.isfinite(g0)) or not np.any(g0 != 0.0):
        raise ValueError("design_gradient must be a finite non-zero (2,) array")

    corner = points[tri]
    edge1 = corner[:, 1] - corner[:, 0]
    edge2 = corner[:, 2] - corner[:, 0]
    jac = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    if np.any(jac == 0.0):
        raise ValueError("triangles must have non-zero area")
    rise1 = field[tri[:, 1]] - field[tri[:, 0]]
    rise2 = field[tri[:, 2]] - field[tri[:, 0]]
    grad = np.column_stack([(rise1 * edge2[:, 1] - rise2 * edge1[:, 1]) / jac,
                            (rise2 * edge1[:, 0] - rise1 * edge2[:, 0]) / jac])
    grad_sq = np.sum(grad * grad, axis=1)
    if np.any(grad_sq == 0.0):
        raise ValueError("the temperature gradient vanishes on some triangle")

    # Lambda = (|G0| / |G|^2) [G u^T + (kappa_0 / kappa_P) G_perp u_perp^T],
    # i.e. lambda R S with lambda = |G0|/|G|, R the rotation onto G and
    # S = diag(1, kappa_0 / kappa_P) in the frame of the design direction u.
    g0_norm = float(np.hypot(g0[0], g0[1]))
    unit = g0 / g0_norm
    unit_perp = np.array([-unit[1], unit[0]])
    grad_perp = np.column_stack([-grad[:, 1], grad[:, 0]])
    stretch = float(background_conductivity) / kappa_p
    jacobians = (np.einsum("ei,j->eij", grad, unit)
                 + stretch[:, None, None] * np.einsum("ei,j->eij", grad_perp, unit_perp))
    return jacobians * (g0_norm / grad_sq)[:, None, None]

import numpy as np

def realize_duality_laminate(
    jacobians: "np.ndarray",
    background_conductivity: float,
    high_fraction: "float | np.ndarray" = 0.5,
) -> "np.ndarray":
    """Reference implementation (closed-form principal axes of the transformation medium)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    jac = np.asarray(jacobians, dtype=float)
    if (jac.ndim != 3 or jac.shape[1:] != (2, 2) or jac.shape[0] == 0
            or not np.all(np.isfinite(jac))):
        raise ValueError("jacobians must be a non-empty finite (e, 2, 2) array")
    if not (_is_number(background_conductivity) and background_conductivity > 0.0):
        raise ValueError("background_conductivity must be a finite positive number")
    f = np.asarray(high_fraction, dtype=float)
    if f.ndim == 0:
        f = np.full(jac.shape[0], float(f))
    if (f.shape != (jac.shape[0],) or not np.all(np.isfinite(f))
            or np.any((f <= 0.0) | (f >= 1.0))):
        raise ValueError("high_fraction must be scalar or (e,) with 0 < f < 1")

    # The 2D transformation is independent of each Jacobian's scalar scale.
    scale = np.max(np.abs(jac), axis=(1, 2))
    if np.any(scale == 0.0):
        raise ValueError("every Jacobian must have a positive determinant")
    reduced = jac / scale[:, None, None]
    det = reduced[:, 0, 0] * reduced[:, 1, 1] - reduced[:, 0, 1] * reduced[:, 1, 0]
    if np.any(det <= 0.0):
        raise ValueError("every Jacobian must have a positive determinant")

    kappa_0 = float(background_conductivity)
    a11, a12 = reduced[:, 0, 0], reduced[:, 0, 1]
    a21, a22 = reduced[:, 1, 0], reduced[:, 1, 1]
    p = kappa_0 * (a11 * a11 + a12 * a12) / det
    r = kappa_0 * (a21 * a21 + a22 * a22) / det
    q = kappa_0 * (a11 * a21 + a12 * a22) / det
    centre = 0.5 * (p + r)
    radius = np.hypot(0.5 * (p - r), q)
    large = centre + radius
    # det(K_D) = kappa_0**2 avoids subtracting nearly equal eigenvalues.
    small = kappa_0 * (kappa_0 / large)
    theta = np.mod(0.5 * np.arctan2(2.0 * q, p - r), np.pi)
    theta = np.where(theta >= np.pi, theta - np.pi, theta)
    isotropic = 2.0 * radius <= 1e-9 * (2.0 * centre)
    theta = np.where(isotropic, 0.0, theta)

    # Solve the weighted arithmetic/harmonic matching problem. Normalize
    # by the larger eigenvalue and rationalize the low-material root.
    ratio = small / large
    gap = np.maximum(1.0 - ratio, 0.0)
    disc = np.sqrt(gap * (gap + 4.0 * f * (1.0 - f) * ratio))
    first = large * (1.0 - ratio * (1.0 - 2.0 * f) + disc) / (2.0 * f)
    second = small * (2.0 * (1.0 - f)) / (1.0 - ratio * (2.0 * f - 1.0) + disc)
    first = np.where(isotropic, centre, first)
    second = np.where(isotropic, centre, second)
    return np.column_stack([first, second, theta])

import numpy as np

def compute_worst_direction_disturbance(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    laminate: "np.ndarray",
    far_threshold: float,
    high_fraction: "float | np.ndarray" = 0.5,
) -> float:
    """Reference implementation (two drives and the largest eigenvalue)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(nodes, dtype=float)
    layers = np.asarray(laminate, dtype=float)
    tri = np.asarray(triangles)
    if (layers.ndim != 2 or layers.shape[1] != 3 or tri.ndim != 2
            or layers.shape[0] != tri.shape[0] or not np.all(np.isfinite(layers))):
        raise ValueError("laminate must be a finite (e, 3) array")
    a, b, theta = layers[:, 0], layers[:, 1], layers[:, 2]
    if np.any(b <= 0.0) or np.any(a < b):
        raise ValueError("every laminate row must satisfy a >= b > 0")
    if not (_is_number(far_threshold) and 0.0 <= far_threshold < 1.0):
        raise ValueError("far_threshold must be a finite number in [0, 1)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("nodes must have shape (n, 2)")

    f = np.asarray(high_fraction, dtype=float)
    if f.ndim == 0:
        f = np.full(layers.shape[0], float(f))
    if (f.shape != (layers.shape[0],) or not np.all(np.isfinite(f))
            or np.any((f <= 0.0) | (f >= 1.0))):
        raise ValueError("high_fraction must be scalar or (e,) with 0 < f < 1")
    along = f * a + (1.0 - f) * b
    across = 1.0 / (f / a + (1.0 - f) / b)
    c, s = np.cos(theta), np.sin(theta)
    tensors = np.empty((layers.shape[0], 2, 2))
    tensors[:, 0, 0] = along * c * c + across * s * s
    tensors[:, 1, 1] = along * s * s + across * c * c
    tensors[:, 0, 1] = (along - across) * c * s
    tensors[:, 1, 0] = tensors[:, 0, 1]

    inside = ((np.abs(points[:, 0]) < 1.0 - 1e-12)
              & (np.abs(points[:, 1]) < 1.0 - 1e-12))
    far = inside & (np.maximum(np.abs(points[:, 0]), np.abs(points[:, 1]))
                    > far_threshold)
    if not np.any(far):
        raise ValueError("there is no far node")
    # The problem is linear in the boundary data, so T_phi = cos(phi) T_0 + sin(phi) T_90
    # and D(phi)^2 is a quadratic form whose maximum is its largest eigenvalue.
    dev_x = solve_steady_conduction(points, tri, tensors, 0.0) - points[:, 0]
    dev_y = solve_steady_conduction(points, tri, tensors, 0.5 * np.pi) - points[:, 1]
    dx, dy = dev_x[far], dev_y[far]
    count = float(dx.size)
    g_xx, g_yy, g_xy = dx @ dx / count, dy @ dy / count, dx @ dy / count
    largest = 0.5 * (g_xx + g_yy) + np.hypot(0.5 * (g_xx - g_yy), g_xy)
    return float(np.sqrt(max(largest, 0.0)))

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

def _cloak_coordinate_jet(nodes, node_derivatives):
    jet = np.zeros((3, len(nodes), 2))
    jet[0] = nodes
    if node_derivatives is not None:
        motion = np.asarray(node_derivatives, float)
        if motion.shape != (2, len(nodes), 2) or not np.all(np.isfinite(motion)):
            raise ValueError('node derivatives must be finite (2,n,2)')
        boundary = np.max(np.abs(nodes), axis=1) >= 1-1e-12
        if np.any(motion[:, boundary] != 0):
            raise ValueError('boundary nodes must stay fixed')
        jet[1:] = motion
    return jet

def _cloak_geometry_jet(coordinates, triangles):
    p = coordinates[:, triangles]
    a, b = p[:, :, 1]-p[:, :, 0], p[:, :, 2]-p[:, :, 0]
    jac = np.stack([a, b], axis=-1)
    inv = np.empty_like(jac)
    inv[0] = np.linalg.inv(jac[0])
    inv[1] = -inv[0] @ jac[1] @ inv[0]
    inv[2] = 2*inv[0] @ jac[1] @ inv[0] @ jac[1] @ inv[0] - inv[0] @ jac[2] @ inv[0]
    reference = np.array([[-1., -1.], [1., 0.], [0., 1.]])
    basis = np.einsum('ia,reab->reib', reference, inv)
    def _cross(u, v):
        return u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
    determinant = np.stack([_cross(a[0], b[0]),
                            _cross(a[1], b[0])+_cross(a[0], b[1]),
                            _cross(a[2], b[0])+2*_cross(a[1], b[1])+_cross(a[0], b[2])])
    area = .5*np.sign(determinant[0])*determinant
    return area, basis

def _cloak_matrix_jet(triangles, area, basis, tensors, size):
    def _term(i, j, k):
        return np.einsum('eia,eab,ejb->eij', basis[i], tensors[j], basis[k])
    local0 = _term(0, 0, 0)
    local1 = _term(1, 0, 0)+_term(0, 1, 0)+_term(0, 0, 1)
    local2 = (_term(2, 0, 0)+_term(0, 2, 0)+_term(0, 0, 2)
              +2*(_term(1, 1, 0)+_term(1, 0, 1)+_term(0, 1, 1)))
    local = [area[0, :, None, None]*local0,
             area[1, :, None, None]*local0+area[0, :, None, None]*local1,
             area[2, :, None, None]*local0+2*area[1, :, None, None]*local1
             +area[0, :, None, None]*local2]
    rows = np.broadcast_to(triangles[:, :, None], local0.shape).ravel()
    cols = np.broadcast_to(triangles[:, None, :], local0.shape).ravel()
    return [coo_matrix((v.ravel(), (rows, cols)), shape=(size, size)).tocsr() for v in local]

def _cloak_temperature_jet(nodes, triangles, tensor_jet, angle, node_derivatives=None):
    t0 = solve_steady_conduction(nodes, triangles, tensor_jet[0], angle)
    coordinates = _cloak_coordinate_jet(nodes, node_derivatives)
    area, basis = _cloak_geometry_jet(coordinates, triangles)
    mats = _cloak_matrix_jet(triangles, area, basis, tensor_jet, len(nodes))
    interior = np.flatnonzero(np.max(np.abs(nodes), axis=1) < 1-1e-12)
    factor = splu(mats[0][interior][:, interior].tocsc())
    jet = np.zeros((3, len(nodes)))
    jet[0] = t0
    jet[1, interior] = factor.solve(-(mats[1] @ t0)[interior])
    jet[2, interior] = factor.solve(-(mats[2] @ t0+2*mats[1] @ jet[1])[interior])
    return jet

def differentiate_compensation_design(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    labels: "np.ndarray",
    inner_conductivity: float,
    outer_conductivity: float,
    background_conductivity: float,
    node_derivatives: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray]":
    """Differentiate the moving-domain equilibrium and moment constraint."""
    x, tri, lab = np.asarray(nodes, float), np.asarray(triangles), np.asarray(labels)
    if (lab.shape != (len(tri),) or not np.issubdtype(lab.dtype, np.integer)
            or np.any((lab < 0) | (lab > 3))):
        raise ValueError('invalid region labels')
    vals = np.asarray([background_conductivity, outer_conductivity, inner_conductivity], float)
    if vals.shape != (3,) or not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        raise ValueError('conductivities must be finite positive scalars')
    k = np.array([vals[0], vals[1], vals[2], vals[0]])[lab]
    t0 = solve_steady_conduction(x, tri, k[:, None, None]*np.eye(2), 0.)
    coordinates = _cloak_coordinate_jet(x, node_derivatives)
    interior = np.flatnonzero(np.max(np.abs(x), axis=1) < 1-1e-12)
    touched = np.zeros(len(x), bool)
    touched[tri[lab != 0].ravel()] = True
    exterior = (~touched) & (np.max(np.abs(x), axis=1) < 1-1e-12)
    if not np.any(exterior):
        raise ValueError('there is no exterior node')
    area, basis = _cloak_geometry_jet(coordinates, tri)
    kfixed = np.stack([k, (lab == 2)*vals[2], (lab == 2)*vals[2]])
    mats = _cloak_matrix_jet(tri, area, basis, kfixed[:, :, None, None]*np.eye(2), len(x))
    outer = np.zeros_like(kfixed)
    outer[0] = lab == 1
    outer_mats = _cloak_matrix_jet(tri, area, basis, outer[:, :, None, None]*np.eye(2), len(x))
    factor = splu(mats[0][interior][:, interior].tocsc())
    def _response(rhs):
        z = np.zeros(len(x))
        z[interior] = factor.solve(rhs[interior])
        return z
    def _moment(z):
        return np.mean(x[exterior, 0]*z[exterior])
    tq = _response(-outer_mats[0] @ t0)
    slope = _moment(tq)
    if abs(slope) <= 1e-14:
        raise ValueError('singular compensation constraint')
    tf1 = _response(-mats[1] @ t0)
    xx, vx, wx = coordinates[:, exterior, 0]
    m1_fixed = np.mean(vx*(t0[exterior]-xx)+xx*(tf1[exterior]-vx))
    q1 = -m1_fixed/slope
    t1 = tf1+q1*tq
    a1 = mats[1]+q1*outer_mats[0]
    # The second total stiffness includes shape/material mixed derivatives.
    a2_fixed = mats[2]+2*q1*outer_mats[1]
    tf2 = _response(-a2_fixed @ t0-2*a1 @ t1)
    m2_fixed = np.mean(wx*(t0[exterior]-xx)+2*vx*(t1[exterior]-vx)
                       +xx*(tf2[exterior]-wx))
    q2 = -m2_fixed/slope
    t2 = tf2+q2*tq
    return np.array([outer_conductivity, q1, q2]), np.stack([t0, t1, t2])

import numpy as np

def differentiate_duality_jacobian(
    conductivity_jet: "np.ndarray",
    gradient_jet: "np.ndarray",
    background_conductivity: float,
) -> "np.ndarray":
    """Differentiate the scaled, stretched field-aligned coordinate map."""
    k = np.asarray(conductivity_jet, dtype=float)
    g = np.asarray(gradient_jet, dtype=float)
    if (k.ndim != 2 or k.shape[0] != 3 or k.shape[1] == 0
            or g.shape != (3,k.shape[1],2) or not np.all(np.isfinite(k))
            or not np.all(np.isfinite(g)) or np.any(k[0] <= 0)):
        raise ValueError('invalid material or physical-gradient jets')
    if (not np.isscalar(background_conductivity)
            or not np.isfinite(background_conductivity) or background_conductivity <= 0):
        raise ValueError('background must be finite and positive')
    s = np.sum(g[0]*g[0],axis=1)
    if np.any(s <= 1e-28):
        raise ValueError('generating gradient is zero')
    s1 = 2*np.sum(g[0]*g[1],axis=1)
    s2 = 2*np.sum(g[1]*g[1]+g[0]*g[2],axis=1)
    stretch = background_conductivity/k[0]
    stretch1 = -stretch*k[1]/k[0]
    stretch2 = stretch*(2*(k[1]/k[0])**2-k[2]/k[0])
    perp = np.stack([-g[:,:,1],g[:,:,0]],axis=2)
    f = np.empty((3,k.shape[1],2,2))
    f[:,:,:,0] = g
    f[0,:,:,1] = stretch[:,None]*perp[0]
    f[1,:,:,1] = stretch1[:,None]*perp[0]+stretch[:,None]*perp[1]
    f[2,:,:,1] = (stretch2[:,None]*perp[0]+2*stretch1[:,None]*perp[1]
                  +stretch[:,None]*perp[2])
    result = np.empty_like(f)
    result[0] = f[0]/s[:,None,None]
    result[1] = (f[1]-s1[:,None,None]*result[0])/s[:,None,None]
    result[2] = (f[2]-s2[:,None,None]*result[0]
                 -2*s1[:,None,None]*result[1])/s[:,None,None]
    return result

import numpy as np

def differentiate_duality_tensor(
    jacobian_jet: "np.ndarray",
    background_conductivity: float,
) -> "np.ndarray":
    """Differentiate the conductivity push-forward and its determinant."""
    j = np.asarray(jacobian_jet, dtype=float)
    if (j.ndim != 4 or j.shape[0] != 3 or j.shape[1] == 0
            or j.shape[2:] != (2,2) or not np.all(np.isfinite(j))):
        raise ValueError('invalid Jacobian jet')
    if (not np.isscalar(background_conductivity)
            or not np.isfinite(background_conductivity) or background_conductivity <= 0):
        raise ValueError('background must be finite and positive')
    # A fixed elementwise rescaling cancels from the two-dimensional law.
    scale = np.max(np.abs(j[0]),axis=(1,2))
    if np.any(scale == 0):
        raise ValueError('baseline Jacobian must preserve orientation')
    j = j/scale[None,:,None,None]
    a,b,c = j
    d = a[:,0,0]*a[:,1,1]-a[:,0,1]*a[:,1,0]
    if np.any(d <= 0):
        raise ValueError('baseline Jacobian must preserve orientation')
    d1 = (b[:,0,0]*a[:,1,1]+a[:,0,0]*b[:,1,1]
          -b[:,0,1]*a[:,1,0]-a[:,0,1]*b[:,1,0])
    d2 = (c[:,0,0]*a[:,1,1]+2*b[:,0,0]*b[:,1,1]+a[:,0,0]*c[:,1,1]
          -c[:,0,1]*a[:,1,0]-2*b[:,0,1]*b[:,1,0]-a[:,0,1]*c[:,1,0])
    def _product(left,right):
        return left @ right.transpose(0,2,1)
    h = _product(a,a)
    h1 = _product(b,a)+_product(a,b)
    h2 = _product(c,a)+2*_product(b,b)+_product(a,c)
    result = np.empty_like(j)
    result[0] = background_conductivity*h/d[:,None,None]
    result[1] = (background_conductivity*h1-d1[:,None,None]*result[0])/d[:,None,None]
    result[2] = (background_conductivity*h2-d2[:,None,None]*result[0]
                 -2*d1[:,None,None]*result[1])/d[:,None,None]
    return result

import numpy as np

def differentiate_duality_response(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    conductivity_jet: "np.ndarray",
    temperature_jet: "np.ndarray",
    background_conductivity: float,
    far_threshold: float,
    node_derivatives: "np.ndarray | None" = None,
) -> "np.ndarray":
    """Differentiate the gradient projector, FEM equilibria and simple eigenvalue."""
    x,tri=np.asarray(nodes,float),np.asarray(triangles)
    k,t=np.asarray(conductivity_jet,float),np.asarray(temperature_jet,float)
    if (k.shape!=(3,len(tri)) or t.shape!=(3,len(x))
            or not np.all(np.isfinite(k)) or not np.all(np.isfinite(t)) or np.any(k[0]<=0)):
        raise ValueError('invalid material or temperature jets')
    if not np.isfinite(background_conductivity) or background_conductivity<=0:
        raise ValueError('background must be finite and positive')
    if not np.isfinite(far_threshold) or not 0<=far_threshold<1:
        raise ValueError('far threshold must be in [0,1)')
    identity=np.eye(2)
    original=k[:,:,None,None]*identity
    # The first reference solve validates mesh geometry before its reuse.
    original_x=_cloak_temperature_jet(x,tri,original,0.,node_derivatives)
    coordinates=_cloak_coordinate_jet(x,node_derivatives)
    area,basis=_cloak_geometry_jet(coordinates,tri)
    g=np.empty((3,len(tri),2))
    g[0]=np.einsum('ej,eja->ea',t[0,tri],basis[0])
    g[1]=(np.einsum('ej,eja->ea',t[1,tri],basis[0])
          +np.einsum('ej,eja->ea',t[0,tri],basis[1]))
    g[2]=(np.einsum('ej,eja->ea',t[2,tri],basis[0])
          +2*np.einsum('ej,eja->ea',t[1,tri],basis[1])
          +np.einsum('ej,eja->ea',t[0,tri],basis[2]))
    jacjet=differentiate_duality_jacobian(k,g,background_conductivity)
    conv=differentiate_duality_tensor(jacjet,background_conductivity)
    radius=np.max(np.abs(x),axis=1)
    far=(radius<1-1e-12)&(radius>far_threshold)
    if not np.any(far):
        raise ValueError('there is no far node')
    result=[]
    for tensors,tx in [(original,original_x),(conv,None)]:
        if tx is None:
            tx=_cloak_temperature_jet(x,tri,tensors,0.,node_derivatives)
        ty=_cloak_temperature_jet(x,tri,tensors,np.pi/2,node_derivatives)
        e=np.stack([tx[:,far],ty[:,far]],axis=-1)
        e-=coordinates[:,far]
        c=e[0].T@e[0]/far.sum()
        c1=(e[1].T@e[0]+e[0].T@e[1])/far.sum()
        c2=(e[2].T@e[0]+2*e[1].T@e[1]+e[0].T@e[2])/far.sum()
        eig,vec=np.linalg.eigh(c)
        lam,gap=eig[-1],eig[-1]-eig[0]
        if lam<=1e-24 or gap<=1e-10*lam:
            raise ValueError('nonregular worst direction')
        v,w=vec[:,1],vec[:,0]
        l1=v@c1@v
        l2=v@c2@v+2*(w@c1@v)**2/gap
        d=np.sqrt(lam)
        result.append([l1/(2*d),l2/(2*d)-l1*l1/(4*d**3)])
    return np.array(result)

import numpy as np

def estimate_duality_curvature(
    n_nodes: int = 121,
    core_boundary: tuple = ((0.20, 0.0), (0.0, 0.0), (0.04, 0.0)),
    inner_boundary: tuple = ((0.30, 0.0), (0.0, 0.0), (0.06, 0.0), (0.0, 0.0), (0.015, 0.0)),
    outer_boundary: tuple = ((0.45, 0.0), (0.0, 0.0), (0.09, 0.0), (0.02, 0.0)),
    inner_conductivity: float = 0.1,
    background_conductivity: float = 1.0,
    bracket: tuple[float, float] = (1.0, 20.0),
    tolerance: float = 1e-10,
    far_threshold: float = 0.61,
    high_fraction: "float | np.ndarray" = 0.5,
    shape_strength: float = 1.0,
) -> float:
    """Compose the ten reference stages and differentiate the log ratio."""
    x,tri,lab=build_cloak_mesh(n_nodes,core_boundary,inner_boundary,outer_boundary)
    q=calibrate_compensation_shell(x,tri,lab,inner_conductivity,
                                          background_conductivity,bracket,tolerance)
    if not np.isscalar(shape_strength) or not np.isfinite(shape_strength):
        raise ValueError('shape strength must be finite')
    u,v=x.T
    envelope=(1-u*u)*(1-v*v)
    motion=shape_strength*np.stack([
        envelope[:,None]*np.column_stack([.18*u+.11*v,-.13*u+.16*v]),
        envelope[:,None]*np.column_stack([.09*np.sin(2*u+v),.07*np.cos(u-2*v)])])
    qjet,tjet=differentiate_compensation_design(x,tri,lab,inner_conductivity,
                                                       q,background_conductivity,motion)
    kjet=np.empty((3,len(tri)))
    kjet[0]=np.array([background_conductivity,q,inner_conductivity,background_conductivity])[lab]
    for order in [1,2]:
        kjet[order]=np.array([0.,qjet[order],inner_conductivity,0.])[lab]
    jac=compute_duality_jacobian(x,tri,tjet[0],kjet[0],background_conductivity,
                                        np.array([1.,0.]))
    layers=realize_duality_laminate(jac,background_conductivity,high_fraction)
    original=np.column_stack([kjet[0],kjet[0],np.zeros(len(tri))])
    d=np.array([compute_worst_direction_disturbance(x,tri,original,far_threshold),
                compute_worst_direction_disturbance(x,tri,layers,far_threshold,high_fraction)])
    derivatives=differentiate_duality_response(x,tri,kjet,tjet,
                                                       background_conductivity,far_threshold,motion)
    curvature=derivatives[:,1]/d-(derivatives[:,0]/d)**2
    return float(curvature[1]-curvature[0])
SCICODE_GOLD_EOF
