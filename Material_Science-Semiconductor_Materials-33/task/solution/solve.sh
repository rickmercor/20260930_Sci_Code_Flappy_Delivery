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


def _finite_array(value, dtype, name):
    """Convert numeric input without modifying it and reject invalid data."""
    try:
        if dtype is float and np.iscomplexobj(value):
            raise ValueError(f"{name} must be real")
        result = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_scalar(value, name, minimum=0.0):
    """Validate a finite real scalar with an inclusive lower bound."""
    result = _finite_array(value, float, name)
    if result.ndim != 0 or result < minimum:
        raise ValueError(f"{name} must be a scalar >= {minimum}")
    return float(result)


def _integer_scalar(value, name, minimum):
    """Validate an integer parameter, excluding booleans."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def solve_bands(
    k_points: "np.ndarray", lattice: "np.ndarray", model: "np.ndarray"
) -> "tuple[np.ndarray, np.ndarray]":
    """Reference Bloch eigensystem; arbitrary normalized band phases."""
    k = _finite_array(k_points, float, "k_points")
    a = _finite_array(lattice, float, "lattice")
    pars = _finite_array(model, float, "model")
    if k.ndim != 2 or k.shape[1] != 2 or len(k) < 1:
        raise ValueError("k_points must have shape (K, 2), K >= 1")
    if a.shape != (2,) or np.any(a <= 0):
        raise ValueError("lattice must contain two positive lengths")
    if pars.shape != (7,) or pars[0] <= abs(pars[1]) + abs(pars[2]):
        raise ValueError("model must satisfy the documented mass bound")
    x, y = (k * a).T
    m, bx, by, t0, tx, ty, lam = pars
    hz = m + bx * np.cos(x) + by * np.cos(y)
    off = t0 + tx * np.exp(-1j * x) + ty * np.exp(-1j * y)
    off += 1j * lam * np.cos(x - y)
    h = np.zeros((len(k), 2, 2), dtype=complex)
    h[:, 0, 0], h[:, 1, 1] = hz, -hz
    h[:, 0, 1], h[:, 1, 0] = off, off.conj()
    return np.linalg.eigh(h)

import numpy as np


def thickness_averages(
    magnitudes: "np.ndarray", heights: "np.ndarray", thickness: float
) -> "np.ndarray":
    """Stable closed forms of the two independently defined averages."""
    p = _finite_array(magnitudes, float, "magnitudes")
    z = _finite_array(heights, float, "heights")
    d = _finite_scalar(thickness, "thickness")
    if p.ndim != 1 or len(p) < 1 or np.any(p < 0):
        raise ValueError("magnitudes must be a nonempty nonnegative vector")
    if z.ndim != 1 or len(z) < 1 or np.any(np.abs(z) > d / 2):
        raise ValueError("heights must lie inside the slab")
    result = np.ones((len(p), len(z) + 1))
    if d > 0:
        x = p * d
        nonzero = p > 0
        result[nonzero, :-1] = (
            -np.expm1(-p[nonzero, None] * (d / 2 - z))
            - np.expm1(-p[nonzero, None] * (d / 2 + z))
        ) / x[nonzero, None]
        small = nonzero & (x < 1e-3)
        xs = x[small]
        result[small, -1] = (
            1 - xs / 3 + xs**2 / 12 - xs**3 / 60 + xs**4 / 360 - xs**5 / 2520
        )
        large = nonzero & ~small
        result[large, -1] = (
            2 * (np.expm1(-x[large]) + x[large]) / x[large] ** 2
        )
    return result

import numpy as np


def density_vertices(
    left_vectors: "np.ndarray",
    right_vectors: "np.ndarray",
    transfer: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    orbital_averages: "np.ndarray",
) -> "np.ndarray":
    """Point-orbital matrix elements in the stated Fourier convention."""
    left = _finite_array(left_vectors, complex, "left_vectors")
    right = _finite_array(right_vectors, complex, "right_vectors")
    q = _finite_array(transfer, float, "transfer")
    gs = _finite_array(g_vectors, float, "g_vectors")
    tau = _finite_array(centres, float, "centres")
    alpha = _finite_array(orbital_averages, float, "orbital_averages")
    if (
        left.ndim != 3
        or left.shape[1:] != (2, 2)
        or len(left) < 1
        or right.shape != left.shape
    ):
        raise ValueError("coefficient arrays must have equal shape (K, 2, 2)")
    if q.shape != (2,) or gs.ndim != 2 or gs.shape[1] != 2 or len(gs) < 1:
        raise ValueError("invalid transfer or reciprocal-vector shape")
    if tau.shape != (2, 2) or alpha.shape != (len(gs), 2):
        raise ValueError("invalid orbital positions or averages shape")
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("orbital averages must lie in [0, 1]")
    phase = np.exp(-1j * ((q + gs) @ tau.T))
    return np.einsum(
        "kan,kam,ga->kgnm", left.conj(), right, phase * np.sqrt(alpha)
    )

import numpy as np


def static_response(
    left_energies: "np.ndarray",
    right_energies: "np.ndarray",
    vertices: "np.ndarray",
    spin_degeneracy: int = 1,
) -> "np.ndarray":
    """Full two-channel response; no assumed equality of the two channels."""
    left = _finite_array(left_energies, float, "left_energies")
    right = _finite_array(right_energies, float, "right_energies")
    vertex = _finite_array(vertices, complex, "vertices")
    spin = _integer_scalar(spin_degeneracy, "spin_degeneracy", 1)
    if (
        left.ndim != 2
        or left.shape[1] != 2
        or len(left) < 1
        or right.shape != left.shape
    ):
        raise ValueError("energies must have equal shape (K, 2)")
    if (
        vertex.ndim != 4
        or vertex.shape[0] != len(left)
        or vertex.shape[1] < 1
        or vertex.shape[2:] != (2, 2)
    ):
        raise ValueError("vertices must have shape (K, G, 2, 2)")
    gap01 = right[:, 1] - left[:, 0]
    gap10 = left[:, 1] - right[:, 0]
    if (
        np.any(np.diff(left, axis=1) <= 0)
        or np.any(np.diff(right, axis=1) <= 0)
        or np.any(gap01 <= 0)
        or np.any(gap10 <= 0)
    ):
        raise ValueError(
            "all within- and cross-momentum gaps must be positive"
        )
    v01, v10 = vertex[:, :, 0, 1], vertex[:, :, 1, 0]
    response = -np.einsum("kg,kh,k->gh", v01, v01.conj(), 1 / gap01)
    response -= np.einsum("kg,kh,k->gh", v10, v10.conj(), 1 / gap10)
    return spin * response / len(left)

import numpy as np


def screened_interaction(
    response: "np.ndarray",
    magnitudes: "np.ndarray",
    area: float,
    coupling: float,
    pair_averages: "np.ndarray",
) -> "np.ndarray":
    """Symmetric Q2D dielectric inversion with separate external factors."""
    chi = _finite_array(response, complex, "response")
    p = _finite_array(magnitudes, float, "magnitudes")
    beta = _finite_array(pair_averages, float, "pair_averages")
    cell_area = _finite_scalar(area, "area")
    constant = _finite_scalar(coupling, "coupling")
    if p.ndim != 1 or len(p) < 1 or np.any(p < 0):
        raise ValueError("magnitudes must be a nonempty nonnegative vector")
    if chi.shape != (len(p), len(p)) or beta.shape != p.shape:
        raise ValueError("response or pair-average shape mismatch")
    if cell_area <= 0 or np.any(beta < 0) or np.any(beta > 1):
        raise ValueError("area must be positive and averages in [0, 1]")
    if not np.allclose(chi, chi.T.conj(), rtol=0, atol=1e-10):
        raise ValueError("response must be Hermitian")
    chi = (chi + chi.T.conj()) / 2
    if np.linalg.eigvalsh(chi)[-1] > 1e-10:
        raise ValueError("response must be negative semidefinite")
    v = np.zeros(len(p))
    positive = p > 0
    v[positive] = 2 * np.pi * constant / (cell_area * p[positive])
    root = np.sqrt(v)
    epsilon = np.eye(len(p)) - root[:, None] * chi * root[None, :]
    inverse = np.linalg.solve(epsilon, np.eye(len(p)))
    outer_root = np.sqrt(v * beta)
    screened = outer_root[:, None] * inverse * outer_root[None, :]
    return np.stack((epsilon, inverse, screened))

import numpy as np


def direct_kernel(
    vectors: "np.ndarray",
    k_points: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    interactions: "np.ndarray",
    pair_indices: "np.ndarray",
) -> "np.ndarray":
    """Contract the real-space direct matrix element using the stated signs."""
    u = _finite_array(vectors, complex, "vectors")
    k = _finite_array(k_points, float, "k_points")
    gs = _finite_array(g_vectors, float, "g_vectors")
    tau = _finite_array(centres, float, "centres")
    ws = _finite_array(interactions, complex, "interactions")
    ids = _finite_array(pair_indices, float, "pair_indices")
    if (
        u.ndim != 3
        or u.shape[1:] != (2, 2)
        or len(u) < 1
        or k.shape != (len(u), 2)
        or tau.shape != (2, 2)
    ):
        raise ValueError("invalid band or geometry shape")
    if gs.ndim != 2 or gs.shape[1] != 2 or len(gs) < 1:
        raise ValueError("g_vectors must have shape (G, 2)")
    if ws.ndim != 3 or len(ws) < 1 or ws.shape[1:] != (len(gs), len(gs)):
        raise ValueError("interactions must have shape (T, G, G)")
    original_ids = np.asarray(pair_indices)
    if (
        ids.shape != (len(u), len(u))
        or original_ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids >= len(ws))
    ):
        raise ValueError(
            "pair_indices must be integer indices into interactions"
        )
    ids = original_ids.astype(np.intp, copy=False)
    transfers = k[:, None, :] - k[None, :, :]
    phase = np.exp(
        1j * np.einsum("ijgd,ad->ijga", transfers[:, :, None, :] + gs, tau)
    )
    cproduct = u[:, None, :, 1].conj() * u[None, :, :, 1]
    vproduct = u[:, None, :, 0].conj() * u[None, :, :, 0]
    cv = np.einsum("ija,ijga->ijg", cproduct, phase)
    vv = np.einsum("ija,ijga->ijg", vproduct, phase)
    return np.einsum("ijg,ijgh,ijh->ij", cv, ws[ids], vv.conj()) / len(u)

import numpy as np


def lowest_exciton_energy(
    energies: "np.ndarray", direct: "np.ndarray"
) -> float:
    """Diagonal band differences minus the attractive direct kernel."""
    e = _finite_array(energies, float, "energies")
    d = _finite_array(direct, complex, "direct")
    if e.ndim != 2 or e.shape[1] != 2 or len(e) < 1:
        raise ValueError("energies must have shape (K, 2)")
    if d.shape != (len(e), len(e)) or np.any(e[:, 1] <= e[:, 0]):
        raise ValueError("invalid direct shape or band ordering")
    if not np.allclose(d, d.T.conj(), rtol=0, atol=1e-10):
        raise ValueError("direct must be Hermitian")
    h = np.diag(e[:, 1] - e[:, 0]) - (d + d.T.conj()) / 2
    return float(np.linalg.eigvalsh(h)[0])

import numpy as np


def head_regularization(
    inverse_head_x: float,
    inverse_head_y: float,
    transfer_x: float,
    transfer_y: float,
    fraction: float,
    area: float,
    coupling: float,
) -> float:
    """Source's small-circle regularization of the screened head at Gamma.

    The screening parameters are the finite-difference slopes of the head of the
    inverse dielectric matrix from Gamma to the mesh-adjacent transfer along each
    axis, r0x = (1 - inverse_head_x)/transfer_x and likewise for y. The circle
    radius is q0 = fraction * k0 with k0 the smaller of the two transfers. The
    average of the strict-2D Coulomb potential 2*pi*C/(area*p) over that circle is
    4*pi*C/(area*q0), and the source's closed form multiplies it by
    (1 - q0*(r0x + r0y)/4).
    """
    ex = _finite_scalar(inverse_head_x, "inverse_head_x", -np.inf)
    ey = _finite_scalar(inverse_head_y, "inverse_head_y", -np.inf)
    kx = _finite_scalar(transfer_x, "transfer_x", 0.0)
    ky = _finite_scalar(transfer_y, "transfer_y", 0.0)
    fr = _finite_scalar(fraction, "fraction", 0.0)
    ar = _finite_scalar(area, "area", 0.0)
    c = _finite_scalar(coupling, "coupling", 0.0)
    if kx <= 0.0 or ky <= 0.0:
        raise ValueError("transfers must be strictly positive")
    if not (0.0 < fr <= 1.0):
        raise ValueError("fraction must lie in (0, 1]")
    if ar <= 0.0:
        raise ValueError("area must be positive")
    if not (0.0 < ex <= 1.0 and 0.0 < ey <= 1.0):
        raise ValueError("inverse heads must lie in (0, 1]")
    r0x = (1.0 - ex) / kx
    r0y = (1.0 - ey) / ky
    q0 = fr * min(kx, ky)
    average = 4.0 * np.pi * c / (ar * q0)
    return float(average * (1.0 - q0 * (r0x + r0y) / 4.0))

from scipy.special import gammainc
import numpy as np


def _slab_moment(order, extent, argument):
    """Integrate s**order * exp(-argument*s) from zero to extent."""
    length, x = np.broadcast_arrays(extent, argument)
    result = np.zeros_like(x, dtype=float)
    small = x * length <= 0.5
    scaled = x[small] * length[small]
    term = np.ones_like(scaled)
    total = term / (order + 1)
    for index in range(1, 24):
        term = term * (-scaled / index)
        total += term / (order + index + 1)
    result[small] = length[small] ** (order + 1) * total
    large = ~small
    factorial = (1, 1, 2, 6)[order]
    result[large] = (
        factorial
        * gammainc(order + 1, x[large] * length[large])
        / x[large] ** (order + 1)
    )
    return result


def _slab_average_derivatives(magnitudes, fractions, thickness):
    """Differentiate the exact normalized slab integrals."""
    p = magnitudes
    argument = p * thickness
    result = np.empty((2, len(p), len(fractions) + 1))
    for order in (1, 2):
        left = _slab_moment(order, 0.5 + fractions[None, :], argument[:, None])
        right = _slab_moment(
            order, 0.5 - fractions[None, :], argument[:, None]
        )
        result[order - 1, :, :-1] = (-p[:, None]) ** order * (left + right)
        result[order - 1, :, -1] = (
            2
            * (-p) ** order
            * (
                _slab_moment(order, 1.0, argument)
                - _slab_moment(order + 1, 1.0, argument)
            )
        )
    return result


def _cross_response(left, right, energies, shifted_energies):
    """Bilinear response used only for derivative product-rule terms."""
    result = np.zeros((left.shape[1], left.shape[1]), dtype=complex)
    for first, second in ((0, 1), (1, 0)):
        factor = (1 - 2 * first) / (
            energies[:, first] - shifted_energies[:, second]
        )
        result += np.einsum(
            "kg,kh,k->gh",
            left[:, :, first, second],
            right[:, :, first, second].conj(),
            factor,
        )
    return result / len(energies)


def screening_thickness_derivatives(
    left_energies: "np.ndarray",
    right_energies: "np.ndarray",
    left_vectors: "np.ndarray",
    right_vectors: "np.ndarray",
    transfer: "np.ndarray",
    g_vectors: "np.ndarray",
    centres: "np.ndarray",
    height_fractions: "np.ndarray",
    thickness: float,
    area: float,
    coupling: float,
) -> "np.ndarray":
    """Propagate ordinary thickness derivatives through microscopic RPA."""
    eta = _finite_array(height_fractions, float, "height_fractions")
    q = _finite_array(transfer, float, "transfer")
    gs = _finite_array(g_vectors, float, "g_vectors")
    tau = _finite_array(centres, float, "centres")
    u = _finite_array(left_vectors, complex, "left_vectors")
    ur = _finite_array(right_vectors, complex, "right_vectors")
    e = _finite_array(left_energies, float, "left_energies")
    er = _finite_array(right_energies, float, "right_energies")
    d = _finite_scalar(thickness, "thickness")
    ar = _finite_scalar(area, "area")
    c = _finite_scalar(coupling, "coupling")
    if eta.shape != (2,) or np.any(np.abs(eta) > 0.5):
        raise ValueError("height_fractions must lie in [-0.5, 0.5]")
    if q.shape != (2,) or gs.ndim != 2 or gs.shape[1] != 2:
        raise ValueError("invalid transfer or reciprocal vectors")
    p = np.linalg.norm(q + gs, axis=1)
    averages = thickness_averages(p, d * eta, d)
    vertices = density_vertices(u, ur, q, gs, tau, averages[:, :2])
    response = static_response(e, er, vertices, 1)
    base = screened_interaction(response, p, ar, c, averages[:, -1])
    average_derivatives = _slab_average_derivatives(p, eta, d)
    a0 = averages[:, :2]
    a1, a2 = average_derivatives[:, :, :2]
    root = np.sqrt(a0)
    root1 = a1 / (2 * root)
    root2 = a2 / (2 * root) - a1**2 / (4 * root**3)
    phase = np.exp(-1j * ((q + gs) @ tau.T))
    v1 = np.einsum("kan,kam,ga->kgnm", u.conj(), ur, phase * root1)
    v2 = np.einsum("kan,kam,ga->kgnm", u.conj(), ur, phase * root2)
    chi1 = _cross_response(v1, vertices, e, er)
    chi1 += _cross_response(vertices, v1, e, er)
    chi2 = _cross_response(v2, vertices, e, er)
    chi2 += 2 * _cross_response(v1, v1, e, er)
    chi2 += _cross_response(vertices, v2, e, er)
    bare = np.zeros_like(p)
    positive = p > 0
    bare[positive] = 2 * np.pi * c / (ar * p[positive])
    bare_root = np.sqrt(bare)
    eps1 = -bare_root[:, None] * chi1 * bare_root[None, :]
    eps2 = -bare_root[:, None] * chi2 * bare_root[None, :]
    inverse = base[1]
    inv1 = -inverse @ eps1 @ inverse
    inv2 = 2 * inverse @ eps1 @ inverse @ eps1 @ inverse
    inv2 -= inverse @ eps2 @ inverse
    b0 = averages[:, -1]
    b1, b2 = average_derivatives[:, :, -1]
    outer = np.sqrt(bare * b0)
    outer1 = outer * b1 / (2 * b0)
    outer2 = outer * (b2 / (2 * b0) - b1**2 / (4 * b0**2))

    def _sandwich(left, middle, right):
        return left[:, None] * middle * right[None, :]

    w1 = _sandwich(outer1, inverse, outer)
    w1 += _sandwich(outer, inv1, outer)
    w1 += _sandwich(outer, inverse, outer1)
    w2 = _sandwich(outer2, inverse, outer)
    w2 += _sandwich(outer, inv2, outer)
    w2 += _sandwich(outer, inverse, outer2)
    w2 += 2 * _sandwich(outer1, inv1, outer)
    w2 += 2 * _sandwich(outer1, inverse, outer1)
    w2 += 2 * _sandwich(outer, inv1, outer1)
    return np.stack(
        (base, np.stack((eps1, inv1, w1)), np.stack((eps2, inv2, w2)))
    )

import numpy as np


def exciton_energy_derivatives(
    energies: "np.ndarray",
    direct_derivatives: "np.ndarray",
    gap_tolerance: float = 1e-9,
) -> "np.ndarray":
    """Evaluate the spectral response with the reduced resolvent."""
    e = _finite_array(energies, float, "energies")
    ds = _finite_array(direct_derivatives, complex, "direct_derivatives")
    tol = _finite_scalar(gap_tolerance, "gap_tolerance")
    if e.ndim != 2 or e.shape[1] != 2 or len(e) < 1:
        raise ValueError("energies must have shape (K, 2)")
    if ds.shape != (3, len(e), len(e)) or tol <= 0:
        raise ValueError("invalid derivative shape or gap tolerance")
    if not np.allclose(ds, ds.conj().transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError("all direct derivatives must be Hermitian")
    ds = (ds + ds.conj().transpose(0, 2, 1)) / 2
    energy = lowest_exciton_energy(e, ds[0])
    h0 = np.diag(e[:, 1] - e[:, 0]) - ds[0]
    values, states = np.linalg.eigh(h0)
    if len(e) > 1 and values[1] - values[0] <= tol:
        raise ValueError("lowest excitation is not sufficiently isolated")
    ground = states[:, 0]
    h1, h2 = -ds[1], -ds[2]
    slope = np.vdot(ground, h1 @ ground).real
    curvature = np.vdot(ground, h2 @ ground).real
    amplitudes = states[:, 1:].conj().T @ h1 @ ground
    curvature += 2 * np.sum(np.abs(amplitudes) ** 2 / (energy - values[1:]))
    return np.array([energy, slope, curvature], dtype=float)

import numpy as np


def thickness_exciton_curvature(
    mesh_size: int,
    lattice: "np.ndarray",
    model: "np.ndarray",
    centres: "np.ndarray",
    thickness: float,
    coupling: float,
    reciprocal_extent: int = 1,
    fraction: float = 0.5,
    gap_tolerance: float = 1e-9,
) -> float:
    """Propagate the slab dilation through screening and the BSE."""
    n = _integer_scalar(mesh_size, "mesh_size", 3)
    extent = _integer_scalar(reciprocal_extent, "reciprocal_extent", 0)
    a = _finite_array(lattice, float, "lattice")
    pars = _finite_array(model, float, "model")
    tau = _finite_array(centres, float, "centres")
    d = _finite_scalar(thickness, "thickness")
    c = _finite_scalar(coupling, "coupling")
    fr = _finite_scalar(fraction, "fraction")
    tol = _finite_scalar(gap_tolerance, "gap_tolerance")
    if n % 2 == 0 or d <= 0 or tol <= 0 or not 0 < fr <= 1:
        raise ValueError("invalid mesh, thickness, fraction or tolerance")
    if a.shape != (2,) or np.any(a <= 0) or tau.shape != (2, 3):
        raise ValueError("invalid lattice or centres")
    if np.any(np.abs(tau[:, 2]) > d / 2):
        raise ValueError("orbital heights must lie within the slab")
    eta = tau[:, 2] / d
    indices = np.array([(i, j) for i in range(n) for j in range(n)])
    step = 2 * np.pi / (n * a)
    points = (indices - (n - 1) / 2) * step
    gs = np.array(
        [
            (i, j)
            for i in range(-extent, extent + 1)
            for j in range(-extent, extent + 1)
        ]
    ) * (2 * np.pi / a)
    shifts = np.array(
        [(i, j) for i in range(1 - n, n) for j in range(1 - n, n)]
    )
    differences = indices[:, None, :] - indices[None, :, :]
    pair_ids = (
        (differences[..., 0] + n - 1) * (2 * n - 1)
        + differences[..., 1]
        + n
        - 1
    )
    e, u = solve_bands(points, a, pars)
    head = extent * (2 * extent + 1) + extent
    area = float(np.prod(a))
    screened = []
    axis_heads = {}
    for shift in shifts:
        q = shift * step
        er, ur = solve_bands(points + q, a, pars)
        derivatives = screening_thickness_derivatives(
            e, er, u, ur, q, gs, tau[:, :2], eta, d, area, c
        )
        screened.append(derivatives[:, 2])
        if tuple(shift) in ((1, 0), (0, 1)):
            axis_heads[tuple(shift)] = derivatives[:, 1, head, head].real
    zero_index = (n - 1) * (2 * n - 1) + n - 1
    screened[zero_index] = screened[zero_index].copy()
    hx, hy = axis_heads[1, 0], axis_heads[0, 1]
    screened[zero_index][0, head, head] = head_regularization(
        float(hx[0]), float(hy[0]), float(step[0]), float(step[1]), fr, area, c
    )
    screened[zero_index][1:, head, head] = (
        np.pi * c / area * (hx[1:] / step[0] + hy[1:] / step[1])
    )
    ws = np.asarray(screened)
    direct = np.array(
        [
            direct_kernel(
                u, points, gs, tau[:, :2], ws[:, order], pair_ids
            )
            for order in range(3)
        ]
    )
    derivatives = exciton_energy_derivatives(e, direct, tol)
    return float(1000 * derivatives[2])
SCICODE_GOLD_EOF
