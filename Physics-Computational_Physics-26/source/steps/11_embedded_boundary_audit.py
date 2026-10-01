"""
Run the whole audit on the prescribed instance at the given refinement by chaining the earlier steps, and return the eight diagnostics that characterise it. The instance is the one the problem statement fixes; refine = 1 is that instance and larger integers subdivide each direction by that factor while keeping the same extents and the same sphere.

The refinement argument exists so the reported error can be checked against the second-order behaviour the reconstruction is designed to have.

Returns
-------
ndarray of shape (8,), float64: the largest absolute ghost-cell reconstruction error, the mean absolute error, the number of ghost cells, the number of ordering passes, the number of derivative-type ghost cells, the smallest normal-direction thickness over the ghost cells, the number of ghost cells with a widened stencil in at least one direction, and the number of ghost centres lying in the fluid.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def embedded_boundary_audit(refine: int) -> "np.ndarray":
    """Run the whole audit on the prescribed instance at the given refinement by chaining the earlier steps, and return the eight diagnostics that characterise it. The instance is the one the problem statement fixes; refine = 1 is that instance and larger integers subdivide each direction by that factor while keeping the same extents and the same sphere.

    Returns
    -------
    ndarray of shape (8,), float64: the largest absolute ghost-cell reconstruction error, the mean absolute error, the number of ghost cells, the number of ordering passes, the number of derivative-type ghost cells, the smallest normal-direction thickness over the ghost cells, the number of ghost cells with a widened stencil in at least one direction, and the number of ghost centres lying in the fluid.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_embedded_boundary_audit(refine: int) -> "np.ndarray":
    refine = int(refine)
    if refine < 1:
        raise ValueError("refine must be a positive integer")
    # Evaluation configuration: GIVEN in the prompt.
    nx = ny = nz = 30 * refine + 1
    x = np.linspace(-1.5, 1.5, nx)
    y = np.linspace(-1.2, 1.2, ny)
    z = np.linspace(-0.9, 0.9, nz)
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0]); dz = float(z[1] - z[0])
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    cx, cy, cz, R = 0.07, -0.11, 0.05, 0.61

    psi = _oracle_signed_normal_distance(X, Y, Z, cx, cy, cz, R)
    nrm = _oracle_boundary_normal(X, Y, Z, cx, cy, cz)
    th = _oracle_normal_cell_thickness(nrm, dx, dy, dz)
    tags = _oracle_hybrid_ghost_tags(psi, th)
    bpt = _oracle_boundary_point(X, Y, Z, cx, cy, cz, R)
    mult = _oracle_stencil_multipliers(psi, nrm, dx, dy, dz)
    wv = _oracle_trilinear_value_weights(X, Y, Z, bpt, nrm, mult, dx, dy, dz)
    wdv = _oracle_trilinear_derivative_weights(X, Y, Z, bpt, nrm, mult, dx, dy, dz)
    lev = _oracle_dependency_levels(tags, nrm, mult)

    # Manufactured field, exact everywhere.  The boundary data is MIXED: the NORMAL
    # DERIVATIVE is prescribed on the part of the sphere whose outward normal has a
    # non-negative x component, and the field value is prescribed on the rest.  The arcs
    # are that way round so the largest error lands on a value-type cell, which is what
    # makes the reported answer depend on the mixed data at all.
    exact = lambda a, b, c: np.sin(1.3 * a) * np.cos(0.9 * b) * np.sin(0.7 * c) + 0.45 * a * b * c
    d_dx = lambda a, b, c: 1.3 * np.cos(1.3 * a) * np.cos(0.9 * b) * np.sin(0.7 * c) + 0.45 * b * c
    d_dy = lambda a, b, c: -0.9 * np.sin(1.3 * a) * np.sin(0.9 * b) * np.sin(0.7 * c) + 0.45 * a * c
    d_dz = lambda a, b, c: 0.7 * np.sin(1.3 * a) * np.cos(0.9 * b) * np.cos(0.7 * c) + 0.45 * a * b

    xs, ys, zs = bpt[0], bpt[1], bpt[2]
    phi_s = exact(xs, ys, zs)
    dphi_dn = d_dx(xs, ys, zs) * nrm[0] + d_dy(xs, ys, zs) * nrm[1] + d_dz(xs, ys, zs) * nrm[2]
    ghost = (tags == 2.0) | (tags == -2.0)
    deriv = nrm[0] >= 0.0

    # The two boundary-condition types do not share a stencil.  The value-type form carries
    # the three single weights, minus their three pairwise products, plus their triple
    # product; the derivative-type form carries seven independent weights built from the
    # normal components and the coordinate offsets.  One weight set drives the same sweep.
    v1, v2, v3 = wv[0], wv[1], wv[2]
    W = np.stack([
        np.where(deriv, wdv[0], v1),
        np.where(deriv, wdv[1], v2),
        np.where(deriv, wdv[2], v3),
        np.where(deriv, wdv[3], -v1 * v2),
        np.where(deriv, wdv[4], -v2 * v3),
        np.where(deriv, wdv[5], -v1 * v3),
        np.where(deriv, wdv[6], v1 * v2 * v3),
        np.where(deriv, wdv[7] * dphi_dn, (1.0 - v1) * (1.0 - v2) * (1.0 - v3) * phi_s),
    ], axis=0)

    seeded = np.where(ghost, 0.0, exact(X, Y, Z))
    rec = _oracle_sweep_reconstruct(seeded, tags, lev, W, nrm, mult)
    err = np.abs(rec - exact(X, Y, Z))[ghost]

    n_ghost = float(ghost.sum())
    n_levels = float(lev[ghost].max() + 1) if ghost.any() else 0.0
    n_deriv = float((ghost & deriv).sum())
    n_wide = float((ghost & (mult > 1.0).any(axis=0)).sum())
    n_fluid = float((tags == -2.0).sum())
    return np.array([float(err.max()), float(err.mean()), n_ghost, n_levels,
                     n_deriv, float(th[ghost].min()), n_wide, n_fluid])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the step's test specifications."""
    return [
        {
            "setup": 'import numpy as np',
            "call": 'embedded_boundary_audit(1)',
            "gold_call": '_oracle_embedded_boundary_audit(1)',
        },
        {
            "setup": 'import numpy as np',
            "call": 'embedded_boundary_audit(2)',
            "gold_call": '_oracle_embedded_boundary_audit(2)',
        },
        {
            "setup": 'import numpy as np',
            "call": 'embedded_boundary_audit(3)',
            "gold_call": '_oracle_embedded_boundary_audit(3)',
        },
    ]
