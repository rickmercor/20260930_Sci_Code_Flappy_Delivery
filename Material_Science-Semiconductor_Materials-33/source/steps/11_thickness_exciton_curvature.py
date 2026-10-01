"""
Compose the microscopic screening and spectral thickness response.

The deformation scales both orbital heights with the slab thickness while

holding the lattice, Hamiltonian and in-plane centres fixed. The strict-2D

reference is fixed and hence contributes zero thickness curvature. This

is the final orchestrator and must compose all ten preceding steps.

Returns
-------
a finite float: Q2D-minus-2D lowest-exciton curvature in meV/angstrom^2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Return the analytic second thickness derivative of the exciton shift.

    Parameters
    ----------
    mesh_size : int
        Odd integer n >= 3; use K=n*n equal-weight mesh points
        (2*pi*(i-(n-1)/2)/(n*a_x), 2*pi*(j-(n-1)/2)/(n*a_y)),
        i,j=0,...,n-1, in row-major order. Booleans are invalid.
    lattice : "np.ndarray"
        Positive finite real array (2,) of lattice lengths in angstroms.
    model : "np.ndarray"
        Finite real array (7,) containing solve_bands parameters, obeying
        its strict mass bound. The Hamiltonian stays fixed under dilation.
    centres : "np.ndarray"
        Finite real array (2, 3) of orbital coordinates in angstroms at the
        evaluation thickness. Heights lie inside [-thickness/2, thickness/2].
        In-plane centres stay fixed; at variable h the heights become
        centres[:, 2]*h/thickness, so the fractional heights are constant.
    thickness : float
        Finite strictly positive evaluation thickness in angstroms.
    coupling : float
        Finite nonnegative Coulomb constant C in eV angstroms.
    reciprocal_extent : int
        Integer R >= 0, excluding booleans. Include every
        G=(2*pi*r/a_x, 2*pi*s/a_y), r,s=-R,...,R, in row-major order.
    fraction : float
        Finite real number in (0, 1]; the Gamma circle radius is this
        fraction times the nearest nonzero mesh momentum.
    gap_tolerance : float
        Positive finite lowest-exciton isolation tolerance, in eV, as in
        exciton_energy_derivatives.

    Returns
    -------
    curvature : float
        1000*d^2(E_Q2D(h)-E_2D)/dh^2 at h=thickness, in meV/angstrom^2.
        E_2D is the fixed strict-2D reference with zero heights and thickness.
        Evaluate bands at k+q with unreduced q=k_i-k_j and retain all G in
        screening and the direct channel. Use both interband directions
        with spin one, direct attraction and no exchange.
        At q=0 all orders of W's wings vanish and its screened body is
        retained. Its order-0 head is head_regularization, using this
        Q2D calculation's inverse heads at the mesh-adjacent axis transfers.
        At variable h the same head formula and fixed circle radius apply;
        differentiate its inverse-head slopes with respect to h as well.
        Use analytic screening and eigenvalue derivatives; include the
        exciton eigenstate response. Thickness finite differences are
        excluded. Compose all ten preceding public functions, transitively
        where appropriate, and use their outputs.

    Raises
    ------
    ValueError
        If an integer, shape, finite/real input, mass, lattice, thickness,
        coupling, fraction or height condition fails, or the lowest
        excitation is not isolated as required by gap_tolerance.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_thickness_exciton_curvature(
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
    e, u = _oracle_solve_bands(points, a, pars)
    head = extent * (2 * extent + 1) + extent
    area = float(np.prod(a))
    screened = []
    axis_heads = {}
    for shift in shifts:
        q = shift * step
        er, ur = _oracle_solve_bands(points + q, a, pars)
        derivatives = _oracle_screening_thickness_derivatives(
            e, er, u, ur, q, gs, tau[:, :2], eta, d, area, c
        )
        screened.append(derivatives[:, 2])
        if tuple(shift) in ((1, 0), (0, 1)):
            axis_heads[tuple(shift)] = derivatives[:, 1, head, head].real
    zero_index = (n - 1) * (2 * n - 1) + n - 1
    screened[zero_index] = screened[zero_index].copy()
    hx, hy = axis_heads[1, 0], axis_heads[0, 1]
    screened[zero_index][0, head, head] = _oracle_head_regularization(
        float(hx[0]), float(hy[0]), float(step[0]), float(step[1]), fr, area, c
    )
    screened[zero_index][1:, head, head] = (
        np.pi * c / area * (hx[1:] / step[0] + hy[1:] / step[1])
    )
    ws = np.asarray(screened)
    direct = np.array(
        [
            _oracle_direct_kernel(
                u, points, gs, tau[:, :2], ws[:, order], pair_ids
            )
            for order in range(3)
        ]
    )
    derivatives = _oracle_exciton_energy_derivatives(e, direct, tol)
    return float(1000 * derivatives[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific and boundary fixtures."""
    return [
        {
            "setup": """
import numpy as np

n = 9
a = np.array([3.2, 4.1])
p = np.array([1.8, 0.2, -0.15, 0.45, 0.8, 0.65, 0.27])
t = np.array([[0.0, 0.0, -1.3], [1.05, 0.82, 0.9]])
d, c, r, f = 5.5, 3.59991137, 1, 0.5
""",
            "call": """
thickness_exciton_curvature(n, a.copy(), p.copy(), t.copy(), d, c, r, f)
""",
            "gold_call": """
_oracle_thickness_exciton_curvature(
    n, a.copy(), p.copy(), t.copy(), d, c, r, f
)
""",
        },
        {
            "setup": """
import numpy as np

n = 9
a = np.array([3.2, 4.1])
p = np.array([1.8, 0.2, -0.15, 0.45, 0.8, 0.65, 0.27])
t = np.array([[0.0, 0.0, -1.3], [1.05, 0.82, 0.9]])
d, c, r, f = 5.5, 3.59991137, 1, 0.5
n = 3
""",
            "call": """
thickness_exciton_curvature(n, a.copy(), p.copy(), t.copy(), d, c, r, f)
""",
            "gold_call": """
_oracle_thickness_exciton_curvature(
    n, a.copy(), p.copy(), t.copy(), d, c, r, f
)
""",
        },
        {
            "setup": """
import numpy as np

n = 9
a = np.array([3.2, 4.1])
p = np.array([1.8, 0.2, -0.15, 0.45, 0.8, 0.65, 0.27])
t = np.array([[0.0, 0.0, -1.3], [1.05, 0.82, 0.9]])
d, c, r, f = 5.5, 3.59991137, 1, 0.5
n = 5
a = np.array([2.8, 3.6])
p = np.array([1.6, -0.15, 0.1, 0.3, 0.6, -0.45, -0.18])
t = np.array([[0.15, -0.1, -0.6], [0.95, 0.72, 0.45]])
d, c = 2.6, 2.8
""",
            "call": """
thickness_exciton_curvature(n, a.copy(), p.copy(), t.copy(), d, c, r, f)
""",
            "gold_call": """
_oracle_thickness_exciton_curvature(
    n, a.copy(), p.copy(), t.copy(), d, c, r, f
)
""",
        },
        {
            "setup": """
import numpy as np

n = 9
a = np.array([3.2, 4.1])
p = np.array([1.8, 0.2, -0.15, 0.45, 0.8, 0.65, 0.27])
t = np.array([[0.0, 0.0, -1.3], [1.05, 0.82, 0.9]])
d, c, r, f = 5.5, 3.59991137, 1, 0.5
n = 3
r = 0
""",
            "call": """
thickness_exciton_curvature(n, a.copy(), p.copy(), t.copy(), d, c, r, f)
""",
            "gold_call": """
_oracle_thickness_exciton_curvature(
    n, a.copy(), p.copy(), t.copy(), d, c, r, f
)
""",
        },
        {
            "setup": """
import numpy as np

n = 9
a = np.array([3.2, 4.1])
p = np.array([1.8, 0.2, -0.15, 0.45, 0.8, 0.65, 0.27])
t = np.array([[0.0, 0.0, -1.3], [1.05, 0.82, 0.9]])
d, c, r, f = 5.5, 3.59991137, 1, 0.5
n = 3
c = 0.0
""",
            "call": """
thickness_exciton_curvature(n, a.copy(), p.copy(), t.copy(), d, c, r, f)
""",
            "gold_call": """
_oracle_thickness_exciton_curvature(
    n, a.copy(), p.copy(), t.copy(), d, c, r, f
)
""",
        },
        {
            "setup": """
import numpy as np

n = 9
a = np.array([3.2, 4.1])
p = np.array([1.8, 0.2, -0.15, 0.45, 0.8, 0.65, 0.27])
t = np.array([[0.0, 0.0, -1.3], [1.05, 0.82, 0.9]])
d, c, r, f = 5.5, 3.59991137, 1, 0.5
n = 3
f = 0.8
""",
            "call": """
thickness_exciton_curvature(n, a.copy(), p.copy(), t.copy(), d, c, r, f)
""",
            "gold_call": """
_oracle_thickness_exciton_curvature(
    n, a.copy(), p.copy(), t.copy(), d, c, r, f
)
""",
        },
        {
            "setup": """
import numpy as np

n = 9
a = np.array([3.2, 4.1])
p = np.array([1.8, 0.2, -0.15, 0.45, 0.8, 0.65, 0.27])
t = np.array([[0.0, 0.0, -1.3], [1.05, 0.82, 0.9]])
d, c, r, f = 5.5, 3.59991137, 1, 0.5


def rejected(fn):
    try:
        fn(4, a, p, t, d, c, r, f)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": """
rejected(thickness_exciton_curvature)
""",
            "gold_call": """
rejected(_oracle_thickness_exciton_curvature)
""",
        },
    ]
