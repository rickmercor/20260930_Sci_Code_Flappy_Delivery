"""
Differentiate the paper's symmetric Q2D screening under slab dilation.

At variable thickness h, orbital heights are h*height_fractions, with

in-plane centres and all band data fixed. The first two derivatives include

the orbital-resolved response weights, the full matrix inverse and the

separate external Coulomb average. Derivative order two means d^2/dh^2,

without a factorial. This extends the source construction to a thickness

susceptibility; it does not replace it by a macroscopic screening model.

Returns
-------
complex array (3, 3, G, G): derivative order, matrix kind, G, G'.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Return value, first derivative and second derivative of screening.

    Parameters
    ----------
    left_energies, right_energies : "np.ndarray"
        Real arrays (K, 2), K >= 1, obeying static_response's positive
        within- and cross-momentum gap contract, with spin multiplicity one.
        Energies remain fixed while thickness varies.
    left_vectors, right_vectors : "np.ndarray"
        Finite complex arrays (K, 2, 2), indexed by k, orbital, band.
        Coefficients remain fixed while thickness varies. General finite
        coefficients are supported as in density_vertices.
    transfer : "np.ndarray"
        Finite real Cartesian vector (2,) in inverse angstroms.
    g_vectors : "np.ndarray"
        Finite real array (G, 2), G >= 1, in inverse angstroms.
    centres : "np.ndarray"
        Finite real array (2, 2) of in-plane orbital centres in angstroms.
    height_fractions : "np.ndarray"
        Finite real vector (2,) in [-0.5, 0.5]. At any thickness h >= 0
        orbital a lies at z_a(h) = h*height_fractions[a].
    thickness : float
        Finite nonnegative evaluation thickness in angstroms. At zero,
        return the right-hand derivatives of this dilation family.
    area : float
        Positive finite cell area in square angstroms.
    coupling : float
        Nonnegative finite Coulomb constant C in eV angstroms.

    Returns
    -------
    derivatives : "np.ndarray"
        Complex array (3, 3, G, G). Axis 0 contains orders 0, 1, 2 of
        differentiation with respect to thickness h, evaluated at thickness.
        Axis 1 contains epsilon, epsilon_inverse, W, in that order.
        Order 0 equals screened_interaction applied to the source's
        symmetric, orbital-weighted response. At every |q+G|=0 row and
        column epsilon and its inverse have identity order-0 entries and
        zero higher derivatives; W has zero entries at all three orders.
        The later orchestrator replaces W's zero-transfer head.
        Other entries retain all local-field couplings. Derivatives must
        include both single-coordinate and double-coordinate averages.
        Matrix values have the original units; orders 1 and 2 additionally
        carry inverse angstroms and inverse square angstroms, respectively.
        Differentiate the defining construction, including its matrix
        inverse, analytically. Thickness finite differences are excluded.
        Compose thickness_averages, density_vertices, static_response and
        screened_interaction for the order-0 construction.

    Raises
    ------
    ValueError
        If any documented shape, finiteness, realness, sign, gap or
        fractional-height bound is violated.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

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


def _oracle_screening_thickness_derivatives(
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
    averages = _oracle_thickness_averages(p, d * eta, d)
    vertices = _oracle_density_vertices(u, ur, q, gs, tau, averages[:, :2])
    response = _oracle_static_response(e, er, vertices, 1)
    base = _oracle_screened_interaction(response, p, ar, c, averages[:, -1])
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Deterministic scientific and boundary fixtures."""
    return [
        {
            "setup": """
import numpy as np

rng = np.random.default_rng(9217)
e = np.array([[-1.2, 1.8], [-0.9, 2.1]])
er = np.array([[-1.0, 2.4], [-1.4, 1.7]])
u = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
ur = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
q = np.array([0.21, -0.16])
gs = np.array([[0.0, 0.0], [1.7, 0.2], [-1.3, 0.8]])
tau = np.array([[0.2, -0.1], [1.05, 0.82]])
eta = np.array([-1.3 / 5.5, 0.9 / 5.5])
d, area, coupling = 5.5, 13.12, 3.59991137
""",
            "call": """
screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
            "gold_call": """
_oracle_screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
        },
        {
            "setup": """
import numpy as np

rng = np.random.default_rng(9217)
e = np.array([[-1.2, 1.8], [-0.9, 2.1]])
er = np.array([[-1.0, 2.4], [-1.4, 1.7]])
u = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
ur = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
q = np.array([0.21, -0.16])
gs = np.array([[0.0, 0.0], [1.7, 0.2], [-1.3, 0.8]])
tau = np.array([[0.2, -0.1], [1.05, 0.82]])
eta = np.array([-1.3 / 5.5, 0.9 / 5.5])
d, area, coupling = 5.5, 13.12, 3.59991137
d = 0.0
""",
            "call": """
screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
            "gold_call": """
_oracle_screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
        },
        {
            "setup": """
import numpy as np

rng = np.random.default_rng(9217)
e = np.array([[-1.2, 1.8], [-0.9, 2.1]])
er = np.array([[-1.0, 2.4], [-1.4, 1.7]])
u = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
ur = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
q = np.array([0.21, -0.16])
gs = np.array([[0.0, 0.0], [1.7, 0.2], [-1.3, 0.8]])
tau = np.array([[0.2, -0.1], [1.05, 0.82]])
eta = np.array([-1.3 / 5.5, 0.9 / 5.5])
d, area, coupling = 5.5, 13.12, 3.59991137
d = 1e-8
""",
            "call": """
screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
            "gold_call": """
_oracle_screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
        },
        {
            "setup": """
import numpy as np

rng = np.random.default_rng(9217)
e = np.array([[-1.2, 1.8], [-0.9, 2.1]])
er = np.array([[-1.0, 2.4], [-1.4, 1.7]])
u = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
ur = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
q = np.array([0.21, -0.16])
gs = np.array([[0.0, 0.0], [1.7, 0.2], [-1.3, 0.8]])
tau = np.array([[0.2, -0.1], [1.05, 0.82]])
eta = np.array([-1.3 / 5.5, 0.9 / 5.5])
d, area, coupling = 5.5, 13.12, 3.59991137
q = np.zeros(2)
ur = u.copy()
er = e.copy()
""",
            "call": """
screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
            "gold_call": """
_oracle_screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
        },
        {
            "setup": """
import numpy as np

rng = np.random.default_rng(9217)
e = np.array([[-1.2, 1.8], [-0.9, 2.1]])
er = np.array([[-1.0, 2.4], [-1.4, 1.7]])
u = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
ur = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
q = np.array([0.21, -0.16])
gs = np.array([[0.0, 0.0], [1.7, 0.2], [-1.3, 0.8]])
tau = np.array([[0.2, -0.1], [1.05, 0.82]])
eta = np.array([-1.3 / 5.5, 0.9 / 5.5])
d, area, coupling = 5.5, 13.12, 3.59991137
eta = np.array([-0.5, 0.5])
d = 30.0
""",
            "call": """
screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
            "gold_call": """
_oracle_screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
        },
        {
            "setup": """
import numpy as np

rng = np.random.default_rng(9217)
e = np.array([[-1.2, 1.8], [-0.9, 2.1]])
er = np.array([[-1.0, 2.4], [-1.4, 1.7]])
u = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
ur = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
q = np.array([0.21, -0.16])
gs = np.array([[0.0, 0.0], [1.7, 0.2], [-1.3, 0.8]])
tau = np.array([[0.2, -0.1], [1.05, 0.82]])
eta = np.array([-1.3 / 5.5, 0.9 / 5.5])
d, area, coupling = 5.5, 13.12, 3.59991137
coupling = 0.0
""",
            "call": """
screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
            "gold_call": """
_oracle_screening_thickness_derivatives(
    e.copy(),
    er.copy(),
    u.copy(),
    ur.copy(),
    q.copy(),
    gs.copy(),
    tau.copy(),
    eta.copy(),
    d,
    area,
    coupling,
)
""",
        },
        {
            "setup": """
import numpy as np

rng = np.random.default_rng(9217)
e = np.array([[-1.2, 1.8], [-0.9, 2.1]])
er = np.array([[-1.0, 2.4], [-1.4, 1.7]])
u = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
ur = np.array(
    [
        np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))[0]
        for _ in range(2)
    ]
)
q = np.array([0.21, -0.16])
gs = np.array([[0.0, 0.0], [1.7, 0.2], [-1.3, 0.8]])
tau = np.array([[0.2, -0.1], [1.05, 0.82]])
eta = np.array([-1.3 / 5.5, 0.9 / 5.5])
d, area, coupling = 5.5, 13.12, 3.59991137

eta = np.array([-0.6, 0.2])


def rejected(fn):
    try:
        fn(e, er, u, ur, q, gs, tau, eta, d, area, coupling)
    except ValueError:
        return 1.0
    return 0.0
""",
            "call": """
rejected(screening_thickness_derivatives)
""",
            "gold_call": """
rejected(_oracle_screening_thickness_derivatives)
""",
        },
    ]
