"""
Layered-sphere electric and magnetic scattering coefficients.

A jet stores a value in its last-axis slot 0 and its first derivatives

with respect to K real parameters in slots 1,...,K. Products, quotients,

exponentials and conjugates carry their ordinary chain-rule derivatives.

B denotes the batch size. All arrays are finite and represent nonsingular

states; K is between 1 and 9 and B between 1 and 32. Complex differentiation

is with respect to these real parameters, including conjugation where used.

Returns
-------
ndarray, shape (nmax, 2, B, K+1), complex128: Coefficient jets in multipole (starting at 1), polarization, batch and jet-slot order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def layered_coefficient_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    m: "np.ndarray",
    dm: "np.ndarray",
    composite: "np.ndarray",
    radial: "np.ndarray",
) -> "np.ndarray":
    """Return electric a_n and magnetic b_n jets at the outer boundary.

    For n=1,...,nmax, use h=H_a/m for a_n and h=m*H_b for b_n;
    each coefficient is [(h+n/x)*psi_n-psi_(n-1)] /
    [(h+n/x)*xi_n-xi_(n-1)]. Propagate material and geometric tangents.
    The convention is outgoing h_n^(1) with e^(-i*omega*t).

    Parameters
    ----------
    x : ndarray, shape (B,), float64
        Outer host size parameters, 0.2 <= x <= 3.
    dx : ndarray, shape (B, K), float64
        Derivatives of the outer size parameters.
    m : ndarray, shape (B,), complex128
        Outer material index, 1 <= Re(m) <= 4.5, 0 <= Im(m) <= 0.2.
    dm : ndarray, shape (B, K), complex128
        Derivatives of the outer index.
    composite : ndarray, shape (nmax+1, 2, B, K+1), complex128
        Electric/magnetic composite jets at the outer boundary.
    radial : ndarray, shape (nmax+1, 2, B, K+1), complex128
        Host psi/xi jets. Inputs have 1 <= nmax <= 16 and nonzero
        scattering denominators.

    Returns
    -------
    ndarray, shape (nmax, 2, B, K+1), complex128
        Coefficient jets in multipole (starting at 1), polarization,
        batch and jet-slot order.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _jet(value, derivative):
    return np.concatenate(
        (
            np.asarray(value, complex)[..., None],
            np.asarray(derivative, complex),
        ),
        axis=-1,
    )


def _mul(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] * b[..., 0]
    out[..., 1:] = a[..., :1] * b[..., 1:] + a[..., 1:] * b[..., :1]
    return out


def _div(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.empty(a.shape, dtype=complex)
    out[..., 0] = a[..., 0] / b[..., 0]
    out[..., 1:] = (a[..., 1:] - out[..., :1] * b[..., 1:]) / b[..., :1]
    return out


def _oracle_layered_coefficient_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    m: "np.ndarray",
    dm: "np.ndarray",
    composite: "np.ndarray",
    radial: "np.ndarray",
) -> "np.ndarray":
    z, material = _jet(x, dx), _jet(m, dm)
    one = _jet(np.ones_like(x), np.zeros_like(dx))
    result = np.empty((len(composite) - 1, 2) + z.shape, dtype=complex)
    for order in range(1, len(composite)):
        n_over_z = _div(order * one, z)
        for pol in range(2):
            h = (
                _div(composite[order, pol], material)
                if pol == 0
                else _mul(composite[order, pol], material)
            )
            prefactor = h + n_over_z
            num = _mul(prefactor, radial[order, 0]) - radial[order - 1, 0]
            den = _mul(prefactor, radial[order, 1]) - radial[order - 1, 1]
            result[order - 1, pol] = _div(num, den)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "setup": """import numpy as np

m1 = np.full(3, 2.2 + 0.05j)
m2 = np.full(3, 2.8 + 0.08j)
dm1 = np.full((3, 2), 0.1 + 0.02j)
dm2 = np.full((3, 2), -0.04 + 0.03j)
x1 = np.array([0.25, 0.45, 0.75])
dx1 = np.array([[0.03, 0.05], [-0.02, 0.04], [0.06, -0.01]])
x2 = 1.4 * x1
dx2 = 1.4 * dx1 + 0.03

z = m2 * x1
z2 = m2 * x2
dz = dm2 * x1[:, None] + m2[:, None] * dx1
dz2 = dm2 * x2[:, None] + m2[:, None] * dx2
zprev = m1 * x1
dzprev = dm1 * x1[:, None] + m1[:, None] * dx1

inner_model = logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_model = logarithmic_wave_jets(z2.copy(), dz2.copy(), 8, 80)
q_model = shell_ratio_jets(
    z.copy(),
    dz.copy(),
    z2.copy(),
    dz2.copy(),
    inner_model.copy(),
    outer_model.copy(),
)
core_model = logarithmic_wave_jets(zprev.copy(), dzprev.copy(), 8, 80)
h_model = np.stack((core_model[:, 0], core_model[:, 0]), axis=1)
h_model = composite_impedance_jets(
    h_model.copy(),
    m1.copy(),
    dm1.copy(),
    m2.copy(),
    dm2.copy(),
    inner_model.copy(),
    outer_model.copy(),
    q_model.copy(),
)
x_model = x2.copy()
dx_model = dx2.copy()
logs_model = logarithmic_wave_jets(x_model.copy(), dx_model.copy(), 8, 80)
rad_model = host_radial_jets(
    x_model.copy(),
    dx_model.copy(),
    logs_model.copy(),
)

inner_ref = _oracle_logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_ref = _oracle_logarithmic_wave_jets(z2.copy(), dz2.copy(), 8, 80)
q_ref = _oracle_shell_ratio_jets(
    z.copy(),
    dz.copy(),
    z2.copy(),
    dz2.copy(),
    inner_ref.copy(),
    outer_ref.copy(),
)
core_ref = _oracle_logarithmic_wave_jets(zprev.copy(), dzprev.copy(), 8, 80)
h_ref_seed = np.stack((core_ref[:, 0], core_ref[:, 0]), axis=1)
h_ref = _oracle_composite_impedance_jets(
    h_ref_seed.copy(),
    m1.copy(),
    dm1.copy(),
    m2.copy(),
    dm2.copy(),
    inner_ref.copy(),
    outer_ref.copy(),
    q_ref.copy(),
)
x_ref = x2.copy()
dx_ref = dx2.copy()
logs_ref = _oracle_logarithmic_wave_jets(x_ref.copy(), dx_ref.copy(), 8, 80)
rad_ref = _oracle_host_radial_jets(
    x_ref.copy(),
    dx_ref.copy(),
    logs_ref.copy(),
)


def compare(value):
    scale = np.maximum(1.0, np.abs(value))
    return np.stack(
        (
            np.log1p(np.abs(value)),
            value.real / scale,
            value.imag / scale,
        ),
        axis=-1,
    )
""",
            "call": """compare(
    layered_coefficient_jets(
        x_model.copy(),
        dx_model.copy(),
        m2.copy(),
        dm2.copy(),
        h_model.copy(),
        rad_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_layered_coefficient_jets(
        x_ref.copy(),
        dx_ref.copy(),
        m2.copy(),
        dm2.copy(),
        h_ref.copy(),
        rad_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np

x1 = np.array([0.25, 0.45, 0.75])
dx1 = np.array([[0.03, 0.05], [-0.02, 0.04], [0.06, -0.01]])
x_model = 1.4 * x1
dx_model = 1.4 * dx1 + 0.03
x_ref = x_model.copy()
dx_ref = dx_model.copy()

logs_model = logarithmic_wave_jets(x_model.copy(), dx_model.copy(), 8, 80)
rad_model = host_radial_jets(
    x_model.copy(),
    dx_model.copy(),
    logs_model.copy(),
)
h_model = np.stack((logs_model[:, 0], logs_model[:, 0]), axis=1)
m_model = np.ones(3, dtype=complex)
dm_model = np.zeros((3, 2), dtype=complex)

logs_ref = _oracle_logarithmic_wave_jets(x_ref.copy(), dx_ref.copy(), 8, 80)
rad_ref = _oracle_host_radial_jets(
    x_ref.copy(),
    dx_ref.copy(),
    logs_ref.copy(),
)
h_ref = np.stack((logs_ref[:, 0], logs_ref[:, 0]), axis=1)
m_ref = np.ones(3, dtype=complex)
dm_ref = np.zeros((3, 2), dtype=complex)


def compare(value):
    scale = np.maximum(1.0, np.abs(value))
    return np.stack(
        (
            np.log1p(np.abs(value)),
            value.real / scale,
            value.imag / scale,
        ),
        axis=-1,
    )
""",
            "call": """compare(
    layered_coefficient_jets(
        x_model.copy(),
        dx_model.copy(),
        m_model.copy(),
        dm_model.copy(),
        h_model.copy(),
        rad_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_layered_coefficient_jets(
        x_ref.copy(),
        dx_ref.copy(),
        m_ref.copy(),
        dm_ref.copy(),
        h_ref.copy(),
        rad_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np

m1 = np.full(3, 2.2 + 0.05j)
m2 = np.full(3, 3.5 + 0.2j)
dm1 = np.full((3, 2), 0.1 + 0.02j)
dm2 = np.full((3, 2), -0.04 + 0.03j)
x1 = np.array([0.25, 0.45, 0.75])
dx1 = np.array([[0.03, 0.05], [-0.02, 0.04], [0.06, -0.01]])
x2 = 1.4 * x1
dx2 = 1.4 * dx1 + 0.03

z = m2 * x1
z2 = m2 * x2
dz = dm2 * x1[:, None] + m2[:, None] * dx1
dz2 = dm2 * x2[:, None] + m2[:, None] * dx2
zprev = m1 * x1
dzprev = dm1 * x1[:, None] + m1[:, None] * dx1

inner_model = logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_model = logarithmic_wave_jets(z2.copy(), dz2.copy(), 8, 80)
q_model = shell_ratio_jets(
    z.copy(),
    dz.copy(),
    z2.copy(),
    dz2.copy(),
    inner_model.copy(),
    outer_model.copy(),
)
core_model = logarithmic_wave_jets(zprev.copy(), dzprev.copy(), 8, 80)
h_model = np.stack((core_model[:, 0], core_model[:, 0]), axis=1)
h_model = composite_impedance_jets(
    h_model.copy(),
    m1.copy(),
    dm1.copy(),
    m2.copy(),
    dm2.copy(),
    inner_model.copy(),
    outer_model.copy(),
    q_model.copy(),
)
x_model = x2.copy()
dx_model = dx2.copy()
logs_model = logarithmic_wave_jets(x_model.copy(), dx_model.copy(), 8, 80)
rad_model = host_radial_jets(
    x_model.copy(),
    dx_model.copy(),
    logs_model.copy(),
)

inner_ref = _oracle_logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_ref = _oracle_logarithmic_wave_jets(z2.copy(), dz2.copy(), 8, 80)
q_ref = _oracle_shell_ratio_jets(
    z.copy(),
    dz.copy(),
    z2.copy(),
    dz2.copy(),
    inner_ref.copy(),
    outer_ref.copy(),
)
core_ref = _oracle_logarithmic_wave_jets(zprev.copy(), dzprev.copy(), 8, 80)
h_ref_seed = np.stack((core_ref[:, 0], core_ref[:, 0]), axis=1)
h_ref = _oracle_composite_impedance_jets(
    h_ref_seed.copy(),
    m1.copy(),
    dm1.copy(),
    m2.copy(),
    dm2.copy(),
    inner_ref.copy(),
    outer_ref.copy(),
    q_ref.copy(),
)
x_ref = x2.copy()
dx_ref = dx2.copy()
logs_ref = _oracle_logarithmic_wave_jets(x_ref.copy(), dx_ref.copy(), 8, 80)
rad_ref = _oracle_host_radial_jets(
    x_ref.copy(),
    dx_ref.copy(),
    logs_ref.copy(),
)


def compare(value):
    scale = np.maximum(1.0, np.abs(value))
    return np.stack(
        (
            np.log1p(np.abs(value)),
            value.real / scale,
            value.imag / scale,
        ),
        axis=-1,
    )
""",
            "call": """compare(
    layered_coefficient_jets(
        x_model.copy(),
        dx_model.copy(),
        m2.copy(),
        dm2.copy(),
        h_model.copy(),
        rad_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_layered_coefficient_jets(
        x_ref.copy(),
        dx_ref.copy(),
        m2.copy(),
        dm2.copy(),
        h_ref.copy(),
        rad_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
    ]
