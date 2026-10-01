"""
Electric and magnetic composite impedances across a shell.

A jet stores a value in its last-axis slot 0 and its first derivatives

with respect to K real parameters in slots 1,...,K. Products, quotients,

exponentials and conjugates carry their ordinary chain-rule derivatives.

B denotes the batch size. All arrays are finite and represent nonsingular

states; K is between 1 and 9 and B between 1 and 32. Complex differentiation

is with respect to these real parameters, including conjugation where used.

Returns
-------
ndarray, shape (nmax+1, 2, B, K+1), complex128: Updated electric/magnetic composite jets.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def composite_impedance_jets(
    previous: "np.ndarray",
    m1: "np.ndarray",
    dm1: "np.ndarray",
    m2: "np.ndarray",
    dm2: "np.ndarray",
    inner: "np.ndarray",
    outer: "np.ndarray",
    ratio: "np.ndarray",
) -> "np.ndarray":
    """Return the source's electric and magnetic composite impedance jets.

    For electric polarization define G1=m2*H_previous-m1*D1_inner and
    G2=m2*H_previous-m1*D3_inner. For magnetic polarization interchange
    m1 and m2 in these two definitions. In either polarization,
    H_new=(G2*D1_outer-Q*G1*D3_outer)/(G2-Q*G1).
    In the core both H polarizations equal D1. Propagate every tangent;
    all products and quotients here act on jets, not independently on slots.

    Parameters
    ----------
    previous : ndarray, shape (nmax+1, 2, B, K+1), complex128
        Previous composite jets; polarization order is electric, magnetic.
    m1, m2 : ndarray, shape (B,), complex128
        Adjacent material indices, 1 <= Re(m) <= 4.5, 0 <= Im(m) <= 0.2.
    dm1, dm2 : ndarray, shape (B, K), complex128
        Their real-parameter derivatives.
    inner, outer : ndarray, shape (nmax+1, 2, B, K+1), complex128
        D1/D3 data for m2 at the inner and outer shell radii.
    ratio : ndarray, shape (nmax+1, B, K+1), complex128
        The shell Q jets. Inputs have 1 <= nmax <= 16 and finite,
        nonzero composite denominators.

    Returns
    -------
    ndarray, shape (nmax+1, 2, B, K+1), complex128
        Updated electric/magnetic composite jets.
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


def _oracle_composite_impedance_jets(
    previous: "np.ndarray",
    m1: "np.ndarray",
    dm1: "np.ndarray",
    m2: "np.ndarray",
    dm2: "np.ndarray",
    inner: "np.ndarray",
    outer: "np.ndarray",
    ratio: "np.ndarray",
) -> "np.ndarray":
    a, b = _jet(m1, dm1), _jet(m2, dm2)
    result = np.empty_like(previous)
    for pol in range(2):
        left, right = (b, a) if pol == 0 else (a, b)
        g1 = _mul(left, previous[:, pol]) - _mul(right, inner[:, 0])
        g2 = _mul(left, previous[:, pol]) - _mul(right, inner[:, 1])
        qg1 = _mul(ratio, g1)
        result[:, pol] = _div(
            _mul(g2, outer[:, 0]) - _mul(qg1, outer[:, 1]),
            g2 - qg1,
        )
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
h_ref = np.stack((core_ref[:, 0], core_ref[:, 0]), axis=1)


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
    composite_impedance_jets(
        h_model.copy(),
        m1.copy(),
        dm1.copy(),
        m2.copy(),
        dm2.copy(),
        inner_model.copy(),
        outer_model.copy(),
        q_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_composite_impedance_jets(
        h_ref.copy(),
        m1.copy(),
        dm1.copy(),
        m2.copy(),
        dm2.copy(),
        inner_ref.copy(),
        outer_ref.copy(),
        q_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np

m2 = np.full(3, 2.8 + 0.08j)
dm2 = np.full((3, 2), -0.04 + 0.03j)
x1 = np.array([0.25, 0.45, 0.75])
dx1 = np.array([[0.03, 0.05], [-0.02, 0.04], [0.06, -0.01]])
x2 = 1.4 * x1
dx2 = 1.4 * dx1 + 0.03

m1 = m2.copy()
dm1 = dm2.copy()

z = m2 * x1
z2 = m2 * x2
dz = dm2 * x1[:, None] + m2[:, None] * dx1
dz2 = dm2 * x2[:, None] + m2[:, None] * dx2

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
h_model = np.stack((inner_model[:, 0], inner_model[:, 0]), axis=1)

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
h_ref = np.stack((inner_ref[:, 0], inner_ref[:, 0]), axis=1)


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
    composite_impedance_jets(
        h_model.copy(),
        m1.copy(),
        dm1.copy(),
        m2.copy(),
        dm2.copy(),
        inner_model.copy(),
        outer_model.copy(),
        q_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_composite_impedance_jets(
        h_ref.copy(),
        m1.copy(),
        dm1.copy(),
        m2.copy(),
        dm2.copy(),
        inner_ref.copy(),
        outer_ref.copy(),
        q_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np

m1 = np.full(3, 1.2 + 0.15j)
m2 = np.full(3, 3.8 + 0.18j)
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
h_ref = np.stack((core_ref[:, 0], core_ref[:, 0]), axis=1)


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
    composite_impedance_jets(
        h_model.copy(),
        m1.copy(),
        dm1.copy(),
        m2.copy(),
        dm2.copy(),
        inner_model.copy(),
        outer_model.copy(),
        q_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_composite_impedance_jets(
        h_ref.copy(),
        m1.copy(),
        dm1.copy(),
        m2.copy(),
        dm2.copy(),
        inner_ref.copy(),
        outer_ref.copy(),
        q_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
    ]
