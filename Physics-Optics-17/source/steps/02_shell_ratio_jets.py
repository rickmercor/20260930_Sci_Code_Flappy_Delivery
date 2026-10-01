"""
Stable propagation of the within-shell radial quotient.

A jet stores a value in its last-axis slot 0 and its first derivatives

with respect to K real parameters in slots 1,...,K. Products, quotients,

exponentials and conjugates carry their ordinary chain-rule derivatives.

B denotes the batch size. All arrays are finite and represent nonsingular

states; K is between 1 and 9 and B between 1 and 32. Complex differentiation

is with respect to these real parameters, including conjugation where used.

Returns
-------
ndarray, shape (nmax+1, B, K+1), complex128: Quotient jets, including order zero. Equal bounding states give Q=1 and zero tangents when their argument tangents also agree.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def shell_ratio_jets(
    z1: "np.ndarray",
    dz1: "np.ndarray",
    z2: "np.ndarray",
    dz2: "np.ndarray",
    inner: "np.ndarray",
    outer: "np.ndarray",
) -> "np.ndarray":
    """Return jets of Q_n=(psi_n(z1)/xi_n(z1))/(psi_n(z2)/xi_n(z2)).

    Here z1=m*k*r_inner and z2=m*k*r_outer for the same shell material.
    Use Q0=exp(2*i*(z2-z1))*expm1(2*i*z1)/expm1(2*i*z2). Then
    Qn=Q_(n-1)*(z1/z2)**2 *
    [(z2*D1_n(z2)+n)*(n-z2*D3_(n-1)(z2))] /
    [(z1*D1_n(z1)+n)*(n-z1*D3_(n-1)(z1))].
    Differentiate both the arguments and supplied logarithmic data.

    Parameters
    ----------
    z1, z2 : ndarray, shape (B,), complex128
        Inner/outer arguments in the preceding step's nonsingular domain,
        with a common complex phase and 0 < abs(z1) <= abs(z2).
    dz1, dz2 : ndarray, shape (B, K), complex128
        Real-parameter tangents of the arguments; their variations may differ.
    inner, outer : ndarray, shape (nmax+1, 2, B, K+1), complex128
        D1/D3 jets at z1/z2 from the preceding step, with 1 <= nmax <= 16.

    Returns
    -------
    ndarray, shape (nmax+1, B, K+1), complex128
        Quotient jets, including order zero. Equal bounding states give Q=1
        and zero tangents when their argument tangents also agree.
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


def _exp(a):
    out = np.empty_like(a)
    out[..., 0] = np.exp(a[..., 0])
    out[..., 1:] = out[..., :1] * a[..., 1:]
    return out


def _expm1(a):
    out = _exp(a)
    out[..., 0] = np.expm1(a[..., 0])
    return out


def _oracle_shell_ratio_jets(
    z1: "np.ndarray",
    dz1: "np.ndarray",
    z2: "np.ndarray",
    dz2: "np.ndarray",
    inner: "np.ndarray",
    outer: "np.ndarray",
) -> "np.ndarray":
    a, b = _jet(z1, dz1), _jet(z2, dz2)
    one = _jet(np.ones_like(z1), np.zeros_like(dz1))
    result = np.empty(inner[:, 0].shape, dtype=complex)
    result[0] = _mul(_exp(2j * (b - a)), _div(_expm1(2j * a), _expm1(2j * b)))
    radii_ratio = _div(a, b)
    for order in range(1, len(result)):
        num = _mul(
            _mul(b, outer[order, 0]) + order * one,
            order * one - _mul(b, outer[order - 1, 1]),
        )
        den = _mul(
            _mul(a, inner[order, 0]) + order * one,
            order * one - _mul(a, inner[order - 1, 1]),
        )
        result[order] = _mul(
            result[order - 1],
            _mul(_mul(radii_ratio, radii_ratio), _div(num, den)),
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

m2 = np.full(3, 2.8 + 0.08j)
dm2 = np.full((3, 2), -0.04 + 0.03j)
x1 = np.array([0.25, 0.45, 0.75])
dx1 = np.array([[0.03, 0.05], [-0.02, 0.04], [0.06, -0.01]])
x2 = 1.4 * x1
dx2 = 1.4 * dx1 + 0.03

z = m2 * x1
z2 = m2 * x2
dz = dm2 * x1[:, None] + m2[:, None] * dx1
dz2 = dm2 * x2[:, None] + m2[:, None] * dx2

inner_model = logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_model = logarithmic_wave_jets(z2.copy(), dz2.copy(), 8, 80)

inner_ref = _oracle_logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_ref = _oracle_logarithmic_wave_jets(z2.copy(), dz2.copy(), 8, 80)


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
    shell_ratio_jets(
        z.copy(),
        dz.copy(),
        z2.copy(),
        dz2.copy(),
        inner_model.copy(),
        outer_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_shell_ratio_jets(
        z.copy(),
        dz.copy(),
        z2.copy(),
        dz2.copy(),
        inner_ref.copy(),
        outer_ref.copy(),
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

z = m2 * x1
dz = dm2 * x1[:, None] + m2[:, None] * dx1
z2 = z.copy()
dz2 = dz.copy()

inner_model = logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_model = inner_model.copy()

inner_ref = _oracle_logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_ref = inner_ref.copy()


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
    shell_ratio_jets(
        z.copy(),
        dz.copy(),
        z2.copy(),
        dz2.copy(),
        inner_model.copy(),
        outer_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_shell_ratio_jets(
        z.copy(),
        dz.copy(),
        z2.copy(),
        dz2.copy(),
        inner_ref.copy(),
        outer_ref.copy(),
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
x2 = 2.0 * x1
dx2 = 2.0 * dx1 + 0.03

z = m2 * x1
z2 = m2 * x2
dz = dm2 * x1[:, None] + m2[:, None] * dx1
dz2 = dm2 * x2[:, None] + m2[:, None] * dx2

inner_model = logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_model = logarithmic_wave_jets(z2.copy(), dz2.copy(), 8, 80)

inner_ref = _oracle_logarithmic_wave_jets(z.copy(), dz.copy(), 8, 80)
outer_ref = _oracle_logarithmic_wave_jets(z2.copy(), dz2.copy(), 8, 80)


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
    shell_ratio_jets(
        z.copy(),
        dz.copy(),
        z2.copy(),
        dz2.copy(),
        inner_model.copy(),
        outer_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_shell_ratio_jets(
        z.copy(),
        dz.copy(),
        z2.copy(),
        dz2.copy(),
        inner_ref.copy(),
        outer_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
    ]
