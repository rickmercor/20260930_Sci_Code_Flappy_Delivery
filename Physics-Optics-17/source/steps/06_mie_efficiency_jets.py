"""
Dimensionless efficiencies and their full shape derivatives.

A jet stores a value in its last-axis slot 0 and its first derivatives

with respect to K real parameters in slots 1,...,K. Products, quotients,

exponentials and conjugates carry their ordinary chain-rule derivatives.

B denotes the batch size. All arrays are finite and represent nonsingular

states; K is between 1 and 9 and B between 1 and 32. Complex differentiation

is with respect to these real parameters, including conjugation where used.

Returns
-------
ndarray, shape (3, B, K+1), float64: Efficiency jets in extinction, scattering, absorption order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mie_efficiency_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    coeff: "np.ndarray",
) -> "np.ndarray":
    """Return extinction, scattering and absorption efficiency jets.

    For w_n=2*n+1 and f=2/x**2, set Qext=f*sum(w_n*Re(a_n+b_n)),
    Qsca=f*sum(w_n*(abs(a_n)**2+abs(b_n)**2)), Qabs=Qext-Qsca.
    Include df=-4*dx/x**3 and d(abs(a)**2)=2*Re(conj(a)*da).
    Efficiencies are cross sections divided by pi*r_outer**2; they are real.

    Parameters
    ----------
    x : ndarray, shape (B,), float64
        Outer size parameters in [0.2, 3].
    dx : ndarray, shape (B, K), float64
        Their derivatives with respect to the real design parameters.
    coeff : ndarray, shape (nmax, 2, B, K+1), complex128
        Electric/magnetic coefficient jets, 1 <= nmax <= 16.

    Returns
    -------
    ndarray, shape (3, B, K+1), float64
        Efficiency jets in extinction, scattering, absorption order.
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


def _oracle_mie_efficiency_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    coeff: "np.ndarray",
) -> "np.ndarray":
    weight = 2 * np.arange(1, len(coeff) + 1) + 1
    extinction = np.sum(
        weight[:, None, None] * np.real(coeff.sum(axis=1)), axis=0
    )
    squares = np.real(_mul(coeff, np.conj(coeff)))
    scattering = np.sum(weight[:, None, None] * squares.sum(axis=1), axis=0)
    data = np.stack((extinction, scattering, extinction - scattering))
    size = _jet(x, dx)
    factor = _div(
        _jet(2 * np.ones_like(x), np.zeros_like(dx)), _mul(size, size)
    )
    return np.real(_mul(data, factor))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    pipeline_setup = """import numpy as np

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
coeff_model = layered_coefficient_jets(
    x_model.copy(),
    dx_model.copy(),
    m2.copy(),
    dm2.copy(),
    h_model.copy(),
    rad_model.copy(),
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
coeff_ref = _oracle_layered_coefficient_jets(
    x_ref.copy(),
    dx_ref.copy(),
    m2.copy(),
    dm2.copy(),
    h_ref.copy(),
    rad_ref.copy(),
)
"""

    return [
        {
            "setup": pipeline_setup,
            "call": """mie_efficiency_jets(
    x_model.copy(),
    dx_model.copy(),
    coeff_model.copy(),
)
""",
            "gold_call": """_oracle_mie_efficiency_jets(
    x_ref.copy(),
    dx_ref.copy(),
    coeff_ref.copy(),
)
""",
            "tol": 1e-09,
        },
        {
            "setup": pipeline_setup + """
coeff_model = coeff_model.copy()
coeff_ref_zeroed = coeff_ref.copy()
coeff_model *= 0
coeff_ref_zeroed *= 0
""",
            "call": """mie_efficiency_jets(
    x_model.copy(),
    dx_model.copy(),
    coeff_model.copy(),
)
""",
            "gold_call": """_oracle_mie_efficiency_jets(
    x_ref.copy(),
    dx_ref.copy(),
    coeff_ref_zeroed.copy(),
)
""",
            "tol": 1e-09,
        },
        {
            "setup": pipeline_setup + """
dx_model = np.zeros_like(dx_model)
dx_ref = np.zeros_like(dx_ref)
""",
            "call": """mie_efficiency_jets(
    x_model.copy(),
    dx_model.copy(),
    coeff_model.copy(),
)
""",
            "gold_call": """_oracle_mie_efficiency_jets(
    x_ref.copy(),
    dx_ref.copy(),
    coeff_ref.copy(),
)
""",
            "tol": 1e-09,
        },
    ]
