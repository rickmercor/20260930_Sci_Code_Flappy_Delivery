"""
Host Riccati-Bessel functions with geometric tangents.

A jet stores a value in its last-axis slot 0 and its first derivatives

with respect to K real parameters in slots 1,...,K. Products, quotients,

exponentials and conjugates carry their ordinary chain-rule derivatives.

B denotes the batch size. All arrays are finite and represent nonsingular

states; K is between 1 and 9 and B between 1 and 32. Complex differentiation

is with respect to these real parameters, including conjugation where used.

Returns
-------
ndarray, shape (nmax+1, 2, B, K+1), complex128: psi/xi jets, in that kind order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def host_radial_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    logarithms: "np.ndarray",
) -> "np.ndarray":
    """Return the regular and outgoing host radial jets.

    Initialize psi_0=sin(x), xi_0=-i*exp(i*x). For n=1,...,nmax,
    psi_n=psi_(n-1)*(n/x-D1_(n-1)) and
    xi_n=xi_(n-1)*(n/x-D3_(n-1)). Propagate all tangents, including the
    host size parameter x. The host is vacuum with outgoing h_n^(1).

    Parameters
    ----------
    x : ndarray, shape (B,), float64
        Host size parameters, 0.2 <= x <= 3, away from radial zeros.
    dx : ndarray, shape (B, K), float64
        Derivatives of x with respect to real design parameters.
    logarithms : ndarray, shape (nmax+1, 2, B, K+1), complex128
        Corresponding D1/D3 jets with 1 <= nmax <= 16.

    Returns
    -------
    ndarray, shape (nmax+1, 2, B, K+1), complex128
        psi/xi jets, in that kind order.
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


def _oracle_host_radial_jets(
    x: "np.ndarray",
    dx: "np.ndarray",
    logarithms: "np.ndarray",
) -> "np.ndarray":
    z = _jet(x, dx)
    one = _jet(np.ones_like(x), np.zeros_like(dx))
    result = np.empty_like(logarithms)
    result[0, 0] = _jet(np.sin(x), np.cos(x)[..., None] * np.asarray(dx))
    result[0, 1] = -1j * _exp(1j * z)
    for order in range(1, len(result)):
        n_over_z = _div(order * one, z)
        for kind in range(2):
            result[order, kind] = _mul(
                result[order - 1, kind],
                n_over_z - logarithms[order - 1, kind],
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

x1 = np.array([0.25, 0.45, 0.75])
dx1 = np.array([[0.03, 0.05], [-0.02, 0.04], [0.06, -0.01]])
x = 1.4 * x1
dx = 1.4 * dx1 + 0.03

logs_model = logarithmic_wave_jets(x.copy(), dx.copy(), 8, 80)
logs_ref = _oracle_logarithmic_wave_jets(x.copy(), dx.copy(), 8, 80)


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
    host_radial_jets(
        x.copy(),
        dx.copy(),
        logs_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_host_radial_jets(
        x.copy(),
        dx.copy(),
        logs_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np

x1 = np.array([0.25, 0.45, 0.75])
dx1 = np.array([[0.03, 0.05], [-0.02, 0.04], [0.06, -0.01]])
x = 1.4 * x1
dx_seed = 1.4 * dx1 + 0.03

logs_model = logarithmic_wave_jets(x.copy(), dx_seed.copy(), 8, 80)
logs_ref = np.array([(2.7395121590837834+0j), 0j, 0j, (1.3715262631452128+0j), 0j, 0j, (0.5736196970642415+0j), 0j, 0j, 1j, 0j, 0j, 1j, 0j, 0j, 1j, 0j, 0j, (5.644039372497591+0j), 0j, 0j, (3.047148639617272+0j), 0j, 0j, (1.6878046156065232+0j), 0j, 0j, (-2.545338848234177+0.10913140311804014j), 0j, 0j, (-1.1363029474562154+0.28412914310258414j), 0j, 0j, (-0.45297548270199894+0.5243757431629014j), 0j, 0j, (8.521331038771631+0j), 0j, 0j, (4.6713318492907145+0j), 0j, 0j, (2.7044402284921345+0j), 0j, 0j, (-5.593235968115838+0.001599386091536072j), 0j, 0j, (-2.9436367107656887+0.0152228560765381j), 0j, 0j, (-1.5006173011904922+0.08988432213436252j), 0j, 0j, (11.389634318590984+0j), 0j, 0j, (6.278923794213779+0j), 0j, 0j, (3.691532884957823+0j), 0j, 0j, (-8.500830359628006+7.971512331007563e-06j), 0j, 0j, (-4.632128535188525+0.00025638234878934554j), 0j, 0j, (-2.62776475882195+0.004731213772886899j), 0j, 0j, (14.25386880652132+0j), 0j, 0j, (7.879075478421013+0j), 0j, 0j, (4.665704295335592+0j), 0j, 0j, (-11.378394307821234+2.0070222932088546e-08j), 0j, 0j, (-6.25814273900024+2.1260714220681025e-06j), 0j, 0j, (-3.6541789855032185+0.00011417378033462696j), 0j, 0j, (17.11591713394523+0j), 0j, 0j, (9.47524899171762+0j), 0j, 0j, (5.6330552176393525+0j), 0j, 0j, (-14.246749363307643+3.0471920596656946e-11j), 0j, 0j, (-7.866058862215127+1.0551845690970026e-08j), 0j, 0j, (-4.643084651526959+1.611930392896941e-06j), 0j, 0j, (19.976655447870804+0j), 0j, 0j, (11.069045556679788+0j), 0j, 0j, (6.596361656904164+0j), 0j, 0j, (-17.110999464112254+3.092630858292982e-14j), 0j, 0j, (-9.466304775768533+3.4892801621870937e-11j), 0j, 0j, (-5.617736110990646+1.502613447977782e-08j), 0j, 0j, (22.83654680825421+0j), 0j, 0j, (12.661308235758995+0j), 0j, 0j, (7.5570707793286465+0j), 0j, 0j, (-19.973053811148656+2.2455502777229405e-17j), 0j, 0j, (-11.062514144149926+8.240511524483254e-14j), 0j, 0j, (-6.585262627895022+9.95724461270374e-11j), 0j, 0j, (25.695859002894966+0j), 0j, 0j, (14.252523347556595+0j), 0j, 0j, (8.516011937625093+0j), 0j, 0j, (-22.83379484387663+1.224116201940278e-20j), 0j, 0j, (-12.656326798033804+1.4595783631939585e-16j), 0j, 0j, (-7.548646453301744+4.935133182559711e-13j), 0j, 0j], dtype=complex).reshape(9, 2, 3, 3)

dx_model = np.zeros_like(dx_seed)
dx_ref = np.zeros_like(dx_seed)
logs_model = logs_model.copy()
logs_model[..., 1:] = 0


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
    host_radial_jets(
        x.copy(),
        dx_model.copy(),
        logs_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_host_radial_jets(
        x.copy(),
        dx_ref.copy(),
        logs_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
        {
            "setup": """import numpy as np

x = np.array([0.2, 2.95])
dx = np.array([[0.2, -0.1], [0.0, 0.3]])

logs_model = logarithmic_wave_jets(x.copy(), dx.copy(), 16, 100)
logs_ref = _oracle_logarithmic_wave_jets(x.copy(), dx.copy(), 16, 100)


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
    host_radial_jets(
        x.copy(),
        dx.copy(),
        logs_model.copy(),
    )
)
""",
            "gold_call": """compare(
    _oracle_host_radial_jets(
        x.copy(),
        dx.copy(),
        logs_ref.copy(),
    )
)
""",
            "tol": 1e-09,
        },
    ]
