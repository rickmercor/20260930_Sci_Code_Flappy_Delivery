"""
Stable logarithmic Riccati-Bessel data and parameter tangents.

A jet stores a value in its last-axis slot 0 and its first derivatives

with respect to K real parameters in slots 1,...,K. Products, quotients,

exponentials and conjugates carry their ordinary chain-rule derivatives.

B denotes the batch size. All arrays are finite and represent nonsingular

states; K is between 1 and 9 and B between 1 and 32. Complex differentiation

is with respect to these real parameters, including conjugation where used.

Returns
-------
ndarray, shape (nmax+1, 2, B, K+1), complex128: Jets in order, kind (D1 then D3), batch, jet-slot order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def logarithmic_wave_jets(
    z: "np.ndarray",
    dz: "np.ndarray",
    nmax: int,
    depth: int,
) -> "np.ndarray":
    """Return D1=psi'/psi and D3=xi'/xi with all first tangents.

    Use psi_n(z)=z*j_n(z), xi_n(z)=z*h_n^(1)(z). Seed D1_depth=0,
    then D1_(n-1)=n/z-1/(D1_n+n/z) for n=depth,...,1. Set D3_0=i,
    P0=(1-exp(2*i*z))/2, and for n=1,...,nmax set
    Pn=P_(n-1)*(n/z-D1_(n-1))*(n/z-D3_(n-1)), D3_n=D1_n+i/Pn.
    Propagate tangents through the finite recurrences, including the zero
    seed. The source uses this product recurrence to avoid separately large
    regular and outgoing radial factors. Evaluate 1-exp(2*i*z) accurately.

    Parameters
    ----------
    z : ndarray, shape (B,), complex128
        Arguments with 0.2 <= Re(z) <= 10 and 0 <= Im(z) <= 2, away
        from zeros of the regular or outgoing radial functions.
    dz : ndarray, shape (B, K), complex128
        Derivatives of z with respect to the K real parameters.
    nmax : int
        Largest returned order, between 1 and 16 inclusive.
    depth : int
        Downward seed order; nmax < depth <= 100 and depth >= 40.

    Returns
    -------
    ndarray, shape (nmax+1, 2, B, K+1), complex128
        Jets in order, kind (D1 then D3), batch, jet-slot order.

    Raises
    ------
    ValueError
        If any argument is zero, or nmax/depth violate their ranges.
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


def _oracle_logarithmic_wave_jets(
    z: "np.ndarray",
    dz: "np.ndarray",
    nmax: int,
    depth: int,
) -> "np.ndarray":
    if (
        not isinstance(nmax, (int, np.integer))
        or not isinstance(depth, (int, np.integer))
        or not 1 <= nmax <= 16
        or not max(nmax + 1, 40) <= depth <= 100
        or np.any(np.asarray(z) == 0)
    ):
        raise ValueError("invalid argument or recurrence order")
    z = _jet(z, dz)
    one = _jet(np.ones_like(z[..., 0]), np.zeros_like(dz))
    regular = np.zeros((depth + 1,) + z.shape, dtype=complex)
    for order in range(depth, 0, -1):
        ratio = _div(order * one, z)
        regular[order - 1] = ratio - _div(one, regular[order] + ratio)
    out = np.empty((nmax + 1, 2) + z.shape, dtype=complex)
    out[:, 0] = regular[: nmax + 1]
    out[0, 1] = 1j * one
    product = -0.5 * _expm1(2j * z)
    for order in range(1, nmax + 1):
        ratio = _div(order * one, z)
        product = _mul(
            product, _mul(ratio - out[order - 1, 0], ratio - out[order - 1, 1])
        )
        out[order, 1] = out[order, 0] + _div(1j * one, product)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    return [
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.55 + 0.01j, 1.3 + 0.12j, 2.2 + 0.04j"
                "])\n"
                "dz = np.array([[0.2 + 0.03j, -0.1], [0.1j, 0.2], [-0"
                ".1, 0.05j]])\n"
                "\n"
                "\n"
                "def compare(value):\n"
                "    scale = np.maximum(1.0, np.abs(value))\n"
                "    return np.stack(\n"
                "        (\n"
                "            np.log1p(np.abs(value)),\n"
                "            value.real / scale,\n"
                "            value.imag / scale,\n"
                "        ),\n"
                "        axis=-1,\n"
                "    )\n"
            ),
            "call": (
                "compare(logarithmic_wave_jets(z.copy(), dz.copy(), 8"
                ", 80))\n"
            ),
            "gold_call": (
                "compare(_oracle_logarithmic_wave_jets(z.copy(), dz.c"
                "opy(), 8, 80))\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.55 + 0.01j, 1.3 + 0.12j, 2.2 + 0.04j"
                "])\n"
                "dz = np.array([[0.2 + 0.03j, -0.1], [0.1j, 0.2], [-0"
                ".1, 0.05j]])\n"
                "dz *= 0\n"
                "\n"
                "\n"
                "def compare(value):\n"
                "    scale = np.maximum(1.0, np.abs(value))\n"
                "    return np.stack(\n"
                "        (\n"
                "            np.log1p(np.abs(value)),\n"
                "            value.real / scale,\n"
                "            value.imag / scale,\n"
                "        ),\n"
                "        axis=-1,\n"
                "    )\n"
            ),
            "call": (
                "compare(logarithmic_wave_jets(z.copy(), dz.copy(), 8"
                ", 80))\n"
            ),
            "gold_call": (
                "compare(_oracle_logarithmic_wave_jets(z.copy(), dz.c"
                "opy(), 8, 80))\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "z = np.array([0.25 + 0.08j, 8.0 + 1.2j])\n"
                "dz = np.array([[0.3j, 0.2, -0.1], [0.1, -0.4j, 0.05]"
                "])\n"
                "\n"
                "\n"
                "def compare(value):\n"
                "    scale = np.maximum(1.0, np.abs(value))\n"
                "    return np.stack(\n"
                "        (\n"
                "            np.log1p(np.abs(value)),\n"
                "            value.real / scale,\n"
                "            value.imag / scale,\n"
                "        ),\n"
                "        axis=-1,\n"
                "    )\n"
            ),
            "call": (
                "compare(logarithmic_wave_jets(z.copy(), dz.copy(), 1"
                "6, 100))\n"
            ),
            "gold_call": (
                "compare(_oracle_logarithmic_wave_jets(z.copy(), dz.c"
                "opy(), 16, 100))\n"
            ),
            "tol": 1e-09,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def raises(fn):\n"
                "    try:\n"
                "        fn(np.array([0j]), np.zeros((1, 2)), 8, 80)\n"
                "    except ValueError:\n"
                "        return 1.0\n"
                "    return 0.0\n"
            ),
            "call": ("raises(logarithmic_wave_jets)\n"),
            "gold_call": ("raises(_oracle_logarithmic_wave_jets)\n"),
            "tol": 0.0,
        },
    ]
