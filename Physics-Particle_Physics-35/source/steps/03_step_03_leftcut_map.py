"""
Evaluate the conformal map that sends the left-hand cut of an intermediate variable to the unit circle. The energy-like argument and both reference arguments may be finite real or complex scalars, and the result is returned as a Python complex number. Invalid input raises ValueError when any argument is not a finite scalar, when the cut and normalization references coincide, when the radicands or the result overflow to a non-finite value, or when the sum of the two principal square roots in the denominator is zero.

Write l = cut_reference and n = normalization_reference. The defining radicands are A = [x(l - 1)^2 - l(x - 1)^2](n - 1)^2 and B = [n(l - 1)^2 - l(n - 1)^2](x - 1)^2. The returned map is (sqrt(A) - sqrt(B)) / (sqrt(A) + sqrt(B)). Each square root is principal, with a nonnegative real part; an exactly negative real radicand uses the positive imaginary root, independent of a signed zero. Apply this convention to the complete radicands, not separately to factors. The algebraically equivalent factorizations are A = (x - l)(1 - lx)(n - 1)^2 and B = (n - l)(1 - ln)(x - 1)^2. The map sends the left-hand cut to the unit circle and vanishes at n. A nonzero radicand that underflows to zero, a non-finite radicand or a non-finite output is outside the supported numerical domain and raises ValueError.

Returns
-------
complex, the conformal variable with the left-hand cut on the unit circle
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def leftcut_map(x, cut_reference, normalization_reference):
    """Return the conformal variable with the left-hand cut on the unit circle.

    Parameters
    ----------
    x : float or complex
        Finite real or complex value of the intermediate map.
    cut_reference : float or complex
        Finite mapped position at which the left-hand cut opens.
    normalization_reference : float or complex
        Finite mapped point at which the returned variable must vanish.

    Returns
    -------
    complex
        Value of the left-hand-cut conformal map.

    Raises
    ------
    ValueError
        If any argument is not a finite real or complex scalar, if the two
        reference arguments coincide, if the radicands or the result overflow
        to a non-finite value, or if the sum of the principal square roots in
        the denominator is zero or a nonzero radicand underflows to zero.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_leftcut_map(x, cut_reference, normalization_reference):
    def _complex_scalar(value, name):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or isinstance(value, (str, bytes)):
            raise ValueError(name + " must be a finite real or complex scalar")
        try:
            numeric = np.asarray(value, dtype=np.complex128).item()
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name + " must be a finite real or complex scalar") from exc
        if not np.isfinite(numeric.real) or not np.isfinite(numeric.imag):
            raise ValueError(name + " must be a finite real or complex scalar")
        return numeric

    value = _complex_scalar(x, "x")
    cut = _complex_scalar(cut_reference, "cut_reference")
    normalization = _complex_scalar(normalization_reference, "normalization_reference")
    if cut == normalization:
        raise ValueError("cut_reference and normalization_reference must differ")

    with np.errstate(all="ignore"):
        try:
            factors_a = (value - cut, 1.0 - cut * value, normalization - 1.0, normalization - 1.0)
            factors_b = (normalization - cut, 1.0 - cut * normalization, value - 1.0, value - 1.0)
            a = factors_a[0] * factors_a[1] * factors_a[2] * factors_a[3]
            b = factors_b[0] * factors_b[1] * factors_b[2] * factors_b[3]
        except OverflowError as exc:
            raise ValueError("the radicands overflow for these arguments") from exc
    for radicand in (a, b):
        if not np.isfinite(radicand.real) or not np.isfinite(radicand.imag):
            raise ValueError("the radicands overflow for these arguments")
    if (a == 0.0 and all(f != 0.0 for f in factors_a)) or (b == 0.0 and all(f != 0.0 for f in factors_b)):
        raise ValueError("a nonzero radicand underflows the supported numeric range")
    # A signed zero is not a choice of lip for a complete negative real radicand.
    root_a = np.sqrt(complex(a.real, 0.0 if a.imag == 0.0 else a.imag))
    root_b = np.sqrt(complex(b.real, 0.0 if b.imag == 0.0 else b.imag))
    denominator = root_a + root_b
    if denominator == 0.0:
        raise ValueError("the sum of the principal square roots must be nonzero")
    result = complex((root_a - root_b) / denominator)
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError("the map is not finite for these arguments")

    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    status_setup = (
        "import numpy as np\n"
        "def status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )

    def _status_case(args):
        # call runs the model's function and gold_call runs the oracle through
        # the same try/except, so the exception contract is compared against the
        # oracle rather than asserted against a constant.
        return {
            "setup": status_setup,
            "call": "status(lambda: leftcut_map" + args + ")",
            "gold_call": "status(lambda: _oracle_leftcut_map" + args + ")",
        }

    return [
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12), int(abs(v) <= 1e-12)])(leftcut_map(0.373912015154, -0.144301069324, 0.373912015154))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12), int(abs(v) <= 1e-12)])(_oracle_leftcut_map(0.373912015154, -0.144301069324, 0.373912015154))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(leftcut_map(0.125, -0.144301069324, 0.373912015154))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_leftcut_map(0.125, -0.144301069324, 0.373912015154))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(leftcut_map(0.21 + 0.17j, -0.144301069324, 0.373912015154))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_leftcut_map(0.21 + 0.17j, -0.144301069324, 0.373912015154))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12), round(abs(v), 12)])(leftcut_map(-0.144301069324, -0.144301069324, 0.373912015154))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12), round(abs(v), 12)])(_oracle_leftcut_map(-0.144301069324, -0.144301069324, 0.373912015154))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(leftcut_map(1.0, -0.144301069324, 0.373912015154))",
            "gold_call": "(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_leftcut_map(1.0, -0.144301069324, 0.373912015154))",
        },
        # valid: signed zero on a negative real radicand does not choose a lip
        {
            "setup": "import numpy as np\n",
            "call": "[(lambda v: [round(v.real, 12), round(v.imag, 12)])(leftcut_map(complex(-1.0, zero), 0.0, 0.5)) for zero in (0.0, -0.0)]",
            "gold_call": "[(lambda v: [round(v.real, 12), round(v.imag, 12)])(_oracle_leftcut_map(complex(-1.0, zero), 0.0, 0.5)) for zero in (0.0, -0.0)]",
        },
        # invalid: references coincide
        _status_case("(0.2, -0.1, -0.1)"),
        # invalid: non-finite argument
        _status_case("(complex(float('inf'), 0.0), -0.1, 0.3)"),
        # invalid: non-finite cut reference
        _status_case("(0.2, float('nan'), 0.3)"),
        # invalid: zero denominator
        _status_case("(2.0, 2.0, 0.5)"),
    ]
