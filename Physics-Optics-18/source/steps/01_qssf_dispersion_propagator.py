"""
Evaluate the dispersion symbol and its linear half-step propagator on the supplied angular-frequency grid, and find the unique real center prescribed by the dispersion–nonlinear-offset window rule. Return all three quantities as the documented numerical array.

Compute the chromatic dispersion symbol, linear half-step propagator, and



prescribed spectral analysis-band center for the Quantum Split-Step Fourier



algorithm.







In nonlinear optical waveguides, pulse propagation is governed by second-



and third-order chromatic dispersion alongside Kerr nonlinearity. The



dispersion symbol is given by:



    D(w) = 0.5 * d2 * w^2 + beta3 * w^3



The symmetric Strang split-step integration utilizes a linear half-step



spectral propagator:



    P(w) = exp(i * D(w) * dz / 2)







The source's analysis-window prescription for input pulse amplitude A0 sets



the center frequency w_RR through the algebraic condition:



    D(w_RR) + 0.5 * gamma * A0^2 = 0



which corresponds to the cubic polynomial:



    beta3 * w^3 + 0.5 * d2 * w^2 + 0.5 * gamma * A0^2 = 0







The name w_RR follows the source notation. Here it labels the prescribed



analysis-band center; no stationary bright-soliton interpretation is assumed.







This routine evaluates the declared discrete dispersion and analysis-window



rule. Here d2 multiplies w^2/2; it is twice the coefficient named d2 in the



source. All inputs must be finite. The frequency vector is real and



nonempty; dz >= 0; d2, A0 and gamma are positive; beta3 is nonzero. This



domain has one real root. Inputs are not modified. Invalid data raises



ValueError.







Parameters



----------



w : np.ndarray



    One-dimensional array of angular frequencies from discrete Fourier



    transform.



dz : float



    Spatial step size along the propagation direction.



d2 : float, default 1.0



    Group velocity dispersion coefficient.



beta3 : float, default 0.08



    Third-order dispersion coefficient.



A0 : float, default 1.5



    Initial sech-pulse amplitude.



gamma : float, default 1.0



    Kerr nonlinear coefficient.







Returns



-------



np.ndarray



    Two-dimensional complex array of shape (3, len(w)) containing:



    - Row 0: Linear half-step spectral propagator P(w).



    - Row 1: Dispersion symbol D(w) (cast to complex).



    - Row 2: Broadcast prescribed analysis-band center w_RR.

Returns
-------
complex ndarray (3, len(w)), rows P, D and broadcast w_RR.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def qssf_dispersion_propagator(
    w: "np.ndarray",
    dz: float,
    d2: float = 1.0,
    beta3: float = 0.08,
    A0: float = 1.5,
    gamma: float = 1.0,
) -> "np.ndarray":
    """
    Compute the linear half-step propagator, dispersion symbol, and
    prescribed analysis-band center.

    Parameters
    ----------
    w : "np.ndarray"
        Array of angular frequencies.
    dz : float
        Spatial step size.
    d2 : float
        Group velocity dispersion coefficient.
    beta3 : float
        Third-order dispersion coefficient.
    A0 : float
        Sech-pulse amplitude.
    gamma : float
        Kerr nonlinearity.

    Returns
    -------
    np.ndarray
        Array of shape (3, len(w)) containing [P, D, w_RR].
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _qssf_array(value, ndim, *, real=False, nonempty=True):
    """
    Validate numerical arrays without changing caller-owned storage.
    """
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "biufc":
            raise ValueError("Expected a numerical dtype")
        if real and np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise ValueError("Expected real data")
        array = np.asarray(
            raw.real if real else raw, dtype=float if real else complex
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Expected numerical array") from error
    if array.ndim != ndim or (nonempty and array.size == 0):
        raise ValueError("Invalid array rank or empty input")
    if not np.isfinite(array).all():
        raise ValueError("Array entries must be finite")
    return array


def _qssf_scalar(value, *, positive=False, nonnegative=False):
    """
    Validate a finite real scalar.
    """
    array = _qssf_array(value, 0, real=True)
    result = float(array)
    if (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError("Scalar is outside the supported domain")
    return result


def _qssf_pair(U, V):
    """
    Validate a pair of equally sized, nonempty square matrices.
    """
    U = _qssf_array(U, 2)
    V = _qssf_array(V, 2)
    if U.shape[0] != U.shape[1] or V.shape != U.shape:
        raise ValueError("U and V must be matching square matrices")
    return U, V


def _oracle_qssf_dispersion_propagator(
    w: "np.ndarray",
    dz: float,
    d2: float = 1.0,
    beta3: float = 0.08,
    A0: float = 1.5,
    gamma: float = 1.0,
) -> "np.ndarray":
    w = _qssf_array(w, 1, real=True)
    dz = _qssf_scalar(dz, nonnegative=True)
    d2 = _qssf_scalar(d2, positive=True)
    beta3 = _qssf_scalar(beta3)
    A0 = _qssf_scalar(A0, positive=True)
    gamma = _qssf_scalar(gamma, positive=True)
    if beta3 == 0:
        raise ValueError("This window rule requires a nonzero cubic term")
    D = 0.5 * d2 * w**2 + beta3 * w**3
    P = np.exp(0.5j * D * dz)
    coeffs = [beta3, 0.5 * d2, 0.0, 0.5 * gamma * A0**2]
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if np.abs(r.imag) < 1e-10]
    if len(real_roots) != 1 or not np.isfinite(D).all():
        raise ValueError(
            "The real band center must be numerically representable"
        )
    w_RR = float(real_roots[0])
    out = np.empty((3, len(w)), dtype=complex)
    out[0] = P
    out[1] = D
    out[2] = w_RR
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:
    """
    Return normal, boundary and edge-case specifications.
    """
    return [
        {
            "setup": (
                "import numpy as np\n\nw = np.array([-1.0, 0.0, 2.0])\n"
            ),
            "call": ("qssf_dispersion_propagator(w.copy(), 0.0, beta3=-0.08)"),
            "gold_call": (
                "_oracle_qssf_dispersion_propagator(\n"
                "    w.copy(), 0.0, beta3=-0.08\n"
                ")"
            ),
            "tol": 1e-10,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "\n"
                "def invalid(fn):\n"
                "    try:\n"
                "        fn(np.array([0.0, 1.0]), 0.01, beta3=0.0)\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "    return 0\n"
            ),
            "call": ("invalid(qssf_dispersion_propagator)"),
            "gold_call": ("invalid(_oracle_qssf_dispersion_propagator)"),
            "tol": 0.0,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "Nt = 128\n"
                "dt = 40.0 / Nt\n"
                "w = 2.0 * np.pi * np.fft.fftfreq(Nt, d=dt)\n"
                "dz = 0.01\n"
            ),
            "call": (
                "qssf_dispersion_propagator(\n"
                "    w, dz, d2=1.0, beta3=0.08, A0=1.5, gamma=1.0\n"
                ")"
            ),
            "gold_call": (
                "_oracle_qssf_dispersion_propagator(\n"
                "    w, dz, d2=1.0, beta3=0.08, A0=1.5, gamma=1.0\n"
                ")"
            ),
            "tol": 1e-10,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "Nt = 64\n"
                "dt = 30.0 / Nt\n"
                "w = 2.0 * np.pi * np.fft.fftfreq(Nt, d=dt)\n"
                "dz = 0.005\n"
            ),
            "call": (
                "qssf_dispersion_propagator(\n"
                "    w, dz, d2=1.2, beta3=0.10, A0=2.0, gamma=1.5\n"
                ")"
            ),
            "gold_call": (
                "_oracle_qssf_dispersion_propagator(\n"
                "    w, dz, d2=1.2, beta3=0.10, A0=2.0, gamma=1.5\n"
                ")"
            ),
            "tol": 1e-10,
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "Nt = 256\n"
                "dt = 50.0 / Nt\n"
                "w = 2.0 * np.pi * np.fft.fftfreq(Nt, d=dt)\n"
                "dz = 0.02\n"
            ),
            "call": (
                "qssf_dispersion_propagator(\n"
                "    w, dz, d2=0.8, beta3=0.05, A0=1.0, gamma=0.8\n"
                ")"
            ),
            "gold_call": (
                "_oracle_qssf_dispersion_propagator(\n"
                "    w, dz, d2=0.8, beta3=0.05, A0=1.0, gamma=0.8\n"
                ")"
            ),
            "tol": 1e-10,
        },
    ]
