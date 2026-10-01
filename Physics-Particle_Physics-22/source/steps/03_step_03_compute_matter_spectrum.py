"""
Compute matter eigenvalue estimates together with reusable cofactor coefficients.

In a real propagation basis, write $d=\Delta m_{31}^2-s_{12}^2\Delta m_{21}^2$.

The electron row of $\operatorname{adj}(\lambda I-K)$ consists of

$\lambda^2-S_{ee}\lambda+T_{ee}$, $-S_{e\mu}\lambda+T_{e\mu}$,

and $-S_{e\tau}\lambda+T_{e\tau}$.

These coefficients do not depend on the matter potential, so they are shared by every layer.



$$

S_{ee}=\Delta m_{21}^2+dc_{13}^2,\quad

S_{e\mu}=-\Delta m_{21}^2c_{13}s_{12}c_{12},\quad

S_{e\tau}=-ds_{13}c_{13},

$$



$$

T_{ee}=\Delta m_{31}^2\Delta m_{21}^2c_{13}^2c_{12}^2,\quad

T_{e\mu}=\Delta m_{31}^2S_{e\mu},\quad

T_{e\tau}=-\Delta m_{31}^2\Delta m_{21}^2s_{13}c_{13}c_{12}^2.

$$



The characteristic polynomial of the real mass-squared Hamiltonian is

$\chi(\lambda)=\lambda^3-A\lambda^2+B\lambda-C$.

Its coefficients are $A=\Delta m_{21}^2+\Delta m_{31}^2+a$,

$B=\Delta m_{21}^2\Delta m_{31}^2+aS_{ee}$, and $C=aT_{ee}$.

The atmospheric branch starts at



$$

\lambda_3^{(0)}=\Delta m_{31}^2+

\frac{d}{2}\left[x-1+\sqrt{(1-x)^2+4s_{13}^2x}\right],\qquad

x=a/d,\quad d=\Delta m_{31}^2-s_{12}^2\Delta m_{21}^2.

$$



Apply the requested number of corrections $\lambda_3\leftarrow\lambda_3-\chi(\lambda_3)/\chi'(\lambda_3)$.

Recover $\lambda_{1,2}=(A-\lambda_3\mp\sqrt{(A-\lambda_3)^2-4C/\lambda_3})/2$.

At $a=0$ the result is exactly $(0,\Delta m_{21}^2,\Delta m_{31}^2)$.

Retain this branch labeling; supported outputs must be strictly increasing.

Returns
-------
A real array of shape (3, 3) containing increasing eigenvalues followed by the two electron-row cofactor coefficient rows.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_matter_spectrum(
    matter_ev2: float,
    s12_sq: float,
    s13_sq: float,
    dm21_ev2: float,
    dm31_ev2: float,
    n_newton: int = 2,
) -> "np.ndarray":
    r"""Compute matter eigenvalue estimates together with reusable cofactor coefficients.

    Parameters
    ----------
    matter_ev2 : float
        Finite signed matter potential in eV squared.
    s12_sq, s13_sq : float
        Squared sines strictly between zero and one.
    dm21_ev2, dm31_ev2 : float
        Finite mass-squared differences with 0 < dm21_ev2 < dm31_ev2.
    n_newton : int, default 2
        Number of cubic corrections, between zero and eight inclusive.

    Returns
    -------
    spectrum : np.ndarray
        Real shape (3, 3) array. Row zero contains increasing eigenvalues in
        eV squared; rows one and two contain the electron-row S coefficients
        in eV squared and T coefficients in eV to the fourth, respectively.

    Raises
    ------
    ValueError
        If inputs violate their domains, a correction has a zero denominator,
        the recovery radicand is materially negative, or recovered roots are
        nonfinite or not strictly increasing. Negative roundoff up to 64
        machine epsilons times the radicand term scale is clipped to zero.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validate_mixing(s12_sq, s13_sq, dm21_ev2, dm31_ev2):
    s12 = _finite_scalar(s12_sq, "s12_sq")
    s13 = _finite_scalar(s13_sq, "s13_sq")
    m21 = _finite_scalar(dm21_ev2, "dm21_ev2")
    m31 = _finite_scalar(dm31_ev2, "dm31_ev2")
    if not 0 < s12 < 1 or not 0 < s13 < 1 or not 0 < m21 < m31:
        raise ValueError(
            "mixing angles or normal-ordering masses are outside the domain"
        )
    return s12, s13, m21, m31


def _spectral_coefficients(
    s12_sq: float, s13_sq: float, dm21_ev2: float, dm31_ev2: float
) -> "np.ndarray":
    s12, s13, m21, m31 = _validate_mixing(s12_sq, s13_sq, dm21_ev2, dm31_ev2)
    c12 = 1.0 - s12
    c13 = 1.0 - s13
    dmee = m31 - s12 * m21
    cross13 = np.sqrt(s13 * c13)
    see = m21 + dmee * c13
    sem = -m21 * np.sqrt(c13 * s12 * c12)
    set_ = -dmee * cross13
    tee = m31 * m21 * c13 * c12
    tem = m31 * sem
    tet = -m31 * m21 * cross13 * c12
    return np.array([[see, sem, set_], [tee, tem, tet]])


def _oracle_compute_matter_spectrum(
    matter_ev2: float,
    s12_sq: float,
    s13_sq: float,
    dm21_ev2: float,
    dm31_ev2: float,
    n_newton: int = 2,
) -> "np.ndarray":
    a = _finite_scalar(matter_ev2, "matter_ev2")
    s12, s13, m21, m31 = _validate_mixing(s12_sq, s13_sq, dm21_ev2, dm31_ev2)
    if (
        isinstance(n_newton, (bool, np.bool_))
        or not isinstance(n_newton, (int, np.integer))
        or not 0 <= n_newton <= 8
    ):
        raise ValueError("n_newton must be an integer between zero and eight")
    coefficients = _spectral_coefficients(s12, s13, m21, m31)
    if a == 0.0:
        return np.vstack((np.array([0.0, m21, m31]), coefficients))
    aa = m21 + m31 + a
    bb = m21 * m31 + a * coefficients[0, 0]
    cc = a * coefficients[1, 0]
    dmee = m31 - s12 * m21
    x = a / dmee
    l3 = m31 + 0.5 * dmee * (x - 1.0 + np.sqrt((1.0 - x) ** 2 + 4.0 * s13 * x))
    for _ in range(n_newton):
        derivative = l3 * (3.0 * l3 - 2.0 * aa) + bb
        if derivative == 0.0:
            raise ValueError("zero characteristic derivative")
        l3 = (l3 * l3 * (2.0 * l3 - aa) + cc) / derivative
    if l3 == 0.0 or not np.isfinite(l3):
        raise ValueError("atmospheric eigenvalue cannot be recovered")
    first = (aa - l3) ** 2
    second = 4.0 * cc / l3
    radicand = first - second
    scale = max(abs(first), abs(second), np.finfo(float).tiny)
    if radicand < -64.0 * np.finfo(float).eps * scale:
        raise ValueError("negative eigenvalue recovery radicand")
    gap = np.sqrt(max(radicand, 0.0))
    l2 = 0.5 * (aa - l3 + gap)
    roots = np.array([l2 - gap, l2, l3])
    if not np.all(np.isfinite(roots)) or np.any(np.diff(roots) <= 0.0):
        raise ValueError("eigenvalues must be finite and strictly increasing")
    return np.vstack((roots, coefficients))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative numerical and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\n\n",
            "call": "compute_matter_spectrum(0.00204, 0.307, 0.02195, 7.49e-5, 2.534e-3, 0)",
            "gold_call": "_oracle_compute_matter_spectrum(0.00204, 0.307, 0.02195, 7.49e-5, 2.534e-3, 0)",
            "tol": 1e-11,
        },
        {
            "setup": "import numpy as np\n\n",
            "call": "compute_matter_spectrum(0.0, 0.307, 0.02195, 7.49e-5, 2.534e-3)",
            "gold_call": "_oracle_compute_matter_spectrum(0.0, 0.307, 0.02195, 7.49e-5, 2.534e-3)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\n\n",
            "call": "compute_matter_spectrum(-0.0047, 0.307, 0.02195, 7.49e-5, 2.534e-3, 0)",
            "gold_call": "_oracle_compute_matter_spectrum(-0.0047, 0.307, 0.02195, 7.49e-5, 2.534e-3, 0)",
            "tol": 1e-11,
        },
        {
            "setup": "import numpy as np\n\n",
            "call": "compute_matter_spectrum(0.0025110057, 0.307, 0.02195, 7.49e-5, 2.534e-3, 0)",
            "gold_call": "_oracle_compute_matter_spectrum(0.0025110057, 0.307, 0.02195, 7.49e-5, 2.534e-3, 0)",
            "tol": 1e-11,
        },
        {
            "setup": "import numpy as np\n\ndef _raises_value_error(function):\n    try:\n        function()\n    except ValueError:\n        return 1\n    return 0\n",
            "call": "_raises_value_error(lambda: compute_matter_spectrum(0.002, 0.307, 0.02195, 7.49e-5, 2.534e-3, -1))",
            "gold_call": "_raises_value_error(lambda: _oracle_compute_matter_spectrum(0.002, 0.307, 0.02195, 7.49e-5, 2.534e-3, -1))",
            "tol": 0.0,
        },
    ]
