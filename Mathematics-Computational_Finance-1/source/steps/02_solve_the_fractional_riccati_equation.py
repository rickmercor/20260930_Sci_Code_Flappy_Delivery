"""
Advance the rough Heston Riccati solution at one complex Fourier argument across the whole grid, consuming the weight arrays from the previous step. Each step predicts a value from the history with the predictor row, evaluates the Riccati right-hand side there, and corrects with the corrector row. The solution is complex but is returned as one flat real array, real parts first, because a step may not return a tuple.

The rough Heston characteristic function is exponential-affine in a function that solves a fractional Riccati equation, and the fractional order is the Hurst exponent shifted by a half. Below one half the volatility path is rougher than Brownian motion, which is what the implied-volatility skew at short maturities calls for, and it is also what removes the closed-form solution that the classical Heston model enjoys.

The right-hand side is quadratic in the unknown, and the correlation enters through the linear coefficient rather than the quadratic one. That placement is easy to get wrong from memory of the classical model, and nothing downstream detects it except the numbers themselves.

Two structural facts are worth using as checks. At a zero Fourier argument the forcing term vanishes and the solution never leaves zero, so the characteristic function collapses to its drift factor. And along the damped contour the magnitude of the solution grows monotonically over the interval, so a nodal sequence that turns around has usually lost the history sum.

Returns
-------
np.ndarray of shape (2*(M+1),) packed as [Re h_0..Re h_M, Im h_0..Im h_M], with h_0 = 0 exactly. At xi = 0 the whole array is zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_riccati_nodal(w_packed, xi_re, xi_im, alpha, dt, M, gam, nu, rho):
    """Nodal values of the rough Heston fractional Riccati solution.

    Solves the Volterra form of the fractional Riccati equation for one
    complex Fourier argument xi = xi_re + 1j*xi_im on the uniform grid
    t_j = j*dt, j = 0..M, by the fractional Adams predictor-corrector scheme,
    reusing the convolution weights produced by the previous step.  The
    Riccati right-hand side is

        F(xi, h) = -0.5*(xi**2 + 1j*xi) + gam*(1j*xi*rho*nu - 1)*h
                   + 0.5*(gam*nu)**2 * h**2,

    the solution starts from h(xi, 0) = 0, and both the predicted and the
    corrected values carry the reciprocal of the Gamma function of the
    fractional order.

    Args:
        w_packed: (2*(M+1)**2,) packed corrector and predictor weights for
            this alpha, dt and M, laid out as in the previous step.
        xi_re, xi_im (float): real and imaginary parts of xi.
        alpha (float): fractional order H + 1/2, in (1/2, 1).
        dt (float): time-step size.
        M (int): number of steps.
        gam, nu, rho (float): mean-reversion speed, volatility of volatility
            and correlation of the rough Heston model.

    Expected return:
        np.ndarray of shape (2*(M+1),) packed as
        [Re h_0, ..., Re h_M, Im h_0, ..., Im h_M].
        Raises:
        ValueError: if M < 1, if alpha is outside (0, 1), or if
        w_packed does not have length 2*(M+1)**2.
    """
    return np.zeros(2 * (int(M) + 1))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_riccati_nodal(w_packed, xi_re, xi_im, alpha, dt, M, gam, nu,
                                rho):
    def _rhs(xi, h, gam, nu, rho):
        return (-0.5 * (xi * xi + 1j * xi)
                + gam * (1j * xi * rho * nu - 1.0) * h
                + 0.5 * (gam * nu) ** 2 * h * h)

    def _gamma(x):
        c = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
             -176.61502916214059, 12.507343278686905, -0.13857109526572012,
             9.9843695780195716e-6, 1.5056327351493116e-7]
        if x < 0.5:
            return np.pi / (np.sin(np.pi * x) * _gamma(1.0 - x))
        x -= 1.0
        a = 0.99999999999980993
        t = x + 7.5
        for i, ci in enumerate(c):
            a += ci / (x + i + 1.0)
        return np.sqrt(2.0 * np.pi) * t ** (x + 0.5) * np.exp(-t) * a

    M = int(M)
    if M < 1:
        raise ValueError("M must be at least 1")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    w = np.asarray(w_packed, dtype=float).reshape(-1)
    if w.size != 2 * (M + 1) ** 2:
        raise ValueError("w_packed does not match M")

    n = M + 1
    a = w[:n * n].reshape(n, n)
    b = w[n * n:].reshape(n, n)
    gam, nu, rho = float(gam), float(nu), float(rho)
    xi = float(xi_re) + 1j * float(xi_im)
    ga = _gamma(float(alpha))

    h = np.zeros(n, dtype=complex)
    Fv = np.zeros(n, dtype=complex)
    Fv[0] = _rhs(xi, 0.0 + 0.0j, gam, nu, rho)
    for jp in range(1, n):
        j = jp - 1
        hp = (b[jp, 0:j + 1] @ Fv[0:j + 1]) / ga
        h[jp] = (a[jp, 0:j + 1] @ Fv[0:j + 1]
                 + a[jp, jp] * _rhs(xi, hp, gam, nu, rho)) / ga
        Fv[jp] = _rhs(xi, h[jp], gam, nu, rho)
    return np.concatenate((h.real, h.imag))

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

W_A = ("w = np.array([0.0, 0.0, 0.0, 0.26134114579090345, 0.4215179770821023, "
       "0.0, 0.17534618045330164, 0.45260508072421984, 0.4215179770821023, "
       "0.0, 0.0, 0.0, 0.6828591228730058, 0.0, 0.0, 0.3666101153866181, "
       "0.6828591228730058, 0.0])\n")

W_B = ("w = np.array([0.0, 0.0, 0.0, 0.0, 0.19066548196385444, "
       "0.2723792599483635, 0.0, 0.0, 0.13963483598188572, "
       "0.34020430654645795, 0.2723792599483635, 0.0, 0.12094715624254751, "
       "0.26556669214038753, 0.34020430654645795, 0.2723792599483635, 0.0, "
       "0.0, 0.0, 0.0, 0.46304474191221795, 0.0, 0.0, 0.0, "
       "0.2891736605644893, 0.46304474191221795, 0.0, 0.0, "
       "0.24687901240104937, 0.2891736605644893, 0.46304474191221795, 0.0])\n")


def test_cases():
    return [
        {
            "setup": (
                W_A
                + "v = solve_riccati_nodal(w, 1.0, 0.0, 0.62, 0.25, 2, "
                  "0.1, 0.331, -0.681)\n"
                  "h = v[:3] + 1j * v[3:]"
            ),
            "call": "np.array([h[1].real, h[1].imag, h[2].real, h[2].imag])",
            "gold_call": ("np.array([-0.23093856035838603, -0.22781355819030943, "
                          "-0.3494584824877605, -0.3417672090218595])"),
        },
        {
            "setup": (
                W_A
                + "v = solve_riccati_nodal(w, 0.0, 0.0, 0.62, 0.25, 2, "
                  "0.1, 0.331, -0.681)"
            ),
            "call": "v",
            "gold_call": "np.zeros(6)",
        },
        {
            "setup": (
                W_B
                + "v = solve_riccati_nodal(w, 0.5, -2.0, 0.7, 0.2, 3, "
                  "1.5, 0.4, -0.5)\n"
                  "h = v[:4] + 1j * v[4:]"
            ),
            "call": "np.array([h[1].real, h[1].imag, h[3].real, h[3].imag])",
            "gold_call": ("np.array([0.18398609893544327, 0.14613137495091738, "
                          "0.29797725756084065, 0.23856847278059493])"),
        },
        {
            "setup": (
                W_A
                + "v = solve_riccati_nodal(w, 1.0, 0.0, 0.62, 0.25, 2, "
                  "0.1, 0.331, -0.681)\n"
                  "h = v[:3] + 1j * v[3:]\n"
                  "grow = bool(abs(h[2]) > abs(h[1]) > 0.0)"
            ),
            "call": "bool(grow and h[0] == 0.0 and np.all(np.isfinite(v)))",
            "gold_call": "True",
        },
        {
            "setup": (
                W_A
                + "v = solve_riccati_nodal(w, 3.0, 0.0, 0.62, 0.25, 2, "
                  "0.1, 0.0, -0.681)\n"
                  "h = v[:3] + 1j * v[3:]\n"
                  "ratio = h[1] / (-0.5 * (9.0 + 3.0j))"
            ),
            "call": "round(float(abs(ratio.imag)), 12)",
            "gold_call": "0.0",
        },
        {
            "setup": (
                W_A
                + "def run_model():\n"
                  "    try:\n"
                  "        solve_riccati_nodal(w, 1.0, 0.0, 0.62, 0.25, 6, "
                  "0.1, 0.331, -0.681)\n"
                  "        return 0\n"
                  "    except ValueError:\n"
                  "        return 1\n"
                  "    except Exception:\n"
                  "        return 2\n"
                  "def run_gold():\n"
                  "    try:\n"
                  "        _oracle_solve_riccati_nodal(w, 1.0, 0.0, 0.62, 0.25, "
                  "6, 0.1, 0.331, -0.681)\n"
                  "        return 0\n"
                  "    except ValueError:\n"
                  "        return 1\n"
                  "    except Exception:\n"
                  "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": (
                W_A
                + "def run_model():\n"
                  "    try:\n"
                  "        solve_riccati_nodal(w, 1.0, 0.0, 1.5, 0.25, 2, "
                  "0.1, 0.331, -0.681)\n"
                  "        return 0\n"
                  "    except ValueError:\n"
                  "        return 1\n"
                  "    except Exception:\n"
                  "        return 2\n"
                  "def run_gold():\n"
                  "    try:\n"
                  "        _oracle_solve_riccati_nodal(w, 1.0, 0.0, 1.5, 0.25, "
                  "2, 0.1, 0.331, -0.681)\n"
                  "        return 0\n"
                  "    except ValueError:\n"
                  "        return 1\n"
                  "    except Exception:\n"
                  "        return 2"
            ),
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
