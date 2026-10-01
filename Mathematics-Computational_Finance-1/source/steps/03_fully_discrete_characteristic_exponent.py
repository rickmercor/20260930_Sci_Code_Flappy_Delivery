"""
Turn the nodal Riccati values into the exponent of the implemented characteristic function at that level. The exponent is the deterministic drift contribution plus the time integral of a combination of the Riccati solution and its own right-hand side, taken on the same grid the solution lives on. The maturity is recovered from the grid rather than passed separately, so this step cannot silently disagree with the solver about the interval.

There are two equivalent ways to write the rough Heston characteristic function, and they stop being equivalent the moment either is discretised. The textbook form adds a plain integral of the Riccati solution to a fractional integral of it, which needs a second, different quadrature with a singular kernel. The form used here exploits the identity that a fractional integral of order one minus the order, composed with one of the order itself, is an ordinary integral, and so replaces the singular quadrature with a single smooth one applied to a combination of the solution and its right-hand side.

That substitution is the reason this step exists. It removes a source of error whose convergence order is hard to control and leaves one quadrature whose order is the same as the solver's, which is what makes the level differences downstream behave like a clean power of the step size. The price is that the combination being integrated is not the Riccati solution itself, and a solver that integrates the solution alone is computing something else.

The quadrature rule is the composite trapezoid, endpoints at half weight. That detail is not cosmetic here: the integrand does not vanish at either end, and dropping the half weights biases the exponent by a term of the order of the step size.

Returns
-------
np.ndarray of shape (2), holding the real and imaginary parts of the exponent. With a vanishing Riccati solution and no variance the exponent reduces to the drift term alone; with a zero Fourier argument and a zero solution it is exactly zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def characteristic_exponent(h_packed, xi_re, xi_im, dt, X0, r, gam, nu, rho,
                            theta, V0):
    """Fully discrete exponent of the rough Heston characteristic function.

    Takes the nodal Riccati values on the grid t_j = j*dt, j = 0..M, produced
    by the previous step, and returns the exponent of the implemented
    characteristic function at that discretization level, so that
    Phi_l(xi) = exp(G_l(xi)).  The maturity is T = M*dt.  The exponent combines
    the deterministic drift contribution with the time integral of

        J(xi, z) = theta*gam*z + V0*F(xi, z),

    where F is the Riccati right-hand side of the previous step, evaluated on
    the same grid and integrated by the quadrature rule the source paper uses
    for the fully discrete exponent.

    Args:
        h_packed: (2*(M+1),) array [Re h_0..Re h_M, Im h_0..Im h_M].
        xi_re, xi_im (float): real and imaginary parts of xi.
        dt (float): time-step size.
        X0 (float): initial log price.
        r (float): risk-free rate.
        gam, nu, rho, theta, V0 (float): rough Heston parameters.

    Expected return:
        np.ndarray of shape (2,) holding [Re G_l(xi), Im G_l(xi)].
        Raises:
        ValueError: if h_packed has odd length or holds fewer than two
        nodes, or if dt <= 0.
    """
    return np.zeros(2)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_characteristic_exponent(h_packed, xi_re, xi_im, dt, X0, r, gam, nu,
                                    rho, theta, V0):
    def _rhs(xi, h, gam, nu, rho):
        return (-0.5 * (xi * xi + 1j * xi)
                + gam * (1j * xi * rho * nu - 1.0) * h
                + 0.5 * (gam * nu) ** 2 * h * h)

    def _trapezoid(v, dt):
        return dt * (0.5 * v[0] + v[1:-1].sum() + 0.5 * v[-1])

    v = np.asarray(h_packed, dtype=float).reshape(-1)
    if v.size % 2 != 0 or v.size < 4:
        raise ValueError("h_packed must hold at least two nodes in [Re; Im] form")
    n = v.size // 2
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")

    h = v[:n] + 1j * v[n:]
    dt = float(dt)
    T = (n - 1) * dt
    xi = float(xi_re) + 1j * float(xi_im)
    J = float(theta) * float(gam) * h + float(V0) * _rhs(
        xi, h, float(gam), float(nu), float(rho))
    G = 1j * xi * (float(X0) + float(r) * T) + _trapezoid(J, dt)
    return np.array([G.real, G.imag])

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    return [
        {
            "setup": (
                "h = np.zeros(10)\n"
                "G = characteristic_exponent(h, 0.0, 0.0, 0.25, 6.9077552789821, "
                "0.0, 0.1, 0.331, -0.681, 0.3156, 0.0392)"
            ),
            "call": "G",
            "gold_call": "np.zeros(2)",
        },
        {
            "setup": (
                "h = np.zeros(10)\n"
                "G = characteristic_exponent(h, 1.0, 0.0, 0.25, 2.0, 0.05, "
                "0.1, 0.331, -0.681, 0.3156, 0.0)"
            ),
            "call": "G",
            "gold_call": "np.array([0.0, 2.05])",
        },
        {
            "setup": (
                "n = 5\n"
                "h = np.concatenate((np.full(n, 0.5), np.zeros(n)))\n"
                "G = characteristic_exponent(h, 0.0, 0.0, 0.25, 0.0, 0.0, "
                "0.2, 0.5, -0.5, 3.0, 0.0)"
            ),
            "call": "np.round(G, 12)",
            "gold_call": "np.array([0.3, 0.0])",
        },
        {
            "setup": (
                "v = np.array([0.0, 0.1, 0.25, 0.4, 0.6, "
                "0.0, -0.05, -0.12, -0.2, -0.3])\n"
                "G = characteristic_exponent(v, 1.5, -2.0, 0.2, 4.0, 0.01, "
                "0.9, 0.35, -0.6, 0.25, 0.04)"
            ),
            "call": "np.round(G, 10)",
            "gold_call": "np.array([8.0474587171, 6.063372351])",
        },
        {
            "setup": (
                "v = np.array([0.0, 0.1, 0.25, 0.4, 0.6, "
                "0.0, -0.05, -0.12, -0.2, -0.3])\n"
                "Ga = characteristic_exponent(v, 1.5, -2.0, 0.2, 4.0, 0.0, "
                "0.9, 0.35, -0.6, 0.25, 0.04)\n"
                "Gb = characteristic_exponent(v, 1.5, -2.0, 0.2, 0.0, 0.0, "
                "0.9, 0.35, -0.6, 0.25, 0.04)\n"
                "shift = Ga - Gb"
            ),
            "call": "np.round(shift, 12)",
            "gold_call": "np.array([8.0, 6.0])",
        },
        {
            "setup": (
                "def run_model():\n"
                "    try:\n"
                "        characteristic_exponent(np.zeros(3), 1.0, 0.0, 0.25, "
                "0.0, 0.0, 0.1, 0.331, -0.681, 0.3156, 0.0392)\n"
                "        return 0\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    except Exception:\n"
                "        return 2\n"
                "def run_gold():\n"
                "    try:\n"
                "        _oracle_characteristic_exponent(np.zeros(3), 1.0, 0.0, "
                "0.25, 0.0, 0.0, 0.1, 0.331, -0.681, 0.3156, 0.0392)\n"
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
