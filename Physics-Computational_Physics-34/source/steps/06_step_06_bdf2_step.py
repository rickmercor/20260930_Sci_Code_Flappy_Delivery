"""
Advance the scheme by one step: given $\\phi^{n-1}$, $\\phi^{n}$, $\\eta^{n}$, the step ratio $\\gamma_{n+1}$ and the step $\\tau_{n+1}$, return the pair $(\\phi^{n+1},\\eta^{n+1})$. This is steps 4 and 5 composed with the reconstruction $\\phi^{n+1}=p^{n+1}+S(\\eta^{n+1})\\,\\tau_{n+1}\\,q^{n+1}$, using $S(\\eta)=\\eta(2-\\eta)$. Passing $\\gamma=0$ (with $\\phi^{n-1}$ ignored) must give the first-order backward-Euler starting step, so the whole run is this one function called once with $\\gamma_1=0$ and then repeatedly with the controller's ratio. The step must conserve the spatial mean of $\\phi$ exactly, to round-off, for every $\\gamma$, $\\tau$ and $\\theta$. The multiplier solve of step 5 is the only stage that can fail, and its failure is **propagated, not absorbed**: if Newton cannot deliver a root the step reports the breakdown instead of returning a state built from an unconverged iterate.

The complete time-stepping procedure is: solve the two decoupled constant-coefficient linear elliptic problems for $p^{n+1}$ and $q^{n+1}$; solve the scalar penalised nonlinear equation for $\\eta^{n+1}$ by Newton's iteration; and recombine linearly. Because the two linear solves are diagonal in Fourier space and the third stage is scalar, the cost of a step is a handful of transforms - which is the point of the decoupling. Mass conservation is structural rather than enforced: the right-hand side of the evolution equation is a Laplacian, so its zero Fourier mode vanishes; the zero mode of the implicit symbol is the bare BDF2 coefficient $(1+2\\gamma)/(1+\\gamma)$; and the zero mode of the explicit right-hand side is $\\big[(1+\\gamma)-\\frac{\\gamma^2}{1+\\gamma}\\big]\\overline{\\phi}=\\frac{1+2\\gamma}{1+\\gamma}\\overline{\\phi}$, so the mean passes through unchanged.




 **Formulas**




With $(\\bar\\phi^{\\,n+1},p^{n+1},q^{n+1})$ from step 4 at $(\\gamma,\\tau)$ and $\\eta^{n+1}$ from step 5,




$$\\phi^{n+1}\\;=\\;p^{n+1}\\;+\\;S(\\eta^{n+1})\\,\\tau\\,q^{n+1}, \\qquad S(\\eta)=\\eta(2-\\eta)=1-(1-\\eta)^{2}.$$




The step satisfies, to round-off, the discrete energy law




$$\\int_\\Omega\\!\\big[F(\\phi^{n+1})-F(\\phi^{n})\\big]+\\theta\\big((\\eta^{n+1})^{2}-(\\eta^{n})^{2}\\big)=S(\\eta^{n+1})\\int_\\Omega\\! F'(\\bar\\phi^{\\,n+1})\\big(\\phi^{n+1}-\\phi^{n}\\big),$$




which is the practical check that the multiplier and the reconstruction agree.

Returns
-------
A `tuple` `(phi_next, eta_next)`: a real `np.ndarray` with the shape of `phi_n`, and a Python `float`. The pair satisfies the discrete energy law of the Formulas section to round-off, which is the check to apply to it. If the step-5 multiplier solve cannot deliver a root, the function raises `RuntimeError` and returns nothing.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def pfc_bdf2_step(phi_n: "np.ndarray", phi_nm1: "np.ndarray", eta_n: float,
                  gamma: float, tau: float, L: "float | Sequence[float]",
                  alpha: float, A: float,
                  theta: float) -> "tuple[np.ndarray, float]":
    """phi_n, phi_nm1: real 2-D arrays of the same shape.
    eta_n: float, the previous Lagrange multiplier.
    gamma: float >= 0, the step ratio (0.0 for the backward-Euler start).
    tau: float > 0, the step tau_{n+1}.  L: float or length-2 sequence.
    alpha, A, theta: floats.
    Return the tuple (phi_next, eta_next): the new state as a real array with
    the shape of phi_n, and the new multiplier as a float.
    Raise RuntimeError if the step-5 multiplier solve breaks down - the Newton
    cap is exhausted, the derivative vanishes, or a residual, derivative or
    iterate becomes non-finite. The failure propagates out of this function; do
    not substitute the last iterate, eta_n, or any fallback value."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pfc_bdf2_step(phi_n: "np.ndarray", phi_nm1: "np.ndarray", eta_n: float,
                          gamma: float, tau: float, L: "float | Sequence[float]", alpha: float,
                          A: float, theta: float) -> "tuple[np.ndarray, float]":
    phi_bar, p, q = _oracle_bdf2_decoupled_fields(phi_n, phi_nm1, gamma, tau,
                                                  L, alpha, A)
    eta = _oracle_solve_multiplier(p, q, np.asarray(phi_n, float), phi_bar,
                                   tau, theta, eta_n, A, L)
    return p + _S(eta) * float(tau) * q, eta

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the backward-Euler starting step of the benchmark run, taken from
        # the seed field itself with eta^0 = 1 and tau_1 = tau_min.
        {"setup": "import numpy as np\nN = 64\nL = 32.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\ns = u[:, None]\nt = u[None, :]\n"
                  "phi0 = (-0.27 + 0.06 * np.cos(16 * s + 0.3) * np.cos(9 * t)\n"
                  "        + 0.05 * np.sin(11 * s) * np.cos(13 * t + 0.7)\n"
                  "        + 0.04 * np.cos(7 * s - 14 * t + 1.1)\n"
                  "        + 0.03 * np.sin(19 * s + 5 * t) * np.sin(6 * t)\n"
                  "        + 0.02 * np.cos(23 * s) * np.sin(21 * t + 0.4))\n"
                  "def pack(r):\n"
                  "    return np.concatenate([np.asarray(r[0], float).ravel(),"
                  " np.asarray([r[1]], float)])\n",
         "call": "pack(pfc_bdf2_step(phi0, phi0, 1.0, 0.0, 1e-4, L, 0.75, 0.37, 5.0))",
         "gold_call": "pack(_oracle_pfc_bdf2_step(phi0, phi0, 1.0, 0.0, 1e-4, L, 0.75, "
                      "0.37, 5.0))"},
        # normal: a mid-run BDF2 step with a ratio near one and a large step.
        {"setup": "import numpy as np\nN = 32\nL = 16.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.1 * np.cos(8 * u)[:, None] * np.sin(5 * u)[None, :]\n"
                  "pm = -0.27 + 0.08 * np.cos(8 * u)[:, None] * np.sin(5 * u)[None, :]\n"
                  "def pack(r):\n"
                  "    return np.concatenate([np.asarray(r[0], float).ravel(),"
                  " np.asarray([r[1]], float)])\n",
         "call": "pack(pfc_bdf2_step(pn, pm, 0.94, 1.02, 0.7, L, 0.75, 0.37, 5.0))",
         "gold_call": "pack(_oracle_pfc_bdf2_step(pn, pm, 0.94, 1.02, 0.7, L, 0.75, 0.37, "
                      "5.0))"},
        # boundary: gamma at the maximal admissible ratio, where the extrapolation
        # reaches far outside the interval and the multiplier is worked hardest.
        {"setup": "import numpy as np\nN = 32\nL = 16.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.1 * np.cos(8 * u)[:, None] * np.sin(5 * u)[None, :]\n"
                  "pm = -0.27 + 0.06 * np.cos(8 * u)[:, None] * np.sin(5 * u)[None, :]\n"
                  "def pack(r):\n"
                  "    return np.concatenate([np.asarray(r[0], float).ravel(),"
                  " np.asarray([r[1]], float)])\n",
         "call": "pack(pfc_bdf2_step(pn, pm, 1.0, 4.86454, 0.02, L, 0.75, 0.37, 5.0))",
         "gold_call": "pack(_oracle_pfc_bdf2_step(pn, pm, 1.0, 4.86454, 0.02, L, 0.75, "
                      "0.37, 5.0))"},
        # boundary: the weak-penalty regime theta = 0.5 at a large step, where the
        # multiplier drifts furthest from one and therefore actually changes the
        # reconstructed state.
        {"setup": "import numpy as np\nN = 24\nL = 12.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.15 * np.cos(4 * u)[:, None] * np.cos(3 * u)[None, :]\n"
                  "pm = -0.27 + 0.10 * np.cos(4 * u)[:, None] * np.cos(3 * u)[None, :]\n"
                  "def pack(r):\n"
                  "    return np.concatenate([np.asarray(r[0], float).ravel(),"
                  " np.asarray([r[1]], float)])\n",
         "call": "pack(pfc_bdf2_step(pn, pm, 0.88, 1.3, 0.9, L, 0.75, 0.37, 0.5))",
         "gold_call": "pack(_oracle_pfc_bdf2_step(pn, pm, 0.88, 1.3, 0.9, L, 0.75, 0.37, "
                      "0.5))"},
        # boundary: a RECTANGULAR grid on an anisotropic box at a different material
        # point, exercising the whole chain with per-direction wave numbers.
        {"setup": "import numpy as np\nNv = (24, 32)\nLv = (20.0 * np.pi, 32.0 * np.pi)\n"
                  "a = np.arange(Nv[0]) * (2 * np.pi / Nv[0])\n"
                  "b = np.arange(Nv[1]) * (2 * np.pi / Nv[1])\n"
                  "pn = -0.27 + 0.11 * np.cos(3 * a)[:, None] * np.sin(4 * b)[None, :]\n"
                  "pm = -0.27 + 0.09 * np.cos(3 * a)[:, None] * np.sin(4 * b)[None, :]\n"
                  "def pack(r):\n"
                  "    return np.concatenate([np.asarray(r[0], float).ravel(),"
                  " np.asarray([r[1]], float)])\n",
         "call": "pack(pfc_bdf2_step(pn, pm, 1.0, 1.5, 0.35, Lv, 0.6, 0.5, 20.0))",
         "gold_call": "pack(_oracle_pfc_bdf2_step(pn, pm, 1.0, 1.5, 0.35, Lv, 0.6, 0.5, "
                      "20.0))"},
        # edge: phi_n identical to phi_nm1, so the extrapolation is exactly phi_n
        # for every gamma and the ratio enters only through the coefficients.
        {"setup": "import numpy as np\nN = 16\nL = 8.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.2 * np.cos(2 * u)[:, None] * np.cos(u)[None, :]\n"
                  "def pack(r):\n"
                  "    return np.concatenate([np.asarray(r[0], float).ravel(),"
                  " np.asarray([r[1]], float)])\n",
         "call": "pack(pfc_bdf2_step(pn, pn, 1.0, 2.5, 0.1, L, 0.75, 0.37, 50.0))",
         "gold_call": "pack(_oracle_pfc_bdf2_step(pn, pn, 1.0, 2.5, 0.1, L, 0.75, 0.37, "
                      "50.0))"},
        # edge: an enormous step at gamma = 1, far beyond any accuracy requirement,
        # where the scheme must still be solvable and mass conserving - the regime
        # the penalty exists to protect.
        {"setup": "import numpy as np\nN = 16\nL = 8.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.2 * np.cos(2 * u)[:, None] * np.cos(u)[None, :]\n"
                  "pm = -0.27 + 0.19 * np.cos(2 * u)[:, None] * np.cos(u)[None, :]\n"
                  "def pack(r):\n"
                  "    return np.concatenate([np.asarray(r[0], float).ravel(),"
                  " np.asarray([r[1]], float)])\n",
         "call": "pack(pfc_bdf2_step(pn, pm, 1.0, 1.0, 100.0, L, 0.75, 0.37, 5.0))",
         "gold_call": "pack(_oracle_pfc_bdf2_step(pn, pm, 1.0, 1.0, 100.0, L, 0.75, 0.37, "
                      "5.0))"},
        # regression: an INDEPENDENT check on the returned pair, not a comparison
        # with the oracle. The setup rebuilds the discrete energy law of the
        # Formulas section from phi_n, phi_nm1, gamma and the parameters with plain
        # numpy, and reports 1.0 only if the returned (phi_next, eta_next) annihilate
        # it to 1e-10 and the spatial mean is conserved to 1e-12. A step built on an
        # unconverged multiplier, or on a reconstruction inconsistent with it, fails
        # here even though it returns an array of the right shape and a plausible
        # scalar. Reference residual 1.5e-15; reference mass drift exactly zero.
        {"setup": "import numpy as np\nN = 32\nL = 16.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.1 * np.cos(8 * u)[:, None] * np.sin(5 * u)[None, :]\n"
                  "pm = -0.27 + 0.08 * np.cos(8 * u)[:, None] * np.sin(5 * u)[None, :]\n"
                  "gam, tau, al, Aa, th, e0 = 1.02, 0.7, 0.75, 0.37, 5.0, 0.94\n"
                  "w0 = (L / N) ** 2\n"
                  "Fb = lambda x: 0.25 * x ** 4 - 0.5 * Aa * x ** 2\n"
                  "Fp = lambda x: x ** 3 - Aa * x\n"
                  "def law(res):\n"
                  "    phi, eta = res\n"
                  "    phi = np.asarray(phi, float)\n"
                  "    pb = pn + gam * (pn - pm)\n"
                  "    S = eta * (2.0 - eta)\n"
                  "    r = (w0 * np.sum(Fb(phi) - Fb(pn)) + th * (eta * eta - e0 * e0)\n"
                  "         - S * w0 * np.sum(Fp(pb) * (phi - pn)))\n"
                  "    m = w0 * np.sum(phi - pn)\n"
                  "    return float(abs(r) < 1e-10 and abs(m) < 1e-12)\n",
         "call": "law(pfc_bdf2_step(pn, pm, e0, gam, tau, L, al, Aa, th))",
         "gold_call": "law(_oracle_pfc_bdf2_step(pn, pm, e0, gam, tau, L, al, Aa, "
                      "th))"},
        # regression: the step-5 breakdown must PROPAGATE, not be absorbed. At
        # theta = 0.05 and tau = 200 the penalty is far too weak for the step, the
        # source theorem's smallness assumption is violated, and Newton does not
        # settle within its ordinary 100-iteration cap. The step must report that
        # rather than return a state reconstructed from the last iterate. The trap
        # catches RuntimeError ONLY and reports -1.0 for it, returning the new
        # multiplier on success, so a step that swallows the breakdown separates
        # immediately; any other exception propagates and fails the case rather
        # than collecting the passing value.
        {"setup": "import numpy as np\nN = 16\nL = 8.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.2 * np.cos(2 * u)[:, None] * np.cos(u)[None, :]\n"
                  "pm = -0.27 + 0.1 * np.cos(2 * u)[:, None] * np.cos(u)[None, :]\n"
                  "def trap(f):\n"
                  "    try:\n"
                  "        return float(f()[1])\n"
                  "    except RuntimeError:\n"
                  "        return -1.0\n",
         "call": "trap(lambda: pfc_bdf2_step(pn, pm, 1.0, 1.0, 200.0, L, 0.75, "
                 "0.37, 0.05))",
         "gold_call": "trap(lambda: _oracle_pfc_bdf2_step(pn, pm, 1.0, 1.0, 200.0, "
                      "L, 0.75, 0.37, 0.05))"},
    ]
