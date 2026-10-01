"""
Return the three fields that make one step of the scheme independent of the multiplier: the explicit extrapolation $\\bar\\phi^{\\,n+1}$ at which the nonlinearity is evaluated, and the two fields $p^{n+1}$ and $q^{n+1}$ obtained from the same constant-coefficient elliptic solve. The new state is then $\\phi^{n+1}=p^{n+1}+S(\\eta^{n+1})\\tau\\,q^{n+1}$, so **neither $p$ nor $q$ may depend on $\\eta$** - that is exactly what makes the scheme fully decoupled and reduces the step to two elementwise divisions in Fourier space plus one scalar root-find. Setting $\\gamma=0$ must reproduce the first-order backward-Euler starting step: the leading coefficient becomes $1$, the extrapolation collapses to $\\phi^n$, and $\\phi^{n-1}$ drops out. That degenerate case is graded.

The scheme writes $D_2\\phi^{n+1}=\\Delta\\mu^{n+1}$ with $\\mu^{n+1}=\\mathcal{L}\\phi^{n+1}+S(\\eta^{n+1})F'(\\bar\\phi^{\\,n+1})$, where $\\mathcal{L}=\\Delta^2+2\\Delta+(\\alpha+A)\\mathcal{I}$ and $F'(\\phi)=\\phi^3-A\\phi$. Eliminating $\\mu^{n+1}$ and multiplying by $\\tau_{n+1}$ collects all the implicit content into one operator $\\mathcal{A}$, which is linear, constant-coefficient and therefore diagonal in Fourier space with a real symbol. Because the nonlinearity is evaluated at the *extrapolated* state $\\bar\\phi^{\\,n+1}$, which is built from data already known, the right-hand side is explicit and the multiplier enters only as a scalar prefactor.




 **Formulas**



With $\\kappa=|\\boldsymbol{k}|^2$, $\\gamma=\\gamma_{n+1}$ and $\\tau=\\tau_{n+1}$,




$$\\bar\\phi^{\\,n+1}=\\phi^{n}+\\gamma\\,(\\phi^{n}-\\phi^{n-1}),$$




$$\\widehat{\\mathcal{A}}(\\boldsymbol{k})=\\frac{1+2\\gamma}{1+\\gamma}-\\tau\\,\\widehat{\\Delta\\mathcal{L}}(\\boldsymbol{k}), \\qquad \\widehat{\\Delta\\mathcal{L}}(\\boldsymbol{k})=-\\kappa^{3}+2\\kappa^{2}-(\\alpha+A)\\,\\kappa,$$




$$p^{n+1}=\\mathcal{F}^{-1}\\!\\left[\\frac{\\mathcal{F}\\big[(1+\\gamma)\\phi^{n}-\\frac{\\gamma^{2}}{1+\\gamma}\\phi^{n-1}\\big]}{\\widehat{\\mathcal{A}}}\\right], \\qquad q^{n+1}=\\mathcal{F}^{-1}\\!\\left[\\frac{-\\kappa\\,\\mathcal{F}\\big[F'(\\bar\\phi^{\\,n+1})\\big]}{\\widehat{\\mathcal{A}}}\\right],$$




both taken as the real part of the inverse transform.

Returns
-------
A `tuple` `(phi_bar, p, q)` of three real `np.ndarray`s, each with the shape of `phi_n`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bdf2_decoupled_fields(phi_n: "np.ndarray", phi_nm1: "np.ndarray", gamma: float,
                          tau: float, L: "float | Sequence[float]", alpha: float,
                          A: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """phi_n, phi_nm1: real 2-D arrays of the same shape, the states at t_n and
    t_{n-1}.  gamma: float >= 0, the step ratio tau_{n+1}/tau_n (use 0.0 for the
    backward-Euler starting step).  tau: float > 0, the step tau_{n+1}.
    L: float, or length-2 sequence (Lx, Ly).  alpha, A: floats.
    Return the tuple (phi_bar, p, q) of three real arrays with the shape of
    phi_n, neither p nor q depending on the Lagrange multiplier."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _Fprime(x, A):
    return x ** 3 - A * x


def _oracle_bdf2_decoupled_fields(phi_n: "np.ndarray", phi_nm1: "np.ndarray", gamma: float,
                                  tau: float, L: "float | Sequence[float]", alpha: float,
                                  A: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    phi_n = np.asarray(phi_n, float)
    phi_nm1 = np.asarray(phi_nm1, float)
    if phi_n.shape != phi_nm1.shape:
        raise ValueError("phi_n and phi_nm1 must have the same shape")
    gamma = float(gamma)
    tau = float(tau)
    if gamma < 0.0:
        raise ValueError("the step ratio gamma must be non-negative")
    if not (tau > 0.0):
        raise ValueError("the time step tau must be positive")
    nv, Lv, hx, hy = _geom(phi_n, L)
    ksq = _oracle_spectral_wavenumbers(nv, Lv)
    # symbol of Delta * Lop, with Lop = Delta^2 + 2 Delta + (alpha + A) I
    dl = -ksq ** 3 + 2.0 * ksq ** 2 - (float(alpha) + float(A)) * ksq
    a_sym = (1.0 + 2.0 * gamma) / (1.0 + gamma) - tau * dl
    phi_bar = phi_n + gamma * (phi_n - phi_nm1)
    rhs_p = (1.0 + gamma) * phi_n - gamma ** 2 / (1.0 + gamma) * phi_nm1
    p = np.real(np.fft.ifft2(np.fft.fft2(rhs_p) / a_sym))
    q = np.real(np.fft.ifft2(-ksq * np.fft.fft2(_Fprime(phi_bar, float(A))) / a_sym))
    return phi_bar, p, q

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: a mid-run BDF2 step on the production grid at the benchmark
        # material point, with a step ratio near one.
        {"setup": "import numpy as np\nN = 64\nL = 32.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.06 * np.cos(16 * u + 0.3)[:, None] "
                  "* np.cos(9 * u)[None, :]\n"
                  "pm = pn - 0.004 * np.sin(11 * u)[:, None] * np.cos(13 * u)[None, :]\n",
         "call": "bdf2_decoupled_fields(pn, pm, 1.03, 0.6, L, 0.75, 0.37)",
         "gold_call": "_oracle_bdf2_decoupled_fields(pn, pm, 1.03, 0.6, L, 0.75, 0.37)"},
        # boundary: gamma = 0, the backward-Euler starting step. The leading
        # coefficient must become exactly 1, phi_bar must collapse to phi_n, and
        # phi_nm1 must drop out of p entirely - here phi_nm1 is deliberately a
        # completely unrelated field, so any residual dependence on it shows up.
        {"setup": "import numpy as np\nN = 32\nL = 16.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.1 * np.cos(5 * u)[:, None] * np.sin(3 * u)[None, :]\n"
                  "pm = 7.0 + np.sin(u)[:, None] * np.cos(u)[None, :]\n",
         "call": "bdf2_decoupled_fields(pn, pm, 0.0, 1e-4, L, 0.75, 0.37)",
         "gold_call": "_oracle_bdf2_decoupled_fields(pn, pm, 0.0, 1e-4, L, 0.75, 0.37)"},
        # boundary: gamma exactly 1, the uniform-step case, where the coefficients
        # collapse to 3/2, -2, 1/2 and the extrapolation to 2 phi_n - phi_nm1.
        {"setup": "import numpy as np\nN = 24\nL = 12.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.3 + 0.2 * np.cos(4 * u)[:, None] * np.cos(2 * u)[None, :]\n"
                  "pm = -0.3 + 0.19 * np.cos(4 * u)[:, None] * np.cos(2 * u)[None, :]\n",
         "call": "bdf2_decoupled_fields(pn, pm, 1.0, 0.05, L, 0.75, 0.37)",
         "gold_call": "_oracle_bdf2_decoupled_fields(pn, pm, 1.0, 0.05, L, 0.75, 0.37)"},
        # boundary: the maximal admissible step ratio 4.86454, where the
        # extrapolation is far outside the interval and the BDF2 coefficients are
        # at their most lopsided.
        {"setup": "import numpy as np\nN = 32\nL = 16.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.08 * np.sin(7 * u)[:, None] * np.cos(5 * u)[None, :]\n"
                  "pm = -0.27 + 0.05 * np.sin(7 * u)[:, None] * np.cos(5 * u)[None, :]\n",
         "call": "bdf2_decoupled_fields(pn, pm, 4.86454, 4.86454e-04, L, 0.75, 0.37)",
         "gold_call": "_oracle_bdf2_decoupled_fields(pn, pm, 4.86454, 4.86454e-04, "
                      "L, 0.75, 0.37)"},
        # boundary: a RECTANGULAR grid on an anisotropic box at a different
        # material point, so every wave number and the whole symbol change.
        {"setup": "import numpy as np\nNv = (24, 32)\nLv = (20.0 * np.pi, 32.0 * np.pi)\n"
                  "a = np.arange(Nv[0]) * (2 * np.pi / Nv[0])\n"
                  "b = np.arange(Nv[1]) * (2 * np.pi / Nv[1])\n"
                  "pn = -0.27 + 0.09 * np.cos(3 * a)[:, None] * np.sin(4 * b)[None, :]\n"
                  "pm = -0.27 + 0.07 * np.cos(3 * a)[:, None] * np.sin(4 * b)[None, :]\n",
         "call": "bdf2_decoupled_fields(pn, pm, 1.4, 0.3, Lv, 0.6, 0.5)",
         "gold_call": "_oracle_bdf2_decoupled_fields(pn, pm, 1.4, 0.3, Lv, 0.6, 0.5)"},
        # edge: a very large step, where the tau * Delta L term dominates the
        # implicit symbol at every non-zero wave number and only the zero mode is
        # controlled by the BDF2 coefficient.
        {"setup": "import numpy as np\nN = 16\nL = 8.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.3 * np.cos(2 * u)[:, None] * np.cos(u)[None, :]\n"
                  "pm = -0.27 + 0.1 * np.cos(2 * u)[:, None] * np.cos(u)[None, :]\n",
         "call": "bdf2_decoupled_fields(pn, pm, 2.0, 50.0, L, 0.75, 0.37)",
         "gold_call": "_oracle_bdf2_decoupled_fields(pn, pm, 2.0, 50.0, L, 0.75, 0.37)"},
        # edge: A = 0, so the modified potential is the bare cubic and the constant
        # term of the implicit operator is alpha alone.
        {"setup": "import numpy as np\nN = 16\nL = 2.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = 0.5 * np.cos(u)[:, None] * np.sin(2 * u)[None, :]\n"
                  "pm = 0.4 * np.cos(u)[:, None] * np.sin(2 * u)[None, :]\n",
         "call": "bdf2_decoupled_fields(pn, pm, 0.8, 0.01, L, 1.0, 0.0)",
         "gold_call": "_oracle_bdf2_decoupled_fields(pn, pm, 0.8, 0.01, L, 1.0, 0.0)"},
    ]
