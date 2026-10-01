"""
Return the Lagrange multiplier $\\eta^{n+1}$ of one step, by solving the penalised scalar nonlinear equation. This is the whole nonlinear content of the step: everything else is linear. The equation is the **discrete energy law itself** - the change in bulk energy plus the change in the penalty term equals the multiplier-weighted work of the extrapolated nonlinearity over the increment - so an implementation is correct if and only if that identity holds to round-off once $\\eta^{n+1}$ is substituted back. Three things are graded and each of them changes the answer: the scaling function is $S(\\eta)=\\eta(2-\\eta)$, not $\\eta$; the penalty term $\\theta(\\eta^2-(\\eta^{n})^2)$ sits **inside** this equation, not only in the energy functional, and removing it makes the iteration break down; and there is **no extra factor of $\\tau$** multiplying $S(\\eta)$ outside the bracket, only the one inside $p+S(\\eta)\\tau q$. Newton is started at $\\eta^{n}$: that is the prescribed **branch-selection convention**, because $g$ may have more than one root and only the branch continuously connected to $\\eta=1$ is the discrete solution. If the iteration cannot deliver such a root - the cap is exhausted, the derivative vanishes, or an iterate goes non-finite - **report the failure**; do not return the last iterate.

In the Lagrange multiplier approach the nonlinear term of the chemical potential is multiplied by $S(\\eta)$, where $\\eta$ is a scalar unknown whose exact value is $1$ and $S$ satisfies $S(1)=1$ and $S'(1)=0$; the second condition is what makes $S(\\eta)$ second-order accurate even though $\\eta$ itself is only first-order accurate. The extra unknown is closed by demanding that the discrete energy law hold exactly, which is a scalar equation. Near equilibrium the physical dissipation rate vanishes, that equation degenerates, and the scheme loses solvability - the failure the global quadratic penalty exists to remove. With the penalty the derivative of the scalar residual picks up the term $2\\theta\\eta$, which is what restores solvability; the source's theorem then places a root in a shrinking neighbourhood $[1-\\sqrt{\\tau},1+\\sqrt{\\tau}]$ of $1$ **under a smallness assumption on $\\tau$**. That assumption is not an idle one, and nothing here should be read as a uniqueness claim: at a weak penalty and $\\tau=O(1)$ the residual can have two roots inside that interval, and the residual can take the same sign at both of its endpoints, so whole-interval bracketing is not merely unreliable - it can fail to find anything at all, and it is not a substitute for the prescribed convention. Newton started at $\\eta^{n}$ tracks the physical branch and is the convention this task prescribes. Note also that unconditional *energy dissipation* and *solvability of the scalar equation* are separate claims and must not be conflated: dissipation of the modified energy is unconditional in $\\tau$ subject only to the step-ratio bound, whereas solvability of this scalar equation holds only under the source theorem's own step-size restriction, and outside that restriction the iteration may simply fail - which is a reportable outcome, not a value to return.




 **Formulas**




Write $S(\\eta)=\\eta(2-\\eta)$, $F(x)=\\tfrac14x^4-\\tfrac{A}{2}x^2$, $F'(x)=x^3-Ax$, $w(\\eta)=p+S(\\eta)\\,\\tau\\,q$, and let $\\langle f\\rangle=h_xh_y\\sum_{\\Omega_h}f$. The residual and its derivative are




$$g(\\eta)=\\big\\langle F(w)\\big\\rangle-\\big\\langle F(\\phi^{n})\\big\\rangle-S(\\eta)\\,\\big\\langle F'(\\bar\\phi^{\\,n+1})\\,(w-\\phi^{n})\\big\\rangle+\\theta\\big(\\eta^{2}-(\\eta^{n})^{2}\\big),$$




$$\\frac{\\mathrm{d}g}{\\mathrm{d}S}=\\tau\\big\\langle F'(w)\\,q\\big\\rangle-\\big\\langle F'(\\bar\\phi^{\\,n+1})(w-\\phi^{n})\\big\\rangle-S(\\eta)\\,\\tau\\,\\big\\langle F'(\\bar\\phi^{\\,n+1})\\,q\\big\\rangle, \\qquad g'(\\eta)=(2-2\\eta)\\frac{\\mathrm{d}g}{\\mathrm{d}S}+2\\theta\\eta .$$




Newton's iteration is started at $\\eta=\\eta^{n}$ and stopped when the correction satisfies $|g/g'|\\le\\texttt{tol}$, within at most $max_iter$ iterations. Exhausting $max_iter$ without meeting that test, meeting $g'=0$, or producing a non-finite residual, derivative or iterate are all **solver failures** and must be raised, not returned.

Returns
-------
A Python `float`: a root $\\eta^{n+1}$ of $g$ - specifically the one Newton reaches from $\\eta^{n}$, which is the branch continuously connected to $\\eta=1$. In the well-resolved regime it sits close to $1$; the source's bound $[1-\\sqrt{\\tau},1+\\sqrt{\\tau}]$ holds under its own step-size restriction and is not guaranteed here for large $\\tau$, nor is the root unique in that interval. If no root is obtained - `max_iter` exhausted, a vanishing derivative, or a non-finite residual, derivative or iterate - the function raises `RuntimeError` instead of returning the last iterate.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_multiplier(p: "np.ndarray", q: "np.ndarray", phi_n: "np.ndarray",
                     phi_bar: "np.ndarray", tau: float, theta: float,
                     eta_n: float, A: float, L: "float | Sequence[float]",
                     tol: float=1e-13, max_iter: int=100) -> float:
    """p, q, phi_n, phi_bar: real 2-D arrays of the same shape, from step 4.
    tau: float > 0, the step tau_{n+1}.  theta: float > 0, the penalty.
    eta_n: float, the previous multiplier (also the Newton starting guess, which
        selects the root branch and is part of the scheme's specification).
    A: float, the stabilisation parameter.  L: float or length-2 sequence.
    tol: float, stopping tolerance on the Newton correction.
    max_iter: int, iteration cap.
    Return the new Lagrange multiplier eta^{n+1} as a float.
    Raise RuntimeError if the iteration cannot produce a root: max_iter is
    exhausted before the correction falls to tol, the derivative vanishes, or
    the residual, derivative or iterate becomes non-finite."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _Fbulk(x, A):
    return 0.25 * x ** 4 - 0.5 * A * x ** 2


def _S(eta):
    return eta * (2.0 - eta)


def _oracle_solve_multiplier(p: "np.ndarray", q: "np.ndarray", phi_n: "np.ndarray", phi_bar: "np.ndarray",
                             tau: float, theta: float, eta_n: float,
                             A: float, L: "float | Sequence[float]", tol: float=1e-13,
                             max_iter: int=100) -> float:
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    phi_n = np.asarray(phi_n, float)
    phi_bar = np.asarray(phi_bar, float)
    tau = float(tau)
    theta = float(theta)
    eta_n = float(eta_n)
    A = float(A)
    if theta <= 0.0:
        raise ValueError("the penalty parameter theta must be positive")
    nv, Lv, hx, hy = _geom(p, L)
    w0 = hx * hy
    fpb = _Fprime(phi_bar, A)
    int_Fn = w0 * np.sum(_Fbulk(phi_n, A))

    def _g_dg(eta):
        s = _S(eta)
        w = p + s * tau * q
        g = (w0 * np.sum(_Fbulk(w, A)) - int_Fn
             - s * w0 * np.sum(fpb * (w - phi_n))
             + theta * (eta * eta - eta_n * eta_n))
        dgds = (tau * w0 * np.sum(_Fprime(w, A) * q)
                - w0 * np.sum(fpb * (w - phi_n))
                - s * tau * w0 * np.sum(fpb * q))
        return g, dgds * (2.0 - 2.0 * eta) + 2.0 * theta * eta

    # Newton from eta^n. The iteration selects the branch continuously connected
    # to eta = 1; it is NOT guaranteed to be the only root of g, so the starting
    # point is part of the scheme's specification, not an implementation detail.
    # Any breakdown is reported, never papered over by returning the last iterate.
    eta = eta_n
    converged = False
    for _ in range(int(max_iter)):
        g, dg = _g_dg(eta)
        if not np.isfinite(g) or not np.isfinite(dg):
            raise RuntimeError("the scalar residual or its derivative is "
                               "non-finite; the multiplier solve broke down")
        if dg == 0.0:
            raise RuntimeError("the Newton derivative vanished; the scalar "
                               "equation is not solvable at this state")
        d = g / dg
        eta = eta - d
        if not np.isfinite(eta):
            raise RuntimeError("the Newton iterate left the finite range; the "
                               "multiplier solve broke down")
        if abs(d) <= tol:
            converged = True
            break
    if not converged:
        raise RuntimeError(
            "Newton did not converge to tol=%g in %d iterations; the last "
            "iterate is not a root of the scalar equation" % (tol, int(max_iter)))
    return float(eta)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the balanced penalty theta = 5 at a moderate step, the regime the
        # source recommends for production runs.
        {"setup": "import numpy as np\nN = 32\nL = 16.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.09 * np.cos(5 * u)[:, None] * np.sin(3 * u)[None, :]\n"
                  "pb = pn + 0.01 * np.cos(2 * u)[:, None]\n"
                  "p = pn + 0.004 * np.sin(4 * u)[None, :]\n"
                  "q = 0.03 * np.cos(u)[:, None] * np.cos(2 * u)[None, :]\n",
         "call": "solve_multiplier(p, q, pn, pb, 0.4, 5.0, 1.0, 0.37, L)",
         "gold_call": "_oracle_solve_multiplier(p, q, pn, pb, 0.4, 5.0, 1.0, 0.37, L)"},
        # normal: the same state with a previous multiplier that is NOT one, so the
        # penalty is measured against (eta^n)^2 and the Newton start is not 1.
        {"setup": "import numpy as np\nN = 32\nL = 16.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.09 * np.cos(5 * u)[:, None] * np.sin(3 * u)[None, :]\n"
                  "pb = pn + 0.01 * np.cos(2 * u)[:, None]\n"
                  "p = pn + 0.004 * np.sin(4 * u)[None, :]\n"
                  "q = 0.03 * np.cos(u)[:, None] * np.cos(2 * u)[None, :]\n",
         "call": "solve_multiplier(p, q, pn, pb, 0.4, 5.0, 0.93, 0.37, L)",
         "gold_call": "_oracle_solve_multiplier(p, q, pn, pb, 0.4, 5.0, 0.93, 0.37, L)"},
        # boundary: a very large penalty, which pins the root essentially at the
        # previous multiplier; the answer is close to eta_n but not equal to it,
        # so an implementation that short-circuits large theta to eta_n fails.
        {"setup": "import numpy as np\nN = 24\nL = 12.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.12 * np.cos(3 * u)[:, None] * np.cos(4 * u)[None, :]\n"
                  "pb = pn + 0.02 * np.sin(u)[:, None]\n"
                  "p = pn + 0.01 * np.cos(3 * u)[None, :]\n"
                  "q = 0.05 * np.sin(2 * u)[:, None] * np.sin(u)[None, :]\n",
         "call": "solve_multiplier(p, q, pn, pb, 0.5, 500.0, 1.0, 0.37, L)",
         "gold_call": "_oracle_solve_multiplier(p, q, pn, pb, 0.5, 500.0, 1.0, 0.37, L)"},
        # boundary: a weak penalty at a large step, the worst-conditioned regime,
        # where the root moves furthest from one and the quadratic S(eta) matters
        # most. An implementation using S(eta) = eta separates here.
        {"setup": "import numpy as np\nN = 24\nL = 12.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.12 * np.cos(3 * u)[:, None] * np.cos(4 * u)[None, :]\n"
                  "pb = pn + 0.05 * np.sin(u)[:, None]\n"
                  "p = pn + 0.03 * np.cos(3 * u)[None, :]\n"
                  "q = 0.2 * np.sin(2 * u)[:, None] * np.sin(u)[None, :]\n",
         "call": "solve_multiplier(p, q, pn, pb, 0.9, 0.2, 1.0, 0.6, L)",
         "gold_call": "_oracle_solve_multiplier(p, q, pn, pb, 0.9, 0.2, 1.0, 0.6, L)"},
        # boundary: a RECTANGULAR grid on an anisotropic box, so the volume element
        # that weights every integral in the residual changes.
        {"setup": "import numpy as np\nNv = (20, 28)\nLv = (5.0, 13.0)\n"
                  "a = np.arange(Nv[0]) * (2 * np.pi / Nv[0])\n"
                  "b = np.arange(Nv[1]) * (2 * np.pi / Nv[1])\n"
                  "pn = -0.27 + 0.1 * np.cos(2 * a)[:, None] * np.sin(3 * b)[None, :]\n"
                  "pb = pn + 0.02 * np.cos(a)[:, None]\n"
                  "p = pn + 0.01 * np.sin(b)[None, :]\n"
                  "q = 0.04 * np.cos(a)[:, None] * np.cos(b)[None, :]\n",
         "call": "solve_multiplier(p, q, pn, pb, 0.25, 5.0, 1.0, 0.37, Lv)",
         "gold_call": "_oracle_solve_multiplier(p, q, pn, pb, 0.25, 5.0, 1.0, 0.37, Lv)"},
        # edge: q identically zero, so the state does not move at all with eta and
        # the bulk-energy difference vanishes; the residual reduces to the penalty
        # alone and the root is exactly eta_n.
        {"setup": "import numpy as np\nN = 12\nL = 6.0\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.05 * np.cos(u)[:, None] * np.cos(u)[None, :]\n"
                  "pb = pn.copy()\np = pn.copy()\nq = np.zeros_like(pn)\n",
         "call": "solve_multiplier(p, q, pn, pb, 0.3, 5.0, 0.97, 0.37, L)",
         "gold_call": "_oracle_solve_multiplier(p, q, pn, pb, 0.3, 5.0, 0.97, 0.37, L)"},
        # edge: a tiny step, the backward-Euler start, where the root sits within a
        # few times 1e-7 of one and the answer is dominated by cancellation.
        {"setup": "import numpy as np\nN = 16\nL = 8.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.08 * np.cos(2 * u)[:, None] * np.sin(u)[None, :]\n"
                  "pb = pn.copy()\n"
                  "p = pn - 1e-5 * np.cos(u)[:, None]\n"
                  "q = 0.01 * np.sin(u)[:, None] * np.cos(2 * u)[None, :]\n",
         "call": "solve_multiplier(p, q, pn, pb, 1e-4, 5.0, 1.0, 0.37, L)",
         "gold_call": "_oracle_solve_multiplier(p, q, pn, pb, 1e-4, 5.0, 1.0, 0.37, L)"},
        # regression: an INDEPENDENT check that the returned scalar really is a
        # root. The setup rebuilds the residual g from p, q, phi_n, phi_bar and
        # the parameters with plain numpy, and the case reports 1.0 only if
        # |g(eta)| < 1e-10. This does not compare the solver with itself, so an
        # implementation that returns a plausible-looking non-root fails here.
        {"setup": "import numpy as np\nN = 32\nL = 16.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.09 * np.cos(5 * u)[:, None] * np.sin(3 * u)[None, :]\n"
                  "pb = pn + 0.01 * np.cos(2 * u)[:, None]\n"
                  "p = pn + 0.004 * np.sin(4 * u)[None, :]\n"
                  "q = 0.03 * np.cos(u)[:, None] * np.cos(2 * u)[None, :]\n"
                  "tau, th, en, Aa = 0.4, 5.0, 1.0, 0.37\n"
                  "w0 = (L / N) ** 2\n"
                  "Fb = lambda x: 0.25 * x ** 4 - 0.5 * Aa * x ** 2\n"
                  "Fp = lambda x: x ** 3 - Aa * x\n"
                  "def is_root(e):\n"
                  "    s = e * (2.0 - e)\n"
                  "    w = p + s * tau * q\n"
                  "    g = (w0 * np.sum(Fb(w)) - w0 * np.sum(Fb(pn))\n"
                  "         - s * w0 * np.sum(Fp(pb) * (w - pn))\n"
                  "         + th * (e * e - en * en))\n"
                  "    return float(abs(g) < 1e-10)\n",
         "call": "is_root(solve_multiplier(p, q, pn, pb, tau, th, en, Aa, L))",
         "gold_call": "is_root(_oracle_solve_multiplier(p, q, pn, pb, tau, th, en, Aa, L))"},
        # regression: a deliberately insufficient iteration cap. This state needs
        # three Newton corrections to reach tol; with max_iter=2 the last iterate
        # is NOT a root, so the contract requires the call to raise RuntimeError
        # rather than return it. The trap catches RuntimeError ONLY and reports
        # -1.0 for it, returning the multiplier itself on success; an unrelated
        # exception - a NameError, a TypeError, a ValueError from mis-validated
        # input - is deliberately NOT caught and propagates, so it fails the case
        # instead of being rewarded with the passing value.
        {"setup": "import numpy as np\nN = 32\nL = 16.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.09 * np.cos(5 * u)[:, None] * np.sin(3 * u)[None, :]\n"
                  "pb = pn + 0.01 * np.cos(2 * u)[:, None]\n"
                  "p = pn + 0.004 * np.sin(4 * u)[None, :]\n"
                  "q = 0.03 * np.cos(u)[:, None] * np.cos(2 * u)[None, :]\n"
                  "def trap(f):\n"
                  "    try:\n"
                  "        return float(f())\n"
                  "    except RuntimeError:\n"
                  "        return -1.0\n",
         "call": "trap(lambda: solve_multiplier(p, q, pn, pb, 0.4, 5.0, 1.0, "
                 "0.37, L, 1e-13, 2))",
         "gold_call": "trap(lambda: _oracle_solve_multiplier(p, q, pn, pb, 0.4, "
                      "5.0, 1.0, 0.37, L, 1e-13, 2))"},
        # regression: a NATURAL breakdown at the ORDINARY 100-iteration cap - no
        # artificially reduced max_iter is involved. A penalty of 0.05 at tau = 50
        # is far outside the source theorem's step-size restriction, the Newton
        # sequence never settles, and the last iterate carries an O(1) residual,
        # so the contract requires RuntimeError rather than that iterate. The trap
        # catches RuntimeError ONLY and reports -1.0 for it, returning the float on
        # success; any other exception propagates and fails the case.
        {"setup": "import numpy as np\nN = 24\nL = 12.0 * np.pi\n"
                  "u = np.arange(N) * (2 * np.pi / N)\n"
                  "pn = -0.27 + 0.12 * np.cos(3 * u)[:, None] * np.cos(4 * u)[None, :]\n"
                  "pb = pn + 0.05 * np.sin(u)[:, None]\n"
                  "p = pn + 0.03 * np.cos(3 * u)[None, :]\n"
                  "q = 0.2 * np.sin(2 * u)[:, None] * np.sin(u)[None, :]\n"
                  "def trap(f):\n"
                  "    try:\n"
                  "        return float(f())\n"
                  "    except RuntimeError:\n"
                  "        return -1.0\n",
         "call": "trap(lambda: solve_multiplier(p, q, pn, pb, 50.0, 0.05, 1.0, "
                 "0.6, L))",
         "gold_call": "trap(lambda: _oracle_solve_multiplier(p, q, pn, pb, 50.0, "
                      "0.05, 1.0, 0.6, L))"},
    ]
