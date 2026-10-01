"""
Evaluate the ****piecewise**** embedding function $F(\\rho)$ of the same parameterisation, and its first derivative $F'(\\rho)$, at an array of local electron densities. Return the two stacked along a new leading axis, in the order $[F,\\, F']$, so that the shape is `(2,) + rho.shape`. The three branches are selected by $\\rho < \\rho_n$, $\\rho_n \\le \\rho < \\rho_0$ and $\\rho_0 \\le \\rho$, with $\\rho_n = 0.85\\rho_e$ and $\\rho_0 = 1.15\\rho_e$; the half-open convention at both knots is graded. ****All three branches matter****: the third one, the only place where $\\eta$ and $F_e$ appear, is what makes those two parameters identifiable at all, and a candidate who implements only the polynomial branches will find later steps failing with a singular information matrix rather than returning a wrong number.

The embedding energy is the nonlinear part of the EAM: it is what distinguishes the model from a pair potential and what makes the force on an atom depend on its neighbours' densities as well as its own. Near the reference density $\\rho_e$ the function is a cubic in the relative density excess; far below it, a different cubic in the excess relative to $\\rho_n$; far above it, an analytic form $F_e[1-\\eta\\ln(\\rho/\\rho_s)](\\rho/\\rho_s)^{\\eta}$ chosen so that the embedding energy keeps rising smoothly under strong compression instead of turning over the way an unconstrained polynomial would. That analytic branch has the convenient property that at $\\rho = \\rho_s$ its value is exactly $F_e$ and its derivative is exactly $0$, which is why the tabulated $F_0$ and $F_1$ are so close to $F_e$ and to zero. Differentiating it gives $dF/d\\rho = -F_e\\eta^2\\ln(\\rho/\\rho_s)(\\rho/\\rho_s)^{\\eta-1}/\\rho_s$ - the two terms of the product rule collapse to a single term, which is a useful check.

****--- Formulas ---****

With $t_n = \\rho/\\rho_n - 1$, $t_e = \\rho/\\rho_e - 1$ and $u = \\rho/\\rho_s$,

$$F(\\rho) = \\begin{cases} \\sum_{i=0}^{3}F_{ni}\\,t_n^{\\,i}, & \\rho < \\rho_n = 0.85\\rho_e,\\\\ \\sum_{i=0}^{3}F_{i}\\,t_e^{\\,i}, & \\rho_n \\le \\rho < \\rho_0 = 1.15\\rho_e,\\\\ F_e\\big[1-\\eta\\ln u\\big]u^{\\eta}, & \\rho_0 \\le \\rho, \\end{cases}$$

$$F'(\\rho) = \\begin{cases} \\dfrac{1}{\\rho_n}\\sum_{i=1}^{3}i\\,F_{ni}\\,t_n^{\\,i-1}, & \\rho < \\rho_n,\\\\ \\dfrac{1}{\\rho_e}\\sum_{i=1}^{3}i\\,F_{i}\\,t_e^{\\,i-1}, & \\rho_n \\le \\rho < \\rho_0,\\\\ -\\dfrac{F_e\\,\\eta^{2}\\ln(u)\\,u^{\\eta-1}}{\\rho_s}, & \\rho_0 \\le \\rho. \\end{cases}$$

The parameters are read from the length-20 vector at indices $2\\!:\\!\\rho_e$, $3\\!:\\!\\rho_s$, $10\\!-\\!13\\!:\\!F_{n0}\\ldots F_{n3}$, $14\\!-\\!17\\!:\\!F_{0}\\ldots F_{3}$, $18\\!:\\!\\eta$, $19\\!:\\!F_e$.

Returns
-------
`np.ndarray` of shape `(2,) + np.shape(rho)`, real and finite. The value and the first derivative are continuous across both knots to better than $10^{-6}$ for the tantalum parameter set.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def eam_embedding(rho, params):
    """rho: array of local electron densities (any shape, all >= 0).
    params: the same length-20 EAM parameter vector as in step 1.
    Return the real array [F, dF/drho] of shape (2,) + rho.shape."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_eam_embedding(rho, params):
    p = _check_params(params)
    rho = np.asarray(rho, float)
    if np.any(rho < 0.0):
        raise ValueError("the electron density must be non-negative")
    rhoe, rhos, eta, Fe = p[2], p[3], p[18], p[19]
    Fn, Fi = p[10:14], p[14:18]
    rn, r0 = 0.85 * rhoe, 1.15 * rhoe
    val = np.zeros(rho.shape, float)
    der = np.zeros(rho.shape, float)
    m1, m2, m3 = rho < rn, (rho >= rn) & (rho < r0), rho >= r0
    if m1.any():
        t = rho[m1] / rn - 1.0
        val[m1] = sum(Fn[i] * t ** i for i in range(4))
        der[m1] = sum(i * Fn[i] * t ** (i - 1) for i in range(1, 4)) / rn
    if m2.any():
        t = rho[m2] / rhoe - 1.0
        val[m2] = sum(Fi[i] * t ** i for i in range(4))
        der[m2] = sum(i * Fi[i] * t ** (i - 1) for i in range(1, 4)) / rhoe
    if m3.any():
        t = rho[m3] / rhos
        val[m3] = Fe * (1.0 - eta * np.log(t)) * t ** eta
        der[m3] = -Fe * eta ** 2 * np.log(t) * t ** (eta - 1.0) / rhos
    return np.stack([val, der])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    TA = ("TA = np.array([2.860082, 3.086341, 33.787168, 33.787168, 8.489528, 4.527748,\n"
          "              0.611679, 1.032101, 0.176977, 0.353954,\n"
          "              -5.103845, -0.405524, 1.112997, -3.585325,\n"
          "              -5.14, 0.0, 1.640098, 0.221375, 0.848843, -5.141526])\n")
    return [
        # normal: entirely inside the middle branch, the range the BCC equilibrium visits.
        {"setup": "import numpy as np\n" + TA + "rho = np.linspace(29.0, 38.0, 19)\n",
         "call": "eam_embedding(rho, TA)",
         "gold_call": "_oracle_eam_embedding(rho, TA)"},
        # boundary: densities in all three branches at once, including both knots exactly.
        {"setup": "import numpy as np\n" + TA + "rho = np.array([20.0, 28.0, 28.719092799999998, "
                  "30.0, 33.787168, 38.855243199999995, 42.0, 55.0])\n",
         "call": "eam_embedding(rho, TA)",
         "gold_call": "_oracle_eam_embedding(rho, TA)"},
        # boundary: the two knots computed from the parameters rather than typed, which
        # pins the half-open convention rho_n <= rho < rho_0 for the middle branch.
        {"setup": "import numpy as np\n" + TA + "rho = np.array([0.85*TA[2], 1.15*TA[2]])\n",
         "call": "eam_embedding(rho, TA)",
         "gold_call": "_oracle_eam_embedding(rho, TA)"},
        # edge: from zero density to well past twice the reference density, so the first
        # branch is evaluated at t = -1 and the third far into the analytic regime.
        {"setup": "import numpy as np\n" + TA + "rho = np.linspace(0.0, 90.0, 16)\n",
         "call": "eam_embedding(rho, TA)",
         "gold_call": "_oracle_eam_embedding(rho, TA)"},
        # boundary: eta and F_e moved, which changes ONLY the third branch. An
        # implementation that never reaches it returns the tantalum values and fails.
        {"setup": "import numpy as np\n" + TA + "P = TA.copy()\nP[18] *= 1.25\nP[19] *= 0.85\n"
                  "rho = np.linspace(20.0, 70.0, 21)\n",
         "call": "eam_embedding(rho, P)",
         "gold_call": "_oracle_eam_embedding(rho, P)"},
        # edge: a two-dimensional rho spanning two branches, so the branch masks must be
        # applied elementwise and the output shape must be (2,) + rho.shape.
        {"setup": "import numpy as np\n" + TA + "rho = np.linspace(24.0, 52.0, 12).reshape(3, 4)\n",
         "call": "eam_embedding(rho, TA)",
         "gold_call": "_oracle_eam_embedding(rho, TA)"},
    ]
