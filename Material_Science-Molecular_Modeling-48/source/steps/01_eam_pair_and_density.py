"""
Evaluate the embedded-atom pair potential $\\phi(r)$ and the electron-density function $f(r)$ of the source paper's EAM parameterisation, ****together with their first derivatives**** with respect to $r$, at an array of interatomic distances. Return all four stacked along a new leading axis, in the order $[\\phi,\\, f,\\, \\phi',\\, f']$, so that the shape is `(4,) + r.shape`. Both functions are ****hard-truncated****: every one of the four is exactly $0$ where $r \\ge r_{\\mathrm{cut}}$, with no shifting and no smoothing. The derivatives are graded because the forces of step 3 and the equilibrium lattice constant of step 4 are built from them analytically, not by differencing.

The repulsive and attractive parts of $\\phi$ and the density function $f$ share one structure: a decaying exponential in the reduced distance $x = r/r_e$, divided by a very steep cutoff denominator $1 + (x - s)^{20}$ whose exponent $20$ makes the function fall to numerical zero within a fraction of $r_e$ beyond $x = s + 1$. The attractive term of $\\phi$ and the whole of $f$ share the same decay rate $\\beta$ and the same cutoff offset $\\lambda$ and differ only in their prefactor, which is why $B$ and $f_e$ enter the potential in structurally identical ways. Differentiating a term $C\\,e^{-k(x-1)}/(1+(x-s)^{20})$ requires both the exponential and the denominator: with $u$ the term itself, $du/dr = \\big[-k\\,u - 20\\,u\\,(x-s)^{19}/(1+(x-s)^{20})\\big]/r_e$.

****--- Formulas ---****

With $x = r/r_e$,

$$\\phi(r) = \\frac{A\\,e^{-\\alpha(x-1)}}{1+(x-\\kappa)^{20}} - \\frac{B\\,e^{-\\beta(x-1)}}{1+(x-\\lambda)^{20}}, \\qquad f(r) = \\frac{f_e\\,e^{-\\beta(x-1)}}{1+(x-\\lambda)^{20}},$$

and, writing $T(C,k,s) = C\\,e^{-k(x-1)}\\big/\\big(1+(x-s)^{20}\\big)$ and

$$T'(C,k,s) = \\frac{1}{r_e}\\left[-k\\,T(C,k,s) - \\frac{20\\,(x-s)^{19}}{1+(x-s)^{20}}\\,T(C,k,s)\\right],$$

one has $\\phi' = T'(A,\\alpha,\\kappa) - T'(B,\\beta,\\lambda)$ and $f' = T'(f_e,\\beta,\\lambda)$. All four are set to exactly $0$ wherever $r \\ge r_{\\mathrm{cut}}$. The parameters are read from the length-20 vector at indices $0\\!:\\!r_e$, $1\\!:\\!f_e$, $4\\!:\\!\\alpha$, $5\\!:\\!\\beta$, $6\\!:\\!A$, $7\\!:\\!B$, $8\\!:\\!\\kappa$, $9\\!:\\!\\lambda$.

Returns
-------
`np.ndarray` of shape `(4,) + np.shape(r)`, real and finite, with every one of the four entries exactly $0$ wherever $r \\ge r_{\\mathrm{cut}}$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def eam_pair_and_density(r, params, r_cut=6.0):
    """r: array of interatomic distances (any shape, all >= 0).
    params: length-20 EAM parameter vector in the order
      (r_e, f_e, rho_e, rho_s, alpha, beta, A, B, kappa, lambda,
       F_n0..F_n3, F_0..F_3, eta, F_e).
    r_cut: hard cutoff; every returned quantity is exactly 0 where r >= r_cut.
    Return the real array [phi, f, dphi/dr, df/dr] of shape (4,) + r.shape."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _check_params(params):
    p = np.asarray(params, float).ravel()
    if p.size != 20:
        raise ValueError("params must be a length-20 vector in the stated order")
    if not np.all(np.isfinite(p)):
        raise ValueError("params must be finite")
    if p[0] <= 0.0:
        raise ValueError("r_e must be positive")
    if p[2] <= 0.0 or p[3] <= 0.0:
        raise ValueError("rho_e and rho_s must be positive")
    return p

def _oracle_eam_pair_and_density(r, params, r_cut=6.0):
    p = _check_params(params)
    r = np.asarray(r, float)
    if np.any(r < 0.0):
        raise ValueError("interatomic distances must be non-negative")
    if not np.isfinite(r_cut) or r_cut <= 0.0:
        raise ValueError("r_cut must be a finite positive scalar")
    re, fe, alpha, beta = p[0], p[1], p[4], p[5]
    A, B, kap, lam = p[6], p[7], p[8], p[9]
    x = r / re
    er, ea = np.exp(-alpha * (x - 1.0)), np.exp(-beta * (x - 1.0))
    dr, da = 1.0 + (x - kap) ** 20, 1.0 + (x - lam) ** 20
    rep, att, fd = A * er / dr, B * ea / da, fe * ea / da

    def dterm(val, k, s):
        return (-k * val - val * 20.0 * (x - s) ** 19 / (1.0 + (x - s) ** 20)) / re

    phi = rep - att
    dphi = dterm(rep, alpha, kap) - dterm(att, beta, lam)
    dfd = dterm(fd, beta, lam)
    m = r < r_cut
    z = np.zeros_like(x)
    return np.stack([np.where(m, phi, z), np.where(m, fd, z),
                     np.where(m, dphi, z), np.where(m, dfd, z)])

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
        # normal: the whole physically relevant range of the production cutoff.
        {"setup": "import numpy as np\n" + TA + "r = np.linspace(2.0, 5.8, 25)\n",
         "call": "eam_pair_and_density(r, TA)",
         "gold_call": "_oracle_eam_pair_and_density(r, TA)"},
        # normal: r_e itself, the BCC nearest-neighbour distance and the lattice constant.
        {"setup": "import numpy as np\n" + TA + "r = np.array([2.4, 2.860082, 3.3025, 4.0])\n",
         "call": "eam_pair_and_density(r, TA)",
         "gold_call": "_oracle_eam_pair_and_density(r, TA)"},
        # boundary: distances straddling the cutoff. The convention is r < r_cut kept,
        # r >= r_cut exactly zero, with NO shifting; a shifted implementation fails here.
        {"setup": "import numpy as np\n" + TA + "r = np.array([5.90, 5.999, 6.0, 6.001, 7.5])\n",
         "call": "eam_pair_and_density(r, TA)",
         "gold_call": "_oracle_eam_pair_and_density(r, TA)"},
        # boundary: a two-dimensional r with a non-default cutoff, so the shape must be
        # (4,) + r.shape and the truncation must follow r_cut rather than a hard-coded 6.
        {"setup": "import numpy as np\n" + TA + "r = np.linspace(1.5, 5.0, 12).reshape(3, 4)\n",
         "call": "eam_pair_and_density(r, TA, 4.0)",
         "gold_call": "_oracle_eam_pair_and_density(r, TA, 4.0)"},
        # boundary: five of the seven master parameters moved at once, so no branch can
        # be short-circuited with the tabulated tantalum values.
        {"setup": "import numpy as np\n" + TA + "P = TA.copy()\nP[0] *= 1.05\nP[5] *= 0.90\n"
                  "P[6] *= 1.30\nP[7] *= 0.80\nP[8] *= 1.40\nr = np.linspace(2.0, 5.5, 20)\n",
         "call": "eam_pair_and_density(r, P)",
         "gold_call": "_oracle_eam_pair_and_density(r, P)"},
        # edge: r = 0 and the deep repulsive wall, where x < kappa makes (x - kappa)
        # negative and the twentieth power must still be taken correctly.
        {"setup": "import numpy as np\n" + TA + "r = np.array([0.0, 0.5, 1.0, 1.8])\n",
         "call": "eam_pair_and_density(r, TA)",
         "gold_call": "_oracle_eam_pair_and_density(r, TA)"},
        # edge: a cutoff so large that nothing is truncated, isolating the functional form.
        {"setup": "import numpy as np\n" + TA + "r = np.array([2.7, 3.1])\n",
         "call": "eam_pair_and_density(r, TA, 12.0)",
         "gold_call": "_oracle_eam_pair_and_density(r, TA, 12.0)"},
    ]
