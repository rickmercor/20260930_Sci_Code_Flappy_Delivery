"""
Evaluate the stabilization condition that the energy-dissipation proof requires, and return the four diagnostics $[C_q,\\;C_{\\mathrm{req}},\\;S_{\\min},\\;C_q-C_{\\mathrm{req}}]$ as a length-4 array. Here $C_q$ is the step-size-independent constant the argument produces from the stabilization parameter $S$, $C_{\\mathrm{req}}$ is the value it has to reach, $S_{\\min}$ is the smallest $S$ for which it does, and the last entry is the margin, non-negative exactly when the condition holds. The inputs are the order $q$, the two constants $\\kappa_q>0$ and $\\eta_q$ of the two quadratic decompositions, the model parameter $\\varepsilon$, and the stabilization parameter $S\\ge0$. The formulas are valid for $q\\ge3$.

The per-step energy budget of the scheme collects four contributions: a positive $\\tfrac{\\kappa_q}{2\\tau}\\|\\delta_\\tau\\phi^n\\|_{-1}^2$ from the time-derivative decomposition, a positive $\\tfrac{S\\tau^{q-1}\\kappa_q}{2}\\|\\nabla_h\\delta_\\tau\\phi^n\\|^2$ from the stabilization, a positive $\\tfrac12\\|(1+\\Delta_h)\\delta_\\tau\\phi^n\\|^2$ from the convex quadratic term, and a NEGATIVE $-\\varepsilon(\\eta_q+\\tfrac12)\\|\\delta_\\tau\\phi^n\\|^2$ from the extrapolation. The whole difficulty is that the negative term is $O(1)$ while the stabilization is $O(\\tau^{q-1})$, so a direct comparison would force $S$ to grow like $\\tau^{-(q-1)}$ and the condition would not be unconditional at all. The repair spends the convex term first: the identity $\\|\\nabla_hv\\|^2=\\|v\\|^2-\\langle(1+\\Delta_h)v,v\\rangle$ - exact wherever $\\nabla_h$ and $\\Delta_h$ are exact adjoints, hence for any field band-limited strictly below Nyquist under the convention used here - with Young's inequality gives the embedding $\\|\\nabla_hv\\|^2\\le\\tfrac54\\|v\\|^2+\\|(1+\\Delta_h)v\\|^2$, which converts the convex term into a gradient term at the cost of $-\\tfrac58\\|\\delta_\\tau\\phi^n\\|^2$. The combined gradient coefficient $\\tfrac12(1+S\\kappa_q\\tau^{q-1})$ is then split as $\\tfrac12\\big[\\tfrac{q-2}{q-1}\\cdot\\tfrac{q-1}{q-2}+\\tfrac{1}{q-1}\\cdot(q-1)S\\kappa_q\\tau^{q-1}\\big]$ and bounded below by the weighted arithmetic-geometric-mean inequality, which trades the two weights $\\tfrac{q-2}{q-1}$ and $\\tfrac1{q-1}$ for a single power and leaves $\\tfrac12C_q\\tau$ with $C_q$ as defined below - an $O(\\tau)$ coefficient with a $\\tau$-free constant. That single factor of $\\tau$ then cancels against the $\\tfrac1\\tau$ in front of the $H^{-1}$ term through $\\tfrac{\\kappa_q}{2\\tau}\\|v\\|_{-1}^2+\\tfrac{C_q\\tau}{2}\\|\\nabla_hv\\|^2\\ge\\sqrt{\\kappa_qC_q}\\,\\|v\\|_{-1}\\|\\nabla_hv\\|\\ge\\sqrt{\\kappa_qC_q}\\,\\|v\\|^2$, the last step being the interpolation inequality $\\|v\\|^2\\le\\|\\nabla_hv\\|\\|v\\|_{-1}$, available because mass conservation makes every first difference mean-zero. What survives is $\\sqrt{\\kappa_qC_q}\\ge\\varepsilon\\eta_q+\\varepsilon/2+5/8$, which contains neither $\\tau$ nor the mesh. Note what this does and does not say: it is a *sufficient* condition assembled from three separately generous inequalities, so the $S$ it licenses can be orders of magnitude below the $S$ a practitioner actually uses, and the energy may well decay for $S$ below the threshold - including at $S=0$, where $C_q=0$ and the condition fails outright.

$$C_q:=\\left[\\Big(\\frac{q-1}{q-2}\\Big)^{q-2}S\\,\\kappa_q\\,(q-1)\\right]^{\\frac{1}{q-1}},\\qquad C_{\\mathrm{req}}:=\\frac{\\big(\\varepsilon\\eta_q+\\varepsilon/2+5/8\\big)^2}{\\kappa_q},$$



and the condition is $C_q\\ge C_{\\mathrm{req}}$. $S_{\\min}$ is the value of $S$ at which that inequality first holds, i.e. the $S$ that makes the margin vanish. Return `np.array([C_q, C_req, S_min, C_q - C_req])`. Note that $S_{\\min}$ does not depend on $S$, and that $C_q=0$ when $S=0$.

Returns
-------
`np.ndarray` of shape `(4,)`, real and finite. Entry $0$ is $\\ge0$ and vanishes exactly when $S=0$; entry $1$ is $>0$ and independent of $S$ and of $q$ except through $\\kappa_q,\\eta_q$; entry $2$ is $>0$ and independent of $S$; entry $3$ is entry $0$ minus entry $1$, and the condition holds exactly when it is $\\ge0$, equivalently when $S\\ge S_{\\min}$. Substituting $S=S_{\\min}$ makes entries $0$ and $1$ equal and entry $3$ zero.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sqpfc_stabilization_certificate(q: int, kappa_q: float,
                                    eta_q: float, eps: float,
                                    S: float) -> "np.ndarray":
    """q: BDF order, at least 3.
    kappa_q: the positive constant of the BDF quadratic decomposition.
    eta_q: the constant of the extrapolation quadratic decomposition.
    eps: the parameter epsilon.  S: the stabilization parameter, >= 0.
    Return np.array([C_q, C_required, S_min, C_q - C_required]).
    Raise ValueError if q is not a whole number of at least 3, if kappa_q
    is not positive, or if S is negative."""
    # Implement per the specification above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_sqpfc_stabilization_certificate(q: int, kappa_q: float,
                                            eta_q: float, eps: float,
                                            S: float) -> "np.ndarray":
    """[C_q, C_required, S_min, margin]."""
    if q != int(q) or int(q) < 3:
        raise ValueError("q must be a whole number and at least 3")
    if not float(kappa_q) > 0.0:
        raise ValueError("kappa_q must be positive")
    if float(S) < 0.0:
        raise ValueError("S must be non-negative")
    q = int(q)
    kappa_q = float(kappa_q)
    eta_q = float(eta_q)
    eps = float(eps)
    S = float(S)
    pref = ((q - 1.0) / (q - 2.0)) ** (q - 2) * kappa_q * (q - 1.0)
    C = (pref * S) ** (1.0 / (q - 1.0))
    C_req = (eps * eta_q + eps / 2.0 + 5.0 / 8.0) ** 2 / kappa_q
    S_min = C_req ** (q - 1.0) / pref
    return np.array([C, C_req, S_min, C - C_req])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # the two graded q = 3 settings, with kappa_3 = 95/48 and eta_3 = 1/2
        {"setup": "",
         "call": "sqpfc_stabilization_certificate(3, 95/48, 0.5, 0.25, 5.0)",
         "gold_call": "_oracle_sqpfc_stabilization_certificate(3, 95/48, 0.5, 0.25, 5.0)"},
        {"setup": "",
         "call": "sqpfc_stabilization_certificate(3, 95/48, 0.5, 0.50, 5.0)",
         "gold_call": "_oracle_sqpfc_stabilization_certificate(3, 95/48, 0.5, 0.50, 5.0)"},
        # higher orders, where the (q-1)/(q-2) prefactor is no longer 2
        {"setup": "",
         "call": "sqpfc_stabilization_certificate(4, 2.25, 0.75, 0.40, 10.0)",
         "gold_call": "_oracle_sqpfc_stabilization_certificate(4, 2.25, 0.75, 0.40, 10.0)"},
        {"setup": "",
         "call": "sqpfc_stabilization_certificate(5, 3.125, 1.25, 0.20, 2.0)",
         "gold_call": "_oracle_sqpfc_stabilization_certificate(5, 3.125, 1.25, 0.20, 2.0)"},
        # S = 0 gives C_q = 0 and a strictly negative margin
        {"setup": "",
         "call": "sqpfc_stabilization_certificate(3, 95/48, 0.5, 0.25, 0.0)",
         "gold_call": "_oracle_sqpfc_stabilization_certificate(3, 95/48, 0.5, 0.25, 0.0)"},
        # at S = S_min the margin is exactly zero; Smin is S_min for (4, 2.0, 0.5, 0.3)
        {"setup": "Smin = 0.005799981935854313\n",
         "call": "sqpfc_stabilization_certificate(4, 2.0, 0.5, 0.3, Smin)[3]",
         "gold_call": ("_oracle_sqpfc_stabilization_certificate("
                       "4, 2.0, 0.5, 0.3, Smin)[3]")},
        # contract: the prefactor ((q-1)/(q-2))^(q-2) is singular at q = 2, so the
        # certificate is defined only for q >= 3 and must raise below it.
        {"setup": ("def trap(f):\n"
                  "    try:\n"
                  "        return float(np.asarray(f()).ravel()[0])\n"
                  "    except ValueError:\n"
                  "        return -12345.0\n"),
         "call": "trap(lambda: sqpfc_stabilization_certificate(2, 95/48, 0.5, 0.25, 5.0))",
         "gold_call": ("trap(lambda: _oracle_sqpfc_stabilization_certificate("
                       "2, 95/48, 0.5, 0.25, 5.0))")},
    ]
