"""
Return the Jacobian of an arbitrary vector-valued model output with respect to the ****logarithms**** of a chosen subset of the potential's parameters, by central differences. Column $j$ of the result is $\\partial(\\text{output})/\\partial\\theta_j$ with $\\theta_j = \\log|p_j|$ at fixed sign, evaluated as $\\big[\\mathrm{func}(p\\ \\text{with}\\ p_j \\to p_je^{h}) - \\mathrm{func}(p\\ \\text{with}\\ p_j \\to p_je^{-h})\\big]/(2h)$, every other entry of `params` held fixed. ****The perturbation is multiplicative, not additive in $\\log p$****, which is the only formulation that works for the one master parameter that is negative; taking a logarithm of it directly is a domain error. The output of `func` is flattened to one dimension before differencing, so a scalar-valued `func` gives a Jacobian with one row.

Potential parameters carry different physical units and differ by orders of magnitude, so a Jacobian taken with respect to the raw parameters is badly scaled and the Fisher information matrix built from it is badly conditioned. Working in the logarithms of the parameters makes every column dimensionless in the parameter direction and turns a relative parameter change into an absolute one: $\\partial g/\\partial\\log p = p\\,\\partial g/\\partial p$. That is why a purely multiplicative perturbation implements exactly the logarithmic derivative, and why the sign of the parameter is irrelevant - only its magnitude is exponentiated. The step size trades truncation, which falls as $h^2$, against cancellation, which grows as $1/h$; here it is prescribed rather than tuned, because the graded quantity is the value of a fixed recipe.

****--- Formulas ---****

For each selected index $j$,

$$\\big[J\\big]_{\\cdot\\,j} \\;=\\; \\frac{\\mathrm{func}\\big(\\ldots,\\, p_je^{h},\\, \\ldots\\big) - \\mathrm{func}\\big(\\ldots,\\, p_je^{-h},\\, \\ldots\\big)}{2h} \\;=\\; \\frac{\\partial\\,\\mathrm{func}}{\\partial\\log|p_j|} + O(h^{2}),$$

the columns being ordered exactly as $master_idx$ lists them. If `func` returns an array of $n$ entries after flattening, the result has shape $(n, |\\text{master\\_idx}|)$.

Returns
-------
`np.ndarray` of shape `(n_out, len(master_idx))`, real and finite, where `n_out` is the size of `func(params)` after flattening. Column order follows `master_idx`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def log_parameter_jacobian(func, params, master_idx=(0, 5, 6, 7, 8, 18, 19), h=5e-3):
    """func: callable taking a length-20 parameter vector and returning a scalar or an
    array of model outputs.
    params: the length-20 EAM parameter vector to differentiate about.
    master_idx: the indices of params to differentiate with respect to, in order.
    h: the central-difference step in the logarithm of each parameter.
    Return the real (n_out, n_master) Jacobian with respect to log|p_j|."""
    # Implement per the formulas above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

MASTER_IDX = (0, 5, 6, 7, 8, 18, 19)

def _oracle_log_parameter_jacobian(func, params, master_idx=MASTER_IDX, h=5e-3):
    p = np.asarray(params, float).ravel().copy()
    idx = [int(j) for j in np.atleast_1d(np.asarray(master_idx)).ravel()]
    if len(idx) < 1:
        raise ValueError("master_idx must contain at least one index")
    if any(j < 0 or j >= p.size for j in idx):
        raise ValueError("every master index must address an entry of params")
    if len(set(idx)) != len(idx):
        raise ValueError("master_idx must not repeat an index")
    if np.any(p[idx] == 0.0):
        raise ValueError("a master parameter of zero has no logarithm")
    if np.ndim(h) != 0 or not np.isfinite(float(h)) or float(h) <= 0.0:
        raise ValueError("h must be a finite positive scalar")
    h = float(h)
    cols = []
    for j in idx:
        pp = p.copy(); pp[j] = p[j] * np.exp(h)
        pm = p.copy(); pm[j] = p[j] * np.exp(-h)
        cols.append((np.atleast_1d(np.asarray(func(pp), float)).ravel()
                     - np.atleast_1d(np.asarray(func(pm), float)).ravel()) / (2.0 * h))
    return np.stack(cols, axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    TA = ("TA = np.array([2.860082, 3.086341, 33.787168, 33.787168, 8.489528, 4.527748,\n"
          "              0.611679, 1.032101, 0.176977, 0.353954,\n"
          "              -5.103845, -0.405524, 1.112997, -3.585325,\n"
          "              -5.14, 0.0, 1.640098, 0.221375, 0.848843, -5.141526])\n")
    CFG = ("def _cfg(m, n_cell=2):\n"
           "    a = 3.05 + 0.06 * (m % 8); L = a * n_cell\n"
           "    amp = 0.08 + 0.04 * (m % 4)\n"
           "    c = np.arange(n_cell, dtype=float)\n"
           "    I, J, K = np.meshgrid(c, c, c, indexing='ij')\n"
           "    b = np.stack([I, J, K], -1).reshape(-1, 3)\n"
           "    R = np.concatenate([b, b + 0.5]) * a\n"
           "    i = np.arange(R.shape[0], dtype=float)\n"
           "    u = amp * np.stack([np.sin(1.7*i + 0.9*m + 0.3),\n"
           "                        np.sin(2.3*i + 1.4*m + 1.1),\n"
           "                        np.sin(3.1*i + 2.2*m + 1.9)], axis=1)\n"
           "    return (R + u) % L, np.full(3, L)\n")
    return [
        # normal: the QoI Jacobian of the production instance, 5 rows by 7 columns. Its
        # last two columns are EXACTLY zero, which is the identifiability control.
        {"setup": "import numpy as np\n" + TA +
                  "f = lambda q: bcc_indicator_properties(q)\n",
         "call": "log_parameter_jacobian(f, TA)",
         "gold_call": "_oracle_log_parameter_jacobian(f, TA)"},
        # normal: the data Jacobian of one configuration, 49 rows by 7 columns.
        {"setup": "import numpy as np\n" + TA + CFG + "pos, cell = _cfg(6)\n"
                  "f = lambda q: eam_energy_forces(pos, cell, q)\n",
         "call": "log_parameter_jacobian(f, TA)",
         "gold_call": "_oracle_log_parameter_jacobian(f, TA)"},
        # boundary: a closed-form func with a NEGATIVE master parameter in the index set.
        # d/dlog|p| of p^2 q is 2 p^2 q, so this has an analytic check, and an
        # implementation that writes np.log(p) instead of a multiplicative step fails.
        {"setup": "import numpy as np\n" + TA +
                  "f = lambda q: np.array([q[0]**2 * q[6], np.log(abs(q[19])) * q[5]])\n",
         "call": "log_parameter_jacobian(f, TA, (0, 5, 6, 19), 1e-4)",
         "gold_call": "_oracle_log_parameter_jacobian(f, TA, (0, 5, 6, 19), 1e-4)"},
        # boundary: only the two third-branch parameters, differentiated through the
        # embedding function at densities that straddle rho_0.
        {"setup": "import numpy as np\n" + TA +
                  "f = lambda q: eam_embedding(np.array([31.0, 40.0]), q).ravel()\n",
         "call": "log_parameter_jacobian(f, TA, (18, 19), 2e-3)",
         "gold_call": "_oracle_log_parameter_jacobian(f, TA, (18, 19), 2e-3)"},
        # edge: a SCALAR-valued func and a three-index subset, so the result must be a
        # (1, 3) matrix rather than a flat vector.
        {"setup": "import numpy as np\n" + TA +
                  "f = lambda q: float(bcc_indicator_properties(q)[1])\n",
         "call": "log_parameter_jacobian(f, TA, (0, 6, 7))",
         "gold_call": "_oracle_log_parameter_jacobian(f, TA, (0, 6, 7))"},
        # edge: a coarse step of 1e-2 through step 1, where truncation is visible; the
        # graded quantity is the value of the fixed recipe, not the exact derivative.
        {"setup": "import numpy as np\n" + TA +
                  "f = lambda q: eam_pair_and_density(np.array([2.6, 3.4]), q).ravel()\n",
         "call": "log_parameter_jacobian(f, TA, (0, 5, 6, 7, 8, 18, 19), 1e-2)",
         "gold_call": "_oracle_log_parameter_jacobian(f, TA, (0, 5, 6, 7, 8, 18, 19), 1e-2)"},
    ]
