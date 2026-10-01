"""
Build the two ****pooled Fisher information matrices**** of the candidate training data: one accumulated over the configuration-energy data, one over the per-atom force data. Return them stacked as an array of shape $(2, P, P)$ with $P$ the number of master parameters, energy first. ****The per-datum Fisher information is not given: derive it**** from the definition in the source paper for the Gaussian model implied by the paper's weighted sum of squared energy and force residuals, and note that the three Cartesian components of one atom's force share that atom's single weight, so an atom is one datum, not three. The matrices returned here are **unweighted** pools - the weights are the common scales that step 7 determines - so each is the plain sum of its data's per-datum matrices.

For a model whose data are independent and Gaussian about the model prediction, the expected Hessian of the negative log-likelihood separates into a term quadratic in the prediction gradient and a term proportional to the residual times the prediction's second derivative. The second term has zero expectation, because the residual does, so the Fisher information of one datum reduces to the outer product of its prediction gradient with itself, scaled by the inverse variance - which is what the weight in a weighted least-squares loss is. Information from independent data is additive, so the total is a weighted sum of those outer products; grouping data that share a weight lets that weight be factored out of the group, which is what makes a common scale per data type meaningful. A datum that is a **vector** of $k$ components sharing one weight contributes the sum of $k$ outer products, i.e. $G^{\\mathsf T}G$ for the $k \\times P$ block $G$ of its rows in the Jacobian, of rank at most $k$.

****--- Formulas ---****

For configuration $m$ let $J^{(m)}$ be the $(3N_m+1)\\times P$ Jacobian of step 5 applied to step 3's output. Its first row is the energy gradient $\\boldsymbol g^{(m)}_E$ and rows $1+3i \\ldots 3+3i$ are the $3\\times P$ block $G^{(m)}_i$ of atom $i$. Then

$$\\mathcal I_E = \\sum_{m}\\boldsymbol g^{(m)}_E\\big(\\boldsymbol g^{(m)}_E\\big)^{\\mathsf T}, \\qquad \\mathcal I_F = \\sum_{m}\\sum_{i=0}^{N_m-1}\\big(G^{(m)}_i\\big)^{\\mathsf T}G^{(m)}_i ,$$

both symmetric positive semidefinite and $P\\times P$, returned as `np.stack([I_E, I_F])`.

Returns
-------
`np.ndarray` of shape `(2, P, P)`, real, both slices symmetric and positive semidefinite. Entry `[0]` is the pooled configuration-energy information, entry `[1]` the pooled per-atom force information.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def data_fisher_matrices(positions_list, cells, params,
                         master_idx=(0, 5, 6, 7, 8, 18, 19), h=5e-3, r_cut=6.0):
    """positions_list: sequence of (N_m, 3) position arrays, one per configuration.
    cells: sequence of box specifications, one per configuration (each a scalar or a
    length-3 vector); a single scalar is broadcast to every configuration.
    params: the length-20 EAM parameter vector to evaluate the information at.
    master_idx, h: passed through to the parameter Jacobian.
    r_cut: the hard cutoff.
    Return the real array np.stack([I_E, I_F]) of shape (2, P, P), P = len(master_idx)."""
    # Implement per the principle above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

MASTER_IDX = (0, 5, 6, 7, 8, 18, 19)

def _oracle_data_fisher_matrices(positions_list, cells, params,
                                 master_idx=MASTER_IDX, h=5e-3, r_cut=6.0):
    p = _check_params(params)
    plist = list(positions_list)
    if len(plist) < 1:
        raise ValueError("at least one configuration is required")
    cl = list(cells) if not np.isscalar(cells) else [cells] * len(plist)
    if len(cl) != len(plist):
        raise ValueError("positions_list and cells must have the same length")
    n = len(list(np.atleast_1d(np.asarray(master_idx)).ravel()))
    IE = np.zeros((n, n))
    IF = np.zeros((n, n))
    for pos, cell in zip(plist, cl):
        pos = np.asarray(pos, float)
        Jm = _oracle_log_parameter_jacobian(
            lambda q, _p=pos, _c=cell: _oracle_eam_energy_forces(_p, _c, q, r_cut),
            p, master_idx, h)
        IE += np.outer(Jm[0], Jm[0])
        for i in range(pos.shape[0]):
            G = Jm[1 + 3 * i:4 + 3 * i]
            IF += G.T @ G
    return np.stack([IE, IF])

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
        # normal: three configurations spanning all three branches of the embedding
        # function (densities 46.6-51.9, 32.8-36.2 and 23.2-31.1), so the pooled
        # matrices are non-singular in all seven directions.
        {"setup": "import numpy as np\n" + TA + CFG +
                  "C = [_cfg(m) for m in (0, 4, 7)]\nPL = [c[0] for c in C]\n"
                  "CL = [c[1] for c in C]\n",
         "call": "data_fisher_matrices(PL, CL, TA)",
         "gold_call": "_oracle_data_fisher_matrices(PL, CL, TA)"},
        # boundary: a SINGLE configuration, where the energy pool is rank one and the
        # force pool rank at most 7; the accumulation must still be correct.
        {"setup": "import numpy as np\n" + TA + CFG +
                  "C = [_cfg(3)]\nPL = [c[0] for c in C]\nCL = [c[1] for c in C]\n",
         "call": "data_fisher_matrices(PL, CL, TA)",
         "gold_call": "_oracle_data_fisher_matrices(PL, CL, TA)"},
        # boundary: 3x3x3 supercells, 54 atoms each, so the number of per-atom force
        # data per configuration is not the 16 of the production pool.
        {"setup": "import numpy as np\n" + TA + CFG +
                  "C = [_cfg(m, 3) for m in (1, 5)]\nPL = [c[0] for c in C]\n"
                  "CL = [c[1] for c in C]\n",
         "call": "data_fisher_matrices(PL, CL, TA)",
         "gold_call": "_oracle_data_fisher_matrices(PL, CL, TA)"},
        # edge: a three-parameter subset with a finer Jacobian step, so the output is
        # (2, 3, 3) and the column selection is not the default.
        {"setup": "import numpy as np\n" + TA + CFG +
                  "C = [_cfg(m) for m in (2, 9)]\nPL = [c[0] for c in C]\n"
                  "CL = [c[1] for c in C]\n",
         "call": "data_fisher_matrices(PL, CL, TA, (0, 5, 6), 1e-3)",
         "gold_call": "_oracle_data_fisher_matrices(PL, CL, TA, (0, 5, 6), 1e-3)"},
        # edge: a short cutoff and a coarse step together, which changes both the
        # neighbour set inside step 3 and the differencing inside step 5.
        {"setup": "import numpy as np\n" + TA + CFG +
                  "C = [_cfg(m) for m in (6, 11)]\nPL = [c[0] for c in C]\n"
                  "CL = [c[1] for c in C]\n",
         "call": "data_fisher_matrices(PL, CL, TA, (0, 5, 6, 7, 8, 18, 19), 1e-2, 4.5)",
         "gold_call": "_oracle_data_fisher_matrices(PL, CL, TA, (0, 5, 6, 7, 8, 18, 19), "
                      "1e-2, 4.5)"},
    ]
