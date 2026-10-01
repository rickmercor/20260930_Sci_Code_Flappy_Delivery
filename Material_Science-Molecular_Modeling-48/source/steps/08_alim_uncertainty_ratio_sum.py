"""
Run the whole benchmark. Build the pool of $n_conf$ candidate configurations from the closed form below; evaluate the pooled energy and force information matrices at the fixed EAM parameters; combine them into one pooled matrix using the fixed ratio of the energy weight to the force weight; evaluate the Jacobian of the five indicator properties; find the smallest common weight scale that satisfies the information-matching condition; form the total information matrix at that scale; ****propagate it back to a predicted uncertainty for each of the five indicator properties**** using the source paper's propagation formula; and return the ****sum of the five ratios of predicted to target uncertainty****. Setting `data='E'` restricts the pool to configuration energies and `data='F'` to per-atom forces; $delta=None$ means the source paper's own target uncertainties for the five indicator properties, in the order of step 4.

This step is the whole workflow: an outer construction of candidate data wrapped around the information machinery of steps 5 to 7, with the potential of steps 1 to 3 underneath. Two structural points govern it. First, the two information matrices enter through their **weighted** sum, and under the paper's simplest weighting scheme the weights carry only one free scale, so the matching condition determines them completely - a single number, obtained once, fixes the whole training-set weighting. Second, the guarantee the condition buys is visible in the output: because the total information dominates each target's own rank-one requirement, every propagated uncertainty is strictly smaller than its target, so the returned sum is strictly less than the number of quantities of interest. That bound is a free self-check, and an implementation that violates it has an error upstream. The sum is also invariant under a uniform rescaling of all five target uncertainties, so only their ratios matter.

****--- Formulas ---****

Configuration $m$ is a $n_cell$$^3$ BCC supercell of lattice constant $a_m = 3.05 + 0.06\\,(m \\bmod 8)$ A in a cube of edge $L_m = a_m\\,$$n_cell$, built as the $n_cell$$^3$ cube corners $(i,j,k)a_m$ in `numpy.meshgrid(..., indexing='ij')` order followed by the same points offset by $(\\tfrac12,\\tfrac12,\\tfrac12)a_m$, atom $n$ of that list then displaced by

$$\\boldsymbol u_n = \\mathcal A_m\\big[\\sin(1.7n + 0.9m + 0.3),\\; \\sin(2.3n + 1.4m + 1.1),\\; \\sin(3.1n + 2.2m + 1.9)\\big], \\quad \\mathcal A_m = 0.08 + 0.04\\,(m \\bmod 4),$$

and the result wrapped into $[0, L_m)$. With $\\mathcal I_E, \\mathcal I_F$ from step 6, $H$ the QoI Jacobian of steps 4 and 5, and $w = $ $w_ratio$,

$$\\mathcal I_{\\mathrm{pool}} = \\begin{cases}\\mathcal I_E, & \\texttt{data='E'},\\\\ \\mathcal I_F, & \\texttt{data='F'},\\\\ w\\,\\mathcal I_E + \\mathcal I_F, & \\texttt{data='EF'},\\end{cases} \\qquad t^{\\star} = \\texttt{minimal\\_information\\_scale}\\big(\\mathcal I_{\\mathrm{pool}}, H, \\boldsymbol\\delta\\big),$$

$$\\mathcal I = t^{\\star}\\,\\mathcal I_{\\mathrm{pool}}, \\qquad \\sigma_n = \\sqrt{\\boldsymbol h_n^{\\mathsf T}\\,\\mathcal I^{-1}\\,\\boldsymbol h_n}, \\qquad \\text{answer} = \\sum_{n=1}^{5}\\frac{\\sigma_n}{\\delta_n}.$$

Returns
-------
`float`: the summed uncertainty ratio. It is strictly between $0$ and the number of quantities of interest whenever the matching condition is satisfiable, it is unchanged when every entry of `delta` is multiplied by one common factor, and it is *not* additive over `data`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def alim_uncertainty_ratio_sum(params=None, n_conf=12, n_cell=2, data="EF",
                               w_ratio=1.0e-2, r_cut=6.0, h=5.0e-3, delta=None,
                               master_idx=(0, 5, 6, 7, 8, 18, 19),
                               a_start=3.30, h_a=1.0e-4, n_newton=8, h_strain=1.0e-2):
    """params: length-20 EAM parameter vector; None means the fixed tantalum set.
    n_conf, n_cell: number of candidate configurations and the supercell repeat.
    data: 'E', 'F' or 'EF' - which data types enter the pooled information.
    w_ratio: the ratio of the common energy weight to the common force weight.
    r_cut: the hard cutoff. h: the parameter-Jacobian step.
    delta: the five target uncertainties; None means the source paper's values.
    master_idx: the adjustable parameter indices.
    a_start, h_a, n_newton, h_strain: passed to the indicator-property step.
    Return the sum over the five indicator properties of sigma_n / delta_n, a float."""
    # Implement per the principle above.
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

MASTER_IDX = (0, 5, 6, 7, 8, 18, 19)

TA_PARAMS = np.array([
    2.860082, 3.086341, 33.787168, 33.787168, 8.489528, 4.527748,
    0.611679, 1.032101, 0.176977, 0.353954,
    -5.103845, -0.405524, 1.112997, -3.585325,
    -5.14, 0.0, 1.640098, 0.221375, 0.848843, -5.141526])

DELTA_QOI = (0.0075, 1.0869, 9.6143, 5.9992, 6.4602)

def _alim_configuration(m, n_cell=2):
    a = 3.05 + 0.06 * (m % 8)
    L = a * n_cell
    amp = 0.08 + 0.04 * (m % 4)
    c = np.arange(n_cell, dtype=float)
    I, J, K = np.meshgrid(c, c, c, indexing="ij")
    b = np.stack([I, J, K], -1).reshape(-1, 3)
    R = np.concatenate([b, b + 0.5]) * a
    i = np.arange(R.shape[0], dtype=float)
    u = amp * np.stack([np.sin(1.7 * i + 0.9 * m + 0.3),
                        np.sin(2.3 * i + 1.4 * m + 1.1),
                        np.sin(3.1 * i + 2.2 * m + 1.9)], axis=1)
    return (R + u) % L, np.full(3, L)

def _oracle_alim_uncertainty_ratio_sum(params=None, n_conf=12, n_cell=2, data="EF",
                                       w_ratio=1.0e-2, r_cut=6.0, h=5.0e-3,
                                       delta=None, master_idx=MASTER_IDX,
                                       a_start=3.30, h_a=1.0e-4, n_newton=8,
                                       h_strain=1.0e-2):
    p = _check_params(TA_PARAMS if params is None else params)
    if not (isinstance(n_conf, (int, np.integer)) or float(n_conf).is_integer()):
        raise ValueError("n_conf must be an integer")
    if int(n_conf) < 1:
        raise ValueError("n_conf must be a positive integer")
    if int(n_cell) < 1:
        raise ValueError("n_cell must be a positive integer")
    if str(data) not in ("E", "F", "EF"):
        raise ValueError("data must be one of 'E', 'F', 'EF'")
    if np.ndim(w_ratio) != 0 or not np.isfinite(float(w_ratio)) or float(w_ratio) < 0.0:
        raise ValueError("w_ratio must be a finite non-negative scalar")
    delta = DELTA_QOI if delta is None else delta
    confs = [_alim_configuration(m, int(n_cell)) for m in range(int(n_conf))]
    IE, IF = _oracle_data_fisher_matrices([c[0] for c in confs], [c[1] for c in confs],
                                          p, master_idx, h, r_cut)
    H = _oracle_log_parameter_jacobian(
        lambda q: _oracle_bcc_indicator_properties(q, r_cut, a_start, h_a,
                                                   n_newton, h_strain),
        p, master_idx, h)
    if str(data) == "E":
        Ipool = IE
    elif str(data) == "F":
        Ipool = IF
    else:
        Ipool = float(w_ratio) * IE + IF
    t = _oracle_minimal_information_scale(Ipool, H, delta)
    if t <= 0.0:
        raise ValueError("the target information is empty; the matching scale vanishes")
    Iinv = np.linalg.inv(t * Ipool)
    d = np.atleast_1d(np.asarray(delta, float)).ravel()
    sig = np.sqrt(np.einsum("ni,ij,nj->n", H, Iinv, H))
    return float(np.sum(sig / d))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the production instance, the graded observable.
        {"setup": "import numpy as np\n",
         "call": "alim_uncertainty_ratio_sum()",
         "gold_call": "_oracle_alim_uncertainty_ratio_sum()"},
        # control: configuration ENERGIES only. A candidate who never implements the
        # per-atom force pooling returns this value for case 1.
        {"setup": "import numpy as np\n",
         "call": "alim_uncertainty_ratio_sum(data='E')",
         "gold_call": "_oracle_alim_uncertainty_ratio_sum(data='E')"},
        # control: per-atom FORCES only. A candidate who never implements the energy
        # pooling, or who sets w_ratio to zero, returns this value for case 1.
        {"setup": "import numpy as np\n",
         "call": "alim_uncertainty_ratio_sum(data='F')",
         "gold_call": "_oracle_alim_uncertainty_ratio_sum(data='F')"},
        # control: energies and forces weighted EQUALLY. A candidate who ignores the
        # stated 1e-2 ratio returns this value for case 1.
        {"setup": "import numpy as np\n",
         "call": "alim_uncertainty_ratio_sum(w_ratio=1.0)",
         "gold_call": "_oracle_alim_uncertainty_ratio_sum(w_ratio=1.0)"},
        # boundary: a four-configuration pool, so the pooled matrices are built from a
        # different set of lattice constants and the answer is not the production one.
        {"setup": "import numpy as np\n",
         "call": "alim_uncertainty_ratio_sum(n_conf=4)",
         "gold_call": "_oracle_alim_uncertainty_ratio_sum(n_conf=4)"},
        # boundary: 3x3x3 supercells, 54 atoms each, three configurations. Every atom
        # count, box length and image-shell count changes at once.
        {"setup": "import numpy as np\n",
         "call": "alim_uncertainty_ratio_sum(n_conf=3, n_cell=3)",
         "gold_call": "_oracle_alim_uncertainty_ratio_sum(n_conf=3, n_cell=3)"},
        # boundary: target uncertainties supplied explicitly and NOT proportional to the
        # paper's, so the invariance under uniform rescaling cannot hide a wrong delta.
        {"setup": "import numpy as np\n",
         "call": "alim_uncertainty_ratio_sum(delta=(0.01, 1.0, 10.0, 6.0, 6.0))",
         "gold_call": "_oracle_alim_uncertainty_ratio_sum(delta=(0.01, 1.0, 10.0, 6.0, 6.0))"},
        # edge: the five identifiable master parameters only, dropping eta and F_e. The
        # matrices become 5x5 and the QoI Jacobian is then full rank.
        {"setup": "import numpy as np\n",
         "call": "alim_uncertainty_ratio_sum(master_idx=(0, 5, 6, 7, 8))",
         "gold_call": "_oracle_alim_uncertainty_ratio_sum(master_idx=(0, 5, 6, 7, 8))"},
        # edge: every numerical knob off its default at once - pool size, cutoff, the
        # parameter step and the strain step.
        {"setup": "import numpy as np\n",
         "call": "alim_uncertainty_ratio_sum(n_conf=5, r_cut=4.5, h=1e-2, h_strain=5e-3)",
         "gold_call": "_oracle_alim_uncertainty_ratio_sum(n_conf=5, r_cut=4.5, h=1e-2, "
                      "h_strain=5e-3)"},
    ]
