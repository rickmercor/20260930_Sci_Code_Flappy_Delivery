"""
Compute the tolerance to which each inner Sternheimer solve may be run at the current outer iteration, using the source paper's guaranteed prefactor. This rule is the paper's contribution reduced to one function.

An outer Krylov iteration whose operator is applied only approximately does not need accurate applications throughout. Early on the outer residual is large and the outer iterate is far from the answer, so an inner solve performed to high accuracy is wasted work; as the outer residual falls the tolerance must tighten to match. The paper makes this precise: if each inner solve at outer iteration $i$ is run to a tolerance that scales inversely with the previous outer residual norm, with a per-state prefactor and a factor built from the smallest singular value of the Krylov Hessenberg matrix divided by three times the restart size, the final outer residual still meets the requested target.




Two features of the prefactor deserve care because they are where a transcription goes wrong. It contains the norm of the kernel applied to the current Krylov vector, so the tolerance is **not** a function of the iteration counters alone - it depends on the vector being operated on, and a state-independent schedule cannot reproduce it. And it contains a mixed norm of the orbitals, written $\\|\\cdot\\|_{2,\\infty}$, which is not a matrix norm in the usual sense: it is the ordinary Euclidean norm taken down one axis and the maximum taken over the other. Which axis is which is fixed by the fact that the quantity must bound the size of a single orbital on the grid, so the Euclidean norm runs over the grid index and the maximum over the states. The singular value the bound calls for belongs to the Hessenberg matrix of the final iteration and so is not available while the iteration runs; it is supplied here as a running estimate carried from the outer routine, which is a lower bound to it and therefore conservative.




****--- Formulas ---****

For the guaranteed strategy the per-state prefactor and the schedule are




$$C_n = \\frac{\\sqrt{|\\Omega|}}{2 f_n\\, \\|K v_i\\|\\, \\big\\|\\operatorname{Re}\\Phi\\big\\|_{2,\\infty}\\, \\sqrt{N_g\\, n_{\\mathrm{occ}}}}, \\qquad \\big\\|\\operatorname{Re}\\Phi\\big\\|_{2,\\infty} = \\max_{n}\\left(\\sum_{j} \\operatorname{Re}(\\Phi_{jn})^2\\right)^{1/2},$$




$$\\tau^{(0)}_n = C_n\\,\\frac{\\tau}{3}, \\qquad \\tau^{(i)}_n = C_n\\,\\frac{s}{3\\,m}\\,\\frac{\\tau}{\\|r_{i-1}\\|} \\quad (i > 0),$$




with $\\tau$ the requested outer accuracy, $s$ the running singular-value estimate, $m$ the restart size, $\\|r_{i-1}\\|$ the previous outer residual norm, $|\\Omega|$ the cell volume, $N_g$ the grid size and $n_{\\mathrm{occ}}$ the number of retained states. A state with $f_n \\le 10^{-14}$ never enters the density and takes $C_n = \\infty$.

Returns
-------
`np.ndarray` of shape `(Nocc,)`, float: one tolerance per retained state for this outer iteration, `np.inf` for any state of vanishing occupation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def adaptive_cg_tolerance(target_tol, residual_norm, sigma_min, iteration, restart_size,
                          occupations, orbitals, kernel_vector, cell_volume, n_grid, n_occ):
    """target_tol: float > 0, requested outer accuracy. residual_norm: float > 0, previous
    outer residual norm. sigma_min: float >= 0, running estimate of the smallest singular
    value of the Krylov Hessenberg matrix. iteration: int >= 0. restart_size: int > 0, the
    restart size m. occupations: (Nocc,) float in [0,1]. orbitals: (Ng,Nocc) complex retained
    states. kernel_vector: (Ng,) real, the kernel applied to the current Krylov vector.
    cell_volume: float > 0. n_grid: int. n_occ: int.
    Return the (Nocc,) array of per-state tolerances, np.inf where the occupation vanishes."""
    # Implement per the formulas above.
    tolerances = None
    return tolerances

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_adaptive_cg_tolerance(target_tol, residual_norm, sigma_min, iteration,
                                  restart_size, occupations, orbitals, kernel_vector,
                                  cell_volume, n_grid, n_occ):
    tau = float(target_tol)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("target_tol must be finite and positive")
    f = np.asarray(occupations, dtype=float)
    if f.ndim != 1 or f.shape[0] < 1 or not np.all(np.isfinite(f)):
        raise ValueError("occupations must be a finite 1-D array")
    if np.any(f < -1e-12) or np.any(f > 1.0 + 1e-12):
        raise ValueError("occupations must lie in [0, 1]")
    Phi = np.asarray(orbitals)
    if Phi.ndim != 2 or Phi.shape[1] != f.shape[0]:
        raise ValueError("orbitals must have shape (Ng, Nocc) matching occupations")
    Kv = np.asarray(kernel_vector, dtype=float)
    if Kv.ndim != 1 or Kv.shape[0] < 1 or not np.all(np.isfinite(Kv)):
        raise ValueError("kernel_vector must be a finite 1-D array")
    vol = float(cell_volume)
    if not np.isfinite(vol) or vol <= 0.0:
        raise ValueError("cell_volume must be finite and positive")
    if int(n_grid) < 1 or int(n_occ) < 1:
        raise ValueError("n_grid and n_occ must be positive integers")
    it = int(iteration)
    if it < 0:
        raise ValueError("iteration must be a non-negative integer")
    m = int(restart_size)
    if m < 1:
        raise ValueError("restart_size must be a positive integer")

    nrmK = float(np.linalg.norm(Kv))
    if nrmK <= 0.0:
        raise ValueError("kernel_vector must have positive norm")
    orb = float(np.max(np.linalg.norm(np.real(Phi), axis=0)))
    if orb <= 0.0:
        raise ValueError("orbitals must have a non-zero real part")
    C = np.empty_like(f)
    small = f <= 1e-14
    C[~small] = np.sqrt(vol) / (2.0 * f[~small] * nrmK * orb
                                * np.sqrt(float(n_grid) * float(n_occ)))
    C[small] = np.inf
    if it == 0:
        return C * (tau / 3.0)
    rn = float(residual_norm)
    if not np.isfinite(rn) or rn <= 0.0:
        raise ValueError("residual_norm must be finite and positive")
    sm = float(sigma_min)
    if not np.isfinite(sm) or sm < 0.0:
        raise ValueError("sigma_min must be finite and non-negative")
    return C * (sm / (3.0 * m)) * (tau / rn)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the first outer iteration, where the rule takes its separate form.
        {"setup": "import numpy as np\nrng = np.random.default_rng(1)\nocc = np.array([0.99, 0.7, 0.3, 0.02])\nPhi = rng.normal(size=(12,4)) + 1j*rng.normal(size=(12,4))\nKv = rng.normal(size=12)\n",
         "call": "adaptive_cg_tolerance(1e-2, 1.0, 1.0, 0, 5, occ, Phi, Kv, 7.0, 12, 4)",
         "gold_call": "_oracle_adaptive_cg_tolerance(1e-2, 1.0, 1.0, 0, 5, occ, Phi, Kv, 7.0, 12, 4)"},
        # normal: a later iteration, where the adaptive factor is active.
        {"setup": "import numpy as np\nrng = np.random.default_rng(1)\nocc = np.array([0.99, 0.7, 0.3, 0.02])\nPhi = rng.normal(size=(12,4)) + 1j*rng.normal(size=(12,4))\nKv = rng.normal(size=12)\n",
         "call": "adaptive_cg_tolerance(1e-2, 0.05, 0.4, 3, 5, occ, Phi, Kv, 7.0, 12, 4)",
         "gold_call": "_oracle_adaptive_cg_tolerance(1e-2, 0.05, 0.4, 3, 5, occ, Phi, Kv, 7.0, 12, 4)"},
        # boundary: a different system, a different restart size.
        {"setup": "import numpy as np\nrng = np.random.default_rng(7)\nocc = np.array([0.95, 0.55, 0.11])\nPhi = rng.normal(size=(16,3)) + 1j*rng.normal(size=(16,3))\nKv = rng.normal(size=16)\n",
         "call": "adaptive_cg_tolerance(5e-2, 0.011, 0.17, 6, 3, occ, Phi, Kv, 11.0, 16, 3)",
         "gold_call": "_oracle_adaptive_cg_tolerance(5e-2, 0.011, 0.17, 6, 3, occ, Phi, Kv, 11.0, 16, 3)"},
        # edge: the tolerance is inversely proportional to the kernel-vector norm.
        {"setup": "import numpy as np\nrng = np.random.default_rng(1)\nocc = np.array([0.99, 0.7, 0.3, 0.02])\nPhi = rng.normal(size=(12,4)) + 1j*rng.normal(size=(12,4))\nKv = rng.normal(size=12)\n",
         "call": "int(np.allclose(adaptive_cg_tolerance(1e-2, 0.05, 0.4, 3, 5, occ, Phi, 2.0*Kv, 7.0, 12, 4), 0.5*adaptive_cg_tolerance(1e-2, 0.05, 0.4, 3, 5, occ, Phi, Kv, 7.0, 12, 4), rtol=1e-12, atol=0))",
         "gold_call": "1"},
        # edge: proportional to the requested accuracy, inverse in the previous residual.
        {"setup": "import numpy as np\nrng = np.random.default_rng(1)\nocc = np.array([0.99, 0.7, 0.3, 0.02])\nPhi = rng.normal(size=(12,4)) + 1j*rng.normal(size=(12,4))\nKv = rng.normal(size=12)\n",
         "call": "int(np.allclose(adaptive_cg_tolerance(2e-2, 0.10, 0.4, 3, 5, occ, Phi, Kv, 7.0, 12, 4), adaptive_cg_tolerance(1e-2, 0.05, 0.4, 3, 5, occ, Phi, Kv, 7.0, 12, 4), rtol=1e-12, atol=0))",
         "gold_call": "1"},
        # edge: at fixed restart size the rule does not depend on the iteration index.
        {"setup": "import numpy as np\nrng = np.random.default_rng(1)\nocc = np.array([0.99, 0.7, 0.3, 0.02])\nPhi = rng.normal(size=(12,4)) + 1j*rng.normal(size=(12,4))\nKv = rng.normal(size=12)\n",
         "call": "int(np.allclose(adaptive_cg_tolerance(1e-2, 0.05, 0.4, 3, 5, occ, Phi, Kv, 7.0, 12, 4), adaptive_cg_tolerance(1e-2, 0.05, 0.4, 9, 5, occ, Phi, Kv, 7.0, 12, 4), rtol=1e-12, atol=0))",
         "gold_call": "1"},
        # edge: a state of vanishing occupation is excluded, the others stay finite.
        {"setup": "import numpy as np\nrng = np.random.default_rng(5)\nocc = np.array([0.9, 0.5, 1e-20, 0.2])\nPhi = rng.normal(size=(9,4)) + 1j*rng.normal(size=(9,4))\nKv = rng.normal(size=9)\n",
         "call": "int(bool(np.isinf(adaptive_cg_tolerance(1e-2, 0.5, 0.8, 2, 4, occ, Phi, Kv, 5.0, 9, 4)[2]) and np.all(np.isfinite(adaptive_cg_tolerance(1e-2, 0.5, 0.8, 2, 4, occ, Phi, Kv, 5.0, 9, 4)[[0,1,3]]))))",
         "gold_call": "1"},
    ]
