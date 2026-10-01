"""
Solve the preconditioned Dyson equation by restarted inexact GMRES: apply the Kerker preconditioner to both sides, run the Arnoldi process with an operator that is deliberately applied only approximately, maintain the running singular-value estimate the tolerance rule needs, and restart when the cycle is exhausted. The screening kernel is built here.

This is the paper's algorithm assembled. The outer iteration is a restarted GMRES on the preconditioned system, and three things distinguish it from a textbook implementation. First, the operator is never assembled: each application costs one inner Sternheimer solve per retained state, and each of those is run only to the tolerance the previous step prescribes. Second, the error budget is split in three equal parts, one for the initial residual, one shared across the inexact operator applications and one for the outer iteration, so the outer iteration is entitled only to its own third of the requested accuracy. Third, the singular value the tolerance rule needs belongs to the final Hessenberg matrix and is unknown while the cycle runs, so the algorithm carries a running estimate and updates it at each restart to the smallest value encountered; convergence is accepted only when that estimate is genuinely a lower bound to the current cycle's singular value, and otherwise the method restarts again with the estimate corrected.




One implementation detail from the paper matters for accuracy rather than cost. The susceptibility is linear, so applying the dielectric operator to a Krylov vector can be written by pulling the norm of the kernel-applied vector out of the susceptibility and putting it back afterwards. This keeps the right-hand side of every inner solve of order one regardless of how small the kernel-applied vector has become, which avoids underflow late in the iteration when the Krylov vectors are small.




****--- Formulas ---****

The preconditioned system and the kernel are




$$\\mathbf{T}\\big(\\mathbf{I} - \\chi_0 K\\big)\\delta n = \\mathbf{T} b, \\qquad (K\\delta n)(x_j) = \\frac{1}{N_g}\\sum_{p} v(G_p) e^{\\mathrm{i}G_p x_j}\\sum_{q} e^{-\\mathrm{i}G_p x_q}\\delta n(x_q) + c_{\\mathrm{xc}}\\delta n(x_j),$$




with $\\mathbf{T}$ the preconditioner of the earlier step and $v(G)$ the Coulomb factor of the grid step. Start each cycle from the current iterate $x_0$, form $r_0 = \\mathbf{T}b - \\mathbf{T}\\mathcal{E}x_0$, and run at most $m$ Arnoldi steps. At step $k$ set $Kv = K q_k$, take the inner tolerances from the previous step evaluated with $Kv$, and apply the operator as




$$\\mathcal{E}q_k = q_k - \\|Kv\\|\\, \\chi_0\\!\\left[\\frac{Kv}{\\|Kv\\|}\\right], \\qquad \\text{then precondition: } v \\leftarrow \\mathbf{T}\\big(q_k - \\mathcal{E}\\text{-image}\\big).$$




Orthogonalise against the previous basis vectors, solve the small least-squares problem for $y$, and stop the cycle as soon as the estimated residual $\\|\\mathbf{H}_k y - \\beta_0 e_1\\|$ falls to $\\tau/3$ or below. At the end of a cycle set $x = x_0 + Q y$, compute $\\sigma_{\\min}$ of the Hessenberg block just built, and return $x$ only if the cycle converged **and** the carried estimate $s$ satisfies $s \\le \\sigma_{\\min}$; otherwise set $s = \\min(s, \\sigma_{\\min})$, take $x$ as the new starting iterate and restart.

Returns
-------
`np.ndarray` of shape `(Ng,)`, real: the self-consistent screened first-order density response, the solution of the Dyson equation to the requested outer tolerance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def preconditioned_gmres(hamiltonian, orbitals, eigenvalues, occupations, rhs,
                         coulomb_gspace, xc_local, g_squared, alpha, beta, target_tol,
                         cell_volume, restart_size=5, max_cycles=12, cg_max_iter=400):
    """hamiltonian, orbitals, eigenvalues, occupations: as in the susceptibility step.
    rhs: (Ng,) real right-hand side. coulomb_gspace: (Ng,) real Coulomb factor.
    xc_local: float contact coefficient. g_squared: (Ng,) real squared wavevectors.
    alpha: float > 0, Kerker parameter. beta: float > 0. target_tol: float > 0, requested
    outer accuracy. cell_volume: float > 0. restart_size: int > 0, the restart size m.
    max_cycles: int > 0, cap on restarts. cg_max_iter: int > 0, inner cap.
    Return the (Ng,) real interacting density response. The screening kernel is built here."""
    # Implement per the formulas above.
    delta_n = None
    return delta_n

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_preconditioned_gmres(hamiltonian, orbitals, eigenvalues, occupations, rhs,
                                 coulomb_gspace, xc_local, g_squared, alpha, beta,
                                 target_tol, cell_volume, restart_size=5,
                                 max_cycles=12, cg_max_iter=400):
    b = np.asarray(rhs, dtype=float)
    if b.ndim != 1 or b.shape[0] < 1 or not np.all(np.isfinite(b)):
        raise ValueError("rhs must be a finite 1-D array")
    tau = float(target_tol)
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("target_tol must be finite and positive")
    vol = float(cell_volume)
    if not np.isfinite(vol) or vol <= 0.0:
        raise ValueError("cell_volume must be finite and positive")
    m = int(restart_size)
    if m < 1 or int(max_cycles) < 1 or int(cg_max_iter) < 1:
        raise ValueError("restart_size, max_cycles and cg_max_iter must be positive")
    ng = b.shape[0]
    Phi = np.asarray(orbitals)
    nocc = Phi.shape[1]
    f = np.asarray(occupations, dtype=float)

    vGa = np.asarray(coulomb_gspace, dtype=float)
    if vGa.shape != b.shape or not np.all(np.isfinite(vGa)):
        raise ValueError("coulomb_gspace must be finite and the shape of rhs")
    if not np.isfinite(float(xc_local)):
        raise ValueError("xc_local must be finite")
    jgrid = np.arange(ng)
    Wf = np.exp(-2j * np.pi * np.outer(jgrid, jgrid) / ng) / np.sqrt(ng)

    def _kernel(d):
        d = np.asarray(d, dtype=float)
        return np.real(Wf.conj().T @ (vGa * (Wf @ d.astype(complex)))) + float(xc_local) * d

    Tb = _oracle_kerker_precondition(b, g_squared, alpha)
    x0 = np.zeros(ng, dtype=float)
    s = np.inf
    for _cycle in range(int(max_cycles)):
        if np.any(x0):
            Ex0 = x0 - _oracle_apply_chi0(
                hamiltonian, orbitals, eigenvalues, occupations,
                _kernel(x0), beta,
                tau / 3.0, cg_max_iter)
            r0 = Tb - _oracle_kerker_precondition(Ex0, g_squared, alpha)
        else:
            r0 = Tb
        beta0 = float(np.linalg.norm(r0))
        if beta0 == 0.0:
            return x0
        Q = np.zeros((ng, m + 1), dtype=float)
        Hh = np.zeros((m + 1, m), dtype=float)
        Q[:, 0] = r0 / beta0
        res = beta0
        y = np.zeros(1)
        kk = 0
        for k in range(m):
            kk = k
            sig = 1.0 if not np.isfinite(s) else s
            Kv = _kernel(Q[:, k])
            nk = float(np.linalg.norm(Kv))
            if nk == 0.0:
                chi = np.zeros(ng, dtype=float)
            else:
                t = _oracle_adaptive_cg_tolerance(tau, res, sig, k, m, f, Phi, Kv,
                                                  vol, ng, nocc)
                chi = nk * _oracle_apply_chi0(hamiltonian, orbitals, eigenvalues,
                                              occupations, Kv / nk, beta, t, cg_max_iter)
            v = _oracle_kerker_precondition(Q[:, k] - chi, g_squared, alpha)
            for i in range(k + 1):
                Hh[i, k] = v @ Q[:, i]
                v = v - Hh[i, k] * Q[:, i]
            Hh[k + 1, k] = float(np.linalg.norm(v))
            if Hh[k + 1, k] > 1e-14:
                Q[:, k + 1] = v / Hh[k + 1, k]
            e1 = np.zeros(k + 2)
            e1[0] = beta0
            y, *_ = np.linalg.lstsq(Hh[:k + 2, :k + 1], e1, rcond=None)
            res = float(np.linalg.norm(Hh[:k + 2, :k + 1] @ y - e1))
            if res <= tau / 3.0:
                break
        sig_i = float(np.linalg.svd(Hh[:kk + 2, :kk + 1], compute_uv=False)[-1])
        x = x0 + Q[:, :kk + 1] @ y
        if res <= tau / 3.0 and s <= sig_i:
            return x
        s = min(s, sig_i)
        x0 = x
    return x0

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the eight-point model at a tight outer accuracy.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 6.0)\ngs = g[:, 1]\nvG = g[:, 2]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc, xc = 1.2, 8.0, 5, -0.9\npk = _oracle_ground_state_orbitals(H, mu, beta)\nPhi = pk[:, :Ng][:, :nocc]\nwo = pk[:, Ng].real[:nocc]\nfo = pk[:, Ng+1].real[:nocc]\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\nbb = _oracle_apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 400)\n",
         "call": "preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 1.5, beta, 1e-11, L, 5)",
         "gold_call": "_oracle_preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 1.5, beta, 1e-11, L, 5)"},
        # normal: the same system with a short restart, so several cycles actually run.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 6.0)\ngs = g[:, 1]\nvG = g[:, 2]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc, xc = 1.2, 8.0, 5, -0.9\npk = _oracle_ground_state_orbitals(H, mu, beta)\nPhi = pk[:, :Ng][:, :nocc]\nwo = pk[:, Ng].real[:nocc]\nfo = pk[:, Ng+1].real[:nocc]\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\nbb = _oracle_apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 400)\n",
         "call": "preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 1.5, beta, 1e-11, L, 2)",
         "gold_call": "_oracle_preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 1.5, beta, 1e-11, L, 2)"},
        # boundary: a loose outer accuracy, where the stopping rule and schedule still bite.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 6.0)\ngs = g[:, 1]\nvG = g[:, 2]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc, xc = 1.2, 8.0, 5, -0.9\npk = _oracle_ground_state_orbitals(H, mu, beta)\nPhi = pk[:, :Ng][:, :nocc]\nwo = pk[:, Ng].real[:nocc]\nfo = pk[:, Ng+1].real[:nocc]\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\nbb = _oracle_apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 400)\n",
         "call": "preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 1.5, beta, 1e-3, L, 5)",
         "gold_call": "_oracle_preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 1.5, beta, 1e-3, L, 5)"},
        # boundary: a stronger Kerker parameter on the same system.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 6.0)\ngs = g[:, 1]\nvG = g[:, 2]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc, xc = 1.2, 8.0, 5, -0.9\npk = _oracle_ground_state_orbitals(H, mu, beta)\nPhi = pk[:, :Ng][:, :nocc]\nwo = pk[:, Ng].real[:nocc]\nfo = pk[:, Ng+1].real[:nocc]\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\nbb = _oracle_apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 400)\n",
         "call": "preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 3.0, beta, 1e-11, L, 4)",
         "gold_call": "_oracle_preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 3.0, beta, 1e-11, L, 4)"},
        # normal: the inexactness check - the Krylov answer must match a dense direct solve.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 6.0)\ngs = g[:, 1]\nvG = g[:, 2]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc, xc = 1.2, 8.0, 5, -0.9\npk = _oracle_ground_state_orbitals(H, mu, beta)\nPhi = pk[:, :Ng][:, :nocc]\nwo = pk[:, Ng].real[:nocc]\nfo = pk[:, Ng+1].real[:nocc]\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\nbb = _oracle_apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 400)\nEm = np.column_stack([\n    np.eye(Ng)[:, c] - _oracle_apply_chi0(H, Phi, wo, fo, np.real(Fm.conj().T @ (vG*(Fm @ np.eye(Ng)[:, c].astype(complex)))) + xc*np.eye(Ng)[:, c], beta, 1e-13, 400)\n    for c in range(Ng)\n])\nddirect = np.linalg.solve(Em, bb)\n",
         "call": "int(np.max(np.abs(preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 1.5, beta, 1e-11, L, 5) - ddirect)) < 1e-9)",
         "gold_call": "1"},
        # edge: the screened response conserves charge.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 6.0)\ngs = g[:, 1]\nvG = g[:, 2]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc, xc = 1.2, 8.0, 5, -0.9\npk = _oracle_ground_state_orbitals(H, mu, beta)\nPhi = pk[:, :Ng][:, :nocc]\nwo = pk[:, Ng].real[:nocc]\nfo = pk[:, Ng+1].real[:nocc]\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\nbb = _oracle_apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 400)\n",
         "call": "int(abs(np.sum(preconditioned_gmres(H, Phi, wo, fo, bb, vG, xc, gs, 1.5, beta, 1e-11, L, 5))) < 1e-10)",
         "gold_call": "1"},
        # edge: a vanishing right-hand side gives a vanishing response.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 6.0)\ngs = g[:, 1]\nvG = g[:, 2]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc, xc = 1.2, 8.0, 5, -0.9\npk = _oracle_ground_state_orbitals(H, mu, beta)\nPhi = pk[:, :Ng][:, :nocc]\nwo = pk[:, Ng].real[:nocc]\nfo = pk[:, Ng+1].real[:nocc]\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\nbb = _oracle_apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 400)\n",
         "call": "int(np.max(np.abs(preconditioned_gmres(H, Phi, wo, fo, 0.0*bb, vG, xc, gs, 1.5, beta, 1e-11, L, 5))) < 1e-14)",
         "gold_call": "1"},
    ]
