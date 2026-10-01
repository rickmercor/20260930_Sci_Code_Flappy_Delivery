"""
Apply the independent-particle susceptibility to a local potential perturbation, at finite electronic temperature and at fixed total electron number, without ever assembling it as a matrix.

The independent-particle susceptibility maps a change in the local potential onto the first-order change in the electron density it induces, before any screening. At zero temperature with integer occupations this is a sum over occupied-to-empty transitions. At finite temperature in a metal it is not, for two reasons.




First, transitions between two states that are both partially occupied contribute in proportion to the difference of their occupations divided by the difference of their energies. That ratio is finite everywhere, including where two levels coincide, where it becomes the derivative of the occupation function; treating such a pair as a vanishing denominator rather than working out the limit is wrong.




Second, the occupations themselves change. A perturbation that raises a level pushes electrons off it, and in a closed system those electrons must go somewhere, so the Fermi level moves. The shift is not a free parameter: it is fixed by the requirement that the total electron number does not change. Omitting it describes a different physical system, one connected to a particle reservoir, and changes the answer at leading order whenever any state is fractionally occupied. If no retained state has a non-zero occupation derivative, which is the zero-compressibility limit of a gapped system at low temperature, the occupations cannot respond at all, every occupation change vanishes, and the shift is taken to be zero.




The coupling of each retained state to the states outside the retained set is not summed explicitly; it comes from the projected linear solve of the previous step, so that the unretained spectrum is never touched. The tolerance passed to those inner solves is an argument here and may be given per state.




****--- Formulas ---****

With $B_{mn} = \\langle\\phi_m|\\,\\delta V\\,|\\phi_n\\rangle$ and $f'(\\varepsilon) = -\\beta f(1-f)$,




$$\\delta n \\;=\\; \\sum_{\\substack{m,n\\\\ m \\neq n}} w_{mn}\\,\\operatorname{Re}\\!\\big(\\phi_m \\phi_n^{*} B_{mn}\\big) \\;+\\; \\sum_n 2 f_n \\operatorname{Re}\\!\\big(\\phi_n^{*}\\,\\delta\\phi_n\\big) \\;+\\; \\sum_n \\delta f_n\\,|\\phi_n|^{2},$$




in which $w_{mn} = (f_n - f_m)/(\\varepsilon_n - \\varepsilon_m)$ tends to $f'(\\varepsilon_n)$ when the two eigenvalues coincide, $\\delta\\phi_n$ is the projected Sternheimer solution of the previous step, and




$$\\delta f_n = f'(\\varepsilon_n)\\big(B_{nn} - \\delta\\varepsilon_F\\big), \\qquad \\delta\\varepsilon_F = \\frac{\\sum_n f'(\\varepsilon_n) B_{nn}}{\\sum_n f'(\\varepsilon_n)} \\quad\\text{so that}\\quad \\sum_n \\delta f_n = 0 .$$




When $\\sum_n f'(\\varepsilon_n)$ vanishes identically, take $\\delta f_n = 0$ for every $n$.

Returns
-------
`np.ndarray` of shape `(Nb,)`, real: the first-order density response to the given potential perturbation at fixed total electron number.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def apply_chi0(hamiltonian, orbitals, eigenvalues, occupations, delta_V, beta,
               tol=1e-10, max_iter=200):
    """hamiltonian: (Nb,Nb) complex Hermitian. orbitals: (Nb,Nocc) complex retained states.
    eigenvalues: (Nocc,) float. occupations: (Nocc,) float in [0,1]. delta_V: (Nb,) real.
    beta: float > 0. tol: float or (Nocc,) array of tolerances for the inner solves.
    max_iter: int, cap for each inner solve.
    Return the (Nb,) real first-order density response at fixed electron number."""
    # Implement per the formulas above.
    delta_rho = None
    return delta_rho

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_apply_chi0(hamiltonian, orbitals, eigenvalues, occupations, delta_V, beta,
                       tol=1e-10, max_iter=200):
    H = np.asarray(hamiltonian)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("hamiltonian must be a square 2-D array")
    if not np.all(np.isfinite(H)):
        raise ValueError("hamiltonian must be finite")
    nb = H.shape[0]
    Phi = np.asarray(orbitals, dtype=complex)
    if Phi.ndim != 2 or Phi.shape[0] != nb or Phi.shape[1] < 1 or Phi.shape[1] > nb:
        raise ValueError("orbitals must have shape (Nb, Nocc) with 1 <= Nocc <= Nb")
    nocc = Phi.shape[1]
    w = np.asarray(eigenvalues, dtype=float)
    if w.ndim != 1 or w.shape[0] != nocc or not np.all(np.isfinite(w)):
        raise ValueError("eigenvalues must be a finite 1-D array of length Nocc")
    f = np.asarray(occupations, dtype=float)
    if f.ndim != 1 or f.shape[0] != nocc or not np.all(np.isfinite(f)):
        raise ValueError("occupations must be a finite 1-D array of length Nocc")
    if np.any(f < -1e-12) or np.any(f > 1.0 + 1e-12):
        raise ValueError("occupations must lie in [0, 1]")
    dV = np.asarray(delta_V, dtype=float)
    if dV.ndim != 1 or dV.shape[0] != nb or not np.all(np.isfinite(dV)):
        raise ValueError("delta_V must be a finite 1-D array of length Nb")
    if not np.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and positive")
    tarr = np.atleast_1d(np.asarray(tol, dtype=float))
    if tarr.size not in (1, nocc):
        raise ValueError("tol must be a scalar or an array of length Nocc")
    if np.any(np.isnan(tarr)) or np.any(tarr <= 0.0):
        raise ValueError("every tolerance must be positive")
    tols = np.broadcast_to(tarr, (nocc,))

    beta = float(beta)
    fp = -beta * f * (1.0 - f)
    B = Phi.conj().T @ (dV[:, None] * Phi)
    drho = np.zeros(nb, dtype=float)

    for n in range(nocc):
        for m in range(nocc):
            if m == n:
                continue
            de = w[n] - w[m]
            wt = fp[n] if abs(de) < 1e-10 else (f[n] - f[m]) / de
            drho += wt * np.real(Phi[:, m] * np.conj(Phi[:, n]) * B[m, n])

    for n in range(nocc):
        if abs(f[n]) < 1e-14:
            continue
        dphi = _oracle_sternheimer_solve(hamiltonian, Phi, w, n, dV, tols[n], max_iter)
        drho += 2.0 * f[n] * np.real(np.conj(Phi[:, n]) * dphi)

    diagB = np.real(np.diag(B))
    denom = float(np.sum(fp))
    if denom == 0.0:
        df = np.zeros(nocc, dtype=float)      # zero-compressibility limit
    else:
        df = fp * (diagB - float(np.sum(fp * diagB)) / denom)
    drho += (np.abs(Phi) ** 2) @ df
    return drho

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the eight-point metallic model.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\n",
         "call": "apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 300)",
         "gold_call": "_oracle_apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 300)"},
        # normal: charge conservation - the response of a closed system integrates to zero.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\n",
         "call": "int(abs(np.sum(apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 300))) < 1e-11)",
         "gold_call": "1"},
        # normal: rigid-shift invariant - a constant perturbation gives exactly no response.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndVc = 0.37*np.ones(Ng)\n",
         "call": "int(np.max(np.abs(apply_chi0(H, Phi, wo, fo, dVc, beta, 1e-12, 300))) < 1e-11)",
         "gold_call": "1"},
        # normal: linearity in the perturbation.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\nd1 = 0.4*np.cos(2*np.pi*x/L) + 0.05\nd2 = -0.25*np.sin(6*np.pi*x/L) + 0.11\n",
         "call": "int(np.max(np.abs(apply_chi0(H, Phi, wo, fo, 1.7*d1 - 0.6*d2, beta, 1e-12, 300) - (1.7*apply_chi0(H, Phi, wo, fo, d1, beta, 1e-12, 300) - 0.6*apply_chi0(H, Phi, wo, fo, d2, beta, 1e-12, 300)))) < 1e-10)",
         "gold_call": "1"},
        # edge: a vanishing perturbation gives a vanishing response.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = np.zeros(Ng)\n",
         "call": "int(np.max(np.abs(apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 300))) < 1e-14)",
         "gold_call": "1"},
        # boundary: a second model system at a different temperature.
        {"setup": "import numpy as np\nNg, L = 6, 4.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.55*np.cos(2*np.pi*x/L) - 0.22*np.sin(2*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.5, 6.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = 0.3*np.cos(2*np.pi*x/L) + 0.12\n",
         "call": "apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 300)",
         "gold_call": "_oracle_apply_chi0(H, Phi, wo, fo, dV, beta, 1e-12, 300)"},
        # edge: zero-compressibility limit - exactly integer occupations, so the sum of occupation derivatives vanishes identically.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\nfint = np.array([1.0, 1.0, 1.0, 0.0, 0.0])\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\n",
         "call": "apply_chi0(H, Phi, wo, fint, dV, beta, 1e-12, 300)",
         "gold_call": "_oracle_apply_chi0(H, Phi, wo, fint, dV, beta, 1e-12, 300)"},
        # edge: the rigid-shift invariant must still hold in the zero-compressibility limit, and must not return NaN.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\nfint = np.array([1.0, 1.0, 1.0, 0.0, 0.0])\ndVc = 0.37*np.ones(Ng)\n",
         "call": "int(np.max(np.abs(apply_chi0(H, Phi, wo, fint, dVc, beta, 1e-12, 300))) < 1e-11)",
         "gold_call": "1"},
    ]
