"""
Solve the projected Sternheimer equation for the first-order response of one retained state to a local potential perturbation, by conjugate gradients run to a caller-supplied tolerance. This is the primitive whose cost the adaptive rule of the later steps controls.

The first-order change of an occupied state under a perturbation of the potential formally requires a sum over the whole remaining spectrum weighted by inverse energy differences. Computing that spectrum is exactly what a large calculation cannot afford. The standard alternative is to obtain the same quantity by solving a linear system for each occupied state: the shifted Hamiltonian acting on the unknown response equals minus the perturbation acting on the state, with both sides confined to the subspace orthogonal to the states treated explicitly. Restricting to that subspace does two jobs at once. It removes the null direction that would otherwise make the system singular, since $\\varepsilon_n$ is an eigenvalue of $H$, and it removes precisely the contributions accounted for separately by the retained-pair term, so that nothing is double counted. The projected operator is Hermitian, so a conjugate-gradient iteration solves it, with every iterate kept inside the subspace. The convergence tolerance is an argument rather than a constant: how loosely this system may be solved is decided by the caller, and that decision is the subject of the source paper.




****--- Formulas ---****

With $\\Phi$ the matrix whose columns are the retained states and $Q = 1 - \\Phi\\Phi^{\\dagger}$ the projector orthogonal to them,




$$Q\\,(H - \\varepsilon_n)\\,Q\\;\\delta\\phi_n \\;=\\; -\\,Q\\,\\big(\\delta V \\odot \\phi_n\\big),$$




where $\\odot$ is pointwise multiplication by the local potential. Solve by conjugate gradients started from zero, applying $Q$ to the operator and to the returned iterate so that the solution lies in the range of $Q$, and stop when the residual norm falls below the supplied tolerance or the iteration cap is reached.

Returns
-------
`np.ndarray` of shape `(Nb,)`, complex: the projected first-order response of the requested state, orthogonal to every supplied orbital to within the solve tolerance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def sternheimer_solve(hamiltonian, orbitals, eigenvalues, band_index, delta_V,
                      tol=1e-10, max_iter=200):
    """hamiltonian: (Nb,Nb) complex Hermitian. orbitals: (Nb,Nocc) complex, the retained
    states as orthonormal columns. eigenvalues: (Nocc,) float. band_index: int, which
    retained state to solve for. delta_V: (Nb,) real local potential perturbation.
    tol: float > 0, residual-norm stopping tolerance. max_iter: int, iteration cap.
    Return the (Nb,) complex projected response of that state."""
    # Implement per the formulas above.
    dphi = None
    return dphi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sternheimer_solve(hamiltonian, orbitals, eigenvalues, band_index, delta_V,
                              tol=1e-10, max_iter=200):
    H = np.asarray(hamiltonian)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("hamiltonian must be a square 2-D array")
    if not np.all(np.isfinite(H)):
        raise ValueError("hamiltonian must be finite")
    H = np.asarray(H, dtype=complex)
    H = (H + H.conj().T) / 2.0
    nb = H.shape[0]
    Phi = np.asarray(orbitals, dtype=complex)
    if Phi.ndim != 2 or Phi.shape[0] != nb or Phi.shape[1] < 1 or Phi.shape[1] > nb:
        raise ValueError("orbitals must have shape (Nb, Nocc) with 1 <= Nocc <= Nb")
    if not np.all(np.isfinite(Phi)):
        raise ValueError("orbitals must be finite")
    w = np.asarray(eigenvalues, dtype=float)
    if w.ndim != 1 or w.shape[0] != Phi.shape[1] or not np.all(np.isfinite(w)):
        raise ValueError("eigenvalues must be a finite 1-D array of length Nocc")
    n = int(band_index)
    if n != band_index or n < 0 or n >= Phi.shape[1]:
        raise ValueError("band_index must be an integer in [0, Nocc)")
    dV = np.asarray(delta_V, dtype=float)
    if dV.ndim != 1 or dV.shape[0] != nb or not np.all(np.isfinite(dV)):
        raise ValueError("delta_V must be a finite 1-D array of length Nb")
    if not np.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError("tol must be finite and positive")
    if int(max_iter) < 1:
        raise ValueError("max_iter must be a positive integer")

    def Qop(v):
        return v - Phi @ (Phi.conj().T @ v)

    def Aop(v):
        v = Qop(v)
        return Qop(H @ v - w[n] * v)

    b = -Qop(dV * Phi[:, n])
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = np.real(np.vdot(r, r))
    it = 0
    while np.sqrt(rs) > float(tol) and it < int(max_iter):
        Ap = Aop(p)
        denom = np.real(np.vdot(p, Ap))
        if abs(denom) < 1e-300:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.real(np.vdot(r, r))
        p = r + (rs_new / rs) * p
        rs = rs_new
        it += 1
    return Qop(x)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the lowest retained state of the eight-point model.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\n",
         "call": "sternheimer_solve(H, Phi, wo, 0, dV, 1e-12, 300)",
         "gold_call": "_oracle_sternheimer_solve(H, Phi, wo, 0, dV, 1e-12, 300)"},
        # normal: a partially occupied state near the Fermi level.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\n",
         "call": "sternheimer_solve(H, Phi, wo, 2, dV, 1e-12, 300)",
         "gold_call": "_oracle_sternheimer_solve(H, Phi, wo, 2, dV, 1e-12, 300)"},
        # normal: projection invariant - the result is orthogonal to every retained state.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\n",
         "call": "int(np.max(np.abs(Phi.conj().T @ sternheimer_solve(H, Phi, wo, 1, dV, 1e-12, 300))) < 1e-10)",
         "gold_call": "1"},
        # normal: residual invariant - the result solves the projected equation, sign included.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\nQp = np.eye(Ng, dtype=complex) - Phi @ Phi.conj().T\n",
         "call": "int(np.linalg.norm(Qp @ ((H - wo[1]*np.eye(Ng)) @ sternheimer_solve(H, Phi, wo, 1, dV, 1e-12, 300)) + Qp @ (dV*Phi[:, 1])) < 1e-9)",
         "gold_call": "1"},
        # edge: a vanishing perturbation must give exactly a vanishing response.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.2, 8.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = np.zeros(Ng)\n",
         "call": "int(np.linalg.norm(sternheimer_solve(H, Phi, wo, 0, dV, 1e-12, 300)) < 1e-14)",
         "gold_call": "1"},
        # boundary: a second model system, six points and a different potential.
        {"setup": "import numpy as np\nNg, L = 6, 4.0\nx = np.arange(Ng)*L/Ng\nkn = np.where(np.arange(Ng) < Ng//2, np.arange(Ng), np.arange(Ng)-Ng)\nGv = 2*np.pi*kn/L\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.55*np.cos(2*np.pi*x/L) - 0.22*np.sin(2*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*Gv**2).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\nmu, beta, nocc = 1.5, 6.0, 5\nw, U = np.linalg.eigh(H)\nPhi = U[:, :nocc]; wo = w[:nocc]\nfo = 1.0/(1.0 + np.exp(np.clip(beta*(wo - mu), -700.0, 700.0)))\ndV = 0.3*np.cos(2*np.pi*x/L) + 0.12\n",
         "call": "sternheimer_solve(H, Phi, wo, 1, dV, 1e-11, 300)",
         "gold_call": "_oracle_sternheimer_solve(H, Phi, wo, 1, dV, 1e-11, 300)"},
    ]
