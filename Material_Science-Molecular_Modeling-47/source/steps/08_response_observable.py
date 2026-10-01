"""
Run the whole calculation on the stated configuration and return the screened response of the observable: build the grid, diagonalise, form the bare response, and solve the preconditioned Dyson equation.

Everything is now in place: the ground state, the susceptibility applied through projected inner solves at fixed electron number, the screening kernel, the adaptive inner tolerance and the outer Krylov solve. The remaining work is to run them in the right order, namely build the ground state, retain the requested number of states, form the bare response to the external perturbation, solve the dielectric system for the screened response, and contract it with the observable, and to return one number.




****--- Formulas ---****

$$b = \\chi_0\\big[\\delta V_{\\mathrm{ext}}\\big], \\qquad \\mathbf{T}\\big(1 - \\chi_0 K\\big)\\delta n = \\mathbf{T}b, \\qquad A = \\sum_j O(x_j)\\,\\delta n(x_j).$$




Build the grid and the Coulomb factor from $N_g$, $L$ and the Coulomb scale; diagonalise and occupy; form the bare response with a tight inner tolerance of $10^{-12}$ and an inner cap of $400$; then hand it to the preconditioned outer solve, which chooses its own inner tolerances adaptively. Contract the result with the observable.

Returns
-------
`float`: the screened first-order response of the observable to the external perturbation, as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def response_observable(hamiltonian, delta_V_ext, observable, mu, beta, n_occ, n_grid,
                        cell_length, coulomb_scale, xc_local, alpha, target_tol,
                        restart_size=5):
    """hamiltonian: (Ng,Ng) complex Hermitian. delta_V_ext: (Ng,) real perturbation.
    observable: (Ng,) real. mu: float. beta: float > 0. n_occ: int in [1, Ng].
    n_grid: int. cell_length: float > 0. coulomb_scale: float. xc_local: float.
    alpha: float > 0, Kerker parameter. target_tol: float > 0. restart_size: int > 0.
    Return the screened response of the observable as a float."""
    # Implement per the formulas above.
    A = None
    return A

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_response_observable(hamiltonian, delta_V_ext, observable, mu, beta, n_occ,
                                n_grid, cell_length, coulomb_scale, xc_local, alpha,
                                target_tol, restart_size=5):
    H = np.asarray(hamiltonian)
    nb = H.shape[0]
    O = np.asarray(observable, dtype=float)
    if O.ndim != 1 or O.shape[0] != nb:
        raise ValueError("observable must be a 1-D array of length Ng")
    no = int(n_occ)
    if no < 1 or no > nb:
        raise ValueError("n_occ must be an integer in [1, Ng]")
    grid = _oracle_plane_wave_grid(n_grid, cell_length, coulomb_scale)
    Gsq = grid[:, 1]
    vG = grid[:, 2]
    packed = _oracle_ground_state_orbitals(hamiltonian, mu, beta)
    U = packed[:, :nb]
    w = packed[:, nb].real
    f = packed[:, nb + 1].real
    Phi = U[:, :no]
    b = _oracle_apply_chi0(hamiltonian, Phi, w[:no], f[:no], delta_V_ext, beta, 1e-12, 400)
    dn = _oracle_preconditioned_gmres(hamiltonian, Phi, w[:no], f[:no], b, vG, xc_local,
                                      Gsq, alpha, beta, target_tol, float(cell_length),
                                      restart_size)
    return float(O @ dn)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: the main configuration of the problem statement.
        {"setup": "import numpy as np\nNg, L = 20, 8.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 28.0)\ngs = g[:, 1]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.9*np.cos(2*np.pi*x/L) + 0.35*np.sin(4*np.pi*x/L) + 0.15*np.cos(6*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\ndVe = 0.4*np.cos(2*np.pi*x/L) - 0.25*np.sin(6*np.pi*x/L) + 0.1\nOb = np.cos(2*np.pi*x/L) + 0.4*np.sin(2*np.pi*x/L)\n",
         "call": "response_observable(H, dVe, Ob, 0.55, 10.0, 8, 20, 8.0, 28.0, -1.4, 1.5, 1e-12, 5)",
         "gold_call": "_oracle_response_observable(H, dVe, Ob, 0.55, 10.0, 8, 20, 8.0, 28.0, -1.4, 1.5, 1e-12, 5)"},
        # normal: a second physical system, different grid, temperature and screening.
        {"setup": "import numpy as np\nNg, L = 8, 5.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 6.0)\ngs = g[:, 1]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.8*np.cos(2*np.pi*x/L) + 0.3*np.sin(4*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\ndVe = 0.4*np.cos(2*np.pi*x/L) - 0.2*np.sin(4*np.pi*x/L) + 0.05\nOb = np.sin(2*np.pi*x/L) + 0.25*np.cos(4*np.pi*x/L)\n",
         "call": "response_observable(H, dVe, Ob, 1.2, 8.0, 5, 8, 5.0, 6.0, -0.9, 1.0, 1e-11, 4)",
         "gold_call": "_oracle_response_observable(H, dVe, Ob, 1.2, 8.0, 5, 8, 5.0, 6.0, -0.9, 1.0, 1e-11, 4)"},
        # edge: the rigid-shift invariant end to end - a constant perturbation, no screening.
        {"setup": "import numpy as np\nNg, L = 10, 6.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 0.0)\ngs = g[:, 1]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.6*np.cos(2*np.pi*x/L) - 0.2*np.sin(2*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\ndVe = 0.35*np.ones(Ng)\nOb = np.cos(2*np.pi*x/L) - 0.3*np.sin(4*np.pi*x/L)\n",
         "call": "int(abs(response_observable(H, dVe, Ob, 0.9, 7.0, 6, 10, 6.0, 0.0, 0.0, 1.3, 1e-11, 4)) < 1e-12)",
         "gold_call": "1"},
        # boundary: a small strongly screened cell with a short restart.
        {"setup": "import numpy as np\nNg, L = 6, 4.0\nx = np.arange(Ng)*L/Ng\ng = _oracle_plane_wave_grid(Ng, L, 15.0)\ngs = g[:, 1]\njj = np.arange(Ng)\nFm = np.exp(-2j*np.pi*np.outer(jj, jj)/Ng)/np.sqrt(Ng)\nV = 0.55*np.cos(2*np.pi*x/L) - 0.22*np.sin(2*np.pi*x/L)\nH = Fm.conj().T @ np.diag((0.5*gs).astype(complex)) @ Fm + np.diag(V.astype(complex))\nH = (H + H.conj().T)/2\ndVe = 0.3*np.cos(2*np.pi*x/L) + 0.12\nOb = np.cos(2*np.pi*x/L) + 0.5\n",
         "call": "response_observable(H, dVe, Ob, 1.5, 6.0, 5, 6, 4.0, 15.0, -1.1, 2.0, 1e-11, 3)",
         "gold_call": "_oracle_response_observable(H, dVe, Ob, 1.5, 6.0, 5, 6, 4.0, 15.0, -1.1, 2.0, 1e-11, 3)"},
    ]
