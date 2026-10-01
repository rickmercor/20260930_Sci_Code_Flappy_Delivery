"""
Diagonalize the Kohn-Sham Hamiltonian and evaluate the Fermi-Dirac occupations of its eigenstates. Returns the eigenvectors, the ascending eigenvalues and the occupations packed into a single array, because every later step needs all three and the Studio driver compares one numeric object rather than a tuple.

Every linear-response calculation starts from the unperturbed ground state. For a Hamiltonian discretised on a finite basis this means an eigendecomposition together with the smearing function that distributes electrons over the resulting levels at finite electronic temperature, $f(\\varepsilon) = \\big[1 + e^{\\beta(\\varepsilon - \\mu)}\\big]^{-1}$. In a metal the occupations are fractional, and both the eigenvalues and the occupations are needed downstream: the eigenvalues set the energy denominators of the response, while the occupations weight it and determine how far the Fermi level must move when the perturbation is applied.




****--- Formulas ---****

$$H = U\\,\\mathrm{diag}(\\varepsilon)\\,U^{\\dagger}, \\qquad \\varepsilon_1 \\le \\varepsilon_2 \\le \\dots \\le \\varepsilon_{N_b}, \\qquad f_n = \\frac{1}{1 + e^{\\beta(\\varepsilon_n - \\mu)}}.$$




The Hamiltonian is Hermitized as $(H + H^{\\dagger})/2$ before diagonalisation, and the exponent is clipped to $[-700, 700]$ so that the occupations saturate cleanly at $0$ and $1$ instead of overflowing.

Returns
-------
`np.ndarray` of shape `(Nb, Nb+2)`, complex. Columns $0$ to $N_b-1$ hold the eigenvectors, column $N_b$ holds the ascending eigenvalues in its real part, and column $N_b+1$ holds the Fermi-Dirac occupations in its real part; the two trailing columns have zero imaginary part.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def ground_state_orbitals(hamiltonian, mu, beta):
    """hamiltonian: (Nb,Nb) complex Hermitian. mu: float, chemical potential.
    beta: float > 0, inverse electronic temperature.
    Return one (Nb, Nb+2) complex array packing the eigenvectors in columns 0..Nb-1,
    the ascending eigenvalues in column Nb, and the occupations in column Nb+1.
    Unpack later with n = packed.shape[0]; U = packed[:, :n];
    w = packed[:, n].real; f = packed[:, n+1].real."""
    # Implement per the formulas above.
    packed = None
    return packed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_ground_state_orbitals(hamiltonian, mu, beta):
    H = np.asarray(hamiltonian)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] < 1:
        raise ValueError("hamiltonian must be a square 2-D array")
    if not np.all(np.isfinite(H)):
        raise ValueError("hamiltonian must be finite")
    if not np.isfinite(float(mu)):
        raise ValueError("mu must be finite")
    if not np.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError("beta must be finite and positive")
    H = np.asarray(H, dtype=complex)
    w, U = np.linalg.eigh((H + H.conj().T) / 2.0)
    f = 1.0 / (1.0 + np.exp(np.clip(float(beta) * (w - float(mu)), -700.0, 700.0)))
    n = U.shape[0]
    packed = np.zeros((n, n + 2), dtype=complex)
    packed[:, :n] = U
    packed[:, n] = w.astype(complex)
    packed[:, n + 1] = f.astype(complex)
    return packed

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: eigenvalues and occupations of a generic complex Hermitian matrix.
        {"setup": "import numpy as np\nH = np.array([[0.8,1.2,0.1+0.05j],[1.2,-0.3,0.25j],[0.1-0.05j,-0.25j,1.1]], dtype=complex)\nHh = (H + H.conj().T)/2\nn = 3\nmu, beta = 0.2, 12.0\n",
         "call": "(lambda P: P[:, n:].real)(ground_state_orbitals(H, mu, beta))",
         "gold_call": "(lambda P: P[:, n:].real)(_oracle_ground_state_orbitals(H, mu, beta))"},
        # normal: full self-check of the decomposition, returning 1.
        {"setup": "import numpy as np\nH = np.array([[0.8,1.2,0.1+0.05j],[1.2,-0.3,0.25j],[0.1-0.05j,-0.25j,1.1]], dtype=complex)\nHh = (H + H.conj().T)/2\nn = 3\nmu, beta = 0.2, 12.0\n",
         "call": "(lambda P: int(bool(np.allclose(P[:, :n].conj().T @ P[:, :n], np.eye(n), atol=1e-11) and np.allclose(Hh @ P[:, :n], P[:, :n] * P[:, n].real, atol=1e-11) and np.allclose((P[:, :n] * P[:, n].real) @ P[:, :n].conj().T, Hh, atol=1e-11) and np.allclose(P[:, n+1].real, 1.0/(1.0+np.exp(np.clip(beta*(P[:, n].real-mu), -700.0, 700.0))), atol=1e-11) and bool(np.all(np.diff(P[:, n].real) >= -1e-12)) and np.allclose(P[:, n:].imag, 0.0, atol=1e-12))))(ground_state_orbitals(H, mu, beta))",
         "gold_call": "1"},
        # normal: the density matrix U diag(f) U^H, the gauge-invariant object later steps consume.
        {"setup": "import numpy as np\nH = np.array([[0.8,1.2,0.1+0.05j],[1.2,-0.3,0.25j],[0.1-0.05j,-0.25j,1.1]], dtype=complex)\nn = 3\nmu, beta = 0.2, 12.0\n",
         "call": "(lambda P: (P[:, :n] * P[:, n+1].real) @ P[:, :n].conj().T)(ground_state_orbitals(H, mu, beta))",
         "gold_call": "(lambda P: (P[:, :n] * P[:, n+1].real) @ P[:, :n].conj().T)(_oracle_ground_state_orbitals(H, mu, beta))"},
        # boundary: two-fold degenerate spectrum; eigenvalues and occupations stay unique.
        {"setup": "import numpy as np\nQ, _ = np.linalg.qr(np.random.RandomState(1).randn(3,3) + 1j*np.random.RandomState(2).randn(3,3))\nevals = np.array([-0.5, 0.5, 0.5])\nH = (Q*evals) @ Q.conj().T\nH = (H + H.conj().T)/2\nHh = H\nn = 3\nmu, beta = 0.0, 5.0\n",
         "call": "(lambda P: P[:, n:].real)(ground_state_orbitals(H, mu, beta))",
         "gold_call": "(lambda P: P[:, n:].real)(_oracle_ground_state_orbitals(H, mu, beta))"},
        # boundary: spectral projector onto the degenerate level, basis-independent inside it.
        {"setup": "import numpy as np\nQ, _ = np.linalg.qr(np.random.RandomState(1).randn(3,3) + 1j*np.random.RandomState(2).randn(3,3))\nevals = np.array([-0.5, 0.5, 0.5])\nH = (Q*evals) @ Q.conj().T\nH = (H + H.conj().T)/2\nn = 3\nmu, beta = 0.0, 5.0\n",
         "call": "(lambda P: P[:, 1:n] @ P[:, 1:n].conj().T)(ground_state_orbitals(H, mu, beta))",
         "gold_call": "(lambda P: P[:, 1:n] @ P[:, 1:n].conj().T)(_oracle_ground_state_orbitals(H, mu, beta))"},
        # boundary: the same self-check on the degenerate spectrum.
        {"setup": "import numpy as np\nQ, _ = np.linalg.qr(np.random.RandomState(1).randn(3,3) + 1j*np.random.RandomState(2).randn(3,3))\nevals = np.array([-0.5, 0.5, 0.5])\nH = (Q*evals) @ Q.conj().T\nH = (H + H.conj().T)/2\nHh = H\nn = 3\nmu, beta = 0.0, 5.0\n",
         "call": "(lambda P: int(bool(np.allclose(P[:, :n].conj().T @ P[:, :n], np.eye(n), atol=1e-11) and np.allclose(Hh @ P[:, :n], P[:, :n] * P[:, n].real, atol=1e-11) and np.allclose((P[:, :n] * P[:, n].real) @ P[:, :n].conj().T, Hh, atol=1e-11) and np.allclose(P[:, n+1].real, 1.0/(1.0+np.exp(np.clip(beta*(P[:, n].real-mu), -700.0, 700.0))), atol=1e-11) and bool(np.all(np.diff(P[:, n].real) >= -1e-12)) and np.allclose(P[:, n:].imag, 0.0, atol=1e-12))))(ground_state_orbitals(H, mu, beta))",
         "gold_call": "1"},
        # edge: one-by-one Hamiltonian, where even the sign of the single eigenvector is arbitrary.
        {"setup": "import numpy as np\nH = np.array([[0.4]], dtype=complex)\nn = 1\nmu, beta = 0.1, 60.0\n",
         "call": "(lambda P: P[:, n:].real)(ground_state_orbitals(H, mu, beta))",
         "gold_call": "(lambda P: P[:, n:].real)(_oracle_ground_state_orbitals(H, mu, beta))"},
    ]
