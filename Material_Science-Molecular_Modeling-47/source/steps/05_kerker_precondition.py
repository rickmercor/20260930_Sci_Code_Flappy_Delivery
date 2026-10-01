"""
Apply the charge-conserving Kerker preconditioner to a vector. This is the operator the source paper puts in front of the Dyson equation to stop the condition number growing with system size, and the charge-conserving correction is the part that is easy to miss.

The dielectric operator inherits the Coulomb divergence at long wavelength, which in a metal is not compensated by the susceptibility, so its condition number grows with system size and the outer iteration slows down. The Kerker preconditioner is the standard cure, borrowed from self-consistent-field practice: in reciprocal space it multiplies each component by $|G|^2/(|G|^2 + \\alpha^2)$, which is close to one for short wavelengths and suppresses the long-wavelength components that cause the trouble.




Applied naively that operator has a serious defect. At $G = 0$ the factor is exactly zero, so the preconditioner annihilates the mean of whatever it acts on - it destroys total charge. The paper therefore uses a modified operator that acts as the plain Kerker factor on the charge-neutral part of a vector and as the identity on its mean. The paper's lemma shows the modification is equivalent to something much simpler to implement than its definition suggests: it is the same reciprocal-space multiplication with the $G = 0$ diagonal entry replaced by one. The same lemma establishes that the resulting operator is Hermitian positive definite with largest eigenvalue exactly one, so it has unit norm - which is precisely what allows the inexactness analysis of the previous steps to survive preconditioning unchanged, since multiplying the residual by an operator of unit norm cannot increase it.




****--- Formulas ---****

With $\\mathbf{W}$ the unitary discrete Fourier transform, $\\mathbf{1}$ the vector of ones and $\\mathbf{D}$ diagonal with entries $D_{jj} = |G_j|^2/(|G_j|^2 + \\alpha^2)$, the preconditioner is defined as




$$\\mathbf{T} = \\Big(\\mathbf{I} - \\tfrac{1}{N_g}\\mathbf{1}\\mathbf{1}^{\\top}\\Big)\\mathbf{W}^{-1}\\mathbf{D}\\mathbf{W} + \\tfrac{1}{N_g}\\mathbf{1}\\mathbf{1}^{\\top}.$$




Deriving its action is the point of this step. Because $\\mathbf{1}\\mathbf{1}^{\\top}\\mathbf{W}^{-1}\\mathbf{D}\\mathbf{W} = D_{11}\\mathbf{1}\\mathbf{1}^{\\top}$, the definition collapses to




$$\\mathbf{T} = \\mathbf{W}^{-1}\\big(\\mathbf{D} + (1 - D_{11})\\mathbf{e}_1\\mathbf{e}_1^{\\top}\\big)\\mathbf{W},$$




that is, the same multiplication with the $G = 0$ diagonal entry set to one. Implement that form: transform, multiply by $\\mathbf{D}$ with $D_{11}$ replaced by $1$, transform back, take the real part. Treat $|G_j|^2 \\le 10^{-12}$ as the $G = 0$ entry.

Returns
-------
`np.ndarray` of shape `(Ng,)`, real: the preconditioned vector, with the same total sum as the input.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def kerker_precondition(vector, g_squared, alpha):
    """vector: (Ng,) real vector to precondition. g_squared: (Ng,) real non-negative squared
    wavevector magnitudes, as returned in column 1 of the grid step. alpha: float > 0, the
    Kerker screening parameter.
    Return the (Ng,) real preconditioned vector."""
    # Implement per the formulas above.
    preconditioned = None
    return preconditioned

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_kerker_precondition(vector, g_squared, alpha):
    v = np.asarray(vector, dtype=float)
    if v.ndim != 1 or v.shape[0] < 1 or not np.all(np.isfinite(v)):
        raise ValueError("vector must be a finite 1-D array")
    Gsq = np.asarray(g_squared, dtype=float)
    if Gsq.shape != v.shape or not np.all(np.isfinite(Gsq)) or np.any(Gsq < 0.0):
        raise ValueError("g_squared must be finite, non-negative and the shape of vector")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError("alpha must be finite and positive")
    ng = v.shape[0]
    j = np.arange(ng)
    W = np.exp(-2j * np.pi * np.outer(j, j) / ng) / np.sqrt(ng)
    D = Gsq / (Gsq + a * a)
    D = np.where(Gsq <= 1e-12, 1.0, D)
    return np.real(W.conj().T @ (D * (W @ v.astype(complex))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: a general vector with non-zero mean, where the charge correction matters.
        {"setup": "import numpy as np\nng = 8\nkn = np.where(np.arange(ng) < ng//2, np.arange(ng), np.arange(ng)-ng)\ngs = (2*np.pi*kn/5.0)**2\nv = np.array([0.9, -0.4, 1.3, 0.2, -1.1, 0.6, 0.15, -0.35])\n",
         "call": "kerker_precondition(v, gs, 1.5)",
         "gold_call": "_oracle_kerker_precondition(v, gs, 1.5)"},
        # normal: a longer cell and a weaker screening parameter.
        {"setup": "import numpy as np\nng = 12\nkn = np.where(np.arange(ng) < ng//2, np.arange(ng), np.arange(ng)-ng)\ngs = (2*np.pi*kn/7.0)**2\nv = np.cos(np.arange(ng)) + 0.3\n",
         "call": "kerker_precondition(v, gs, 0.7)",
         "gold_call": "_oracle_kerker_precondition(v, gs, 0.7)"},
        # boundary: strong screening, where every short-wavelength component is suppressed.
        {"setup": "import numpy as np\nng = 10\nkn = np.where(np.arange(ng) < ng//2, np.arange(ng), np.arange(ng)-ng)\ngs = (2*np.pi*kn/4.0)**2\nv = np.sin(2*np.arange(ng)) + 0.8\n",
         "call": "kerker_precondition(v, gs, 4.0)",
         "gold_call": "_oracle_kerker_precondition(v, gs, 4.0)"},
        # edge: the preconditioner conserves total charge exactly.
        {"setup": "import numpy as np\nng = 8\nkn = np.where(np.arange(ng) < ng//2, np.arange(ng), np.arange(ng)-ng)\ngs = (2*np.pi*kn/5.0)**2\nv = np.array([0.9, -0.4, 1.3, 0.2, -1.1, 0.6, 0.15, -0.35])\n",
         "call": "int(abs(np.sum(kerker_precondition(v, gs, 1.5)) - np.sum(v)) < 1e-12)",
         "gold_call": "1"},
        # edge: unit operator norm - the preconditioner never lengthens a vector.
        {"setup": "import numpy as np\nng = 12\nkn = np.where(np.arange(ng) < ng//2, np.arange(ng), np.arange(ng)-ng)\ngs = (2*np.pi*kn/7.0)**2\nv = np.cos(np.arange(ng)) + 0.3\n",
         "call": "int(np.linalg.norm(kerker_precondition(v, gs, 0.7)) <= np.linalg.norm(v) + 1e-12)",
         "gold_call": "1"},
        # edge: in the strong-screening limit only the mean survives.
        {"setup": "import numpy as np\nng = 8\nkn = np.where(np.arange(ng) < ng//2, np.arange(ng), np.arange(ng)-ng)\ngs = (2*np.pi*kn/5.0)**2\nv = np.array([0.9, -0.4, 1.3, 0.2, -1.1, 0.6, 0.15, -0.35])\n",
         "call": "int(np.allclose(kerker_precondition(v, gs, 1e6), np.full(ng, v.mean()), rtol=0, atol=1e-9))",
         "gold_call": "1"},
        # control: on a charge-neutral input the naive form and the correct one coincide.
        {"setup": "import numpy as np\nng = 10\nkn = np.where(np.arange(ng) < ng//2, np.arange(ng), np.arange(ng)-ng)\ngs = (2*np.pi*kn/4.0)**2\nv = np.sin(2*np.arange(ng))\nv = v - v.mean()\n",
         "call": "kerker_precondition(v, gs, 1.2)",
         "gold_call": "_oracle_kerker_precondition(v, gs, 1.2)"},
    ]
