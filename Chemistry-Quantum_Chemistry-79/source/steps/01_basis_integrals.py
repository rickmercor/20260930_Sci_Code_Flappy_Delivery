"""
All one-body integrals of the model in an even-tempered Gaussian basis: overlap, field-free Hamiltonian, complex absorbing potential, and the second spatial moment.

The model is a single particle on the line in atomic units (hbar = m = 1) bound by V(x) = A x^2 exp(-lambda x^2). This potential vanishes at the origin and at infinity and rises to a maximum of A / (lambda e) at x = 1 / sqrt(lambda), so it confines a well behind a finite barrier and supports metastable states above the dissociation threshold at zero energy.

The basis is the set of even-tempered Gaussians phi_i(x) = exp(-alpha_i x^2) with alpha_i = alpha0 * beta^i for i = 0, 1, ..., n_basis - 1. These functions are normalisation-free as written and are not orthogonal, so every quantity here is a matrix in a non-orthogonal basis. Four matrices are needed over this basis: the overlap S, the kinetic energy T taken in the symmetric form (1/2) times the integral of phi_i' phi_j', the potential energy V over the V(x) above, and the second spatial moment X2 over x^2. Every one of them is a Gaussian integral with an elementary closed form, and all of them are even, so each reduces to a standard moment of exp(-p x^2) with p = alpha_i + alpha_j. The field-free Hamiltonian returned here is T + V.

The absorbing potential is the box-shaped CAP used throughout the resonance literature: W(x) = (|x| - c)^2 for |x| >= c and zero inside, where c is the CAP onset. Its matrix elements are the only ones that are not a plain Gaussian moment, because the quadratic is switched on only outside the onset; they reduce to Gaussian moments truncated to |x| >= c, which are complementary-error-function expressions rather than powers of p. Two checks fix the result: a CAP onset of zero must reduce W exactly to the second moment X2, and at large p the elements must fall off like the Gaussian tail beyond the onset rather than diverging.

Returns
-------
numpy.ndarray of shape (4, n_basis, n_basis): the overlap S, the field-free Hamiltonian T + V, the CAP matrix W, and the second-moment matrix X2, stacked in that order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.special import erfc

def basis_integrals(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                    potential_width: float, cap_onset: float) -> "np.ndarray":
    '''Overlap, field-free Hamiltonian, CAP and second-moment matrices.

    Parameters
    ----------
    alpha0 : float
        Exponent of the most diffuse Gaussian; must be positive.
    beta : float
        Even-tempered ratio, alpha_i = alpha0 * beta**i; must exceed 1.
    n_basis : int
        Number of Gaussians; must be at least 2.
    barrier_strength : float
        Coefficient A of the potential V(x) = A x^2 exp(-lambda x^2); must not be negative.
    potential_width : float
        Exponent lambda of the potential; must be positive.
    cap_onset : float
        Onset c of the box CAP W(x) = (|x| - c)^2 for |x| >= c; must not be negative.

    Returns
    -------
    integrals : numpy.ndarray
        Real array of shape (4, n_basis, n_basis) holding, in order, the overlap
        matrix S, the field-free Hamiltonian T + V, the CAP matrix W and the
        second-moment matrix X2, each in the primitive (non-orthogonal) basis.

    Raises
    ------
    ValueError
        If alpha0 is not positive, if beta is not greater than 1, if n_basis is not an
        integer of at least 2, if barrier_strength is negative, if potential_width is
        not positive, or if cap_onset is negative.
    '''
    return integrals

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_basis_integrals(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                            potential_width: float, cap_onset: float) -> "np.ndarray":
    import numpy as np
    from scipy.special import erfc
    if not np.isfinite(alpha0) or alpha0 <= 0.0:
        raise ValueError("alpha0 must be a positive finite number")
    if not np.isfinite(beta) or beta <= 1.0:
        raise ValueError("beta must be greater than 1")
    if int(n_basis) != n_basis or n_basis < 2:
        raise ValueError("n_basis must be an integer of at least 2")
    if not np.isfinite(barrier_strength) or barrier_strength < 0.0:
        raise ValueError("barrier_strength must be a non-negative finite number")
    if not np.isfinite(potential_width) or potential_width <= 0.0:
        raise ValueError("potential_width must be a positive finite number")
    if not np.isfinite(cap_onset) or cap_onset < 0.0:
        raise ValueError("cap_onset must be a non-negative finite number")

    n = int(n_basis)
    alpha = float(alpha0) * float(beta) ** np.arange(n)
    p = alpha[:, None] + alpha[None, :]

    S = np.sqrt(np.pi / p)
    T = (alpha[:, None] * alpha[None, :]) * np.sqrt(np.pi) / p ** 1.5
    q = p + float(potential_width)
    V = float(barrier_strength) * np.sqrt(np.pi) / (2.0 * q ** 1.5)
    X2 = np.sqrt(np.pi) / (2.0 * p ** 1.5)

    c = float(cap_onset)
    J0 = 0.5 * np.sqrt(np.pi / p) * erfc(c * np.sqrt(p))
    J1 = np.exp(-p * c * c) / (2.0 * p)
    J2 = c * np.exp(-p * c * c) / (2.0 * p) + J0 / (2.0 * p)
    W = 2.0 * (J2 - 2.0 * c * J1 + c * c * J0)

    return np.stack([S, T + V, W, X2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the basis and potential used for the production calculation ---
        {
            "setup": """import numpy as np
""",
            "call": "basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)",
            "gold_call": "_oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)",
            "tol": 1e-10,
        },
        # --- Normal: a tighter, shorter even-tempered set with a distant CAP onset ---
        {
            "setup": """import numpy as np
""",
            "call": "basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)",
            "gold_call": "_oracle_basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)",
            "tol": 1e-10,
        },
        # --- Boundary: the smallest admissible basis, onset inside the barrier ---
        {
            "setup": """import numpy as np
""",
            "call": "basis_integrals(0.3, 1.05, 2, 0.75, 1.5, 0.9)",
            "gold_call": "_oracle_basis_integrals(0.3, 1.05, 2, 0.75, 1.5, 0.9)",
            "tol": 1e-12,
        },
        # --- Edge: zero CAP onset, where W must collapse onto the second moment X2 ---
        {
            "setup": """import numpy as np
""",
            "call": "basis_integrals(0.02, 2.0, 12, 1.2, 0.5, 0.0)",
            "gold_call": "_oracle_basis_integrals(0.02, 2.0, 12, 1.2, 0.5, 0.0)",
            "tol": 1e-12,
        },
        # --- Edge: no barrier at all, so the Hamiltonian is purely kinetic ---
        {
            "setup": """import numpy as np
""",
            "call": "basis_integrals(0.01, 3.0, 8, 0.0, 0.25, 5.0)",
            "gold_call": "_oracle_basis_integrals(0.01, 3.0, 8, 0.0, 0.25, 5.0)",
            "tol": 1e-12,
        },
        # --- Edge: a very diffuse set whose CAP integrals are dominated by erfc tails ---
        {
            "setup": """import numpy as np
""",
            "call": "basis_integrals(0.001, 1.8, 14, 1.2, 0.5, 6.0)",
            "gold_call": "_oracle_basis_integrals(0.001, 1.8, 14, 1.2, 0.5, 6.0)",
            "tol": 1e-12,
        },
        # --- Invalid: an even-tempered ratio of exactly 1 gives a singular basis ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        basis_integrals(0.02, 1.0, 20, 1.2, 0.5, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_basis_integrals(0.02, 1.0, 20, 1.2, 0.5, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: a negative CAP onset is not a box CAP ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        basis_integrals(0.02, 2.0, 20, 1.2, 0.5, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
