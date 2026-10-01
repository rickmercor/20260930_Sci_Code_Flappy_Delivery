"""
Return the closed-form eigenvalue spectrum of the Hatano-Nelson chain, selecting

the periodic or the open branch according to the boundary switch.



Inputs

------

n_sites: int, number of chain sites N (>= 2)

gamma: float, hopping energy scale

p: float, non-reciprocity, |p| < 1

alpha_bc: float, boundary switch, 0.0 (OBC) or 1.0 (PBC)



Returns

-------

spectrum: (n_sites,) complex ndarray, ordered by the mode index a = 1 .. N



Raises

------

ValueError: if n_sites is below 2, gamma is not finite, |p| is not below 1, or alpha_bc is neither 0.0 nor 1.0

Both boundary conditions admit closed-form solutions, which is what makes this chain a

useful benchmark.



Under periodic boundary conditions the matrix is circulant, so it is diagonalised by

the discrete Fourier transform and



    eps_a = gamma (1 + p) exp(-2 pi i a / N) + gamma (1 - p) exp(+2 pi i a / N)



for a = 1 .. N. Writing theta = 2 pi a / N this becomes

eps_a = 2 gamma cos(theta) - 2 i gamma p sin(theta), so the spectrum traces an ellipse

centred on the origin with real semi-axis 2 gamma and imaginary semi-axis 2 gamma |p|.

The non-reciprocity is what lifts the spectrum off the real axis.



Under open boundary conditions the matrix is tridiagonal Toeplitz and



    eps_a = 2 gamma sqrt((1 - p)(1 + p)) cos(a pi / (N + 1)).



The geometric mean sqrt((1 - p)(1 + p)) of the two hopping amplitudes is what keeps

this spectrum entirely real despite the broken reciprocity: the open chain can be

brought to Hermitian form by an imaginary gauge transformation, whereas the ring

cannot, because the transformation is obstructed by the loop.



Using the closed forms rather than dense diagonalisation matters here. These matrices

are strongly non-normal, and in the open case the eigenvectors are exponentially

localised, so overlaps computed from numerically obtained eigenvectors lose precision

in the exponential tails.

Returns
-------
np.ndarray, complex array of shape (n_sites,) holding the Hatano-Nelson eigenvalues ordered by the mode index a = 1 .. N
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def analytic_spectrum(n_sites: int, gamma: float, p: float, alpha_bc: float) -> np.ndarray:
    '''Evaluate the closed-form Hatano-Nelson eigenvalues.

    Parameters
    ----------
    n_sites : int
        Number of chain sites N, must be >= 2.
    gamma : float
        Hopping energy scale.
    p : float
        Non-reciprocity of the hoppings, must satisfy |p| < 1.
    alpha_bc : float
        Boundary-condition switch: 0.0 for open, 1.0 for periodic.

    Returns
    -------
    spectrum : np.ndarray
        Complex array of shape (n_sites,) holding the eigenvalues in order a = 1 .. N.

    Raises
    ------
    ValueError
        Raised if n_sites is below 2, gamma is not finite, |p| is not below 1, or alpha_bc is neither 0.0 nor 1.0.
    '''
    return spectrum

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_analytic_spectrum(n_sites: int, gamma: float, p: float, alpha_bc: float) -> np.ndarray:
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 2:
        raise ValueError("n_sites must be an integer >= 2")
    if not np.isfinite(float(gamma)):
        raise ValueError("gamma must be finite")
    if not np.isfinite(float(p)) or abs(float(p)) >= 1.0:
        raise ValueError("p must satisfy |p| < 1")
    if float(alpha_bc) not in (0.0, 1.0):
        raise ValueError("alpha_bc must be 0.0 (OBC) or 1.0 (PBC)")

    n = int(n_sites)
    g = float(gamma)
    pp = float(p)
    a = np.arange(1, n + 1)

    if float(alpha_bc) == 1.0:
        theta = 2.0 * np.pi * a / n
        spectrum = g * (1.0 + pp) * np.exp(-1j * theta) + g * (1.0 - pp) * np.exp(1j * theta)
    else:
        amp = 2.0 * g * np.sqrt((1.0 - pp) * (1.0 + pp))
        spectrum = (amp * np.cos(a * np.pi / (n + 1.0))).astype(complex)

    return spectrum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the task configuration, periodic boundary conditions ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 1.0
""",
            "call": "analytic_spectrum(n_sites, gamma, p, alpha_bc)",
            "gold_call": "_oracle_analytic_spectrum(n_sites, gamma, p, alpha_bc)",
        },
        # --- Normal: open boundary conditions give a real spectrum ---
        {
            "setup": """import numpy as np
n_sites = 100
gamma = 0.85
p = 0.35
alpha_bc = 0.0
""",
            "call": "analytic_spectrum(n_sites, gamma, p, alpha_bc)",
            "gold_call": "_oracle_analytic_spectrum(n_sites, gamma, p, alpha_bc)",
        },
        # --- Boundary: p = 0 collapses both branches onto the Hermitian cosine band ---
        {
            "setup": """import numpy as np
n_sites = 16
gamma = 1.0
p = 0.0
alpha_bc = 1.0
""",
            "call": "analytic_spectrum(n_sites, gamma, p, alpha_bc)",
            "gold_call": "_oracle_analytic_spectrum(n_sites, gamma, p, alpha_bc)",
        },
        # --- Edge: strong non-reciprocity close to the |p| < 1 limit ---
        {
            "setup": """import numpy as np
n_sites = 7
gamma = 1.25
p = 0.95
alpha_bc = 0.0
""",
            "call": "analytic_spectrum(n_sites, gamma, p, alpha_bc)",
            "gold_call": "_oracle_analytic_spectrum(n_sites, gamma, p, alpha_bc)",
        },
        # --- Invalid: |p| >= 1 ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = analytic_spectrum
    try:
        _fn(10, 1.0, -1.5, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_analytic_spectrum
    try:
        _fn(10, 1.0, -1.5, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: alpha_bc outside {0, 1} ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = analytic_spectrum
    try:
        _fn(10, 1.0, 0.2, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_analytic_spectrum
    try:
        _fn(10, 1.0, 0.2, 2.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: n_sites below 2 ---
        {
            "setup": """import numpy as np
def run_model():
    _fn = analytic_spectrum
    try:
        _fn(0, 1.0, 0.2, 1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    _fn = _oracle_analytic_spectrum
    try:
        _fn(0, 1.0, 0.2, 1.0)
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
