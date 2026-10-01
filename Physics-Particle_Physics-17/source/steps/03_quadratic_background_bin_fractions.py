"""
Compute the continuum background of the CMS Upsilon fit: a second-order polynomial in the dimuon mass, normalized to unit integral on the histogram window $[e_0, e_K]$, where $e_0$ and $e_K$ are the first and last bin edges (`edges[0]` and `edges[-1]`). With the reduced variable



$$x=\dfrac{m-m_c}{h},\qquad m_c=\dfrac{e_0+e_K}{2},\qquad h=\dfrac{e_K-e_0}{2},$$



so that $x$ runs from $-1$ to $+1$, the density is



$$p(x)\propto1+b_1x+b_2x^2.$$



The exact bin integrals follow from the primitive $P(x)=x+b_1x^2/2+b_2x^3/3$. Because the fit maximizes a likelihood, the choice of this parametrization of the quadratic family does not change the fitted yields.

Under the three resonances the dimuon spectrum contains a smooth continuum (Drell-Yan and combinatorial pairs from heavy-flavour decays). CMS describes it with a second-order polynomial in the fit window, with its own freely floating yield. What enters the binned likelihood is only the integral of the density over each bin, so the physical requirement is that every bin expectation is positive; a pointwise check of the polynomial is stricter than the likelihood needs and would wrongly exclude valid parameter points.

Returns
-------
A float array of shape $(K,)$ gives the window-normalized quadratic-background bin fractions; the entries sum to 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def quadratic_background_bin_fractions(edges: np.ndarray, b1: float, b2: float) -> np.ndarray:
    r'''Window-normalized bin fractions of $p(x)\propto1+b_1x+b_2x^2$.

    Parameters
    ----------
    edges : np.ndarray
        1D array of $K+1$ strictly increasing, finite bin edges (GeV), with
        $K\geq1$. The reduced variable is $x=(m-m_c)/h$, with $m_c$ the
        window centre and $h$ the window half-width.
    b1 : float
        Linear coefficient $b_1$.
    b2 : float
        Quadratic coefficient $b_2$.

    Returns
    -------
    fractions : np.ndarray
        Shape $(K,)$, the exact integral of the polynomial over each bin
        divided by its integral over the whole window; the entries sum to 1.
        Only bin integrals are constrained: the polynomial itself may vanish
        or dip below zero inside a bin as long as that bin's integral stays
        positive.

    Raises
    ------
    ValueError
        If edges is not a finite, strictly increasing 1D array of length
        $\geq2$, if b1 or b2 is not finite, or if the integral of the
        polynomial over any bin is $\leq0$ (the background density must be
        positive in every bin).
    '''
    return fractions  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_quadratic_background_bin_fractions(edges: np.ndarray, b1: float, b2: float) -> np.ndarray:
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)):
        raise ValueError("edges must be a finite 1D array with at least 2 entries")
    if not np.all(np.diff(edges) > 0):
        raise ValueError("edges must be strictly increasing")
    if not (np.isfinite(b1) and np.isfinite(b2)):
        raise ValueError("b1 and b2 must be finite")
    mc = 0.5 * (edges[0] + edges[-1])
    h = 0.5 * (edges[-1] - edges[0])
    x = (edges - mc) / h
    P = x + b1 * x ** 2 / 2.0 + b2 * x ** 3 / 3.0
    bins = np.diff(P)
    if np.any(bins <= 0):
        raise ValueError("background integral must be positive in every bin")
    return (bins / (P[-1] - P[0])).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: fitted-like coefficients ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
""",
            "call": "quadratic_background_bin_fractions(edges.copy(), -0.6066, 0.1962)",
            "gold_call": "_oracle_quadratic_background_bin_fractions(edges, -0.6066, 0.1962)",
        },
        # --- Boundary: b1 = b2 = 0 gives a flat distribution (1/K per bin) ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
""",
            "call": "quadratic_background_bin_fractions(edges.copy(), 0.0, 0.0)",
            "gold_call": "_oracle_quadratic_background_bin_fractions(edges, 0.0, 0.0)",
        },
        # --- Edge: density touching zero only at the window endpoint (b1 = 1, b2 = 0) ---
        {
            "setup": """import numpy as np
edges = np.array([8.5, 8.6, 9.5, 11.5])
""",
            "call": "quadratic_background_bin_fractions(edges.copy(), 1.0, 0.0)",
            "gold_call": "_oracle_quadratic_background_bin_fractions(edges, 1.0, 0.0)",
        },
        # --- Edge: double root inside the window (density touches zero at x = 0.2) ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
""",
            "call": "quadratic_background_bin_fractions(edges.copy(), -10.0, 25.0)",
            "gold_call": "_oracle_quadratic_background_bin_fractions(edges, -10.0, 25.0)",
        },
        # --- Edge: two close roots inside ONE bin -> pointwise negative dip, positive bin integral (must NOT raise) ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
r1, r2 = 0.205, 0.21
b1 = -(1.0 / r1 + 1.0 / r2)
b2 = 1.0 / (r1 * r2)
""",
            "call": "quadratic_background_bin_fractions(edges.copy(), b1, b2)",
            "gold_call": "_oracle_quadratic_background_bin_fractions(edges, b1, b2)",
        },
        # --- Invalid: density negative over part of the window ---
        {
            "setup": """import numpy as np
edges = np.linspace(8.5, 11.5, 76)
def run_model():
    try:
        quadratic_background_bin_fractions(edges.copy(), 3.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_quadratic_background_bin_fractions(edges, 3.0, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
