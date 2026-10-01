"""
Compute the damped payoff coefficients of a European call struck at K, at resolution level m, for every integer translation index from lam_lo to lam_hi inclusive.

The density and the payoff are represented in a basis of Shannon scaling functions at level m. Because the Fourier transform of that basis function has compact support, each coefficient is a finite-interval frequency integral of the transform of the damped payoff against a single complex exponential, with the interval determined by the resolution level and symmetric about zero. Compute each coefficient with the uniform composite trapezoidal rule, where NQ is the total number of nodes placed across that whole symmetric interval, endpoints included, so that the node spacing is the interval width divided by NQ minus one.

The damped payoff is the call payoff multiplied by a decaying exponential in the log-price at rate alpha. Its Fourier transform is available in closed form, so no numerical integration in the log-price variable is required.

Return the coefficients in increasing order of translation index.

Raises ValueError if lam_lo > lam_hi; if NQ is not an odd integer of at least 3; if alpha <= 1, the condition under which the damped call payoff is integrable; or if K <= 0.

The Shannon scaling function at resolution level m is a dilation of the cardinal sine function, and its translates form an orthonormal basis of square-integrable functions. What makes this basis convenient for transform methods is a duality: the cardinal sine is badly localized in the physical variable, decaying only like the reciprocal of distance, but its Fourier transform is an indicator function, so it is perfectly localized in frequency. Expanding a function in this basis and applying the Parseval identity therefore turns each expansion coefficient into an integral over a bounded frequency interval, whose width doubles with each increase in the resolution level, of the function's transform against a single complex exponential in the translation index.

This is what allows a pricing method to work entirely from a characteristic function. The price is an integral of the payoff against the density; expanding both in the same basis turns that integral into a sum over translation indices of the products of their coefficients, and each family of coefficients is obtained from the corresponding transform without ever forming the density. For a payoff the transform is usually available in closed form, so its coefficients require only one numerical quadrature over the bounded frequency interval.

Exponential damping is what makes the payoff side well defined. A call payoff grows exponentially in the log-price, so it is not integrable and has no ordinary Fourier transform. Multiplying by a decaying exponential at a rate exceeding the growth rate of the payoff restores integrability, and the transform of the damped payoff is then an elementary function of the damping rate and the strike with simple poles at the two exponents where integrability fails. Because the coefficients of the damped payoff inherit the vanishing of the payoff below the log-strike, they are strongly suppressed on that side, which is what allows a truncation rule based on the product of the payoff and density coefficients to discard terms that a density-only rule would retain.

Returns
-------
np.ndarray of shape (lam_hi - lam_lo + 1,), the damped payoff coefficients in increasing translation index as native float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def payoff_coefficients(alpha: float, m: int, lam_lo: int, lam_hi: int,
                        NQ: int, K: float) -> 'np.ndarray':
    '''Damped payoff coefficients of a European call.

    Parameters
    ----------
    alpha : float
        Damping level, greater than 1.
    m : int
        Resolution level.
    lam_lo, lam_hi : int
        Inclusive bounds of the translation range, lam_lo <= lam_hi.
    NQ : int
        Total trapezoidal nodes across the symmetric frequency interval,
        endpoints included; odd and at least 3.
    K : float
        Strike, positive.

    Returns
    -------
    result : np.ndarray
        Shape (lam_hi - lam_lo + 1,), in increasing translation index.

    Raises
    ------
    ValueError
        If lam_lo > lam_hi; if NQ is not an odd integer of at least 3; if
        alpha <= 1, the condition under which the damped call payoff is
        integrable; or if K <= 0.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _h_frequency_grid(m, NQ):
    """Symmetric frequency grid and trapezoidal weights at resolution m."""
    half_width = 2.0 ** m * np.pi
    nodes = np.linspace(-half_width, half_width, int(NQ))
    step = nodes[1] - nodes[0]
    weights = np.full(int(NQ), step)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return nodes, weights


def _h_damped_call_transform(nodes, alpha, K):
    """Fourier transform of the damped call payoff on the frequency grid."""
    s = float(alpha) + 1j * nodes
    return float(K) ** (1.0 - s) / (s * (s - 1.0))


def _oracle_payoff_coefficients(alpha: float, m: int, lam_lo: int,
                                lam_hi: int, NQ: int,
                                K: float) -> 'np.ndarray':
    """Reference implementation."""
    if int(lam_lo) > int(lam_hi):
        raise ValueError("lam_lo must not exceed lam_hi")
    if int(NQ) < 3 or int(NQ) % 2 == 0:
        raise ValueError("NQ must be an odd integer of at least 3")
    if float(alpha) <= 1.0:
        raise ValueError(
            "alpha must exceed 1 for an integrable damped call payoff")
    if float(K) <= 0.0:
        raise ValueError("K must be positive")

    nodes, weights = _h_frequency_grid(int(m), int(NQ))
    transform = _h_damped_call_transform(nodes, float(alpha), float(K))
    indices = np.arange(int(lam_lo), int(lam_hi) + 1)
    phase = np.exp(1j * np.outer(indices / 2.0 ** int(m), nodes))
    scale = 2.0 ** (-int(m) / 2.0) / (2.0 * np.pi)
    return scale * np.real(phase @ (weights * transform))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _setup = """import numpy as np
K = 98.0
"""
    return [
        # --- normal: the locked configuration ---
        {
            "setup": _setup,
            "call": "payoff_coefficients(4.3317046485, 3, 20, 101, 513, K)",
            "gold_call": ("_oracle_payoff_coefficients(4.3317046485, 3, 20, "
                          "101, 513, K)"),
        },
        # --- boundary: damping level just above the integrability threshold,
        #     coarse resolution and a range straddling the origin ---
        {
            "setup": _setup,
            "call": "payoff_coefficients(1.05, 1, -4, 4, 33, K)",
            "gold_call": "_oracle_payoff_coefficients(1.05, 1, -4, 4, 33, K)",
        },
        # --- boundary: the smallest admissible node count ---
        {
            "setup": _setup,
            "call": "payoff_coefficients(2.5, 2, 0, 3, 3, K)",
            "gold_call": "_oracle_payoff_coefficients(2.5, 2, 0, 3, 3, K)",
        },
        # --- edge: a single translation index at a fine resolution, unit
        #     strike so the log-strike sits at the origin ---
        {
            "setup": _setup,
            "call": "payoff_coefficients(3.0, 5, 60, 60, 129, 1.0)",
            "gold_call": ("_oracle_payoff_coefficients(3.0, 5, 60, 60, 129, "
                          "1.0)"),
        },
        # --- structural probe: the coefficient array has the length implied
        #     by the range, the coefficients peak away from both ends of the
        #     range, and raising the damping level suppresses them; exact
        #     integers ---
        {
            "setup": _setup + """
def probe():
    v = payoff_coefficients(4.3317046485, 3, 20, 101, 513, K)
    w = payoff_coefficients(4.8, 3, 20, 101, 513, K)
    peak = int(np.argmax(np.abs(v)))
    return [int(len(v)), peak, int(np.abs(v[peak]) > np.abs(v[0])),
            int(np.abs(v[peak]) > np.abs(v[-1])),
            int(np.abs(w[peak]) < np.abs(v[peak]))]
def probe_gold():
    v = _oracle_payoff_coefficients(4.3317046485, 3, 20, 101, 513, K)
    w = _oracle_payoff_coefficients(4.8, 3, 20, 101, 513, K)
    peak = int(np.argmax(np.abs(v)))
    return [int(len(v)), peak, int(np.abs(v[peak]) > np.abs(v[0])),
            int(np.abs(v[peak]) > np.abs(v[-1])),
            int(np.abs(w[peak]) < np.abs(v[peak]))]
""",
            "call": "probe()",
            "gold_call": "probe_gold()",
        },
        # --- invalid: damping level at the integrability threshold ---
        {
            "setup": _setup + """
def run_model():
    try:
        payoff_coefficients(1.0, 3, 0, 4, 33, K)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_payoff_coefficients(1.0, 3, 0, 4, 33, K)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: even node count, which has no midpoint and so does not
        #     match the intended symmetric grid ---
        {
            "setup": _setup + """
def run_model():
    try:
        payoff_coefficients(2.5, 3, 0, 4, 512, K)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_payoff_coefficients(2.5, 3, 0, 4, 512, K)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: inverted translation range ---
        {
            "setup": _setup + """
def run_model():
    try:
        payoff_coefficients(2.5, 3, 10, 4, 33, K)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_payoff_coefficients(2.5, 3, 10, 4, 33, K)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive strike ---
        {
            "setup": _setup + """
def run_model():
    try:
        payoff_coefficients(2.5, 3, 0, 4, 33, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_payoff_coefficients(2.5, 3, 0, 4, 33, 0.0)
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
