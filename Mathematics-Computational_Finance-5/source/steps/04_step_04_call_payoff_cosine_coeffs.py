"""
Implement $call_payoff_cosine_coeffs$ for a terminal call payoff. On a finite interval $[a_p, b_p]$ of the terminal log-value $y$, define the frequencies $w_k = k * pi / (b_p - a_p)$ for $k = 0, ..., N_in - 1$ and the projections of the payoff $max(exp(y) - K, 0)$ onto the shifted cosine basis, V_k = (2 / (b_p - a_p)) * integral over [a_p, b_p] of max(exp(y) - K, 0) * cos(w_k * (y - a_p)) dy .

Return the vector of those projections. They must be evaluated exactly rather than by numerical quadrature: the interval reaches log-values near 20 in the intended use, the integrand therefore reaches magnitudes near `1e8`, and a quadrature rule loses the leading digits of the alternating sum that the later steps form from these coefficients.

The terminal call payoff has analytic cosine coefficients on the specified log-value interval.

Returns
-------
return np.zeros(int(N_in), dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def call_payoff_cosine_coeffs(K, a_p, b_p, N_in):
    """Exact cosine projections of ``max(exp(y) - K, 0)`` on ``[a_p, b_p]``.

    Parameters
    ----------
    K : float
        Strictly positive strike of the terminal payoff.
    a_p : float
        Left endpoint of the interval.
    b_p : float
        Right endpoint of the interval, ``b_p > a_p``.
    N_in : int
        Number of coefficients, ``N_in >= 1``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(N_in,)`` holding ``V_0, ..., V_{N_in - 1}``. If
        ``log(K) >= b_p`` the payoff vanishes on the interval and the returned
        array is exactly zero.

    Raises
    ------
    ValueError
        If ``K <= 0``, ``b_p <= a_p``, or ``N_in`` is not an integer with
        ``N_in >= 1``.
    """
    return np.zeros(int(N_in), dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_call_payoff_cosine_coeffs(K, a_p, b_p, N_in):
    """Reference implementation."""
    np = __import__("numpy")
    K = float(K)
    a_p = float(a_p)
    b_p = float(b_p)
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError("K must be a finite positive number")
    if not (np.isfinite(a_p) and np.isfinite(b_p)) or b_p <= a_p:
        raise ValueError("require a finite interval with b_p > a_p")
    if isinstance(N_in, bool) or int(N_in) != N_in or int(N_in) < 1:
        raise ValueError("N_in must be an integer >= 1")
    N_in = int(N_in)

    k = np.arange(N_in)
    w = k * np.pi / (b_p - a_p)
    c = max(a_p, np.log(K))
    d = b_p
    if c >= d:
        return np.zeros(N_in)
    wc = w * (c - a_p)
    wd = w * (d - a_p)
    chi = (np.cos(wd) * np.exp(d) - np.cos(wc) * np.exp(c)
           + w * np.sin(wd) * np.exp(d) - w * np.sin(wc) * np.exp(c)) \
        / (1.0 + w ** 2)
    psi = np.empty(N_in)
    psi[0] = d - c
    psi[1:] = (np.sin(wd[1:]) - np.sin(wc[1:])) / w[1:]
    return 2.0 / (b_p - a_p) * (chi - K * psi)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the test-case specifications for this step."""
    return [
        # Case 1 - the target instance, first few coefficients.
        {
            "setup": """import numpy as np
K = 197.22
a_p = -8.491686480930326
b_p = 19.450892115737147
""",
            "call": "np.round(call_payoff_cosine_coeffs(K, a_p, b_p, 8), 6).tolist()",
            "gold_call": "np.round(_oracle_call_payoff_cosine_coeffs(K, a_p, b_p, 8), 6).tolist()",
        },
        # Case 2 - the target instance at full length, sampled entries.
        {
            "setup": """import numpy as np
K = 197.22
a_p = -8.491686480930326
b_p = 19.450892115737147
idx = [0, 1, 2, 17, 512, 1023]
def sample(V):
    V = np.asarray(V, dtype=float)
    return np.round(V[idx], 6).tolist()
""",
            "call": "sample(call_payoff_cosine_coeffs(K, a_p, b_p, 1024))",
            "gold_call": "sample(_oracle_call_payoff_cosine_coeffs(K, a_p, b_p, 1024))",
        },
        # Case 3 - strike below the left endpoint, so the payoff is positive throughout.
        {
            "setup": """import numpy as np
K = 0.5
a_p = 0.0
b_p = 4.0
""",
            "call": "np.round(call_payoff_cosine_coeffs(K, a_p, b_p, 12), 8).tolist()",
            "gold_call": "np.round(_oracle_call_payoff_cosine_coeffs(K, a_p, b_p, 12), 8).tolist()",
        },
        # Case 4 - the jump-free benchmark instance.
        {
            "setup": """import numpy as np
K = 80.0
a_p = 0.7776
b_p = 8.4224
""",
            "call": "np.round(call_payoff_cosine_coeffs(K, a_p, b_p, 64), 8).tolist()",
            "gold_call": "np.round(_oracle_call_payoff_cosine_coeffs(K, a_p, b_p, 64), 8).tolist()",
        },
        # Case 5 - strike above the right endpoint: the payoff vanishes on the interval.
        {
            "setup": """import numpy as np
K = 500.0
a_p = 0.0
b_p = 5.0
def all_zero(V):
    return bool(np.all(np.asarray(V, dtype=float) == 0.0))
""",
            "call": "all_zero(call_payoff_cosine_coeffs(K, a_p, b_p, 16))",
            "gold_call": "all_zero(_oracle_call_payoff_cosine_coeffs(K, a_p, b_p, 16))",
        },
        # Case 6 - invalid: degenerate interval.
        {
            "setup": """import numpy as np
def run_model_iv():
    try:
        call_payoff_cosine_coeffs(100.0, 2.0, 2.0, 8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_iv():
    try:
        _oracle_call_payoff_cosine_coeffs(100.0, 2.0, 2.0, 8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_iv()",
            "gold_call": "run_oracle_iv()",
        },
        # Case 7 - invalid: non-positive strike.
        {
            "setup": """import numpy as np
def run_model_K():
    try:
        call_payoff_cosine_coeffs(0.0, 0.0, 5.0, 8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle_K():
    try:
        _oracle_call_payoff_cosine_coeffs(0.0, 0.0, 5.0, 8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model_K()",
            "gold_call": "run_oracle_K()",
        },
    ]
