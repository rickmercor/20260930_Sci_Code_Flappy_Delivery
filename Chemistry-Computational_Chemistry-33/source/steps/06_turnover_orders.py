"""
The reduced hierarchy carries only the projections p_tilde = Q p of the probability, and that is enough for an observable that takes one value on all the configurations a reduced coordinate covers, such as the adsorbate count on a lattice whose classes fix it. The reaction propensity is not such an observable: two configurations of the same class hold the same adsorbates in different arrangements, so they offer the reaction a different number of pairs and turn over at different rates. Averaging it over the reduced coefficients alone amounts to assuming that the arrangement inside a class never departs from its leading-order form, which is precisely the assumption the expansion was set up to correct, so that route returns the zeroth-order answer dressed up as a corrected one.

The full-space probability therefore has to be rebuilt before the observable is averaged. Each term of the expansion is a linear combination of the reduced coefficient vectors of its own order and below, and the conventions of the closure step fix every coefficient in it: the leading-order map multiplies the vector of the term's own order, and because the hierarchy has the same form at every order, the responses the closure step produced are all the other coefficients the rebuild needs. They are the only place where the arrangement inside a class can move away from its leading-order form.

Contracting each rebuilt term with the reaction propensity gives the coefficients

    o_k = sum_i rate(i) * p^(k)_i,   k = 0, 1, ..., K,

from which the k-th order truncation of the expected reaction rate is recombined afterwards as sum over l <= k of eps^l o_l. The coefficients themselves carry no eps, and o_0 is exactly what a simulation that treats the fast channel as infinitely fast would report. The order K is read off the sizes of the inputs.

Whatever the reduced coefficient vectors mean, the same assembly applies: here they are the sampling-window averages produced by the previous step, so the o_k are the coefficients of the window-averaged reaction rate.

The reduced hierarchy carries only the projections p_tilde = Q p of the probability, and that is enough for an observable that takes one value on all the configurations a reduced coordinate covers, such as the adsorbate count on a lattice whose classes fix it. The reaction propensity is not such an observable: two configurations of the same class hold the same adsorbates in different arrangements, so they offer the reaction a different number of pairs and turn over at different rates. Averaging it over the reduced coefficients alone amounts to assuming that the arrangement inside a class never departs from its leading-order form, which is precisely the assumption the expansion was set up to correct, so that route returns the zeroth-order answer dressed up as a corrected one.

Returns
-------
np.ndarray of float with shape (K + 1,): the coefficients o_0, ..., o_K of the expected reaction rate in the eps expansion, o_k being the reaction propensity averaged over the k-th term of the rebuilt full-space probability.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def turnover_orders(event_rate: "np.ndarray", K0: "np.ndarray", resp: "np.ndarray", coeffs: "np.ndarray") -> "np.ndarray": """Parameters: reaction propensities, leading map, response matrices, and reduced coefficients. Returns: order-by-order reaction-rate coefficients. Raises: ValueError if K0, resp or coeffs is not a 2D array; if event_rate, K0 and resp do not share the configuration count; if resp does not have shape (m, K n) for K >= 1; if coeffs does not have shape (K + 1, n); or if any input holds a non-finite entry."""; return orders

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_turnover_orders(event_rate: "np.ndarray", K0: "np.ndarray", resp: "np.ndarray",
                            coeffs: "np.ndarray") -> "np.ndarray":
    event_rate = np.asarray(event_rate, dtype=float).reshape(-1)
    K0 = np.asarray(K0, dtype=float)
    resp = np.asarray(resp, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    if K0.ndim != 2 or resp.ndim != 2 or coeffs.ndim != 2:
        raise ValueError("K0, resp and coeffs must be 2D arrays")
    m, n = K0.shape
    if event_rate.size != m or resp.shape[0] != m:
        raise ValueError("event_rate, K0 and resp must share the configuration count")
    if n < 1 or resp.shape[1] % n != 0 or resp.shape[1] < n:
        raise ValueError("resp must have shape (m, K n) with K >= 1")
    order = resp.shape[1] // n
    if coeffs.shape != (order + 1, n):
        raise ValueError("coeffs must have shape (K + 1, n)")
    for M in (event_rate, K0, resp, coeffs):
        if not np.all(np.isfinite(M)):
            raise ValueError("inputs must be finite")
    G = [K0] + [resp[:, k * n:(k + 1) * n] for k in range(order)]
    out = np.empty(order + 1)
    for k in range(order + 1):
        term = sum(G[k - l] @ coeffs[l] for l in range(k + 1))
        out[k] = float(event_rate @ term)
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    setup_common = """import numpy as np
def _mk(m, n, seed, K=3):
    g = np.random.default_rng(seed)
    K0 = np.abs(g.normal(size=(m, n))) + 0.1
    K0 /= K0.sum(axis=0, keepdims=True)
    resp = g.normal(size=(m, K * n))
    for blk in range(K):
        col = resp[:, blk * n:(blk + 1) * n]
        col -= col.mean(axis=0, keepdims=True)
    rate = np.abs(g.normal(size=m)) * 3.0
    coeffs = g.normal(size=(K + 1, n)) * 0.2
    coeffs[0] = np.abs(coeffs[0]); coeffs[0] /= coeffs[0].sum()
    for k in range(1, K + 1):
        coeffs[k] -= coeffs[k].mean()
    return rate, K0, resp, coeffs
def _fp(res):
    v = np.asarray(res, dtype=float).ravel()
    return float(np.sum(v * np.cos(np.arange(v.size, dtype=float))) + 1e-3 * v.size)
"""
    return [
        # --- Normal: a moderate state space with four classes at third order ---
        {"setup": setup_common + "rate, K0, resp, coeffs = _mk(24, 4, 11)\n",
         "call": "_fp(turnover_orders(rate, K0, resp, coeffs))",
         "gold_call": "_fp(_oracle_turnover_orders(rate, K0, resp, coeffs))"},
        # --- Normal: many classes with few members each ---
        {"setup": setup_common + "rate, K0, resp, coeffs = _mk(30, 10, 12)\n",
         "call": "_fp(turnover_orders(rate, K0, resp, coeffs))",
         "gold_call": "_fp(_oracle_turnover_orders(rate, K0, resp, coeffs))"},
        # --- Normal: a fifth-order expansion ---
        {"setup": setup_common + "rate, K0, resp, coeffs = _mk(18, 3, 21, K=5)\n",
         "call": "_fp(turnover_orders(rate, K0, resp, coeffs))",
         "gold_call": "_fp(_oracle_turnover_orders(rate, K0, resp, coeffs))"},
        # --- Boundary: one class holding the whole state space ---
        {"setup": setup_common + "rate, K0, resp, coeffs = _mk(9, 1, 13)\n",
         "call": "_fp(turnover_orders(rate, K0, resp, coeffs))",
         "gold_call": "_fp(_oracle_turnover_orders(rate, K0, resp, coeffs))"},
        # --- Boundary: a first-order expansion ---
        {"setup": setup_common + "rate, K0, resp, coeffs = _mk(12, 3, 31, K=1)\n",
         "call": "_fp(turnover_orders(rate, K0, resp, coeffs))",
         "gold_call": "_fp(_oracle_turnover_orders(rate, K0, resp, coeffs))"},
        # --- Boundary: an observable constant across the state space ---
        {"setup": setup_common + "rate, K0, resp, coeffs = _mk(24, 4, 14)\nrate = np.full(24, 2.5)\n",
         "call": "_fp(turnover_orders(rate, K0, resp, coeffs))",
         "gold_call": "_fp(_oracle_turnover_orders(rate, K0, resp, coeffs))"},
        # --- Edge: vanishing response matrices ---
        {"setup": setup_common + "rate, K0, resp, coeffs = _mk(24, 4, 15)\nresp = np.zeros_like(resp)\n",
         "call": "_fp(turnover_orders(rate, K0, resp, coeffs))",
         "gold_call": "_fp(_oracle_turnover_orders(rate, K0, resp, coeffs))"},
        # --- Edge: higher-order reduced coefficients all zero ---
        {"setup": setup_common + "rate, K0, resp, coeffs = _mk(24, 4, 16)\ncoeffs[1:] = 0.0\n",
         "call": "_fp(turnover_orders(rate, K0, resp, coeffs))",
         "gold_call": "_fp(_oracle_turnover_orders(rate, K0, resp, coeffs))"},
        # --- Invalid: coefficients for a different order than the responses ---
        {"setup": setup_common + """
rate, K0, resp, coeffs = _mk(24, 4, 17)
bad = coeffs[:3]
def run_model():
    try:
        turnover_orders(rate, K0, resp, bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_turnover_orders(rate, K0, resp, bad)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]
