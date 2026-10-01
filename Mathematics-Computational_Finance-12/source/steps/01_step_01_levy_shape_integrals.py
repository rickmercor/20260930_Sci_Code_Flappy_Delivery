"""
Compute three integrals of the normalized asymmetric tempered-stable jump-size shape that the rest of the pipeline consumes: the mean excitation of the activity process, the compensated exponential jump integral that enters the risk-neutral drift restriction, and the jump-entropy integral that enters the sufficient true-martingale condition.

The shape is parameterised by p, M, G and a_ts, where p in (0,1) is the share of the second jump moment carried by positive jumps, M and G are the positive and negative exponential tempering rates, and a_ts in (0,2) is the stable index. The shape is normalized so that its second moment is one. The excitation function is g(y) = 1 - exp(-a_exc * y**2), with a_exc > 0. In the drift-restriction integral only jumps of size below one are compensated: the integrand is exp(y) - 1 - y on the small-jump region and exp(y) - 1 on the tail, with the reflected forms on the negative half-line.

Every Levy integral in this task, in this step and in all later steps, is evaluated with the following protocol. Map the negative half-line onto the positive one by reflection. Truncate the positive half-line at y_cut = 10 and split it into the small-jump region (0, 1) and the tail (1, y_cut). On the small-jump region use 128-node Gauss-Jacobi quadrature with the endpoint weight induced by the near-origin behaviour of the compensated integrands. On the tail use 128-node Gauss-Legendre quadrature.

Return the three values in the order: mean excitation, drift-restriction integral, martingale-condition integral.

Raises ValueError if p is not in (0,1); if M <= 0 or G <= 0; if a_ts is not in (0,2); if a_exc <= 0; or if M <= 1, the positive-tail condition under which the drift-restriction integral is finite.

Infinite-activity jump models describe returns with a Levy measure that has a non-integrable singularity at the origin, so that infinitely many small jumps occur on any interval, while large jumps are damped by exponential tempering on each tail separately. Tempered-stable specifications of this kind combine a stable-like small-jump structure with finite moments of every order, and the asymmetry between the two tempering rates controls the skew of the return distribution.

When such a jump component is combined with a stochastic activity scale, it is convenient to normalize the jump-size measure so that its second moment is one. The activity state then has a direct interpretation as the local rate at which jumps contribute quadratic variation, and the normalizing constants for each tail follow from the tempering rates and the stable index through the Gamma function. Under that normalization the two tails carry shares p and 1 - p of the second moment.

Three integrals against this shape recur throughout any pricing calculation built on such a model. The first is the average of the excitation function, which measures how much realized jump variation feeds back into future activity and therefore governs the stability of the activity process. The second is the compensated exponential jump integral, which is what the drift must offset for the discounted price to be a martingale, and which converges only when the positive tempering rate is large enough to dominate the exponential growth of the integrand. The third is a jump-entropy integral that appears in exponential-martingale criteria for processes with jumps.

All three integrands share a numerical difficulty. Each is compensated near the origin, so it vanishes to second order there, while the Levy density diverges. The product is integrable, but evaluating the two factors separately at quadrature nodes close to the origin subtracts nearly equal quantities. A quadrature rule whose weight function matches the true endpoint behaviour of the product avoids this entirely, which is why the rule used on the small-jump region is chosen to carry that weight rather than applied to the raw integrand.

Returns
-------
np.ndarray of shape (3,), the mean excitation, drift-restriction and martingale-condition integrals as native float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def levy_shape_integrals(p: float, M: float, G: float, a_ts: float,
                         a_exc: float) -> 'np.ndarray':
    '''Compute the three shape integrals of the jump-size law.

    Parameters
    ----------
    p : float
        Share of the second jump moment carried by positive jumps, in (0,1).
    M : float
        Positive-jump exponential tempering rate, greater than 1.
    G : float
        Negative-jump exponential tempering rate, positive.
    a_ts : float
        Stable index of the jump-size shape, in (0,2).
    a_exc : float
        Excitation parameter, positive.

    Returns
    -------
    result : np.ndarray
        Shape (3,): mean excitation, drift-restriction integral,
        martingale-condition integral, in that order. The drift-restriction
        integral compensates only jumps of size below one, so its integrand is
        exp(y) - 1 - y on the small-jump region and exp(y) - 1 on the tail.

    Raises
    ------
    ValueError
        If p is not in (0,1); if M <= 0 or G <= 0; if a_ts is not in (0,2);
        if a_exc <= 0; or if M <= 1, the positive-tail condition under which
        the drift-restriction integral is finite.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.special import roots_jacobi, roots_legendre, gamma as _gamma_fn


def _h_nodes(a_ts, n_small=128, n_tail=128, y_cut=10.0):
    """Quadrature nodes and weights for the split protocol."""
    jac_a, jac_b = 0.0, 1.0 - a_ts
    xj, wj = roots_jacobi(n_small, jac_a, jac_b)
    ys = (xj + 1.0) / 2.0
    ws = wj / 2.0 ** (jac_a + jac_b + 1.0)
    xl, wl = roots_legendre(n_tail)
    yt = 0.5 * (y_cut - 1.0) * xl + 0.5 * (y_cut + 1.0)
    wt = 0.5 * (y_cut - 1.0) * wl
    return ys, ws, yt, wt


def _h_density(p, M, G, a_ts, a_exc, ys, yt):
    """Shape constants, density factors and the excitation on both node sets."""
    c_pos = p * M ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    c_neg = (1.0 - p) * G ** (2.0 - a_ts) / _gamma_fn(2.0 - a_ts)
    return {
        "sp": c_pos * np.exp(-M * ys),
        "sm": c_neg * np.exp(-G * ys),
        "tp": c_pos * np.exp(-M * yt) * yt ** (-1.0 - a_ts),
        "tm": c_neg * np.exp(-G * yt) * yt ** (-1.0 - a_ts),
        "gs": 1.0 - np.exp(-a_exc * ys ** 2),
        "gt": 1.0 - np.exp(-a_exc * yt ** 2),
    }


def _h_integrate(ws, wt, dens, ys, bs_pos, bt_pos, bs_neg, bt_neg):
    """Integrate over the whole real line.

    bs_* are the integrand on the small nodes and are divided by y**2 here,
    because the y**(1 - a_ts) part of the kernel sits in the Jacobi weight.
    bt_* are the integrand on the tail nodes, used directly.
    """
    return (
        np.sum(ws * dens["sp"] * bs_pos / ys ** 2)
        + np.sum(wt * dens["tp"] * bt_pos)
        + np.sum(ws * dens["sm"] * bs_neg / ys ** 2)
        + np.sum(wt * dens["tm"] * bt_neg)
    )


def _oracle_levy_shape_integrals(p: float, M: float, G: float, a_ts: float,
                                 a_exc: float) -> 'np.ndarray':
    """Reference implementation."""
    if not (0.0 < float(p) < 1.0):
        raise ValueError("p must lie in (0,1)")
    if float(M) <= 0.0 or float(G) <= 0.0:
        raise ValueError("M and G must be positive")
    if not (0.0 < float(a_ts) < 2.0):
        raise ValueError("a_ts must lie in (0,2)")
    if float(a_exc) <= 0.0:
        raise ValueError("a_exc must be positive")
    if float(M) <= 1.0:
        raise ValueError(
            "M must exceed 1 for a finite drift-restriction integral")

    ys, ws, yt, wt = _h_nodes(float(a_ts))
    dens = _h_density(float(p), float(M), float(G), float(a_ts),
                      float(a_exc), ys, yt)

    g_small = -np.expm1(-float(a_exc) * ys ** 2)
    g_tail = -np.expm1(-float(a_exc) * yt ** 2)
    mean_exc = _h_integrate(ws, wt, dens, ys,
                            g_small, g_tail, g_small, g_tail)

    drift = _h_integrate(
        ws, wt, dens, ys,
        np.expm1(ys) - ys, np.exp(yt) - 1.0,
        np.expm1(-ys) + ys, np.exp(-yt) - 1.0)

    entropy = _h_integrate(
        ws, wt, dens, ys,
        ys * np.expm1(ys) - (np.expm1(ys) - ys),
        yt * np.exp(yt) - np.exp(yt) + 1.0,
        -ys * np.expm1(-ys) - (np.expm1(-ys) + ys),
        -yt * np.exp(-yt) - np.exp(-yt) + 1.0)

    return np.array([mean_exc, drift, entropy], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- normal: the locked configuration ---
        {
            "setup": """import numpy as np
p, M, G, a_ts, a_exc = 0.55, 6.5, 3.0, 1.1, 0.8
""",
            "call": "levy_shape_integrals(p, M, G, a_ts, a_exc)",
            "gold_call": "_oracle_levy_shape_integrals(p, M, G, a_ts, a_exc)",
        },
        # --- boundary: stable index just below one, where the small jumps
        #     switch from infinite to finite variation ---
        {
            "setup": """import numpy as np
p, M, G, a_ts, a_exc = 0.55, 6.5, 3.0, 0.999, 0.8
""",
            "call": "levy_shape_integrals(p, M, G, a_ts, a_exc)",
            "gold_call": "_oracle_levy_shape_integrals(p, M, G, a_ts, a_exc)",
        },
        # --- edge: symmetric shape, equal tempering, strong excitation ---
        {
            "setup": """import numpy as np
p, M, G, a_ts, a_exc = 0.5, 4.0, 4.0, 1.1, 5.0
""",
            "call": "levy_shape_integrals(p, M, G, a_ts, a_exc)",
            "gold_call": "_oracle_levy_shape_integrals(p, M, G, a_ts, a_exc)",
        },
        # --- edge: tempering rate only just above the positive-tail bound.
        #     Compared at 1e-6: at a_ts = 1.7 the Jacobi nodes reach 2.1e-5, where
        #     a literal evaluation of the compensated integrands loses up to 1e-7 ---
        {
            "setup": """import numpy as np
p, M, G, a_ts, a_exc = 0.2, 1.05, 9.0, 1.7, 0.1
""",
            "call": "levy_shape_integrals(p, M, G, a_ts, a_exc)",
            "gold_call": "_oracle_levy_shape_integrals(p, M, G, a_ts, a_exc)",
            "tol": 1e-6,
        },
        # --- structural probe: monotone response to the excitation parameter,
        #     positivity of both compensated integrals, and the locked mean
        #     excitation, all returned as exact integers ---
        {
            "setup": """import numpy as np
def probe():
    v = levy_shape_integrals(0.55, 6.5, 3.0, 1.1, 0.8)
    w = levy_shape_integrals(0.55, 6.5, 3.0, 1.1, 8.0)
    return [int(v[0] < w[0]), int(v[1] > 0.0), int(v[2] > 0.0),
            int(round(1e6 * float(v[0])))]
def probe_gold():
    v = _oracle_levy_shape_integrals(0.55, 6.5, 3.0, 1.1, 0.8)
    w = _oracle_levy_shape_integrals(0.55, 6.5, 3.0, 1.1, 8.0)
    return [int(v[0] < w[0]), int(v[1] > 0.0), int(v[2] > 0.0),
            int(round(1e6 * float(v[0])))]
""",
            "call": "probe()",
            "gold_call": "probe_gold()",
        },
        # --- invalid: tempering rate at or below the positive-tail bound ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        levy_shape_integrals(0.55, 0.9, 3.0, 1.1, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_levy_shape_integrals(0.55, 0.9, 3.0, 1.1, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: stable index outside (0,2) ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        levy_shape_integrals(0.55, 6.5, 3.0, 2.4, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_levy_shape_integrals(0.55, 6.5, 3.0, 2.4, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: moment share outside (0,1) ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        levy_shape_integrals(1.0, 6.5, 3.0, 1.1, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_levy_shape_integrals(1.0, 6.5, 3.0, 1.1, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- invalid: non-positive excitation parameter ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        levy_shape_integrals(0.55, 6.5, 3.0, 1.1, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_levy_shape_integrals(0.55, 6.5, 3.0, 1.1, 0.0)
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
