"""
Given the intercepts $b_n^{(q)}$ of one band, integrate the predicted integer bulk response $n(h)$ over the continuation interval $[h_a,h_b]$, where $n(h)$ is the Fourier order that dominates the limiting envelope of the logarithmic coefficient lines at height $h$.

Each Fourier order contributes one straight line of integer slope; the order on top of their upper envelope sets the predicted integer slope of the growth rate, and a crossing between the dominant order and its successor is the paper's candidate phase boundary. The paper's boundary predictor uses neither a fitted Lyapunov intercept nor a measured response midpoint. When the dominant order skips an integer, the paper's general crossing relation, not the adjacent-order one, is the appropriate comparison; consult the paper's own crossing equations. The integral is the exact area under the resulting piecewise-constant integer function.

Returns
-------
float: the integral of the predicted dominant order $n(h)$ over $[h_a,h_b]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def integrated_dominant_order(b: "np.ndarray", h_a: float, h_b: float) -> float:
    r"""Area under the predicted piecewise-constant integer response.

    Args:
        b (np.ndarray): real intercepts $[b_0,b_1,\dots,b_{n_{\max}}]$; index = Fourier order.
        h_a (float): lower end $h_a$ of the continuation interval.
        h_b (float): upper end $h_b$, with $h_b>h_a$.

    Raises:
        ValueError: if $h_b\le h_a$.

    Expected return:
        float: between $(\min n)(h_b-h_a)$ and $(\max n)(h_b-h_a)$ over the retained orders $n$; equal to
        $n\,(h_b-h_a)$ when one order $n$ dominates throughout the interval.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: integrated_dominant_order
def _oracle_integrated_dominant_order(b: "np.ndarray", h_a: float, h_b: float) -> float:
    b = np.asarray(b, dtype=float)
    if not h_b > h_a:
        raise ValueError("need h_b > h_a")
    orders = np.arange(len(b))
    pts = [h_a, h_b]
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            x = (b[i] - b[j]) / (j - i)
            if h_a < x < h_b:
                pts.append(x)
    pts = np.sort(np.array(pts))
    total = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (lo + hi)
        total += int(np.argmax(b + orders * mid)) * (hi - lo)
    return float(total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'import numpy as np\n'
                'v = integrated_dominant_order(np.array([0.0152599, -0.5988219, -1.3226223, -2.2937098, -3.4920013]), 0.60, 1.10)'
            ),
            "call": 'round(v, 8)',
            "gold_call": 'round(_oracle_integrated_dominant_order(np.array([0.0152599, -0.5988219, -1.3226223, -2.2937098, -3.4920013]), 0.60, 1.10), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = integrated_dominant_order(np.array([0.0164905, -0.6035127, -1.2604040, -2.2742122, -3.3931359]), 0.60, 1.10)'
            ),
            "call": 'round(v, 8)',
            "gold_call": 'round(_oracle_integrated_dominant_order(np.array([0.0164905, -0.6035127, -1.2604040, -2.2742122, -3.3931359]), 0.60, 1.10), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = integrated_dominant_order(np.array([0.0, -2.0, -1.0]), 0.0, 1.5)'
            ),
            "call": 'round(v, 8)',
            "gold_call": 'round(_oracle_integrated_dominant_order(np.array([0.0, -2.0, -1.0]), 0.0, 1.5), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = integrated_dominant_order(np.array([0.0, -0.3, -0.9, -1.8]), 0.1, 0.6)'
            ),
            "call": 'round(v, 8)',
            "gold_call": 'round(_oracle_integrated_dominant_order(np.array([0.0, -0.3, -0.9, -1.8]), 0.1, 0.6), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = integrated_dominant_order(np.array([1.0, -0.2, -1.7, -3.0]), -0.5, 2.5)'
            ),
            "call": 'round(v, 8)',
            "gold_call": 'round(_oracle_integrated_dominant_order(np.array([1.0, -0.2, -1.7, -3.0]), -0.5, 2.5), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def run_model():\n'
                '    try:\n'
                '        integrated_dominant_order(np.array([0.0, -1.0]), 1.0, 1.0)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_integrated_dominant_order(np.array([0.0, -1.0]), 1.0, 1.0)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2'
            ),
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
    ]
