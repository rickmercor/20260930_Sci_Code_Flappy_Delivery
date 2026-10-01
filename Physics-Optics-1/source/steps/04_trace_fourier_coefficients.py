"""
Compute the complex Fourier amplitudes $C_n^{(q)}(h_0)$, $n=0,1,\dots,n_{\max}$, of the approximant cell trace at reference continuation height $h_0$, sampling the trace at $K$ equally spaced real phases over one of its periods.

Because a shift of the initial phase by $p/q$ cyclically permutes the cell factors, the cell trace has a shorter period than the unit phase circle, so its Fourier expansion contains only a restricted set of harmonics. The paper fixes a specific sign convention for the harmonic exponent, a specific normalization of the coefficient integral, and a specific labeling $n$ of the harmonics; with the paper's convention the positive orders are the ones that grow with $h$. Consult the paper's own Fourier-expansion equation for all three; do not assume the ordinary unit-period Fourier convention. The $K$-node rule over one period is the trapezoidal discretization of that integral.

Returns
-------
np.ndarray of shape (n_max + 1,), complex: $[C_0^{(q)}(h_0),\dots,C_{n_{\max}}^{(q)}(h_0)]$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def trace_fourier_coefficients(h0: float, p: int, q: int, eps_R: float, d_eps: float,
                               w0tau: float, n_max: int, K: int) -> "np.ndarray":
    r"""Fourier amplitudes of the approximant cell trace at height $h_0$.

    Args:
        h0 (float): reference continuation height $h_0$, strictly inside the strip.
        p (int): approximant numerator $p$ (coprime with $q$).
        q (int): approximant denominator $q$.
        eps_R (float): background relative permittivity $\varepsilon_R$.
        d_eps (float): modulation-amplitude parameter $\delta\varepsilon$ at $h=0$.
        w0tau (float): reference phase $\omega_0\tau$ accumulated per layer.
        n_max (int): highest retained non-negative Fourier order $n_{\max}\ge 0$.
        K (int): number $K$ of equally spaced real phase nodes over one period of the
            trace; must satisfy $K\ge 2n_{\max}+2$.

    Raises:
        ValueError: if $n_{\max}<0$ or $K<2n_{\max}+2$ (plus any error raised by the
            cell-trace evaluation).

    Expected return:
        np.ndarray of shape (n_max + 1,), complex. Index $n$ holds the order-$n$ amplitude.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: trace_fourier_coefficients
def _oracle_trace_fourier_coefficients(h0: float, p: int, q: int, eps_R: float, d_eps: float,
                                       w0tau: float, n_max: int, K: int) -> "np.ndarray":
    if n_max < 0 or K < 2 * n_max + 2:
        raise ValueError("need n_max >= 0 and K >= 2 * n_max + 2")
    phis = np.arange(K) / (K * q)
    F = np.array([_oracle_approximant_cell_trace(x, h0, p, q, eps_R, d_eps, w0tau) for x in phis])
    return np.array([np.mean(F * np.exp(2j * np.pi * n * q * phis)) for n in range(n_max + 1)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'import numpy as np\n'
                'v = trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.54, 4, 64)'
            ),
            "call": 'np.round(np.log(np.abs(v)), 6)',
            "gold_call": 'np.round(np.log(np.abs(_oracle_trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.54, 4, 64))), 6)',
            "tol": 1e-05,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.82, 4, 64)'
            ),
            "call": 'np.round(np.log(np.abs(v)), 6)',
            "gold_call": 'np.round(np.log(np.abs(_oracle_trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.82, 4, 64))), 6)',
            "tol": 1e-05,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.82, 4, 64)'
            ),
            "call": 'np.round(np.cos(np.angle(v)), 6)',
            "gold_call": 'np.round(np.cos(np.angle(_oracle_trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.82, 4, 64))), 6)',
            "tol": 1e-05,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = trace_fourier_coefficients(1.2, 13, 21, 3.4, 1.0, 9.03, 3, 32)'
            ),
            "call": 'np.round(np.log(np.abs(v)), 6)',
            "gold_call": 'np.round(np.log(np.abs(_oracle_trace_fourier_coefficients(1.2, 13, 21, 3.4, 1.0, 9.03, 3, 32))), 6)',
            "tol": 1e-05,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def _phase_components(z):\n'
                '    phase = np.angle(z)\n'
                '    return np.round(np.concatenate([np.cos(phase), np.sin(phase)]), 6)\n'
                'v = trace_fourier_coefficients(1.0, 3, 5, 3.4, 1.0, 9.03, 2, 16)'
            ),
            "call": '_phase_components(v)',
            "gold_call": '_phase_components(_oracle_trace_fourier_coefficients(1.0, 3, 5, 3.4, 1.0, 9.03, 2, 16))',
            "tol": 1e-05,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def run_model():\n'
                '    try:\n'
                '        trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.54, 4, 8)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_trace_fourier_coefficients(0.8, 21, 34, 3.4, 1.0, 15.54, 4, 8)\n'
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
