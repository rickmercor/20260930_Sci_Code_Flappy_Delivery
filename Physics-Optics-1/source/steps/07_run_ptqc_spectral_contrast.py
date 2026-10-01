"""
Chain the earlier steps into the full prediction: check the continuation interval and the reference height against the nonsingular strip, build the approximant trace Fourier amplitudes and their intercepts for each of the two input bands, integrate each band's predicted integer response over $[h_a,h_b]$, and convert the integrated integer difference (band 2 minus band 1) into the change of the two-band output power ratio, in decibels, after $N$ layers. The reference implementation calls the earlier public functions by name rather than reproducing them.

In the paper's long-propagation, narrowband limit the logarithmic sensitivity of the two-band power ratio to $h$ is fixed by the difference of the two bands' integer responses and grows in proportion to the number of layers; integrated across phase boundaries it gives the predicted change of the output spectral contrast. Consult the paper's own leading-order relation for the prefactor and for how the integer difference enters when a band crosses a boundary inside the interval. The result is the paper's approximant prediction, not the finite-length transfer-matrix or PSTD output.

Returns
-------
float: predicted change of $10\log_{10}(P_2/P_1)$, in dB, between $h_a$ and $h_b$ after $N$ layers.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_ptqc_spectral_contrast(w1tau: float, w2tau: float, eps_R: float, d_eps: float,
                               p: int, q: int, h0: float, h_a: float, h_b: float, N: int,
                               n_max: int, K: int) -> float:
    r"""Predicted change of the two-band output power ratio (dB).

    Args:
        w1tau (float): reference phase per layer $\omega_1\tau$ at the centre of band 1.
        w2tau (float): reference phase per layer $\omega_2\tau$ at the centre of band 2.
        eps_R (float): background relative permittivity $\varepsilon_R$.
        d_eps (float): modulation-amplitude parameter $\delta\varepsilon$ at $h=0$.
        p (int): approximant numerator $p$.
        q (int): approximant denominator $q$.
        h0 (float): reference continuation height $h_0$ for the Fourier amplitudes.
        h_a (float): lower end $h_a$ of the continuation interval, $h_a\ge 0$.
        h_b (float): upper end $h_b$, with $h_a<h_b<h_\star$ (the strip half-width).
        N (int): number of propagated layers, $N>0$.
        n_max (int): highest retained Fourier order $n_{\max}$.
        K (int): phase nodes $K$ per trace period.

    Raises:
        ValueError: if the interval or $h_0$ is not inside the nonsingular strip, if
            $N\le 0$, or if any earlier step rejects its inputs.

    Expected return:
        float: dB change; changes sign when the two bands are swapped and is
        proportional to $N$.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: run_ptqc_spectral_contrast
def _oracle_run_ptqc_spectral_contrast(w1tau: float, w2tau: float, eps_R: float, d_eps: float,
                                       p: int, q: int, h0: float, h_a: float, h_b: float, N: int,
                                       n_max: int, K: int) -> float:
    h_star = _oracle_analytic_strip_halfwidth(eps_R, d_eps)
    if not (0.0 <= h_a < h_b < h_star) or not (abs(h0) < h_star):
        raise ValueError("interval and reference height must lie inside the nonsingular strip")
    if N <= 0:
        raise ValueError("N must be positive")
    integrals = []
    for w0tau in (w1tau, w2tau):
        coeffs = _oracle_trace_fourier_coefficients(h0, p, q, eps_R, d_eps, w0tau, n_max, K)
        b = _oracle_coefficient_intercepts(coeffs, q, h0)
        integrals.append(_oracle_integrated_dominant_order(b, h_a, h_b))
    return float(20.0 * N / np.log(10.0) * (integrals[1] - integrals[0]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": 'import numpy as np',
            "call": 'round(run_ptqc_spectral_contrast(15.54, 15.82, 3.4, 1.0, 21, 34, 0.8, 0.60, 1.10, 120, 4, 64), 6)',
            "gold_call": 'round(_oracle_run_ptqc_spectral_contrast(15.54, 15.82, 3.4, 1.0, 21, 34, 0.8, 0.60, 1.10, 120, 4, 64), 6)',
            "tol": 1e-05,
        },
        {
            "setup": 'import numpy as np',
            "call": 'round(run_ptqc_spectral_contrast(15.82, 15.54, 3.4, 1.0, 21, 34, 0.8, 0.60, 1.10, 120, 4, 64), 6)',
            "gold_call": 'round(_oracle_run_ptqc_spectral_contrast(15.82, 15.54, 3.4, 1.0, 21, 34, 0.8, 0.60, 1.10, 120, 4, 64), 6)',
            "tol": 1e-05,
        },
        {
            "setup": 'import numpy as np',
            "call": 'round(run_ptqc_spectral_contrast(15.54, 15.82, 3.4, 1.0, 13, 21, 0.8, 0.60, 1.10, 120, 4, 64), 6)',
            "gold_call": 'round(_oracle_run_ptqc_spectral_contrast(15.54, 15.82, 3.4, 1.0, 13, 21, 0.8, 0.60, 1.10, 120, 4, 64), 6)',
            "tol": 1e-05,
        },
        {
            "setup": 'import numpy as np',
            "call": 'round(run_ptqc_spectral_contrast(9.03, 8.50, 3.4, 1.0, 21, 34, 1.2, 1.00, 1.40, 64, 4, 64), 6)',
            "gold_call": 'round(_oracle_run_ptqc_spectral_contrast(9.03, 8.50, 3.4, 1.0, 21, 34, 1.2, 1.00, 1.40, 64, 4, 64), 6)',
            "tol": 1e-05,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def run_model():\n'
                '    try:\n'
                '        run_ptqc_spectral_contrast(15.54, 15.82, 3.4, 1.0, 21, 34, 0.8, 0.60, 1.95, 120, 4, 64)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_run_ptqc_spectral_contrast(15.54, 15.82, 3.4, 1.0, 21, 34, 0.8, 0.60, 1.95, 120, 4, 64)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2'
            ),
            "call": 'run_model()',
            "gold_call": 'run_gold()',
        },
        {
            "setup": (
                'import numpy as np\n'
                'def run_model():\n'
                '    try:\n'
                '        run_ptqc_spectral_contrast(15.54, 15.82, 3.4, 1.0, 21, 34, 0.8, 0.60, 1.10, 0, 4, 64)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_run_ptqc_spectral_contrast(15.54, 15.82, 3.4, 1.0, 21, 34, 0.8, 0.60, 1.10, 0, 4, 64)\n'
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
