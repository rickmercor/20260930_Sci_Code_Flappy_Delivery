"""
Compute the half-width $h_\star$ of the nonsingular analytic strip of the complex-continued quasiperiodic permittivity sequence of the time-switched dielectric, i.e. the largest imaginary phase displacement $|h|$ for which the continued single-harmonic permittivity stays nonzero for every real modulation phase.

The layer transfer matrices of the temporal layer sequence are analytic in the continuation coordinate $h$ only while the continued permittivity avoids zero; the reciprocal permittivity in the displacement-field wave equation is singular at such a zero. Every downstream Fourier-coefficient construction and every integer-phase statement in the paper is restricted to this strip. Consult the paper's own analysis of the nearest complex zero of the sinusoidal waveform for the exact expression.

Returns
-------
float: $h_\star>0$, the half-width of the nonsingular strip in the continuation coordinate $h$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def analytic_strip_halfwidth(eps_R: float, d_eps: float) -> float:
    r"""Half-width of the nonsingular strip of the continued permittivity.

    Args:
        eps_R (float): background (reference) relative permittivity $\varepsilon_R$, real and positive.
        d_eps (float): modulation-amplitude parameter $\delta\varepsilon$ at $h=0$, real and positive.

    Raises:
        ValueError: unless $\varepsilon_R>\delta\varepsilon>0$.

    Expected return:
        float: strictly positive; grows without bound as $\delta\varepsilon\to 0$ and tends to $0$
        as $\delta\varepsilon\to\varepsilon_R$ from below.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: analytic_strip_halfwidth
def _oracle_analytic_strip_halfwidth(eps_R: float, d_eps: float) -> float:
    if not (eps_R > d_eps > 0.0):
        raise ValueError("requires eps_R > d_eps > 0")
    return float(np.arccosh(eps_R / d_eps))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'import numpy as np\n'
                'v = analytic_strip_halfwidth(3.4, 1.0)'
            ),
            "call": 'round(v, 8)',
            "gold_call": 'round(_oracle_analytic_strip_halfwidth(3.4, 1.0), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = analytic_strip_halfwidth(5.0, 2.0)'
            ),
            "call": 'round(v, 8)',
            "gold_call": 'round(_oracle_analytic_strip_halfwidth(5.0, 2.0), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'v = analytic_strip_halfwidth(2.0, 1.9)'
            ),
            "call": 'round(v, 8)',
            "gold_call": 'round(_oracle_analytic_strip_halfwidth(2.0, 1.9), 8)',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def run_model():\n'
                '    try:\n'
                '        analytic_strip_halfwidth(1.0, 2.0)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_analytic_strip_halfwidth(1.0, 2.0)\n'
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
                '        analytic_strip_halfwidth(3.4, -1.0)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_analytic_strip_halfwidth(3.4, -1.0)\n'
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
