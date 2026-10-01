"""
Compute the trace $F_q(\varphi,h)$ of the $q$-layer periodic-cell propagator of the paper's auxiliary rational approximant $p/q$ of the golden-mean rotation, for the complex-continued single-harmonic permittivity, at real initial phase $\varphi$ (in cycles) and continuation coordinate $h$.

The paper predicts boundaries between integer phases of the irrational quasicrystal from an auxiliary periodic approximant whose modulation phase advances by $p/q$ per layer. The cell propagator is an ordered product of $q$ one-layer propagators whose permittivities sample the same analytic waveform, continued into the complex phase plane by $h$. The layer ordering, the per-layer phase shift, and the way $h$ enters the waveform are the paper's own conventions; consult its equations for the continued permittivity and the periodic-cell propagator. The trace, not the matrix, is the object used downstream. Evaluations with $|h|$ at or beyond the nonsingular strip half-width must be rejected.

Returns
-------
complex: the trace of the $q$-layer periodic-cell propagator at $(\varphi,h)$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def approximant_cell_trace(phi: float, h: float, p: int, q: int, eps_R: float,
                           d_eps: float, w0tau: float) -> complex:
    r"""Trace of the rational-approximant periodic-cell propagator.

    Args:
        phi (float): real initial modulation phase $\varphi$, in cycles.
        h (float): continuation coordinate $h$ (imaginary phase displacement).
        p (int): approximant numerator $p$, coprime with $q$.
        q (int): approximant denominator $q$ (cell length in layers), $q\ge 1$.
        eps_R (float): background relative permittivity $\varepsilon_R$.
        d_eps (float): modulation-amplitude parameter $\delta\varepsilon$ at $h=0$.
        w0tau (float): reference phase $\omega_0\tau$ accumulated per layer.

    Raises:
        ValueError: if $q<1$, if $p$ and $q$ are not coprime integers, or if $|h|$ is not
            strictly inside the nonsingular strip.

    Expected return:
        complex: invariant under $\varphi\to\varphi+1/q$; real when $h=0$ and the
        permittivities are real; its magnitude grows rapidly with $h$.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: approximant_cell_trace
def _oracle_approximant_cell_trace(phi: float, h: float, p: int, q: int, eps_R: float,
                                   d_eps: float, w0tau: float) -> complex:
    if int(q) != q or int(p) != p or q < 1:
        raise ValueError("p, q must be integers with q >= 1")
    if np.gcd(int(p), int(q)) != 1:
        raise ValueError("p and q must be coprime")
    if abs(h) >= _oracle_analytic_strip_halfwidth(eps_R, d_eps):
        raise ValueError("h outside the nonsingular analytic strip")
    M = np.eye(2, dtype=complex)
    for j in range(int(q)):
        eps = eps_R + d_eps * np.sin(2.0 * np.pi * (phi + j * p / q) + 1j * h)
        M = _oracle_layer_transfer_matrix(eps, eps_R, w0tau) @ M
    return complex(np.trace(M))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'import numpy as np\n'
                'def _pack(z):\n'
                '    z = complex(z)\n'
                '    return np.round(np.array([np.log(abs(z)), np.cos(np.angle(z)), np.sin(np.angle(z))]), 6)\n'
                'v = approximant_cell_trace(0.1, 0.8, 21, 34, 3.4, 1.0, 15.54)'
            ),
            "call": '_pack(v)',
            "gold_call": '_pack(_oracle_approximant_cell_trace(0.1, 0.8, 21, 34, 3.4, 1.0, 15.54))',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def _pack(z):\n'
                '    z = complex(z)\n'
                '    return np.round(np.array([np.log(abs(z)), np.cos(np.angle(z)), np.sin(np.angle(z))]), 6)\n'
                'v = approximant_cell_trace(0.37, 0.65, 21, 34, 3.4, 1.0, 15.82)'
            ),
            "call": '_pack(v)',
            "gold_call": '_pack(_oracle_approximant_cell_trace(0.37, 0.65, 21, 34, 3.4, 1.0, 15.82))',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def _pack(z):\n'
                '    z = complex(z)\n'
                '    return np.round(np.array([np.log(abs(z)), np.cos(np.angle(z)), np.sin(np.angle(z))]), 6)\n'
                'v = approximant_cell_trace(0.2, 1.2, 13, 21, 3.4, 1.0, 9.03)'
            ),
            "call": '_pack(v)',
            "gold_call": '_pack(_oracle_approximant_cell_trace(0.2, 1.2, 13, 21, 3.4, 1.0, 9.03))',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def _pack(z):\n'
                '    z = complex(z)\n'
                '    return np.round(np.array([np.log(abs(z)), np.cos(np.angle(z)), np.sin(np.angle(z))]), 6)\n'
                'v = approximant_cell_trace(0.05, 0.9, 3, 5, 3.4, 1.0, 9.03)'
            ),
            "call": '_pack(v)',
            "gold_call": '_pack(_oracle_approximant_cell_trace(0.05, 0.9, 3, 5, 3.4, 1.0, 9.03))',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def _pack(z):\n'
                '    z = complex(z)\n'
                '    return np.round(np.array([np.log(abs(z)), np.cos(np.angle(z)), np.sin(np.angle(z))]), 6)\n'
                'v = approximant_cell_trace(0.1 + 1.0/34.0, 0.8, 21, 34, 3.4, 1.0, 15.54)'
            ),
            "call": '_pack(v)',
            "gold_call": '_pack(_oracle_approximant_cell_trace(0.1, 0.8, 21, 34, 3.4, 1.0, 15.54))',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def run_model():\n'
                '    try:\n'
                '        approximant_cell_trace(0.1, 0.8, 2, 4, 3.4, 1.0, 9.03)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_approximant_cell_trace(0.1, 0.8, 2, 4, 3.4, 1.0, 9.03)\n'
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
                '        approximant_cell_trace(0.1, 1.95, 21, 34, 3.4, 1.0, 9.03)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_approximant_cell_trace(0.1, 1.95, 21, 34, 3.4, 1.0, 9.03)\n'
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
