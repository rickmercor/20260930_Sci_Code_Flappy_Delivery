"""
Compute the $2\times 2$ propagator of one temporal layer of constant (possibly complex) relative permittivity $\varepsilon_m$ and dimensionless duration $\omega_0\tau$, acting on the paper's main-text two-component propagation state built from the displacement field and its scaled time derivative.

Within one layer the displacement-field amplitude of a fixed wavevector obeys a constant-coefficient Hill equation; the displacement field and the magnetic induction are continuous across temporal interfaces, so no separate interface matrix is needed. The propagator must be written in the paper's main-text state basis (not the $(D,\,dD/ds)$ basis), must have unit determinant, and must not depend on the square-root branch chosen for the local frequency ratio. Consult the paper's own layer-propagator equations for the exact form and basis.

Returns
-------
np.ndarray of shape (2, 2), complex: the one-layer propagator $S_m$ in the paper's main-text state basis.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def layer_transfer_matrix(eps_m: complex, eps_R: float, w0tau: float) -> "np.ndarray":
    r"""One-layer propagator of the temporal layer sequence.

    Args:
        eps_m (complex): relative permittivity $\varepsilon_m$ of the layer (nonzero, may be complex).
        eps_R (float): real positive reference relative permittivity $\varepsilon_R$ that defines $\omega_0$.
        w0tau (float): phase $\omega_0\tau$ accumulated over one layer in the unmodulated reference
            medium, $\omega_0\tau>0$.

    Raises:
        ValueError: if $\varepsilon_m=0$, $\varepsilon_R\le 0$ or $\omega_0\tau\le 0$.

    Expected return:
        np.ndarray of shape (2, 2), complex, determinant $1$. For real $\varepsilon_m$ equal to
        $\varepsilon_R$ it is a pure phase rotation of the state; its two diagonal entries are equal.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


# Oracle implementation for public function: layer_transfer_matrix
def _oracle_layer_transfer_matrix(eps_m: complex, eps_R: float, w0tau: float) -> "np.ndarray":
    eps_m = complex(eps_m)
    if eps_m == 0 or eps_R <= 0.0 or w0tau <= 0.0:
        raise ValueError("need eps_m != 0, eps_R > 0, w0tau > 0")
    z = np.sqrt(eps_R / eps_m)
    c = np.cos(w0tau * z)
    s = np.sin(w0tau * z)
    return np.array([[c, -1j * s / z], [-1j * z * s, c]], dtype=complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": (
                'import numpy as np\n'
                'def _pack(M):\n'
                '    M = np.asarray(M)\n'
                '    return np.round(np.concatenate([M.real.ravel(), M.imag.ravel()]), 8)\n'
                'v = layer_transfer_matrix(3.4 + 0.5j, 3.4, 15.54)'
            ),
            "call": '_pack(v)',
            "gold_call": '_pack(_oracle_layer_transfer_matrix(3.4 + 0.5j, 3.4, 15.54))',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def _pack(M):\n'
                '    M = np.asarray(M)\n'
                '    return np.round(np.concatenate([M.real.ravel(), M.imag.ravel()]), 8)\n'
                'v = layer_transfer_matrix(2.4 - 1.2j, 3.4, 9.03)'
            ),
            "call": '_pack(v)',
            "gold_call": '_pack(_oracle_layer_transfer_matrix(2.4 - 1.2j, 3.4, 9.03))',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def _pack(M):\n'
                '    M = np.asarray(M)\n'
                '    return np.round(np.concatenate([M.real.ravel(), M.imag.ravel()]), 8)\n'
                'v = layer_transfer_matrix(5.0, 3.4, 15.82)'
            ),
            "call": '_pack(v)',
            "gold_call": '_pack(_oracle_layer_transfer_matrix(5.0, 3.4, 15.82))',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def _pack(M):\n'
                '    M = np.asarray(M)\n'
                '    return np.round(np.concatenate([M.real.ravel(), M.imag.ravel()]), 8)\n'
                'v = layer_transfer_matrix(1.5 + 2.0j, 3.4, 0.7)'
            ),
            "call": '_pack(v)',
            "gold_call": '_pack(_oracle_layer_transfer_matrix(1.5 + 2.0j, 3.4, 0.7))',
            "tol": 1e-06,
        },
        {
            "setup": (
                'import numpy as np\n'
                'def run_model():\n'
                '    try:\n'
                '        layer_transfer_matrix(0.0, 3.4, 9.03)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_layer_transfer_matrix(0.0, 3.4, 9.03)\n'
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
                '        layer_transfer_matrix(3.0, 3.4, -1.0)\n'
                '        return 0\n'
                '    except ValueError:\n'
                '        return 1\n'
                '    except Exception:\n'
                '        return 2\n'
                'def run_gold():\n'
                '    try:\n'
                '        _oracle_layer_transfer_matrix(3.0, 3.4, -1.0)\n'
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
