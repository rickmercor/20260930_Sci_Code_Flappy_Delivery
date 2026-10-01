"""
Implement model_jet. The model fixes signed electronic couplings and reference energies. Analytic control derivatives start from a Hamiltonian jet whose mixed slot is zero even though the final energy response is not.

The model fixes signed electronic couplings and reference energies. Analytic control derivatives start from a Hamiltonian jet whose mixed slot is zero even though the final energy response is not.

Returns
-------
np.ndarray of shape (4,4,4), the Hamiltonian jet.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def model_jet(a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """Construct the four-configuration Hamiltonian and its control derivatives.

    Parameters
    ----------
    a, b : float
        Finite real scalar controls convertible to float.

        All energies are expressed in units of |t|. The ordered basis is
        (alpha, alpha_prime, beta, beta_prime), indexed by (0, 1, 2, 3).

        The real symmetric Hamiltonian is defined by

            H(a, b) = [
                [0,       3.2 + a, -1,      -1],
                [3.2 + a, 0,       -1.5,    -1.5],
                [-1,      -1.5,     3,       1.5 + b],
                [-1,      -1.5,     1.5 + b, 6]
            ].

        Only the explicitly indicated entries depend on the controls;
        all other entries remain fixed.

    Returns
    -------
    result : np.ndarray
        Floating-point array of shape (4, 4, 4), indexed as
        result[d, i, j]. The FIRST axis selects the derivative slot,
        and the last two axes index Hamiltonian rows and columns:

        result[0, :, :] = H(a, b)
        result[1, :, :] = partial H / partial a
        result[2, :, :] = partial H / partial b
        result[3, :, :] = partial^2 H / (partial a partial b)

        All derivatives are evaluated at the supplied controls.
        Entries are actual derivatives, not Taylor coefficients.

    Raises
    ------
    ValueError
        If either control is not a finite real scalar convertible to
        float, or if constructing the Hamiltonian produces any
        nonfinite entry.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_model_jet(a: float = 0.0, b: float = 0.0) -> np.ndarray:
    """Construct the four-configuration Hamiltonian and its control derivatives.

    Parameters
    ----------
    a, b : float
        Finite real scalar controls. In units of |t|, K_alpha=3.2+a,
        K_beta=1.5+b, t=-1, t_prime=-1.5, U=3, U_prime=6.
        Basis order is (alpha, alpha_prime, beta, beta_prime).

    Returns
    -------
    result : np.ndarray
        Shape (4,4,4), with slots (H, H_a, H_b, H_ab). Slots contain
        actual derivatives, not Taylor coefficients. The diagonal is
        (0,0,3,6); both reference configurations couple to both perturbers
        with t and t_prime respectively.

    Raises
    ------
    ValueError
        If either control is not a finite real scalar convertible to float,
        or construction produces nonfinite entries.
    """
    try:
        x = np.asarray([a, b])
        if x.shape != (2,) or np.iscomplexobj(x):
            raise ValueError('real scalar controls required')
        a, b = np.asarray(x, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid controls') from exc
    if not np.isfinite([a, b]).all():
        raise ValueError('finite controls required')
    result = np.zeros((4, 4, 4))
    result[0] = [[0,3.2+a,-1,-1], [3.2+a,0,-1.5,-1.5],
                 [-1,-1.5,3,1.5+b], [-1,-1.5,1.5+b,6]]
    result[1,0,1] = result[1,1,0] = 1
    result[2,2,3] = result[2,3,2] = 1
    if not np.isfinite(result).all():
        raise ValueError('nonfinite Hamiltonian')
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Three valid cases (normal, boundary, edge), then one invalid case."""
    return [{'setup': 'import numpy as np\na,b=0.,0.',
      'call': 'model_jet(a,b)',
      'gold_call': '_oracle_model_jet(a,b)'},
     {'setup': 'import numpy as np\na,b=0.001,-0.001',
      'call': 'model_jet(a,b)',
      'gold_call': '_oracle_model_jet(a,b)'},
     {'setup': 'import numpy as np\na,b=-3.2,-1.5',
      'call': 'model_jet(a,b)',
      'gold_call': '_oracle_model_jet(a,b)'},
     {'setup': 'import numpy as np\n'
               'a,b=np.nan,0.\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        model_jet(a,b)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_model_jet(a,b)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
