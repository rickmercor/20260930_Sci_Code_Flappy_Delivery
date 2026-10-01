"""
Construct the normalized thermal density matrix of a finite many-body Hamiltonian at inverse temperature beta. The returned matrix is real symmetric with unit trace, and every subsequent expectation value in this calculation is a trace against it. Invalid input raises ValueError: H must be a two-dimensional square symmetric array and beta must be positive and finite.

The thermal state of a finite system follows from its complete spectrum rather than from its ground state alone, so exact diagonalization must supply every eigenpair and the density matrix is the sum over eigenstates weighted by Boltzmann factors and normalized by the partition function. Measuring energies relative to the ground state before exponentiating is what keeps the calculation finite at large beta, since the raw exponentials would otherwise overflow; the shift cancels exactly between the weights and the partition function, so the state itself is unchanged.

Returns
-------
numpy.ndarray, the real symmetric grand canonical thermal density matrix with trace one
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def thermal_density_matrix(H, beta):
    """Construct the normalized grand canonical thermal density matrix.

    Parameters
    ----------
    H : numpy.ndarray
        Real symmetric many-body Hamiltonian with shape ``(dim, dim)``.
    beta : float
        Positive finite inverse temperature.

    Returns
    -------
    rho : numpy.ndarray
        Real symmetric thermal density matrix with shape ``(dim, dim)`` and
        unit trace.
    
    Raises
    ------
    ValueError
        If ``H`` is not a two-dimensional square symmetric array, or if
        ``beta`` is not a positive finite scalar.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_thermal_density_matrix(H, beta):
    if not isinstance(H, np.ndarray) or H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a two-dimensional square numpy array")
    if not np.allclose(H, H.T):
        raise ValueError("H must be symmetric")
    try:
        valid_beta = np.asarray(beta).ndim == 0 and bool(np.isfinite(beta)) and beta > 0
    except (TypeError, ValueError):
        valid_beta = False
    if not valid_beta:
        raise ValueError("beta must be positive and finite")
    ev, evec = np.linalg.eigh(H)
    w = np.exp(-beta * (ev - ev[0]))
    w = w / w.sum()
    return (evec * w) @ evec.T

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nH = np.diag([0.0, 1.0, 2.0, 3.0])\n',
      'call': 'round(float(np.trace(thermal_density_matrix(H, 8.0))), 10)',
      'gold_call': 'round(float(np.trace(_oracle_thermal_density_matrix(H, 8.0))), 10)'},
     {'setup': 'import numpy as np\n'
               'H = np.array([[0.0, 0.2, 0.0], [0.2, 1.0, -0.1], [0.0, -0.1, 2.0]], dtype=float)\n',
      'call': 'int(np.allclose(thermal_density_matrix(H, 2.5), thermal_density_matrix(H, 2.5).T))',
      'gold_call': 'int(np.allclose(_oracle_thermal_density_matrix(H, 2.5), _oracle_thermal_density_matrix(H, '
                   '2.5).T))'},
     {'setup': 'import numpy as np\nH = np.diag([0.0, 1.0, 2.0, 3.0])\n',
      'call': 'round(float(thermal_density_matrix(H, 1.0e-8)[0, 0]), 8)',
      'gold_call': 'round(float(_oracle_thermal_density_matrix(H, 1.0e-8)[0, 0]), 8)'},
     {'setup': 'import numpy as np\nH = np.diag([0.0, 1.0, 2.0, 3.0])\n',
      'call': 'round(float(np.linalg.eigvalsh(thermal_density_matrix(H, 100.0))[-1]), 8)',
      'gold_call': 'round(float(np.linalg.eigvalsh(_oracle_thermal_density_matrix(H, 100.0))[-1]), 8)'},
     {'setup': 'import numpy as np\n'
               '\n'
               '\n'
               'import numpy as np\n'
               'def _exception_code(function, *args, **kwargs):\n'
               '    try:\n'
               '        function(*args, **kwargs)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': '_exception_code(thermal_density_matrix, np.eye(4), -1.0)',
      'gold_call': '_exception_code(_oracle_thermal_density_matrix, np.eye(4), -1.0)'}]
