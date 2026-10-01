"""
Encode a recorded or erased fidelity distribution.

The source method treats the input and observed Bell outcome jointly. For the declared resource family both assessment fidelities are axial in the input Bloch coordinate z. Mode 0 records outcomes; mode 1 erases them after correction.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def conditional_fidelity_law(rho: np.ndarray, mode: int = 0) -> np.ndarray:
    """Encode a recorded or erased fidelity distribution.

    rho : ndarray, shape (4,4)
        A density matrix in the damped Phi+ family, in basis 00,01,10,11.
    mode : int
        0 for recorded outcomes R; 1 for the erased channel E.
    Returns
    -------
    ndarray, shape (4,5), float
        Each row is [q0,q1,q2,p0,p1]. In mode 0 the rows are Bell outcomes
        Phi+,Phi-,Psi+,Psi-. Mode 1 encodes four identical artificial rows,
        each with p0=1/4 and p1=0, whose fidelity is the erased channel
        fidelity. These rows represent one normalized distribution.
        The outcome probability at Bloch
        coordinate z is p0+p1*z, and its conditional fidelity is
        (q0+q1*z+q2*z*z)/(p0+p1*z). The joint measure is
        (p0+p1*z)*dz/2. Zero-probability endpoints have zero measure.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def _oracle_conditional_fidelity_law(rho: np.ndarray, mode: int=0) -> np.ndarray:
    if mode not in (0, 1):
        raise ValueError('Mode must be 0 (recorded) or 1 (erased).')
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4) or not np.isfinite(rho).all():
        raise ValueError('Expected a finite 4 by 4 density matrix.')
    if not np.allclose(rho, rho.conj().T, atol=1e-11) or abs(np.trace(rho) - 1) > 1e-10 or np.linalg.eigvalsh(rho)[0] < -1e-10:
        raise ValueError('Expected a physical density matrix.')
    a = float((rho[0, 0] + rho[1, 1] - rho[2, 2] - rho[3, 3]).real)
    b = float((rho[0, 0] - rho[1, 1] + rho[2, 2] - rho[3, 3]).real)
    c = float(2 * rho[0, 3].real)
    w = float((rho[0, 0] - rho[1, 1] - rho[2, 2] + rho[3, 3]).real)
    if not np.allclose(rho, _oracle_damped_resource(a, b), atol=1e-10, rtol=0):
        raise ValueError('The state must be in the declared amplitude-damped Bell family.')
    if mode == 1:
        return np.tile([(1 + c) / 8, 0.0, (w - c) / 8, 0.25, 0.0], (4, 1))
    return np.array([[(1 + c) / 8, s * (a + b) / 8, (w - c) / 8, 0.25, s * a / 4] for s in (1, 1, -1, -1)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'rho=np.array([[0.5, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.0, '
               '0.0, 0.5]], dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, '
               '0.0, 0.0]], dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '0.0, 0.0]], dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '0.0, 0.0]], dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5851, 0.0, 0.0, 0.22371857321197094], [0.0, 0.029900000000000003, 0.0, '
               '0.0], [0.0, 0.0, 0.2849, 0.0], [0.22371857321197094, 0.0, 0.0, 0.10010000000000001]], '
               'dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5851, 0.0, 0.0, 0.22371857321197094], [0.0, 0.2849, 0.0, 0.0], [0.0, 0.0, '
               '0.029900000000000003, 0.0], [0.22371857321197094, 0.0, 0.0, 0.10010000000000001]], '
               'dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.625, 0.0, 0.0, 0.25], [0.0, 0.125, 0.0, 0.0], [0.0, 0.0, 0.125, 0.0], '
               '[0.25, 0.0, 0.0, 0.125]], dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.500000005, 0.0, 0.0, 4.999999987561898e-05], [0.0, 5.000000025123797e-17, '
               '0.0, 0.0], [0.0, 0.0, 0.49999999, 0.0], [4.999999987561898e-05, 0.0, 0.0, '
               '4.9999999751237955e-09]], dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5004995, 0.0, 0.0, 0.015803480629279117], [0.0, 0.4990005, 0.0, 0.0], '
               '[0.0, 0.0, 5.000000000000005e-07, 0.0], [0.015803480629279117, 0.0, 0.0, '
               '0.0004995000000000005]], dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5004995, 0.0, 0.0, 0.015803480629279117], [0.0, 5.000000000000005e-07, '
               '0.0, 0.0], [0.0, 0.0, 0.4990005, 0.0], [0.015803480629279117, 0.0, 0.0, '
               '0.0004995000000000005]], dtype=float)',
      'call': 'conditional_fidelity_law(rho)',
      'gold_call': '_oracle_conditional_fidelity_law(rho)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.0, '
               '0.0, 0.5]], dtype=float)',
      'call': 'conditional_fidelity_law(rho, 1)',
      'gold_call': '_oracle_conditional_fidelity_law(rho, 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '0.0, 0.0]], dtype=float)',
      'call': 'conditional_fidelity_law(rho, 1)',
      'gold_call': '_oracle_conditional_fidelity_law(rho, 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, '
               '0.0, 0.0]], dtype=float)',
      'call': 'conditional_fidelity_law(rho, 1)',
      'gold_call': '_oracle_conditional_fidelity_law(rho, 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5800000000000001, 0.0, 0.0, 0.19999999999999998], [0.0, '
               '0.019999999999999997, 0.0, 0.0], [0.0, 0.0, 0.32000000000000006, 0.0], '
               '[0.19999999999999998, 0.0, 0.0, 0.07999999999999999]], dtype=float)',
      'call': 'conditional_fidelity_law(rho, 1)',
      'gold_call': '_oracle_conditional_fidelity_law(rho, 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.5800000000000001, 0.0, 0.0, 0.19999999999999998], [0.0, '
               '0.32000000000000006, 0.0, 0.0], [0.0, 0.0, 0.019999999999999997, 0.0], '
               '[0.19999999999999998, 0.0, 0.0, 0.07999999999999999]], dtype=float)',
      'call': 'conditional_fidelity_law(rho, 1)',
      'gold_call': '_oracle_conditional_fidelity_law(rho, 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'rho=np.array([[0.52, 0.0, 0.0, 0.4], [0.0, 0.08000000000000002, 0.0, 0.0], [0.0, 0.0, '
               '0.08000000000000002, 0.0], [0.4, 0.0, 0.0, 0.32000000000000006]], dtype=float)',
      'call': 'conditional_fidelity_law(rho, 1)',
      'gold_call': '_oracle_conditional_fidelity_law(rho, 1)',
      'tol': 2e-07},
     {'setup': 'import numpy as np\n'
               'def _exception_code(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0',
      'call': '_exception_code(lambda: conditional_fidelity_law(np.eye(4)))',
      'gold_call': '_exception_code(lambda: _oracle_conditional_fidelity_law(np.eye(4)))',
      'tol': 0.0}]
