"""
Orchestrate the complete fixed-reference IAO plus finite-temperature interacting-bath variance comparison.

The fixed occupied reference defines IAOs, while fractional occupations define the thermal bath. Their distinct roles persist through projection, two-field number matching and quantum impurity fluctuations. The single output is the unrounded MEB-minus-EVB variance difference.

Returns
-------
float, the unrounded MEB-minus-EVB impurity charge-variance difference as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def charge_variance_difference(F: "np.ndarray", M: "np.ndarray", U: "np.ndarray", beta: float = 2.3, ne: float = 4.0, nocc: int = 2) -> float:
    """Return the complete task's MEB-minus-EVB impurity variance difference.

    Parameters
    ----------
    F : np.ndarray
        Fixed real finite symmetric (n,n) reference and physical one-body
        matrix, n >= 4; symmetry tolerance 1e-10, symmetrized if accepted.
    M : np.ndarray
        Real finite (n,m) minimal-basis coefficients, 2 <= m <= n.
    U : np.ndarray
        Real finite shape (n,) onsite opposite-spin interactions.
    beta : float
        Real finite scalar, 0 < beta <= 100.
    ne : float
        Real finite scalar, 1e-6 <= ne <= 2*n-1e-6.
    nocc : int
        Fixed reference occupied spatial count, 1 <= nocc < n and
        nocc <= m, excluding bool; the occupied boundary gap must exceed
        1e-10. All three IAO Gram matrices must have eigenvalues > 1e-12.

    Returns
    -------
    difference : float
        Native unrounded float. Construct all IAOs before selecting the
        first two as active impurity; the second is the only MEB seed.
        Use thermal_reference and compare_baths with their exact contracts,
        requiring rank-two baths and simultaneous exact number matching.
        No F update, trace-one density, frozen-environment term, or thermal
        sector truncation.

    Raises
    ------
    ValueError
        On any nonreal/nonfinite/nonconvertible input or stated shape,
        parameter, reference-gap or Gram-spectrum violation; either bath
        not having rank two; a number target outside its open spectral
        interval; or inability to fit both thermal number constraints to
        absolute residual 1e-9. These include the inherited contracts of
        intrinsic_orbitals, thermal_reference, and compare_baths.
    
    """
    return None  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_charge_variance_difference(F: "np.ndarray", M: "np.ndarray", U: "np.ndarray", beta: float = 2.3, ne: float = 4.0, nocc: int = 2) -> float:
    """Deterministic reference implementation."""
    Q=_oracle_intrinsic_orbitals(F,M,nocc)
    F=np.asarray(F,dtype=float)
    if len(F)<4 or Q.shape[1]<2: raise ValueError('at least four primary and two minimal orbitals')
    D,_=_oracle_thermal_reference(F,beta,ne)
    return float(_oracle_compare_baths((F+F.T)/2,U,Q[:,:2],D,beta)[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent normal, boundary, edge and invalid cases."""
    return [{'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n',
  'call': 'charge_variance_difference(F,M,U)',
  'gold_call': '_oracle_charge_variance_difference(F,M,U)'},
 {'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n',
  'call': 'charge_variance_difference(F,M,np.zeros(6))',
  'gold_call': '_oracle_charge_variance_difference(F,M,np.zeros(6))'},
 {'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n',
  'call': 'charge_variance_difference(F,M,.4*U,1.2,3.6,2)',
  'gold_call': '_oracle_charge_variance_difference(F,M,.4*U,1.2,3.6,2)'},
 {'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n'
           '\n'
           'def _exception_code(function, *args, **kwargs):\n'
           '    try:\n'
           '        function(*args, **kwargs)\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n',
  'call': '_exception_code(charge_variance_difference, F,M,U,0.)',
  'gold_call': '_exception_code(_oracle_charge_variance_difference, F,M,U,0.)'},
 {'setup': 'import numpy as np\n'
           'F=np.array([[-1.6,-.45,.12,0,.18,-.08],[-.45,-.8,-.35,.16,0,.11],[.12,-.35,-.15,-.4,.14,0],[0,.16,-.4,.35,-.3,.17],[.18,0,.14,-.3,.9,-.25],[-.08,.11,0,.17,-.25,1.4]])\n'
           'M=np.array([[1,.12,0,0],[.08,1,.1,0],[0,.06,1,.09],[.04,0,.08,1],[.3,-.2,.25,.1],[-.15,.28,.05,.32]])\n'
           'U=np.array([2.4,1.7,2.1,1.3,1.9,1.5])\n'
           '\n'
           'def _exception_code(function, *args, **kwargs):\n'
           '    try:\n'
           '        function(*args, **kwargs)\n'
           '    except ValueError:\n'
           '        return 1\n'
           '    except Exception:\n'
           '        return 2\n'
           '    return 0\n',
  'call': '_exception_code(charge_variance_difference, F,M,U,2.3,0.)',
  'gold_call': '_exception_code(_oracle_charge_variance_difference, F,M,U,2.3,0.)'}]
