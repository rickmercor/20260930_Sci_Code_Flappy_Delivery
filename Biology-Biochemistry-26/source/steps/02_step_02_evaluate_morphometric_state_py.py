"""
Evaluate overlap-penalized morphometric solvation energies for aligned states.

Evaluate the overlap-penalized nonpolar solvation contribution of the matching treatment. descriptors has nonempty shape (N,5), with columns [V,A,C,X,L] in the units declared by the main problem. eta, radius, beta and phi are finite scalars with 0<eta<1, radius>0, beta>0 and phi>=0. V, A and L are nonnegative; C and X may have either sign. Recover the solvent coefficients from the matching treatment, using the given parameters. Return a float vector of length N+4: the coefficients in [p,sigma,kappa,kappa_bar] order followed by one energy for each state. Nonfinite values, incorrect shapes or violations of the stated domain raise ValueError. Do not mutate inputs.

Returns
-------
morphometric_state : np.ndarray — float vector of length N+4 containing [p, sigma, kappa, kappa_bar] followed by N state energies.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
 
def evaluate_morphometric_state(
    descriptors: np.ndarray,
    eta: float,
    radius: float,
    beta: float,
    phi: float,
) -> np.ndarray:
    """Return solvent coefficients followed by state-wise geometric energies.
 
    Parameters
    ----------
    descriptors
        Finite array of shape (N, 5) with columns [V, A, C, X, L].
    eta, radius, beta, phi
        Solvent and overlap parameters in the stated physical domain.
 
    Returns
    -------
    np.ndarray
        Float vector of length N+4.
    """
    return morphometric_state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_evaluate_morphometric_state(
    descriptors: np.ndarray,
    eta: float,
    radius: float,
    beta: float,
    phi: float,
) -> np.ndarray:
    import numpy as np
    d=np.asarray(descriptors,dtype=float)
    pars=np.asarray([eta,radius,beta,phi],dtype=float)
    if d.ndim!=2 or d.shape[1]!=5 or not len(d) or not np.all(np.isfinite(d)) or not np.all(np.isfinite(pars)):
        raise ValueError('finite nonempty state by five descriptors and scalar parameters required')
    if not 0<eta<1 or radius<=0 or beta<=0 or phi<0 or np.any(d[:,[0,1,4]]<0):
        raise ValueError('invalid physical parameter domain')
    h=1-eta; g=3*eta/(4*np.pi*beta); log=np.log1p(-eta)
    coeff=np.array([
        g/radius**3*(1+eta+eta**2-eta**3)/h**3,
        g/radius**2*(-(1+2*eta+8*eta**2-5*eta**3)/(3*h**3)-log/(3*eta)),
        g/radius*((4-10*eta+20*eta**2-8*eta**3)/(3*h**3)+4*log/(3*eta)),
        g*((-4+11*eta-13*eta**2+4*eta**3)/(3*h**3)-4*log/(3*eta))])
    return np.concatenate([coeff,d[:,:4]@coeff+phi*d[:,4]])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nd=np.array([[800.,900.,30.,12.3,1.1],[920.,1010.,-12.,-25.,0.]])\n',
      'call': 'evaluate_morphometric_state(d,.22,1.3,.7,2.)',
      'gold_call': '_oracle_evaluate_morphometric_state(d,.22,1.3,.7,2.)'},
     {'setup': 'import numpy as np\nd=np.zeros((1,5))\n',
      'call': 'evaluate_morphometric_state(d,.3665,1.4,1.,0.)',
      'gold_call': '_oracle_evaluate_morphometric_state(d,.3665,1.4,1.,0.)'},
     {'setup': 'import numpy as np\nd=np.array([[2.,3.,-.5,-4.,0.],[4.,8.,2.,1.,.2]])\n',
      'call': 'evaluate_morphometric_state(d,.001,.8,2.1,.4)',
      'gold_call': '_oracle_evaluate_morphometric_state(d,.001,.8,2.1,.4)'},
     {'setup': 'import numpy as np\n'
               'd=np.zeros((2,5))\n'
               '\n'
               'def run_model():\n'
               '    try:\n'
               '        evaluate_morphometric_state(d,1.,1.,1.,0.)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               'def run_gold():\n'
               '    try:\n'
               '        _oracle_evaluate_morphometric_state(d,1.,1.,1.,0.)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n',
      'call': 'run_model()',
      'gold_call': 'run_gold()'}]
