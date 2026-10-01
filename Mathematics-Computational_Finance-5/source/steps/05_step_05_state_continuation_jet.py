"""
Propagate value coefficients and clustering derivatives through queue states. kernel_jet[d,q,j,k] contains the d-th ordinary alpha derivative, d=0,1,2, of the marked transition transform from state q to state j at frequency k. payoff_jet[d,j,k] contains the corresponding derivatives of the real cosine payoff coefficient in terminal state j. The complex diffusion vector is independent of alpha and already includes the window's discount factor.

Return the value and first two derivatives of the complex coefficients C[q,k] = diffusion[k] * sum_j kernel[0,q,j,k]*payoff[0,j,k] These are unprimed coefficients: the half weight of frequency zero is applied when evaluating the trigonometric expansion, not in this step.

State-conditioned continuation coefficients combine the transition kernel with the next payoff. Their derivatives include both factors.

Returns
-------
return np.zeros((3,np.shape(kernel_jet)[1],np.size(diffusion)),dtype=complex)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def state_continuation_jet(kernel_jet,payoff_jet,diffusion):
    """Return complex shape (3,Q_initial,N).

    kernel_jet: finite complex shape (3,Q_initial,Q_terminal,N).
    payoff_jet: finite real shape (3,Q_terminal,N).
    diffusion: finite complex shape (N,). All dimensions are positive.
    Entries at leading index two are second derivatives, not Taylor
    coefficients divided by two. Raise ValueError for inconsistent shapes
    or nonfinite values.
    """
    return np.zeros((3,np.shape(kernel_jet)[1],np.size(diffusion)),dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_state_continuation_jet(kernel_jet,payoff_jet,diffusion):
    np=__import__('numpy')
    K=np.asarray(kernel_jet,complex);H=np.asarray(payoff_jet,float);D=np.asarray(diffusion,complex)
    if K.ndim!=4 or K.shape[0]!=3 or min(K.shape)<1 or H.shape!=(3,K.shape[2],K.shape[3]) or D.shape!=(K.shape[3],):
        raise ValueError('shapes')
    if not all(np.all(np.isfinite(x)) for x in [K,H,D]):raise ValueError('nonfinite coefficients')
    def product(k,h):return np.einsum('qjk,jk->qk',k,h)
    return np.stack([product(K[0],H[0]),product(K[1],H[0])+product(K[0],H[1]),
                     product(K[2],H[0])+2*product(K[1],H[1])+product(K[0],H[2])])*D

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit dictionaries for Studio differential-case preflight."""
    return [{'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(9)\n'
               'K=rng.normal(size=(3,2,3,7))+1j*rng.normal(size=(3,2,3,7))\n'
               'H=rng.normal(size=(3,3,7))\n'
               'D=rng.normal(size=7)+1j*rng.normal(size=7)\n'
               'def pack(z):return np.r_[z.real.ravel(),z.imag.ravel()].tolist()\n',
      'call': 'pack(state_continuation_jet(K,H,D))',
      'gold_call': 'pack(_oracle_state_continuation_jet(K,H,D))'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(17)\n'
               'K=rng.normal(size=(3,4,2,5))+1j*rng.normal(size=(3,4,2,5))\n'
               'H=rng.normal(size=(3,2,5))\n'
               'D=rng.normal(size=5)+1j*rng.normal(size=5)\n'
               'def pack(z):return np.r_[z.real.ravel(),z.imag.ravel()].tolist()\n',
      'call': 'pack(state_continuation_jet(K,H,D))',
      'gold_call': 'pack(_oracle_state_continuation_jet(K,H,D))'},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(23)\n'
               'K=rng.normal(size=(3,1,1,3))+1j*rng.normal(size=(3,1,1,3))\n'
               'H=rng.normal(size=(3,1,3))\n'
               'D=rng.normal(size=3)+1j*rng.normal(size=3)\n'
               'def pack(z):return np.r_[z.real.ravel(),z.imag.ravel()].tolist()\n',
      'call': 'pack(state_continuation_jet(K,H,D))',
      'gold_call': 'pack(_oracle_state_continuation_jet(K,H,D))'},
     {'setup': 'import numpy as np\n'
               'def check(f):\n'
               '    try:\n'
               '        f(np.zeros((3,2,3,4)),np.zeros((3,2,4)),np.ones(4))\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'check(state_continuation_jet)',
      'gold_call': 'check(_oracle_state_continuation_jet)'}]
