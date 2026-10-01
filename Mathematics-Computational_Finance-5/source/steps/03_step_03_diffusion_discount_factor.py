"""
Transform and discount the independent continuous log-value increment. For a window of length tau, the continuous increment is normal with mean (mu-sigma**2/2)*tau and variance sigma**2*tau. Compute its characteristic function at the supplied real frequencies, multiplied by exp(-rho*tau). The drift mu and discount rate rho are separate project parameters.

The continuous log increment is independent of the marked queue, and project discounting uses a separate rate.

Returns
-------
return np.zeros(np.shape(v),dtype=complex)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def diffusion_discount_factor(v,tau,mu,sigma,rho):
    """Return a complex array with the same shape as finite real array v.

    tau and sigma are nonnegative finite scalars. mu and rho are finite.
    Raise ValueError for nonfinite inputs, negative tau or negative sigma.
    """
    return np.zeros(np.shape(v),dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_diffusion_discount_factor(v,tau,mu,sigma,rho):
    np=__import__('numpy');v=np.asarray(v,float)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite([tau,mu,sigma,rho])) or min(tau,sigma)<0:
        raise ValueError('diffusion inputs')
    return np.exp(1j*v*(mu-.5*sigma*sigma)*tau-.5*sigma*sigma*v*v*tau-rho*tau)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit dictionaries for Studio differential-case preflight."""
    return [{'setup': 'import numpy as np\n'
               'args=([0.0, 0.4, 1.7, -2.0], 0.75, 0.04, 0.22, 0.08)\n'
               'def pack(z):\n'
               '    return np.r_[z.real.ravel(),z.imag.ravel()].tolist()',
      'call': 'pack(diffusion_discount_factor(*args))',
      'gold_call': 'pack(_oracle_diffusion_discount_factor(*args))'},
     {'setup': 'import numpy as np\n'
               'args=([[0.0, 1.0], [-1.0, 3.0]], 1.4, -0.03, 0.3, 0.05)\n'
               'def pack(z):\n'
               '    return np.r_[z.real.ravel(),z.imag.ravel()].tolist()',
      'call': 'pack(diffusion_discount_factor(*args))',
      'gold_call': 'pack(_oracle_diffusion_discount_factor(*args))'},
     {'setup': 'import numpy as np\n'
               'args=([0.2, 0.8], 0.0, 0.1, 0.0, -0.01)\n'
               'def pack(z):\n'
               '    return np.r_[z.real.ravel(),z.imag.ravel()].tolist()',
      'call': 'pack(diffusion_discount_factor(*args))',
      'gold_call': 'pack(_oracle_diffusion_discount_factor(*args))'},
     {'setup': 'def check(f):\n'
               '    try:\n'
               '        f([0.,1.],-1.,.04,.22,.08)\n'
               '        return 0\n'
               '    except ValueError:\n'
               '        return 1\n',
      'call': 'check(diffusion_discount_factor)',
      'gold_call': 'check(_oracle_diffusion_discount_factor)'}]
