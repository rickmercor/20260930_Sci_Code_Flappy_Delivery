"""
Compute target Boltzmann microstate masses and their perturbation response.

The confining target potential on the full real line is U_alpha(q)=barrier*(q^2-1)^2+tilt*q+alpha*[exp(-q^2/(2*width^2))+htilt*q]. Integrating out momentum leaves the normalized coordinate Boltzmann density. Its microstate integrals provide the independently fixed populations in the kinetic likelihood. The integration covers both unbounded tail intervals.

Returns
-------
masses : np.ndarray: Float shape (2,L). Row 0 contains normalized equilibrium masses; row 1 their derivatives with respect to alpha, with units 1 and inverse energy. States are ordered left to right. The derivative includes the response of the common normalization integral. Input scales have representable positive microstate masses.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_equilibrium_masses(cuts: "np.ndarray", alpha: float, width: float, htilt: float, thermal: float, barrier: float, tilt: float) -> "np.ndarray":
    """Compute target Boltzmann microstate masses and their perturbation response.

    Parameters
    ----------
    cuts : np.ndarray
        Finite strictly increasing (L-1,) positions, possibly empty.
    alpha : float
        Target perturbation amplitude in energy units.
    width : float
        Positive Gaussian width in length units.
    htilt : float
        Linear coefficient in dimensionless h, in inverse length units.
    thermal : float
        Positive thermal energy k_B*T.
    barrier : float
        Strictly positive quartic coefficient in energy units.
    tilt : float
        Base linear coefficient in energy per length units.

    Returns
    -------
    masses : np.ndarray
        Float shape (2,L). Row 0 contains normalized equilibrium masses;
        row 1 their derivatives with respect to alpha, with units 1 and
        inverse energy. States are ordered left to right. The derivative
        includes the response of the common normalization integral.
        Input scales have representable positive microstate masses."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad

def _perturbation_value(q, width=0.4, tilt=0.18):
    return np.exp(-0.5*(q/width)**2)+tilt*q

def _oracle_compute_equilibrium_masses(cuts: "np.ndarray", alpha: float, width: float, htilt: float, thermal: float, barrier: float, tilt: float) -> "np.ndarray":
    edges=np.r_[-np.inf,cuts,np.inf]
    def _weight(q):
        return np.exp(-(barrier*(q*q-1)**2+tilt*q+alpha*_perturbation_value(q,width,htilt))/thermal)
    z=np.array([quad(_weight,a,b,epsabs=2e-12,epsrel=2e-12)[0] for a,b in zip(edges[:-1],edges[1:])])
    dz=np.array([quad(lambda q: -_perturbation_value(q,width,htilt)*_weight(q)/thermal,a,b,epsabs=2e-12,epsrel=2e-12)[0] for a,b in zip(edges[:-1],edges[1:])])
    pi=z/z.sum();dpi=dz/z.sum()-pi*dz.sum()/z.sum()
    return np.stack([pi,dpi])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent scientific configurations, derived from the cited method."""
    return [{'setup': '# Case: asymmetric_five_state_landscape\n'
               'import numpy as np\n'
               'cuts=np.array([-1.05,-.45,.15,.8]);alpha=.35;width=.4;htilt=.18;thermal=1.;barrier=2.4;tilt=.25\n',
      'call': 'compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, barrier, tilt)',
      'gold_call': '_oracle_compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, '
                   'barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: symmetric_two_well_masses\n'
               'import numpy as np\n'
               'cuts=np.array([-1.05,-.45,.15,.8]);alpha=.35;width=.4;htilt=.18;thermal=1.;barrier=2.4;tilt=.25\n'
               'cuts=np.array([-1.,-.3,.3,1.]);tilt=0.;htilt=0.\n',
      'call': 'compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, barrier, tilt)',
      'gold_call': '_oracle_compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, '
                   'barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: base_potential_nonzero_response\n'
               'import numpy as np\n'
               'cuts=np.array([-1.05,-.45,.15,.8]);alpha=.35;width=.4;htilt=.18;thermal=1.;barrier=2.4;tilt=.25\n'
               'alpha=0.\n',
      'call': 'compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, barrier, tilt)',
      'gold_call': '_oracle_compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, '
                   'barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: low_temperature_rare_barrier_states\n'
               'import numpy as np\n'
               'cuts=np.array([-1.05,-.45,.15,.8]);alpha=.35;width=.4;htilt=.18;thermal=1.;barrier=2.4;tilt=.25\n'
               'thermal=.24;cuts=np.array([-.8,-.2,.2,.8])\n',
      'call': 'compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, barrier, tilt)',
      'gold_call': '_oracle_compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, '
                   'barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: narrow_localized_barrier\n'
               'import numpy as np\n'
               'cuts=np.array([-1.05,-.45,.15,.8]);alpha=.35;width=.4;htilt=.18;thermal=1.;barrier=2.4;tilt=.25\n'
               'width=.09;htilt=0.;alpha=1.2\n',
      'call': 'compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, barrier, tilt)',
      'gold_call': '_oracle_compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, '
                   'barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: linear_response_broad_gaussian\n'
               'import numpy as np\n'
               'cuts=np.array([-1.05,-.45,.15,.8]);alpha=.35;width=.4;htilt=.18;thermal=1.;barrier=2.4;tilt=.25\n'
               'width=50.;htilt=-.7\n',
      'call': 'compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, barrier, tilt)',
      'gold_call': '_oracle_compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, '
                   'barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: unbounded_tail_populations\n'
               'import numpy as np\n'
               'cuts=np.array([-1.05,-.45,.15,.8]);alpha=.35;width=.4;htilt=.18;thermal=1.;barrier=2.4;tilt=.25\n'
               'cuts=np.array([-1.8,-1.1,-.55,.1,.6,1.6]);thermal=1.7\n',
      'call': 'compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, barrier, tilt)',
      'gold_call': '_oracle_compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, '
                   'barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: single_state_normalization_response\n'
               'import numpy as np\n'
               'cuts=np.array([-1.05,-.45,.15,.8]);alpha=.35;width=.4;htilt=.18;thermal=1.;barrier=2.4;tilt=.25\n'
               'cuts=np.array([])\n',
      'call': 'compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, barrier, tilt)',
      'gold_call': '_oracle_compute_equilibrium_masses(cuts.copy(), alpha, width, htilt, thermal, '
                   'barrier, tilt)',
      'tol': 2e-08}]
