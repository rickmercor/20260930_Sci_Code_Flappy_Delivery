"""
Evaluate conditional path-density increments and their target-potential response on the fixed biased record.

The target potential is U_alpha(q)=U0(q)+alpha*h(q), where h(q)=exp(-q^2/(2*width^2))+htilt*q. Compute the log ratio of the target ABOBA transition density to the simulated biased transition density. The sampled positions, momenta, restraint gradients and biased Gaussian variates are held fixed as alpha changes. The equivalent target noise is determined by reproducing the same recorded transition.

Returns
-------
actions : np.ndarray: Float shape (E,N,2). Last-axis entry 0 is the one-step logarithm of target/biased conditional density; entry 1 is its derivative with respect to alpha at fixed records. Their units are 1 and inverse energy, respectively. All inputs are finite.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_path_actions(records: "np.ndarray", alpha: float, width: float, htilt: float, dt: float, friction: float, mass: float, thermal: float) -> "np.ndarray":
    """Evaluate conditional path-density increments and their target-potential response on the fixed biased record.

    Parameters
    ----------
    records : np.ndarray
        Float shape (E,N,6), ordered as simulate_biased_paths returns.
    alpha : float
        Target perturbation amplitude, in energy units.
    width : float
        Positive Gaussian perturbation width, in length units.
    htilt : float
        Linear coefficient in dimensionless h, in inverse length units.
    dt : float
        Positive simulation step duration.
    friction : float
        Positive simulation friction rate.
    mass : float
        Positive simulation mass.
    thermal : float
        Positive thermal energy k_B*T, unchanged between simulated and target laws.

    Returns
    -------
    actions : np.ndarray
        Float shape (E,N,2). Last-axis entry 0 is the one-step logarithm
        of target/biased conditional density; entry 1 is its derivative
        with respect to alpha at fixed records. Their units are 1 and
        inverse energy, respectively. All inputs are finite."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _perturbation_gradient(q, width=0.4, tilt=0.18):
    return -q/width**2*np.exp(-0.5*(q/width)**2)+tilt

def _oracle_compute_path_actions(records: "np.ndarray", alpha: float, width: float, htilt: float, dt: float, friction: float, mass: float, thermal: float) -> "np.ndarray":
    q=records[:,:,2]; bg=records[:,:,3]; eta=records[:,:,4]
    coeff=0.5*dt*(1+np.exp(-friction*dt))/np.sqrt(thermal*mass*(-np.expm1(-2*friction*dt)))
    delta=coeff*(alpha*_perturbation_gradient(q,width,htilt)-bg)
    delta_prime=coeff*_perturbation_gradient(q,width,htilt)
    ell=-eta*delta-0.5*delta**2
    dell=-(eta+delta)*delta_prime
    return np.stack([ell,dell],axis=-1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent scientific configurations, derived from the cited method."""
    return [{'setup': '# Case: mixed_signed_work\n'
               'import numpy as np\n'
               'records=np.zeros((2,4,6))\n'
               'records[:,:,2]=[[-1.2,-0.4,0.0,0.8],[1.1,0.35,-0.15,-0.7]]\n'
               'records[:,:,3]=[[0.2,-0.3,0.7,-0.5],[0.6,0.0,-0.4,0.1]]\n'
               'records[:,:,4]=[[-0.8,1.2,-0.4,0.5],[1.6,-0.3,0.8,-1.1]]\n'
               'alpha=.35;width=.4;htilt=.18;dt=.025;friction=1.8;mass=1.3;thermal=1.\n',
      'call': 'compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, mass, thermal)',
      'gold_call': '_oracle_compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, '
                   'mass, thermal)',
      'tol': 2e-08},
     {'setup': '# Case: unbiased_base_target\n'
               'import numpy as np\n'
               'records=np.zeros((2,4,6))\n'
               'records[:,:,2]=[[-1.2,-0.4,0.0,0.8],[1.1,0.35,-0.15,-0.7]]\n'
               'records[:,:,3]=[[0.2,-0.3,0.7,-0.5],[0.6,0.0,-0.4,0.1]]\n'
               'records[:,:,4]=[[-0.8,1.2,-0.4,0.5],[1.6,-0.3,0.8,-1.1]]\n'
               'alpha=.35;width=.4;htilt=.18;dt=.025;friction=1.8;mass=1.3;thermal=1.\n'
               'alpha=0.;records[:,:,3]=0.\n',
      'call': 'compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, mass, thermal)',
      'gold_call': '_oracle_compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, '
                   'mass, thermal)',
      'tol': 2e-08},
     {'setup': '# Case: target_matches_simulation_gradient\n'
               'import numpy as np\n'
               'records=np.zeros((2,4,6))\n'
               'records[:,:,2]=[[-1.2,-0.4,0.0,0.8],[1.1,0.35,-0.15,-0.7]]\n'
               'records[:,:,3]=[[0.2,-0.3,0.7,-0.5],[0.6,0.0,-0.4,0.1]]\n'
               'records[:,:,4]=[[-0.8,1.2,-0.4,0.5],[1.6,-0.3,0.8,-1.1]]\n'
               'alpha=.35;width=.4;htilt=.18;dt=.025;friction=1.8;mass=1.3;thermal=1.\n'
               'q=records[:,:,2];records[:,:,3]=alpha*(-q/width**2*np.exp(-.5*(q/width)**2)+htilt)\n',
      'call': 'compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, mass, thermal)',
      'gold_call': '_oracle_compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, '
                   'mass, thermal)',
      'tol': 2e-08},
     {'setup': '# Case: zero_noise_quadratic_action\n'
               'import numpy as np\n'
               'records=np.zeros((2,4,6))\n'
               'records[:,:,2]=[[-1.2,-0.4,0.0,0.8],[1.1,0.35,-0.15,-0.7]]\n'
               'records[:,:,3]=[[0.2,-0.3,0.7,-0.5],[0.6,0.0,-0.4,0.1]]\n'
               'records[:,:,4]=[[-0.8,1.2,-0.4,0.5],[1.6,-0.3,0.8,-1.1]]\n'
               'alpha=.35;width=.4;htilt=.18;dt=.025;friction=1.8;mass=1.3;thermal=1.\n'
               'records[:,:,4]=0.\n',
      'call': 'compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, mass, thermal)',
      'gold_call': '_oracle_compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, '
                   'mass, thermal)',
      'tol': 2e-08},
     {'setup': '# Case: weak_noise_limit_log_weights\n'
               'import numpy as np\n'
               'records=np.zeros((2,4,6))\n'
               'records[:,:,2]=[[-1.2,-0.4,0.0,0.8],[1.1,0.35,-0.15,-0.7]]\n'
               'records[:,:,3]=[[0.2,-0.3,0.7,-0.5],[0.6,0.0,-0.4,0.1]]\n'
               'records[:,:,4]=[[-0.8,1.2,-0.4,0.5],[1.6,-0.3,0.8,-1.1]]\n'
               'alpha=.35;width=.4;htilt=.18;dt=.025;friction=1.8;mass=1.3;thermal=1.\n'
               'friction=1e-10\n',
      'call': 'compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, mass, thermal)',
      'gold_call': '_oracle_compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, '
                   'mass, thermal)',
      'tol': 2e-06},
     {'setup': '# Case: localized_barrier_perturbation\n'
               'import numpy as np\n'
               'records=np.zeros((2,4,6))\n'
               'records[:,:,2]=[[-1.2,-0.4,0.0,0.8],[1.1,0.35,-0.15,-0.7]]\n'
               'records[:,:,3]=[[0.2,-0.3,0.7,-0.5],[0.6,0.0,-0.4,0.1]]\n'
               'records[:,:,4]=[[-0.8,1.2,-0.4,0.5],[1.6,-0.3,0.8,-1.1]]\n'
               'alpha=.35;width=.4;htilt=.18;dt=.025;friction=1.8;mass=1.3;thermal=1.\n'
               'width=.07;htilt=0.;records[:,:,2]/=5\n',
      'call': 'compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, mass, thermal)',
      'gold_call': '_oracle_compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, '
                   'mass, thermal)',
      'tol': 2e-08},
     {'setup': '# Case: broad_perturbation_with_tilt\n'
               'import numpy as np\n'
               'records=np.zeros((2,4,6))\n'
               'records[:,:,2]=[[-1.2,-0.4,0.0,0.8],[1.1,0.35,-0.15,-0.7]]\n'
               'records[:,:,3]=[[0.2,-0.3,0.7,-0.5],[0.6,0.0,-0.4,0.1]]\n'
               'records[:,:,4]=[[-0.8,1.2,-0.4,0.5],[1.6,-0.3,0.8,-1.1]]\n'
               'alpha=.35;width=.4;htilt=.18;dt=.025;friction=1.8;mass=1.3;thermal=1.\n'
               'width=30.;htilt=-.8\n',
      'call': 'compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, mass, thermal)',
      'gold_call': '_oracle_compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, '
                   'mass, thermal)',
      'tol': 2e-08},
     {'setup': '# Case: opposed_target_restraint_forces\n'
               'import numpy as np\n'
               'records=np.zeros((2,4,6))\n'
               'records[:,:,2]=[[-1.2,-0.4,0.0,0.8],[1.1,0.35,-0.15,-0.7]]\n'
               'records[:,:,3]=[[0.2,-0.3,0.7,-0.5],[0.6,0.0,-0.4,0.1]]\n'
               'records[:,:,4]=[[-0.8,1.2,-0.4,0.5],[1.6,-0.3,0.8,-1.1]]\n'
               'alpha=.35;width=.4;htilt=.18;dt=.025;friction=1.8;mass=1.3;thermal=1.\n'
               'alpha=-1.2;records[:,:,3]*=3.\n',
      'call': 'compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, mass, thermal)',
      'gold_call': '_oracle_compute_path_actions(records.copy(), alpha, width, htilt, dt, friction, '
                   'mass, thermal)',
      'tol': 2e-08}]
