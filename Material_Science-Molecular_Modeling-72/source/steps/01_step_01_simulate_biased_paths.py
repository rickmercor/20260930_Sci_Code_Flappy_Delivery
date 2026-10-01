"""
Generate the recorded one-dimensional molecular trajectories under moving harmonic restraints.

Use the ABOBA discretization of underdamped Langevin dynamics, with the restraint frozen at the temporal midpoint of each full step for both kicks. The base potential is U0(q)=barrier*(q^2-1)^2+tilt*q. Restraint j is k_j*(q-c_j(t))^2/2, where c_j(t)=c0_j+A_j*sin(omega_j*t+phase_j). All quantities are in reduced units. This is a constructed finite record for the molecular kinetic estimator.

Returns
-------
records : np.ndarray: Float array (E,N,6). Columns: position before the step, position after the step, spatial midpoint position, restraint gradient at the midpoint, supplied biased noise, momentum after the step. Coordinates and momenta have reduced length and momentum units. The restraint gradient is in energy per length units. Inputs are finite and chosen to give stable trajectories; equivalent input values produce identical records.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_biased_paths(initial: "np.ndarray", bias: "np.ndarray", noise: "np.ndarray", dt: float, friction: float, mass: float, thermal: float, barrier: float, tilt: float) -> "np.ndarray":
    """Generate the recorded one-dimensional molecular trajectories under moving harmonic restraints.

    Parameters
    ----------
    initial : np.ndarray
        Shape (E,2); initial position and momentum in that order; E >= 1.
    bias : np.ndarray
        Shape (E,5); columns k, c0, A, omega, phase. k >= 0; angular phase in radians.
    noise : np.ndarray
        Shape (E,N), N >= 1; supplied standard-normal variates, ensemble then time.
    dt : float
        Positive integration time step; the first step spans times 0 to dt.
    friction : float
        Positive momentum friction rate in inverse time units.
    mass : float
        Positive particle mass.
    thermal : float
        Positive thermal energy k_B*T.
    barrier : float
        Nonnegative coefficient in U0, in energy units.
    tilt : float
        Coefficient of q in U0, in energy per length units.

    Returns
    -------
    records : np.ndarray
        Float array (E,N,6). Columns: position before the step, position after
        the step, spatial midpoint position, restraint gradient at the
        midpoint, supplied biased noise, momentum after the step.
        Coordinates and momenta have reduced length and momentum units.
        The restraint gradient is in energy per length units. Inputs are
        finite and chosen to give stable trajectories; equivalent input
        values produce identical records."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_simulate_biased_paths(initial: "np.ndarray", bias: "np.ndarray", noise: "np.ndarray", dt: float, friction: float, mass: float, thermal: float, barrier: float, tilt: float) -> "np.ndarray":
    e,n=noise.shape
    out=np.empty((e,n,6))
    q,p=initial[:,0].copy(),initial[:,1].copy()
    a=np.exp(-friction*dt)
    sigma=np.sqrt(thermal*mass*(-np.expm1(-2*friction*dt)))
    for t in range(n):
        mid=q+dt*p/(2*mass)
        center=bias[:,1]+bias[:,2]*np.sin(bias[:,3]*(t+0.5)*dt+bias[:,4])
        bg=bias[:,0]*(mid-center)
        grad=4*barrier*mid*(mid**2-1)+tilt+bg
        newp=a*(p-0.5*dt*grad)+sigma*noise[:,t]-0.5*dt*grad
        newq=mid+dt*newp/(2*mass)
        out[:,t,:]=np.column_stack([q,newq,mid,bg,noise[:,t],newp])
        q,p=newq,newp
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent scientific configurations, derived from the cited method."""
    return [{'setup': '# Case: unrestrained_free_particle\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'noise=noise[:,:23]\n'
               'barrier=0.;tilt=0.;bias[:,0]=0.\n',
      'call': 'simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, friction, mass, '
              'thermal, barrier, tilt)',
      'gold_call': '_oracle_simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, '
                   'friction, mass, thermal, barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: single_zero_noise_step\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'noise=noise[:,:23]\n'
               'noise=np.zeros((4,1));bias[:,2]=0.\n',
      'call': 'simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, friction, mass, '
              'thermal, barrier, tilt)',
      'gold_call': '_oracle_simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, '
                   'friction, mass, thermal, barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: multiple_moving_restraints\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'noise=noise[:,:23]\n',
      'call': 'simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, friction, mass, '
              'thermal, barrier, tilt)',
      'gold_call': '_oracle_simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, '
                   'friction, mass, thermal, barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: nearly_hamiltonian_friction\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'noise=noise[:,:23]\n'
               'friction=1e-9\n',
      'call': 'simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, friction, mass, '
              'thermal, barrier, tilt)',
      'gold_call': '_oracle_simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, '
                   'friction, mass, thermal, barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: strong_thermostat_memory_loss\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'noise=noise[:,:23]\n'
               'friction=150.\n',
      'call': 'simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, friction, mass, '
              'thermal, barrier, tilt)',
      'gold_call': '_oracle_simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, '
                   'friction, mass, thermal, barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: static_offcenter_umbrella\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'noise=noise[:,:23]\n'
               'bias[:,2]=0.;initial=initial[:1];bias=bias[:1];noise=noise[:1]\n',
      'call': 'simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, friction, mass, '
              'thermal, barrier, tilt)',
      'gold_call': '_oracle_simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, '
                   'friction, mass, thermal, barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: negative_amplitude_phase_offsets\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'noise=noise[:,:23]\n'
               'bias[:,2]*=-1;bias[:,4]+=0.7\n',
      'call': 'simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, friction, mass, '
              'thermal, barrier, tilt)',
      'gold_call': '_oracle_simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, '
                   'friction, mass, thermal, barrier, tilt)',
      'tol': 2e-08},
     {'setup': '# Case: symmetric_potential_reflected_launch\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'noise=noise[:,:23]\n'
               'tilt=0.;initial*=-1;bias[:,1:3]*=-1;noise*=-1\n',
      'call': 'simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, friction, mass, '
              'thermal, barrier, tilt)',
      'gold_call': '_oracle_simulate_biased_paths(initial.copy(), bias.copy(), noise.copy(), dt, '
                   'friction, mass, thermal, barrier, tilt)',
      'tol': 2e-08}]
