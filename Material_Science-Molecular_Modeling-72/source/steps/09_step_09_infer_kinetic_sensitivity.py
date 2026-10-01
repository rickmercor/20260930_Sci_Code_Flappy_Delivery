"""
Infer the perturbation response of the slow molecular relaxation time from the complete recorded-trajectory reconstruction.

Final orchestrator. Compose simulate_biased_paths, compute_path_actions, accumulate_path_counts, compute_equilibrium_masses, fit_stationary_flux, differentiate_stationary_flux, assemble_transition_response, and compute_relaxation_response. The molecular trajectories are generated under U0 plus their prescribed restraints and held fixed when differentiating the target potential. The inferred operator is the separated-density, fixed-equilibrium reversible maximum-likelihood estimator.

Returns
-------
sensitivity : float: Native finite Python float d(log(t_1))/dalpha at the supplied alpha, in inverse energy units, for lag duration lag*dt. The derivative holds every sampled trajectory fixed and varies the target law.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def infer_kinetic_sensitivity(initial: "np.ndarray", bias: "np.ndarray", noise: "np.ndarray", cuts: "np.ndarray", alpha: float, dt: float, friction: float, mass: float, thermal: float, barrier: float, tilt: float, width: float, htilt: float, lag: int) -> float:
    """Infer the perturbation response of the slow molecular relaxation time from the complete recorded-trajectory reconstruction.

    Parameters
    ----------
    initial : np.ndarray
        (E,2) starting positions and momenta; see simulate_biased_paths.
    bias : np.ndarray
        (E,5) restraint rows k, c0, A, omega, phase; see simulate_biased_paths.
    noise : np.ndarray
        (E,N) supplied biased Gaussian variates, fixed during differentiation.
    cuts : np.ndarray
        (L-1,) strictly increasing position boundaries, L >= 2.
    alpha : float
        Target perturbation amplitude in energy units.
    dt : float
        Positive simulation step duration.
    friction : float
        Positive momentum friction rate.
    mass : float
        Positive particle mass.
    thermal : float
        Positive thermal energy k_B*T.
    barrier : float
        Positive quartic coefficient of U0=barrier*(q^2-1)^2+tilt*q.
    tilt : float
        Base linear-potential coefficient.
    width : float
        Positive width in h(q)=exp(-q^2/(2*width^2))+htilt*q.
    htilt : float
        Linear coefficient in h, in inverse length units.
    lag : int
        Contiguous window length in integration steps, 1 <= lag <= N.
        Valid configurations give finite weights, positive diagonal counts,
        connected observed support, positive representable equilibrium
        masses, and a simple largest nonstationary eigenvalue in (0,1).

    Returns
    -------
    sensitivity : float
        Native finite Python float d(log(t_1))/dalpha at the supplied alpha,
        in inverse energy units, for lag duration lag*dt. The derivative
        holds every sampled trajectory fixed and varies the target law."""
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_infer_kinetic_sensitivity(initial: "np.ndarray", bias: "np.ndarray", noise: "np.ndarray", cuts: "np.ndarray", alpha: float, dt: float, friction: float, mass: float, thermal: float, barrier: float, tilt: float, width: float, htilt: float, lag: int) -> float:
    records = _oracle_simulate_biased_paths(initial, bias, noise, dt, friction, mass, thermal, barrier, tilt)
    action = _oracle_compute_path_actions(records, alpha, width, htilt, dt, friction, mass, thermal)
    C, dC = _oracle_accumulate_path_counts(records, action, cuts, lag)
    pi, dpi = _oracle_compute_equilibrium_masses(cuts, alpha, width, htilt, thermal, barrier, tilt)
    X = _oracle_fit_stationary_flux(C, pi)
    dX = _oracle_differentiate_stationary_flux(C, dC, pi, dpi, X)
    op = _oracle_assemble_transition_response(X, dX, pi, dpi)
    result = _oracle_compute_relaxation_response(op, pi, lag * dt)
    return float(result[2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Independent scientific configurations, derived from the cited method."""
    return [{'setup': '# Case: base_target_without_restraints\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'alpha=0.;bias[:,0]=0.\n',
      'call': 'infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), cuts.copy(), '
              'alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'gold_call': '_oracle_infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), '
                   'cuts.copy(), alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'tol': 2e-06},
     {'setup': '# Case: stationary_umbrella_ensembles\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'bias[:,2]=0.;lag=11\n',
      'call': 'infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), cuts.copy(), '
              'alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'gold_call': '_oracle_infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), '
                   'cuts.copy(), alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'tol': 2e-06},
     {'setup': '# Case: symmetric_energy_and_perturbation\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'tilt=0.;htilt=0.;cuts=np.array([-1.,-.3,.3,1.]);alpha=.6\n',
      'call': 'infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), cuts.copy(), '
              'alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'gold_call': '_oracle_infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), '
                   'cuts.copy(), alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'tol': 2e-06},
     {'setup': '# Case: two_state_projection\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'cuts=np.array([0.]);lag=17\n',
      'call': 'infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), cuts.copy(), '
              'alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'gold_call': '_oracle_infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), '
                   'cuts.copy(), alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'tol': 2e-06},
     {'setup': '# Case: fine_spatial_projection\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'cuts=np.array([-1.25,-.85,-.4,.05,.45,.95]);lag=21\n',
      'call': 'infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), cuts.copy(), '
              'alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'gold_call': '_oracle_infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), '
                   'cuts.copy(), alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'tol': 2e-06},
     {'setup': '# Case: underdamped_crossings\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'friction=.45;lag=9\n',
      'call': 'infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), cuts.copy(), '
              'alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'gold_call': '_oracle_infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), '
                   'cuts.copy(), alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'tol': 2e-06},
     {'setup': '# Case: single_driven_trajectory\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'initial=initial[:1];bias=bias[:1];noise=np.random.default_rng(62).standard_normal((1,12000));lag=15\n',
      'call': 'infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), cuts.copy(), '
              'alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'gold_call': '_oracle_infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), '
                   'cuts.copy(), alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'tol': 2e-06},
     {'setup': '# Case: reversed_driving_and_target_tilt\n'
               'import numpy as np\n'
               'initial = np.array([[-1.15,0.4],[-0.35,-0.6],[0.6,0.2],[1.25,-0.3]])\n'
               'bias = '
               'np.array([[0.7,-0.65,0.5,0.17,0.2],[1.1,0.55,0.65,0.11,1.1],[0.45,-0.1,0.8,0.23,2.2],[0.85,0.35,0.45,0.19,-0.7]])\n'
               'cuts = np.array([-1.05,-0.45,0.15,0.8])\n'
               'alpha=0.35; dt=0.025; friction=1.8; mass=1.3; thermal=1.0\n'
               'barrier=2.4; tilt=0.25; width=0.4; htilt=0.18; lag=13\n'
               'noise=np.random.default_rng(1847).standard_normal((4,3600))\n'
               'bias[:,2]*=-1.;htilt=-.3;alpha=-.25;lag=19\n',
      'call': 'infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), cuts.copy(), '
              'alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'gold_call': '_oracle_infer_kinetic_sensitivity(initial.copy(), bias.copy(), noise.copy(), '
                   'cuts.copy(), alpha, dt, friction, mass, thermal, barrier, tilt, width, htilt, lag)',
      'tol': 2e-06}]
