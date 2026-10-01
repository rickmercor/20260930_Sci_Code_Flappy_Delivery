"""
Update nodal mechanical energy and predict the magnetic norm implied by the energy balance.

At each node, compute the old and new kinetic energies



K_old = ||momentum_old||_2^2 / (2*density),

K_new = ||momentum_new||_2^2 / (2*density).



Apply the nodal source-energy identity



E_new = total_energy_old + K_new - K_old + time_step*joule_power.



With lumped quadrature, define



Delta_E = sum_i lumped_mass_i * (E_new,i - total_energy_old,i).



Compute the magnetic-norm prediction implied by the global energy balance:



magnetic_norm_new_sq = magnetic_norm_old_sq - (2/permeability)*Delta_E.



The updated internal energy is



internal_new = E_new - K_new,



and the total Joule increment is



Joule_heat = sum_i lumped_mass_i * time_step * joule_power_i.



Return the n nodal E_new values followed by [magnetic_norm_new_sq, min(internal_new), Joule_heat].

This scalar is a balance prediction for independent comparison with the magnetic norm produced by the coupled field solve. It does not supply a new magnetic field. The final orchestrator obtains that field from the midpoint equations and checks the prediction against its directly evaluated norm.

Returns
-------
Return one length-(n+3) real NumPy array in documented order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def source_energy_update(lumped_mass, density, momentum_old, momentum_new,
                         total_energy_old, joule_power, time_step,
                         magnetic_norm_old_sq, permeability=1.0):
    """Advance nodal source energy and form an independent magnetic-balance prediction.

    Parameters
    ----------
    lumped_mass : array_like, shape (n,)
        Positive nodal masses.
    density : array_like, shape (n,)
        Positive nodal densities.
    momentum_old : array_like, shape (n,3)
        Cartesian momentum before the source solve.
    momentum_new : array_like, shape (n,3)
        Cartesian momentum after the source solve.
    total_energy_old : array_like, shape (n,)
        Old nodal total energy.
    joule_power : array_like, shape (n,)
        Nonnegative nodal Joule power.
    time_step : float
        Positive source-step size.
    magnetic_norm_old_sq : float
        Nonnegative old magnetic norm squared.
    permeability : float
        Positive magnetic permeability.
    Returns
    -------
    ndarray, shape (n+3,)
        Nodal E_new followed by [magnetic_norm_new_sq,min_internal,Joule_heat].
    """
    return np.zeros(len(lumped_mass)+3, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_source_energy_update(lumped_mass, density, momentum_old,
                                 momentum_new, total_energy_old, joule_power,
                                 time_step, magnetic_norm_old_sq,
                                 permeability=1.0):
    """Apply the nodal source-energy identity and its magnetic-energy balance."""
    import math
    import numpy as np
    mass = np.asarray(lumped_mass, dtype=float)
    rho = np.asarray(density, dtype=float)
    old_m = np.asarray(momentum_old, dtype=float)
    new_m = np.asarray(momentum_new, dtype=float)
    old_e = np.asarray(total_energy_old, dtype=float)
    joule = np.asarray(joule_power, dtype=float)
    tau = float(time_step)
    h_old_sq = float(magnetic_norm_old_sq)
    mu = float(permeability)
    n = mass.size
    if (mass.ndim != 1 or n == 0 or rho.shape != (n,) or old_m.shape != (n, 3)
            or new_m.shape != old_m.shape or old_e.shape != (n,) or joule.shape != (n,)):
        raise ValueError("inconsistent nodal shapes")
    if (not np.all(np.isfinite(mass)) or not np.all(np.isfinite(rho))
            or not np.all(np.isfinite(old_m)) or not np.all(np.isfinite(new_m))
            or not np.all(np.isfinite(old_e)) or not np.all(np.isfinite(joule))
            or not all(math.isfinite(x) for x in (tau, h_old_sq, mu))):
        raise ValueError("inputs must be finite")
    if np.any(mass <= 0) or np.any(rho <= 0) or np.any(joule < 0) or tau <= 0 or h_old_sq < 0 or mu <= 0:
        raise ValueError("invalid source-energy domain")
    kinetic_old = np.sum(old_m * old_m, axis=1) / (2.0 * rho)
    kinetic_new = np.sum(new_m * new_m, axis=1) / (2.0 * rho)
    new_e = old_e + kinetic_new - kinetic_old + tau * joule
    energy_increment = float(np.sum(mass * (new_e - old_e)))
    h_new_sq = h_old_sq - (2.0 / mu) * energy_increment
    internal = new_e - kinetic_new
    heat = float(np.sum(mass * tau * joule))
    return np.concatenate((new_e, np.array([h_new_sq, np.min(internal), heat])))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nm=np.array([.4,.6]); r=np.array([1.,2.]); p=np.array([[1.,0,0],[0,2.,0]]); q=p+.1; e=np.array([2.,3.]); j=np.array([.2,.3])', 'call': 'source_energy_update(m,r,p,q,e,j,.01,4.)', 'gold_call': '_oracle_source_energy_update(m,r,p,q,e,j,.01,4.)'}, {'setup': 'import numpy as np\nm=np.array([1.]); r=np.array([.5]); p=np.zeros((1,3)); q=np.array([[.2,-.1,.3]]); e=np.array([1.]); j=np.array([0.])', 'call': 'source_energy_update(m,r,p,q,e,j,.2,2.)', 'gold_call': '_oracle_source_energy_update(m,r,p,q,e,j,.2,2.)'}, {'setup': 'import numpy as np\nm=np.array([.2,.3,.5]); r=np.array([1.,.8,.6]); p=np.arange(9.).reshape(3,3)/10; q=np.flip(p,axis=0); e=np.array([2.,2.5,3.]); j=np.array([.1,.2,.4])', 'call': 'source_energy_update(m,r,p,q,e,j,.03,8.,2.)', 'gold_call': '_oracle_source_energy_update(m,r,p,q,e,j,.03,8.,2.)'}, {'setup': 'import numpy as np\nm=np.array([.25,.75]); r=np.array([2.,2.]); p=np.array([[1.,2.,0],[-1.,0,1.]]); q=.8*p; e=np.array([5.,4.]); j=np.array([.4,.1])', 'call': 'source_energy_update(m,r,p,q,e,j,.05,10.)', 'gold_call': '_oracle_source_energy_update(m,r,p,q,e,j,.05,10.)'}, {'setup': 'import numpy as np\nm=np.full(4,.25); r=np.linspace(.5,1.1,4); p=np.linspace(-.3,.4,12).reshape(4,3); q=p+np.eye(4,3)*.02; e=np.linspace(1.,1.6,4); j=np.linspace(.01,.04,4)', 'call': 'source_energy_update(m,r,p,q,e,j,.004,3.)', 'gold_call': '_oracle_source_energy_update(m,r,p,q,e,j,.004,3.)'}, {'setup': 'import numpy as np\nm=np.array([.1,.2,.3,.4]); r=np.ones(4); p=np.zeros((4,3)); q=np.ones((4,3))*.01; e=np.ones(4); j=np.arange(4.)/10', 'call': 'source_energy_update(m,r,p,q,e,j,.1,6.,.5)', 'gold_call': '_oracle_source_energy_update(m,r,p,q,e,j,.1,6.,.5)'}, {'setup': 'import numpy as np\nm=np.array([.3,.7]); r=np.array([.3,1.7]); p=np.array([[.5,-.2,.1],[.1,.6,-.4]]); q=np.array([[.2,.1,.3],[-.2,.5,.2]]); e=np.array([3.,2.]); j=np.array([.7,.05])', 'call': 'source_energy_update(m,r,p,q,e,j,.007,9.)', 'gold_call': '_oracle_source_energy_update(m,r,p,q,e,j,.007,9.)'}]
