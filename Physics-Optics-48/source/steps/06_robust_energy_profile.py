"""
Evaluate the componentwise worst uncertainty corner at every energy for one exponent.

Each scenario row is [sigma_delta,eps_nx,eps_ny,A]. For every row and energy, compose the preceding paper-grounded models; then take componentwise maxima of R_x and R_y. Resource columns do not depend on the scenario.

Returns
-------
Return an (N,4) NumPy float array with columns [max_R_x,max_R_y,abs_tau_x,B].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def robust_energy_profile(exponent, energies_gev, scenarios, reference):
    """Return worst-case ratios and resources for one exponent.

    Args:
        exponent: Finite p.
        energies_gev: Length-N increasing positive numerical array in GeV.
        scenarios: Shape-(S,4) array with columns [sigma_delta,eps_nx,eps_ny,A]
            in [1,m rad,m rad,1], with S>=1.
        reference: Seven-element reference array used by scale_staging_lattice.

    Returns:
        numpy.ndarray: Shape-(N,4) float array with columns
        [max_R_x,max_R_y,abs_tau_x,B] in [1,1,m^-1,T].

    Raises:
        ValueError: If arrays have the wrong shape, contain nonfinite or
            nonphysical values, energies are not positive and strictly
            increasing, exponent is nonfinite, or a downstream result is
            outside its documented finite supported domain.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_robust_energy_profile(exponent, energies_gev, scenarios, reference):
    import math
    import numpy as np
    exponent = float(exponent)
    energies = np.asarray(energies_gev, dtype=float)
    scenarios = np.asarray(scenarios, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if energies.ndim != 1 or len(energies) == 0 or not np.all(np.isfinite(energies)):
        raise ValueError("energies must be a nonempty finite vector")
    if np.any(energies <= 0) or np.any(np.diff(energies) <= 0):
        raise ValueError("energies must be positive and strictly increasing")
    if scenarios.ndim != 2 or scenarios.shape[1] != 4 or len(scenarios) == 0 or not np.all(np.isfinite(scenarios)):
        raise ValueError("scenarios must have finite shape (S,4)")
    if np.any(scenarios < 0) or np.any(scenarios[:, 1:3] <= 0):
        raise ValueError("scenario spreads are nonnegative and emittances positive")
    out = np.empty((len(energies), 4), dtype=float)
    for i, energy in enumerate(energies):
        scaled = _oracle_scale_staging_lattice(energy, exponent, reference)
        length, ell, beta, field, tau = scaled[:5]
        worst_x = -math.inf
        worst_y = -math.inf
        for sigma_delta, eps_nx, eps_ny, amplitude in scenarios:
            chromatic = _oracle_second_order_chromatic_growth(sigma_delta, length, ell, beta)
            geometric = _oracle_nonlinear_geometric_growth(
                energy, length, ell, beta, tau, eps_nx, eps_ny)
            isr = _oracle_incoherent_radiation_growth(
                energy, exponent, reference[0], amplitude)
            ratios = _oracle_combine_emittance_increments(chromatic, geometric, isr)
            worst_x = max(worst_x, ratios[0])
            worst_y = max(worst_y, ratios[1])
        out[i] = [worst_x, worst_y, abs(tau), field]
    if not np.all(np.isfinite(out)):
        raise ValueError("profile exceeds the supported finite domain")
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])\nsc=np.array([[.018,8e-6,.1e-6,.025]])", "call":"robust_energy_profile(.6,np.array([50.,500.,5000.]),sc,ref)", "gold_call":"_oracle_robust_energy_profile(.6,np.array([50.,500.,5000.]),sc,ref)"},
        {"setup":"import numpy as np\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])\nsc=np.array([[.018,8e-6,.1e-6,.025],[.032,12e-6,.18e-6,.04]])", "call":"robust_energy_profile(.48,np.geomspace(50.,4000.,9),sc,ref)", "gold_call":"_oracle_robust_energy_profile(.48,np.geomspace(50.,4000.,9),sc,ref)"},
        {"setup":"import numpy as np\nref=np.array([20.,1.4,2.1,.02,.8,-40.,-2e-4])\nsc=np.array([[0.,3e-6,3e-6,0.],[.04,7e-6,.2e-6,.06]])", "call":"robust_energy_profile(0.,np.array([20.,21.]),sc,ref)", "gold_call":"_oracle_robust_energy_profile(0.,np.array([20.,21.]),sc,ref)"},
        {"setup":"import numpy as np\nref=np.array([60.,2.1,3.7,.031,1.1,-52.,-8e-5])\nsc=np.array([[.01,3e-6,.3e-6,.015],[.026,13e-6,.08e-6,.065],[.04,2e-6,2e-6,.01]])", "call":"robust_energy_profile(.32,np.array([60.,77.,155.,620.]),sc,ref)", "gold_call":"_oracle_robust_energy_profile(.32,np.array([60.,77.,155.,620.]),sc,ref)"},
        {"setup":"import numpy as np\nref=np.array([35.,1.5,2.9,.019,.75,-70.,-2e-4])\nsc=np.array([[.02,6e-6,.2e-6,.035],[.038,10e-6,.16e-6,.08]])", "call":"robust_energy_profile(.73,np.geomspace(35.,700.,6),sc,ref)", "gold_call":"_oracle_robust_energy_profile(.73,np.geomspace(35.,700.,6),sc,ref)"},
        {"setup":"import numpy as np\nref=np.array([12.,.7,5.1,.04,.33,19.,4e-5])\nsc=np.array([[.001,.4e-6,8e-6,0.],[.08,11e-6,.06e-6,.12]])", "call":"robust_energy_profile(-.15,np.array([12.,12.5,19.,150.]),sc,ref)", "gold_call":"_oracle_robust_energy_profile(-.15,np.array([12.,12.5,19.,150.]),sc,ref)"},
        {"setup":"import numpy as np\nref=np.array([100.,4.,1.,.08,2.,-15.,-5e-5])\nsc=np.array([[0.,1e-6,1e-6,.2],[.05,20e-6,3e-6,0.]])", "call":"robust_energy_profile(.6,np.array([100.,101.,1000.]),sc,ref)", "gold_call":"_oracle_robust_energy_profile(.6,np.array([100.,101.,1000.]),sc,ref)"}
    ]
