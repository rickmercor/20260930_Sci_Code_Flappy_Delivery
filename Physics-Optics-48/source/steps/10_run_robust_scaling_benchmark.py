"""
Run the whole robust plasma-lens scaling benchmark and return its rounded energy ceiling.

The implementation must call and combine every preceding public subproblem function. Re-evaluate all scenario corners at the selected node, use the robust profile as a consistency check, propagate a nonzero probe-ray batch with transport_nonlinear_lens_pair, and build the fixed validity certificate before returning the selected energy rounded once to 0.1 GeV. The validity certificate is diagnostic and does not change the selected node. For the no-argument path construct exponents=0.25+0.005*k for k=0,...,100; energies_gev=50*exp(j*ln(100)/400) for j=0,...,400; scenarios as the lexicographic itertools.product of sigma_delta=[0.018,0.032], eps_nx=[8e-6,12e-6], eps_ny=[0.10e-6,0.18e-6], A_ISR=[0.025,0.040], with A_ISR varying fastest; reference=[50,sqrt(5),2sqrt(5),0.015sqrt(5),1.0,-61.42,-1e-4]; and limits=[1.25,1.25,520,0.040]. At the selected node use the largest eps_nx and eps_ny in scenarios, quadrature_order=7, offset_fraction=0.20, offset_correlation=-0.35, and gamma=E/0.000511. The nonzero probe batch is [(sqrt(eps_nx/(gamma*beta)),0),(0,sqrt(eps_ny/(gamma*beta))),(sqrt(eps_nx/(gamma*beta)),-sqrt(eps_ny/(gamma*beta)))]. Explicit arguments support independent whole-pipeline cases and use these same fixed validation parameters.

Returns
-------
Return one finite float: the selected energy in GeV rounded once to 0.1 GeV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def run_robust_scaling_benchmark(exponents=None, energies_gev=None, scenarios=None,
                                 reference=None, limits=None):
    """Return the robust energy ceiling for the supplied or canonical benchmark.

    Args:
        exponents: Optional finite nonempty one-dimensional candidate-p array;
            canonical default is 0.25+0.005*k, k=0,...,100.
        energies_gev: Optional positive strictly increasing energy grid in GeV;
            canonical default is 50*exp(j*ln(100)/400), j=0,...,400.
        scenarios: Optional finite shape-(S,4) array [sigma_delta,eps_nx,
            eps_ny,A_ISR]; canonical default is the 16-row Cartesian product
            [0.018,0.032] x [8e-6,12e-6] x [0.10e-6,0.18e-6] x
            [0.025,0.040], with the last coordinate varying fastest.
        reference: Optional seven-entry [E_r,L_r,l_r,beta_r,B_r,tau_r,R56_r];
            canonical default is [50,sqrt(5),2sqrt(5),0.015sqrt(5),1,-61.42,-1e-4].
        limits: Optional positive [R_x_max,R_y_max,abs_tau_max,B_min];
            canonical default is [1.25,1.25,520,0.040].
            Either omit all five arguments or supply all five.

    Returns:
        float: Selected energy in GeV, rounded once to 0.1 GeV.

    Raises:
        ValueError: If only some optional arguments are supplied; an explicit
            array is malformed/nonfinite or violates its documented physical
            domain; no exponent is feasible at the first energy; or a
            downstream calculation leaves its supported finite domain.
        RuntimeError: If the independently recomputed selected-node profile,
            exact-map probe, or fixed validity certificate is internally
            inconsistent with the public subproblem results.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_robust_scaling_benchmark(exponents=None, energies_gev=None, scenarios=None,
                                                 reference=None, limits=None):
    import itertools
    import math
    import numpy as np
    supplied = [exponents, energies_gev, scenarios, reference, limits]
    if all(x is None for x in supplied):
        exponents = np.arange(0.25, 0.7501, 0.005)
        energies_gev = 50.0 * np.exp(np.arange(401) * np.log(100.0) / 400.0)
        scenarios = np.array(list(itertools.product(
            [0.018, 0.032], [8e-6, 12e-6], [0.10e-6, 0.18e-6], [0.025, 0.040])), dtype=float)
        reference = np.array([50.0, np.sqrt(5.0), 2.0 * np.sqrt(5.0),
                              0.015 * np.sqrt(5.0), 1.0, -61.42, -1e-4], dtype=float)
        limits = np.array([1.25, 1.25, 520.0, 0.040], dtype=float)
    elif any(x is None for x in supplied):
        raise ValueError("either omit every argument or supply all five")
    exponents = np.asarray(exponents, dtype=float)
    energies_gev = np.asarray(energies_gev, dtype=float)
    scenarios = np.asarray(scenarios, dtype=float)
    reference = np.asarray(reference, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if (exponents.ndim != 1 or len(exponents) == 0 or
            energies_gev.ndim != 1 or len(energies_gev) == 0 or
            scenarios.ndim != 2 or scenarios.shape[1] != 4 or
            reference.shape != (7,) or limits.shape != (4,) or
            not all(np.all(np.isfinite(x)) for x in
                    (exponents, energies_gev, scenarios, reference, limits))):
        raise ValueError("malformed or non-finite benchmark inputs")
    selection = _oracle_select_minimax_exponent(exponents, energies_gev, scenarios, reference, limits)
    exponent = float(selection[0])
    energy = float(selection[1])
    index = int(selection[2])
    next_code = float(selection[4])
    profile = _oracle_robust_energy_profile(exponent, energies_gev, scenarios, reference)
    scaled = _oracle_scale_staging_lattice(energy, exponent, reference)
    length, ell, beta, field, tau = scaled[:5]
    direct_x = -math.inf
    direct_y = -math.inf
    for sigma_delta, eps_nx, eps_ny, amplitude in np.asarray(scenarios, dtype=float):
        chromatic = _oracle_second_order_chromatic_growth(sigma_delta, length, ell, beta)
        geometric = _oracle_nonlinear_geometric_growth(
            energy, length, ell, beta, tau, eps_nx, eps_ny)
        isr = _oracle_incoherent_radiation_growth(energy, exponent, reference[0], amplitude)
        ratios = _oracle_combine_emittance_increments(chromatic, geometric, isr)
        direct_x = max(direct_x, ratios[0])
        direct_y = max(direct_y, ratios[1])
    worst_x = max(direct_x, profile[index, 0])
    worst_y = max(direct_y, profile[index, 1])
    max_eps_nx = float(np.max(scenarios[:, 1]))
    max_eps_ny = float(np.max(scenarios[:, 2]))
    validity = _oracle_nonlinear_lens_validity_certificate(
        energy, length, ell, beta, tau, max_eps_nx, max_eps_ny,
        0.20, -0.35, 7)
    gamma = energy / 0.000511
    probe = np.array([
        [math.sqrt(max_eps_nx / (gamma * beta)), 0.0],
        [0.0, math.sqrt(max_eps_ny / (gamma * beta))],
        [math.sqrt(max_eps_nx / (gamma * beta)),
         -math.sqrt(max_eps_ny / (gamma * beta))]], dtype=float)
    mapped_probe = _oracle_transport_nonlinear_lens_pair(probe, length, ell, tau)
    linear_probe = _oracle_transport_nonlinear_lens_pair(probe, length, ell, 0.0)
    asym_direct = _oracle_nonlinear_geometric_growth(
        energy, length, ell, beta, tau, max_eps_nx, max_eps_ny)
    if (not np.allclose([worst_x, worst_y, abs(tau), field], profile[index], rtol=1e-12, atol=1e-14)
            or next_code not in (0.0, 1.0, 2.0, 3.0, 4.0)
            or not np.all(np.isfinite(validity))
            or not np.allclose(validity[2:4], asym_direct, rtol=1e-12, atol=1e-14)
            or np.any(validity[:4] < 0.0)
            or np.any(validity[4:6] <= 0.5) or np.any(validity[4:6] >= 3.5)
            or validity[6] <= 0.0 or validity[7] <= 0.0
            or not np.allclose(linear_probe[:, [0, 2]], 0.0, rtol=0.0, atol=1e-14)
            or not np.allclose(linear_probe[:, [1, 3]], probe, rtol=1e-12, atol=1e-14)
            or not np.all(np.isfinite(mapped_probe))):
        raise RuntimeError("inconsistent end-to-end certificate")
    return float(np.round(energy, 1))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np,itertools\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])\nsc=np.array(list(itertools.product([.018,.032],[8e-6,12e-6],[.1e-6,.18e-6],[.025,.04])))", "call":"run_robust_scaling_benchmark(np.arange(.25,.7501,.005),50*np.exp(np.arange(401)*np.log(100)/400),sc,ref,np.array([1.25,1.25,520.,.04]))", "gold_call":"_oracle_run_robust_scaling_benchmark(np.arange(.25,.7501,.005),50*np.exp(np.arange(401)*np.log(100)/400),sc,ref,np.array([1.25,1.25,520.,.04]))"},
        {"setup":"import numpy as np\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])\nsc=np.array([[.02,8e-6,.1e-6,.03],[.03,10e-6,.15e-6,.05]])", "call":"run_robust_scaling_benchmark(np.array([.35,.45,.55,.65]),np.geomspace(50.,1200.,33),sc,ref,np.array([1.22,1.24,430.,.07]))", "gold_call":"_oracle_run_robust_scaling_benchmark(np.array([.35,.45,.55,.65]),np.geomspace(50.,1200.,33),sc,ref,np.array([1.22,1.24,430.,.07]))"},
        {"setup":"import numpy as np\nref=np.array([40.,1.8,3.4,.026,.9,-48.,-1.5e-4])\nsc=np.array([[.015,4e-6,.4e-6,.02],[.035,9e-6,.12e-6,.07]])", "call":"run_robust_scaling_benchmark(np.array([.4,.5,.6]),np.geomspace(40.,900.,29),sc,ref,np.array([1.3,1.3,600.,.05]))", "gold_call":"_oracle_run_robust_scaling_benchmark(np.array([.4,.5,.6]),np.geomspace(40.,900.,29),sc,ref,np.array([1.3,1.3,600.,.05]))"},
        {"setup":"import numpy as np\nref=np.array([20.,1.4,2.1,.02,.8,-40.,-2e-4])\nsc=np.array([[.004,2e-6,2e-6,.005],[.04,7e-6,.2e-6,.06]])", "call":"run_robust_scaling_benchmark(np.array([.1,.3,.5,.7]),np.geomspace(20.,500.,27),sc,ref,np.array([1.4,1.35,550.,.04]))", "gold_call":"_oracle_run_robust_scaling_benchmark(np.array([.1,.3,.5,.7]),np.geomspace(20.,500.,27),sc,ref,np.array([1.4,1.35,550.,.04]))"},
        {"setup":"import numpy as np\nref=np.array([75.,1.2,6.2,.024,.65,-33.,-3e-4])\nsc=np.array([[.005,1e-6,9e-6,.09],[.06,17e-6,.05e-6,.002]])", "call":"run_robust_scaling_benchmark(np.array([.15,.35,.55,.75]),np.geomspace(75.,600.,25),sc,ref,np.array([1.6,1.6,700.,.02]))", "gold_call":"_oracle_run_robust_scaling_benchmark(np.array([.15,.35,.55,.75]),np.geomspace(75.,600.,25),sc,ref,np.array([1.6,1.6,700.,.02]))"},
        {"setup":"import numpy as np\nref=np.array([100.,4.,1.,.08,2.,-15.,-5e-5])\nsc=np.array([[.001,.5e-6,.5e-6,.2],[.05,20e-6,3e-6,.001]])", "call":"run_robust_scaling_benchmark(np.array([.25,.45,.65]),np.array([100.,120.,180.,260.,400.]),sc,ref,np.array([50.,50.,250.,.2]))", "gold_call":"_oracle_run_robust_scaling_benchmark(np.array([.25,.45,.65]),np.array([100.,120.,180.,260.,400.]),sc,ref,np.array([50.,50.,250.,.2]))"},
        {"setup":"import numpy as np\nref=np.array([30.,.8,2.4,.012,1.3,-85.,-9e-5])\nsc=np.array([[.012,3e-6,.7e-6,.01],[.028,6e-6,.11e-6,.03],[.021,9e-6,.2e-6,.02]])", "call":"run_robust_scaling_benchmark(np.array([.31,.47,.63,.79]),np.geomspace(30.,1000.,35),sc,ref,np.array([1.35,1.31,800.,.025]))", "gold_call":"_oracle_run_robust_scaling_benchmark(np.array([.31,.47,.63,.79]),np.geomspace(30.,1000.,35),sc,ref,np.array([1.35,1.31,800.,.025]))"},
        {"setup":"def _ve(fn):\n try: fn()\n except ValueError: return 1.0\n except Exception: return -1.0\n return 0.0", "call":"_ve(lambda: run_robust_scaling_benchmark(exponents=[.5]))", "gold_call":"_ve(lambda: _oracle_run_robust_scaling_benchmark(exponents=[.5]))"}
    ]
