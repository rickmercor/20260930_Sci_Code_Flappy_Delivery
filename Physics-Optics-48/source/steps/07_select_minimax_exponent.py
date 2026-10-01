"""
Apply prefix feasibility and the deterministic minimax tie-break across exponent candidates.

Limits are [R_x_max,R_y_max,abs_tau_max,B_min]. A candidate stops at its first failing energy. Rank by highest last-feasible energy, then smaller normalized load there, then smaller exponent. Failure codes use [R_x,R_y,abs_tau,B_min/B] = [1,2,3,4].

Returns
-------
Return a length-5 NumPy float array [p,E_last,last_index,load,next_code].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def select_minimax_exponent(exponents, energies_gev, scenarios, reference, limits):
    """Return the selected exponent and prefix-feasibility certificate.

    Args:
        exponents: Nonempty finite one-dimensional candidate-p array.
        energies_gev: Positive strictly increasing energy grid in GeV.
        scenarios: Shape-(S,4) array [sigma_delta,eps_nx,eps_ny,A].
        reference: Seven-entry reference-lattice array.
        limits: Positive four-entry array [R_x_max,R_y_max,abs_tau_max,B_min].

    Returns:
        numpy.ndarray: Length-5 float array [p,E_last,last_index,load,next_code],
        where E_last is in GeV, last_index and next_code are exact integer-valued
        floats, load is dimensionless, and next_code is in {0,1,2,3,4}.

    Raises:
        ValueError: If any array is malformed or nonfinite, physical grid or
            limit conditions fail, a downstream result is outside its finite
            supported domain, or no exponent is feasible at the first node.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_minimax_exponent(exponents, energies_gev, scenarios, reference, limits):
    import math
    import numpy as np
    exponents = np.asarray(exponents, dtype=float)
    energies = np.asarray(energies_gev, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if exponents.ndim != 1 or len(exponents) == 0 or not np.all(np.isfinite(exponents)):
        raise ValueError("exponents must be a nonempty finite vector")
    if limits.shape != (4,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("limits must be a positive finite length-4 array")
    best = None
    best_key = None
    for exponent in exponents:
        profile = _oracle_robust_energy_profile(exponent, energies, scenarios, reference)
        last = -1
        for i, row in enumerate(profile):
            loads = np.array([row[0] / limits[0], row[1] / limits[1],
                              row[2] / limits[2], limits[3] / row[3]], dtype=float)
            if np.all(loads <= 1.0):
                last = i
            else:
                break
        if last < 0:
            continue
        row = profile[last]
        loads = np.array([row[0] / limits[0], row[1] / limits[1],
                          row[2] / limits[2], limits[3] / row[3]], dtype=float)
        load = float(np.max(loads))
        if last + 1 < len(energies):
            next_row = profile[last + 1]
            next_loads = np.array([next_row[0] / limits[0], next_row[1] / limits[1],
                                   next_row[2] / limits[2], limits[3] / next_row[3]], dtype=float)
            next_code = int(np.argmax(next_loads)) + 1
        else:
            next_code = 0
        key = (float(energies[last]), -load, -float(exponent))
        if best_key is None or key > best_key:
            best_key = key
            best = np.array([exponent, energies[last], float(last), load, float(next_code)], dtype=float)
    if best is None:
        raise ValueError("no exponent is feasible at the first energy")
    return best

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup":"import numpy as np,itertools\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])\nsc=np.array(list(itertools.product([.018,.032],[8e-6,12e-6],[.1e-6,.18e-6],[.025,.04])))", "call":"select_minimax_exponent(np.array([.45,.48,.51]),np.geomspace(50.,5000.,101),sc,ref,np.array([1.25,1.25,520.,.04]))", "gold_call":"_oracle_select_minimax_exponent(np.array([.45,.48,.51]),np.geomspace(50.,5000.,101),sc,ref,np.array([1.25,1.25,520.,.04]))"},
        {"setup":"import numpy as np\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])\nsc=np.array([[.02,8e-6,.1e-6,.03]])", "call":"select_minimax_exponent(np.array([.6]),np.array([50.,100.,200.]),sc,ref,np.array([1.4,1.4,1000.,.01]))", "gold_call":"_oracle_select_minimax_exponent(np.array([.6]),np.array([50.,100.,200.]),sc,ref,np.array([1.4,1.4,1000.,.01]))"},
        {"setup":"import numpy as np\nref=np.array([50.,np.sqrt(5.),2*np.sqrt(5.),.015*np.sqrt(5.),1.,-61.42,-1e-4])\nsc=np.array([[.032,12e-6,.18e-6,.04]])", "call":"select_minimax_exponent(np.array([.4,.5,.6]),np.geomspace(50.,800.,17),sc,ref,np.array([1.15,1.2,350.,.08]))", "gold_call":"_oracle_select_minimax_exponent(np.array([.4,.5,.6]),np.geomspace(50.,800.,17),sc,ref,np.array([1.15,1.2,350.,.08]))"},
        {"setup":"import numpy as np\nref=np.array([20.,1.4,2.1,.02,.8,-40.,-2e-4])\nsc=np.array([[0.,3e-6,3e-6,0.]])", "call":"select_minimax_exponent(np.array([.2,.2,.4]),np.array([20.,22.,24.]),sc,ref,np.array([2.,2.,1e5,.001]))", "gold_call":"_oracle_select_minimax_exponent(np.array([.2,.2,.4]),np.array([20.,22.,24.]),sc,ref,np.array([2.,2.,1e5,.001]))"},
        {"setup":"import numpy as np\nref=np.array([60.,2.1,3.7,.031,1.1,-52.,-8e-5])\nsc=np.array([[.026,13e-6,.08e-6,.065]])", "call":"select_minimax_exponent(np.array([.68,.56,.44,.32]),np.geomspace(60.,2400.,41),sc,ref,np.array([1.28,1.22,510.,.06]))", "gold_call":"_oracle_select_minimax_exponent(np.array([.68,.56,.44,.32]),np.geomspace(60.,2400.,41),sc,ref,np.array([1.28,1.22,510.,.06]))"},
        {"setup":"import numpy as np\nref=np.array([35.,1.5,2.9,.019,.75,-70.,-2e-4])\nsc=np.array([[.02,6e-6,.2e-6,.035],[.038,10e-6,.16e-6,.08]])", "call":"select_minimax_exponent(np.array([.4,.46,.52,.58]),np.geomspace(35.,700.,31),sc,ref,np.array([1.18,1.26,390.,.08]))", "gold_call":"_oracle_select_minimax_exponent(np.array([.4,.46,.52,.58]),np.geomspace(35.,700.,31),sc,ref,np.array([1.18,1.26,390.,.08]))"},
        {"setup":"import numpy as np\nref=np.array([75.,1.2,6.2,.024,.65,-33.,-3e-4])\nsc=np.array([[.005,1e-6,9e-6,.09],[.06,17e-6,.05e-6,.002]])", "call":"select_minimax_exponent(np.array([-.1,.15,.35,.95]),np.array([75.,90.,150.,300.]),sc,ref,np.array([1.6,1.6,900.,.005]))", "gold_call":"_oracle_select_minimax_exponent(np.array([-.1,.15,.35,.95]),np.array([75.,90.,150.,300.]),sc,ref,np.array([1.6,1.6,900.,.005]))"}
    ]
