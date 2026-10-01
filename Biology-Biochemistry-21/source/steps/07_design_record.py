"""
Evaluate every condition for one biochemical design.

This step composes growth adaptation, nonlinear flux fitting, and one shared hermodynamic fit. Eligibility requires both stated limits.

Returns
-------
a fixed thirteen-value float64 design record.  The limiting state and reaction are the lowest (state, reaction) pair among active reactions whose corrected energy is within 1e-9 of the shared maximum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def design_record(data: dict, design_index: int) -> "np.ndarray":
    """Evaluate one design across all conditions.

    Parameters
    ----------
    data
        Mapping containing the numerical arrays and scalars named in the task:
        affinity, turnover, pathway_weight, base_phi, reaction_sector,
        metabolite_sector, design_multiplier, restriction, growth_ratios,
        metabolome_fraction, uptake, apparent_rate, intercept, coefficients,
        stoichiometry, objective_weight, rho_high, standard_energy,
        uncertainty_map, gas_temperature, driving_force, design_shift,
        active_threshold, residual_limit, thermo_limit, and cost.
    design_index
        Zero-based integer row of the design arrays.

    Returns
    -------
    np.ndarray
        Eligibility indicator; cost-normalized score; c-bar; worst RMS
        residual; shared thermodynamic maximum; robust product yield; limiting
        thermodynamic condition and reaction; four R06 condition fluxes; and
        the limiting-yield condition index.

    Raises
    ------
    ValueError
        If keys are missing, the design index is invalid, design or condition
        arrays are misaligned, cost or uptake is nonpositive, or any composed
        step rejects its inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_design_record(data: dict, design_index: int) -> "np.ndarray":
    required={
        "affinity","turnover","pathway_weight","base_phi","reaction_sector",
        "metabolite_sector","design_multiplier","restriction","growth_ratios",
        "metabolome_fraction","uptake","apparent_rate","intercept","coefficients",
        "stoichiometry","objective_weight","rho_high","standard_energy",
        "uncertainty_map","gas_temperature","driving_force","design_shift",
        "active_threshold","residual_limit","thermo_limit","cost",
    }
    if not isinstance(data,dict) or not required.issubset(data):
        raise ValueError("design data are incomplete")
    if not isinstance(design_index,(int,np.integer)):
        raise ValueError("design index must be an integer")
    multipliers=np.asarray(data["design_multiplier"],dtype=np.float64)
    restrictions=np.asarray(data["restriction"],dtype=np.float64)
    shifts=np.asarray(data["design_shift"],dtype=np.float64)
    costs=np.asarray(data["cost"],dtype=np.float64)
    growth=np.asarray(data["growth_ratios"],dtype=np.float64)
    uptake=np.asarray(data["uptake"],dtype=np.float64)
    if (
        multipliers.ndim!=2 or restrictions.ndim!=2
        or multipliers.shape[0]!=restrictions.shape[0]
        or shifts.shape!=(multipliers.shape[0],)
        or costs.shape!=(multipliers.shape[0],)
        or design_index<0 or design_index>=multipliers.shape[0]
        or growth.ndim!=1 or growth.size!=4 or uptake.shape!=growth.shape
        or not np.isfinite(uptake).all() or np.any(uptake<=0.0)
        or not np.isfinite(costs[design_index]) or costs[design_index]<=0.0
    ):
        raise ValueError("design arrays, conditions, cost or index are invalid")
    base_phi=np.asarray(data["base_phi"],dtype=np.float64)
    modified=base_phi*multipliers[design_index]
    if (modified.ndim!=1 or modified.size==0 or not np.isfinite(modified).all()
            or np.any(modified<=0.0)):
        raise ValueError("modified allocation must be finite and positive")
    phi_high=modified/modified.sum()
    kinetic=_oracle_kinetic_factors(data["affinity"],data["turnover"],data["pathway_weight"])
    packed=_oracle_growth_response(
        phi_high,kinetic,data["reaction_sector"],data["metabolite_sector"],
        restrictions[design_index],growth,data["metabolome_fraction"])
    cbar=float(packed[0]); n_reactions=base_phi.size
    q_rho=packed[1+growth.size*n_reactions:].reshape(growth.size,-1)
    enzyme=_oracle_enzyme_profiles(
        base_phi,multipliers[design_index],data["affinity"],data["turnover"],
        data["pathway_weight"],data["reaction_sector"],data["metabolite_sector"],
        restrictions[design_index],growth,data["metabolome_fraction"])
    states=[]
    for condition in range(growth.size):
        states.append(_oracle_kineflux_state(
            float(uptake[condition]),enzyme[condition],data["apparent_rate"],
            data["intercept"],data["coefficients"],data["stoichiometry"],
            data["objective_weight"]))
    fluxes=np.vstack([state[:n_reactions] for state in states])
    residuals=np.array([state[-2] for state in states],dtype=np.float64)
    rho_high=np.asarray(data["rho_high"],dtype=np.float64)
    if (rho_high.ndim!=1 or rho_high.size!=q_rho.shape[1]
            or not np.isfinite(rho_high).all() or np.any(rho_high<=0.0)):
        raise ValueError("high-growth metabolite concentrations are invalid")
    log_concentrations=np.log(rho_high[None,:]*q_rho)
    thermo=_oracle_shared_thermodynamic_fit(
        fluxes,log_concentrations,data["stoichiometry"],data["standard_energy"],
        data["uncertainty_map"],data["gas_temperature"],data["driving_force"],
        float(shifts[design_index]),data["active_threshold"])
    product=fluxes[:,5]; yields=product/uptake
    limiting_yield=int(np.argmin(yields)); robust_yield=float(yields[limiting_yield])
    worst_residual=float(np.max(residuals))
    eligible=float(worst_residual<=float(data["residual_limit"])
        and thermo[0]<=float(data["thermo_limit"]))
    score=robust_yield/float(costs[design_index])
    return np.array([
        eligible,score,cbar,worst_residual,thermo[0],robust_yield,
        thermo[-2],thermo[-1],product[0],product[1],product[2],product[3],
        float(limiting_yield),
    ],dtype=np.float64)


def _dr7_fixture():
    return {
        "affinity":np.array([.18,.055,.092,.041,.077,.032,.068,.049]),
        "turnover":np.array([54.,31.,47.,28.,39.,22.,18.,25.]),
        "pathway_weight":np.array([1.,.82,.82,.73,.73,.95,.61,.61]),
        "base_phi":np.array([.16,.13,.12,.11,.105,.15,.115,.11]),
        "reaction_sector":np.array([0,0,0,2,2,2,1,1]),
        "metabolite_sector":np.array([0,0,0,2]),
        "design_multiplier":np.array([
            [1.02,1.16,.91,1.08,.93,1.05,.88,1.12],
            [.96,1.04,1.12,.94,1.10,1.08,1.03,.91],
            [1.05,.91,1.18,1.13,.90,.97,1.08,.94],
            [.93,1.14,.98,1.02,1.12,1.04,.92,1.07],
            [1.01,1.08,1.05,1.10,1.02,1.09,.95,.96],
            [1.07,.97,1.09,.96,1.06,1.02,1.10,.90]]),
        "restriction":np.array([[1.,.05,.88],[.94,.08,1.02],[1.08,.02,.91],
            [.89,.12,1.10],[.98,.04,.95],[1.04,.06,.86]]),
        "growth_ratios":np.array([1.,.78,.55,.34]),"metabolome_fraction":.12,
        "uptake":np.array([.82,.70,.57,.46]),
        "apparent_rate":np.array([11.,12.5,12.,13.5,13.,10.5,8.5,8.]),
        "intercept":np.array([-.30,-.10,-.22,.05,-.08,-.18,.12,.09]),
        "coefficients":np.array([[.35,-.12,.08,.04],[.16,.25,-.18,.09],[.18,-.14,.27,.07],[-.05,.22,.03,.28],[.02,-.03,.24,.31],[.08,.04,.06,.38],[-.09,.21,-.02,.05],[.03,-.05,.19,.02]]),
        "stoichiometry":np.array([[1,-1,-1,0,0,0,0,0],[0,1,0,-1,0,0,-1,0],[0,0,1,0,-1,0,0,-1],[0,0,0,1,1,-1,0,0]],float),
        "objective_weight":.012,"rho_high":np.array([.006,.0038,.0044,.0029]),
        "standard_energy":np.array([-9.3,-3.,-2.5,-2.2,-1.8,-20.3,-15.3,-20.]),
        "uncertainty_map":np.array([[.60,.10],[-.25,.55],[.20,-.50],[-.45,-.15]]),
        "gas_temperature":2.4789570296,"driving_force":.08,
        "design_shift":np.array([0.,0.,-.12,0.,0.,0.]),"active_threshold":1e-8,
        "residual_limit":.595,"thermo_limit":.35,
        "cost":np.array([1.,1.,.94,1.,1.0575,.9732]),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup":"""import copy
d=_dr7_fixture(); cd=copy.deepcopy(d); gd=copy.deepcopy(d)""",
            "call":"design_record(cd,0)","gold_call":"_oracle_design_record(gd,0)","tol":1e-7,
        },
        {
            "setup":"""import copy
d=_dr7_fixture(); cd=copy.deepcopy(d); gd=copy.deepcopy(d)""",
            "call":"design_record(cd,1)","gold_call":"_oracle_design_record(gd,1)","tol":1e-7,
        },
        {
            "setup":"""import copy
d=_dr7_fixture(); d['residual_limit']=1.; cd=copy.deepcopy(d); gd=copy.deepcopy(d)""",
            "call":"design_record(cd,2)","gold_call":"_oracle_design_record(gd,2)","tol":1e-7,
        },
        {
            "setup":"""import copy
d=_dr7_fixture(); cd=copy.deepcopy(d); gd=copy.deepcopy(d)
def case_raises(fn,*args):
    try: fn(*args)
    except ValueError: return 1.0
    return 0.0""",
            "call":"case_raises(design_record,cd,8)",
            "gold_call":"case_raises(_oracle_design_record,gd,8)","tol":0.0,
        },
    ]
