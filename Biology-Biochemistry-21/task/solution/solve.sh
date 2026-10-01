#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def kinetic_factors(
    affinity: "np.ndarray",
    turnover: "np.ndarray",
    pathway_weight: "np.ndarray",
) -> "np.ndarray":
    affinity = np.asarray(affinity, dtype=np.float64)
    turnover = np.asarray(turnover, dtype=np.float64)
    pathway_weight = np.asarray(pathway_weight, dtype=np.float64)
    if (
        affinity.ndim != 1
        or affinity.size == 0
        or turnover.shape != affinity.shape
        or pathway_weight.shape != affinity.shape
        or not np.isfinite(affinity).all()
        or not np.isfinite(turnover).all()
        or not np.isfinite(pathway_weight).all()
        or np.any(affinity <= 0.0)
        or np.any(turnover <= 0.0)
        or np.any(pathway_weight <= 0.0)
    ):
        raise ValueError("kinetic inputs must be aligned finite positive vectors")
    return np.sqrt(affinity * turnover / pathway_weight).astype(np.float64)

import numpy as np


def growth_response(
    phi_high: "np.ndarray",
    kinetic_factor: "np.ndarray",
    reaction_sector: "np.ndarray",
    metabolite_sector: "np.ndarray",
    restriction: "np.ndarray",
    growth_ratios: "np.ndarray",
    metabolome_fraction: float,
) -> "np.ndarray":
    phi_high = np.asarray(phi_high, dtype=np.float64)
    kinetic_factor = np.asarray(kinetic_factor, dtype=np.float64)
    reaction_sector = np.asarray(reaction_sector)
    metabolite_sector = np.asarray(metabolite_sector)
    restriction = np.asarray(restriction, dtype=np.float64)
    growth_ratios = np.asarray(growth_ratios, dtype=np.float64)
    if (
        phi_high.ndim != 1
        or phi_high.size == 0
        or kinetic_factor.shape != phi_high.shape
        or reaction_sector.shape != phi_high.shape
        or metabolite_sector.ndim != 1
        or metabolite_sector.size == 0
        or restriction.ndim != 1
        or restriction.size == 0
        or growth_ratios.ndim != 1
        or growth_ratios.size == 0
        or not np.isfinite(phi_high).all()
        or not np.isfinite(kinetic_factor).all()
        or not np.isfinite(restriction).all()
        or not np.isfinite(growth_ratios).all()
        or np.any(phi_high <= 0.0)
        or np.any(kinetic_factor <= 0.0)
        or np.any(restriction < 0.0)
        or np.any(growth_ratios < 0.0)
        or np.any(growth_ratios > 1.0)
        or not np.isfinite(metabolome_fraction)
        or metabolome_fraction <= 0.0
        or not np.isclose(phi_high.sum(), 1.0, rtol=0.0, atol=1e-12)
        or not np.issubdtype(reaction_sector.dtype, np.integer)
        or not np.issubdtype(metabolite_sector.dtype, np.integer)
        or np.any(reaction_sector < 0)
        or np.any(metabolite_sector < 0)
        or np.any(reaction_sector >= restriction.size)
        or np.any(metabolite_sector >= restriction.size)
    ):
        raise ValueError("growth-response inputs violate the stated contract")
    xi_r = restriction[reaction_sector]
    cbar = float(np.sum(phi_high * kinetic_factor * xi_r))
    if not np.isfinite(cbar) or cbar <= 0.0:
        raise ValueError("the mass-weighted kinetic average must be positive")
    r = growth_ratios[:, None]
    q = r + (kinetic_factor[None, :] / cbar) * xi_r[None, :] * (1.0 - r)
    xi_m = restriction[metabolite_sector][None, :]
    numerator = metabolome_fraction * r
    denominator = numerator + xi_m * (1.0 - r)
    qrho = np.divide(
        numerator,
        denominator,
        out=np.ones((growth_ratios.size, metabolite_sector.size), dtype=np.float64),
        where=denominator > 0.0,
    )
    return np.concatenate(([cbar], q.ravel(), qrho.ravel())).astype(np.float64)

import numpy as np


def enzyme_profiles(
    base_phi: "np.ndarray",
    design_multiplier: "np.ndarray",
    affinity: "np.ndarray",
    turnover: "np.ndarray",
    pathway_weight: "np.ndarray",
    reaction_sector: "np.ndarray",
    metabolite_sector: "np.ndarray",
    restriction: "np.ndarray",
    growth_ratios: "np.ndarray",
    metabolome_fraction: float,
) -> "np.ndarray":
    base_phi = np.asarray(base_phi, dtype=np.float64)
    design_multiplier = np.asarray(design_multiplier, dtype=np.float64)
    if (
        base_phi.ndim != 1
        or base_phi.size == 0
        or design_multiplier.shape != base_phi.shape
        or not np.isfinite(base_phi).all()
        or not np.isfinite(design_multiplier).all()
        or np.any(base_phi <= 0.0)
        or np.any(design_multiplier <= 0.0)
    ):
        raise ValueError("base allocation and design multiplier must align and be positive")
    modified = base_phi * design_multiplier
    phi_high = modified / modified.sum()
    kinetic = kinetic_factors(affinity, turnover, pathway_weight)
    packed = growth_response(
        phi_high,
        kinetic,
        reaction_sector,
        metabolite_sector,
        restriction,
        growth_ratios,
        metabolome_fraction,
    )
    n_conditions = np.asarray(growth_ratios).size
    n_reactions = base_phi.size
    q = packed[1 : 1 + n_conditions * n_reactions].reshape(n_conditions, n_reactions)
    return (q * phi_high[None, :]).astype(np.float64)

import numpy as np


def flux_sums(stoichiometry: "np.ndarray", flux: "np.ndarray") -> "np.ndarray":
    stoichiometry = np.asarray(stoichiometry, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if (
        stoichiometry.ndim != 2
        or stoichiometry.shape[0] == 0
        or stoichiometry.shape[1] == 0
        or flux.shape != (stoichiometry.shape[1],)
        or not np.isfinite(stoichiometry).all()
        or not np.isfinite(flux).all()
        or np.any(flux < 0.0)
    ):
        raise ValueError("stoichiometry and irreversible flux must be finite and aligned")
    return (np.maximum(stoichiometry, 0.0) @ flux).astype(np.float64)

import numpy as np
from scipy.optimize import minimize


def _ks5_unpack(uptake, independent):
    r02, r04, r05 = independent
    return np.array([
        uptake, r02, uptake-r02, r04, r05, r04+r05,
        r02-r04, uptake-r02-r05,
    ], dtype=np.float64)


def kineflux_state(
    uptake: float,
    enzyme_abundance: "np.ndarray",
    apparent_rate: "np.ndarray",
    intercept: "np.ndarray",
    coefficients: "np.ndarray",
    stoichiometry: "np.ndarray",
    weight: float,
) -> "np.ndarray":
    enzyme_abundance = np.asarray(enzyme_abundance, dtype=np.float64)
    apparent_rate = np.asarray(apparent_rate, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    stoichiometry = np.asarray(stoichiometry, dtype=np.float64)
    expected = np.array([
        [1,-1,-1,0,0,0,0,0], [0,1,0,-1,0,0,-1,0],
        [0,0,1,0,-1,0,0,-1], [0,0,0,1,1,-1,0,0],
    ], dtype=np.float64)
    if (
        not np.isfinite(uptake) or uptake <= 0.0
        or enzyme_abundance.shape != (8,) or apparent_rate.shape != (8,)
        or intercept.shape != (8,) or coefficients.shape != (8,4)
        or stoichiometry.shape != (4,8)
        or not np.isfinite(enzyme_abundance).all()
        or not np.isfinite(apparent_rate).all()
        or not np.isfinite(intercept).all()
        or not np.isfinite(coefficients).all()
        or not np.isfinite(stoichiometry).all()
        or np.any(enzyme_abundance <= 0.0) or np.any(apparent_rate <= 0.0)
        or not np.array_equal(stoichiometry, expected)
        or not np.isfinite(weight) or weight < 0.0
    ):
        raise ValueError("KineFlux inputs violate the stated contract")
    capacity = enzyme_abundance * apparent_rate

    def _ks5_objective(independent):
        flux = _ks5_unpack(uptake, independent)
        # SLSQP samples infinitesimally outside inequality faces while taking
        # numerical derivatives.  Project only those trial values for the
        # flux-sum evaluation; accepted states remain constrained.
        flux_sum = flux_sums(stoichiometry, np.maximum(flux,0.0))
        linear = np.clip(intercept + coefficients @ flux_sum, -700.0, 700.0)
        eta = 1.0 / (1.0 + np.exp(-linear))
        mismatch = flux - capacity * eta
        return float(np.sum(mismatch*mismatch) + weight*np.sum(flux*flux))

    constraints = []
    for reaction in range(8):
        constraints.append({"type":"ineq", "fun":lambda x, reaction=reaction:
            capacity[reaction]-_ks5_unpack(uptake,x)[reaction]})
        constraints.append({"type":"ineq", "fun":lambda x, reaction=reaction:
            _ks5_unpack(uptake,x)[reaction]})
    starts = (
        np.array([.50*uptake,.25*uptake,.25*uptake]),
        np.array([.70*uptake,.50*uptake,.15*uptake]),
        np.array([.30*uptake,.12*uptake,.50*uptake]),
    )
    fits=[]
    for start in starts:
        fit = minimize(_ks5_objective, start, method="SLSQP", constraints=constraints,
            options={"ftol":1e-13,"maxiter":2000,"disp":False})
        feasible = min(float(rule["fun"](fit.x)) for rule in constraints)
        if fit.success and feasible >= -2e-8:
            fits.append((_ks5_objective(fit.x),fit.x.copy()))
    if not fits:
        raise ValueError("no feasible converged flux state")
    lowest=min(value for value,_ in fits)
    independent=min((x for value,x in fits if value<=lowest+1e-12),
        key=lambda x:tuple(float(v) for v in x))
    flux = np.maximum(_ks5_unpack(uptake, independent),0.0)
    flux_sum = flux_sums(stoichiometry, flux)
    linear = np.clip(intercept + coefficients @ flux_sum, -700.0, 700.0)
    eta = 1.0 / (1.0 + np.exp(-linear))
    mismatch = flux - capacity*eta
    residual = float(np.sqrt(np.mean(mismatch*mismatch)))
    margin = float(np.min(capacity-flux))
    return np.concatenate((flux,eta,[_ks5_objective(independent),residual,margin])).astype(np.float64)


def _ks5_fixture():
    stoich=np.array([[1,-1,-1,0,0,0,0,0],[0,1,0,-1,0,0,-1,0],
        [0,0,1,0,-1,0,0,-1],[0,0,0,1,1,-1,0,0]],float)
    kmax=np.array([11.,12.5,12.,13.5,13.,10.5,8.5,8.])
    beta=np.array([-.30,-.10,-.22,.05,-.08,-.18,.12,.09])
    alpha=np.array([[.35,-.12,.08,.04],[.16,.25,-.18,.09],
        [.18,-.14,.27,.07],[-.05,.22,.03,.28],[.02,-.03,.24,.31],
        [.08,.04,.06,.38],[-.09,.21,-.02,.05],[.03,-.05,.19,.02]])
    return stoich,kmax,beta,alpha

import numpy as np
from scipy.optimize import minimize


def shared_thermodynamic_fit(
    fluxes: "np.ndarray",
    log_concentrations: "np.ndarray",
    stoichiometry: "np.ndarray",
    standard_energy: "np.ndarray",
    uncertainty_map: "np.ndarray",
    gas_temperature: float,
    driving_force: float,
    design_shift: float,
    active_threshold: float,
) -> "np.ndarray":
    fluxes=np.asarray(fluxes,dtype=np.float64)
    log_concentrations=np.asarray(log_concentrations,dtype=np.float64)
    stoichiometry=np.asarray(stoichiometry,dtype=np.float64)
    standard_energy=np.asarray(standard_energy,dtype=np.float64)
    uncertainty_map=np.asarray(uncertainty_map,dtype=np.float64)
    if (
        fluxes.ndim!=2 or fluxes.shape[0]==0 or fluxes.shape[1]==0
        or log_concentrations.ndim!=2 or log_concentrations.shape[0]!=fluxes.shape[0]
        or stoichiometry.shape!=(log_concentrations.shape[1],fluxes.shape[1])
        or standard_energy.shape!=(fluxes.shape[1],)
        or uncertainty_map.ndim!=2
        or uncertainty_map.shape[0]!=log_concentrations.shape[1]
        or uncertainty_map.shape[1]==0
        or not np.isfinite(fluxes).all()
        or not np.isfinite(log_concentrations).all()
        or not np.isfinite(stoichiometry).all()
        or not np.isfinite(standard_energy).all()
        or not np.isfinite(uncertainty_map).all()
        or np.any(fluxes<0.0)
        or not np.isfinite(gas_temperature) or gas_temperature<=0.0
        or not np.isfinite(driving_force) or not np.isfinite(design_shift)
        or not np.isfinite(active_threshold) or active_threshold<0.0
    ):
        raise ValueError("thermodynamic-fit inputs violate the stated contract")
    response=gas_temperature*(stoichiometry.T@uncertainty_map)
    rows=[]; locations=[]
    for condition in range(fluxes.shape[0]):
        baseline=(standard_energy+design_shift
            +gas_temperature*(stoichiometry.T@log_concentrations[condition])
            +driving_force)
        for reaction in np.flatnonzero(fluxes[condition]>active_threshold):
            rows.append((response[reaction].copy(),float(baseline[reaction])))
            locations.append((condition,int(reaction)))
    if not rows:
        raise ValueError("at least one reaction must be active")
    slopes=np.vstack([row[0] for row in rows])
    baselines=np.array([row[1] for row in rows],dtype=np.float64)
    dimensions=uncertainty_map.shape[1]

    def _st6_objective(state):
        return float(state[-1])

    constraints=[{"type":"ineq","fun":lambda state:
        1.0-float(state[:-1]@state[:-1])}]
    for row in range(baselines.size):
        constraints.append({"type":"ineq","fun":lambda state,row=row:
            state[-1]-(baselines[row]+slopes[row]@state[:-1])})
    starts=[np.zeros(dimensions,dtype=np.float64)]
    for coordinate in range(dimensions):
        direction=np.zeros(dimensions,dtype=np.float64); direction[coordinate]=1.0
        starts.extend((direction,-direction))
    fits=[]
    for delta in starts:
        initial=np.concatenate((delta,[np.max(baselines+slopes@delta)]))
        fit=minimize(_st6_objective,initial,method="SLSQP",constraints=constraints,
            options={"ftol":1e-13,"maxiter":2000,"disp":False})
        if fit.success and min(float(rule["fun"](fit.x)) for rule in constraints)>=-2e-8:
            fits.append((float(fit.fun),fit.x.copy()))
    if not fits:
        raise ValueError("shared thermodynamic minimax fit did not converge")
    lowest=min(value for value,_ in fits)
    state=min((x for value,x in fits if value<=lowest+1e-12),
        key=lambda x:tuple(float(v) for v in x[:-1]))
    corrected=baselines+slopes@state[:-1]
    limiting=int(np.flatnonzero(corrected>=np.max(corrected)-1e-9)[0])
    condition,reaction=locations[limiting]
    return np.concatenate(([state[-1]],state[:-1],[float(condition),float(reaction)])).astype(np.float64)

import numpy as np


def design_record(data: dict, design_index: int) -> "np.ndarray":
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
    kinetic=kinetic_factors(data["affinity"],data["turnover"],data["pathway_weight"])
    packed=growth_response(
        phi_high,kinetic,data["reaction_sector"],data["metabolite_sector"],
        restrictions[design_index],growth,data["metabolome_fraction"])
    cbar=float(packed[0]); n_reactions=base_phi.size
    q_rho=packed[1+growth.size*n_reactions:].reshape(growth.size,-1)
    enzyme=enzyme_profiles(
        base_phi,multipliers[design_index],data["affinity"],data["turnover"],
        data["pathway_weight"],data["reaction_sector"],data["metabolite_sector"],
        restrictions[design_index],growth,data["metabolome_fraction"])
    states=[]
    for condition in range(growth.size):
        states.append(kineflux_state(
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
    thermo=shared_thermodynamic_fit(
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

import numpy as np


def evaluate_panel(data: dict) -> "np.ndarray":
    if not isinstance(data,dict) or "design_multiplier" not in data or "restriction" not in data:
        raise ValueError("panel data must contain design arrays")
    multiplier=np.asarray(data["design_multiplier"])
    restriction=np.asarray(data["restriction"])
    if (multiplier.ndim!=2 or multiplier.shape[0]==0 or restriction.ndim!=2
            or restriction.shape[0]!=multiplier.shape[0]):
        raise ValueError("design arrays must have the same nonzero row count")
    return np.vstack([design_record(data,design)
        for design in range(multiplier.shape[0])]).astype(np.float64)

import numpy as np


def _sd9_pick(indices,scores,tolerance):
    maximum=float(np.max(scores)); tied=indices[scores>=maximum-tolerance]
    chosen=int(np.min(tied))
    return chosen,float(scores[np.flatnonzero(indices==chosen)[0]])


def select_design(panel: "np.ndarray", tie_tolerance: float) -> "np.ndarray":
    panel=np.asarray(panel,dtype=np.float64)
    if (panel.ndim!=2 or panel.shape[0]==0 or panel.shape[1]!=13
            or not np.isfinite(panel).all() or not np.isfinite(tie_tolerance)
            or tie_tolerance<0.0
            or np.any((panel[:,0]!=0.0)&(panel[:,0]!=1.0))):
        raise ValueError("selection inputs violate the stated contract")
    eligible=np.flatnonzero(panel[:,0]==1.0)
    if eligible.size==0:
        raise ValueError("at least one design must be eligible")
    winner,winner_score=_sd9_pick(eligible,panel[eligible,1],tie_tolerance)
    remaining=eligible[eligible!=winner]
    if remaining.size==0:
        runner,runner_score=-1,-1.0
    else:
        runner,runner_score=_sd9_pick(remaining,panel[remaining,1],tie_tolerance)
    return np.array([float(winner),winner_score,float(runner),runner_score])

import numpy as np


def run_benchmark(data: dict, tie_tolerance: float) -> float:
    panel=evaluate_panel(data)
    selection=select_design(panel,tie_tolerance)
    return float(selection[1])
SCICODE_GOLD_EOF
