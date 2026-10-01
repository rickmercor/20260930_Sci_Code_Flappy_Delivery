"""
Fit one thermodynamic uncertainty state across all conditions.

The same bounded uncertainty vector must satisfy every active reaction in

every condition. Independent condition fits are therefore not equivalent.

Returns
-------
minimized worst energy, fitted uncertainty coordinates, and the zero-based limiting condition and reaction.  The limiting state and reaction are the lowest (state, reaction) pair among active reactions whose corrected energy is within 1e-9 of the shared maximum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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
    """Minimize the worst corrected energy with shared uncertainty.

    Parameters
    ----------
    fluxes
        Finite nonnegative condition-by-reaction flux matrix.
    log_concentrations
        Finite condition-by-metabolite natural-log concentrations.
    stoichiometry
        Finite metabolite-by-reaction matrix aligned to both inputs.
    standard_energy
        Finite standard-energy vector aligned to reactions.
    uncertainty_map
        Finite metabolite-by-coordinate matrix. The shared coordinate vector
        has Euclidean norm at most one.
    gas_temperature
        Positive finite value of RT.
    driving_force, design_shift
        Finite scalar additions to every corrected reaction energy.
    active_threshold
        Finite nonnegative threshold; flux strictly above it is active.

    Returns
    -------
    np.ndarray
        Worst corrected energy, fitted uncertainty coordinates, limiting
        condition index, and limiting reaction index.

    Raises
    ------
    ValueError
        If shapes, ranges or finite-value requirements are violated, no active
        reaction exists, or the constrained minimax fit does not converge.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import minimize


def _oracle_shared_thermodynamic_fit(
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
v=np.array([[1.,.5,.5],[.8,.3,.5]])
x=np.log(np.array([[.01,.02],[.008,.015]]))
s=np.array([[-1.,1.,0.],[0.,-1.,1.]])
g=np.array([-2.,-1.,-3.])
L=np.array([[.4,-.1],[-.2,.5]])
cargs=(v.copy(),x.copy(),s.copy(),g.copy(),L.copy(),2.48,.05,0.,1e-8)
gargs=(v.copy(),x.copy(),s.copy(),g.copy(),L.copy(),2.48,.05,0.,1e-8)""",
            "call": "shared_thermodynamic_fit(*cargs)",
            "gold_call": "_oracle_shared_thermodynamic_fit(*gargs)",
            "tol": 1e-6,
        },
        {
            "setup": """import numpy as np
v=np.ones((1,2))
x=np.zeros((1,1))
s=np.array([[-1.,1.]])
g=np.array([-1.,-1.])
L=np.array([[.3]])
cargs=(v.copy(),x.copy(),s.copy(),g.copy(),L.copy(),1.,0.,.2,0.)
gargs=(v.copy(),x.copy(),s.copy(),g.copy(),L.copy(),1.,0.,.2,0.)""",
            "call": "shared_thermodynamic_fit(*cargs)",
            "gold_call": "_oracle_shared_thermodynamic_fit(*gargs)",
            "tol": 1e-6,
        },
        {
            "setup": """import numpy as np
v=np.array([[1.,0.],[0.,2.]])
x=np.zeros((2,1))
s=np.array([[1.,-1.]])
g=np.array([-.5,-.7])
L=np.array([[.2]])
cargs=(v.copy(),x.copy(),s.copy(),g.copy(),L.copy(),2.,.1,0.,.5)
gargs=(v.copy(),x.copy(),s.copy(),g.copy(),L.copy(),2.,.1,0.,.5)""",
            "call": "shared_thermodynamic_fit(*cargs)",
            "gold_call": "_oracle_shared_thermodynamic_fit(*gargs)",
            "tol": 1e-6,
        },
        {
            "setup": """import numpy as np
v=np.zeros((2,2))
x=np.zeros((2,1))
s=np.array([[1.,-1.]])
g=np.array([-.5,-.7])
L=np.array([[.2]])
cargs=(v.copy(),x.copy(),s.copy(),g.copy(),L.copy(),2.,.1,0.,0.)
gargs=(v.copy(),x.copy(),s.copy(),g.copy(),L.copy(),2.,.1,0.,0.)
def case_raises(fn,*args):
    try:
        fn(*args)
    except ValueError:
        return 1.0
    return 0.0""",
            "call": "case_raises(shared_thermodynamic_fit,*cargs)",
            "gold_call": "case_raises(_oracle_shared_thermodynamic_fit,*gargs)",
            "tol": 0.0,
        },
    ]
