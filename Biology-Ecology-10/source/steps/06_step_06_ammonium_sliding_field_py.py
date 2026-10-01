"""
Evaluate the one-sided ammonium rates and the Filippov sliding vector field on the critical-ammonium switching surface of the reduced OxyPOM model.

The ammonium-preference rule of Step 2 is discontinuous at the critical concentration $c=0.7$: for $NH_4<c$ the ammonium share of nitrogen uptake is $\chi^-=NH_4/(NH_4+NO_3)$, while for $NH_4\ge c$ it is $\chi^+=1$. The Step-5 right-hand side therefore has two smooth one-sided branches, $\mathbf{f}^-$ (share $\chi^-$) and $\mathbf{f}^+$ (share $\chi^+$), which differ only in how the total uptake $U_N$ is split between the ammonium and nitrate equations. On the switching surface $\Sigma=\{NH_4=c\}$, both branches are evaluated at the same state with $NH_4=c$, so $\chi^-=c/(c+NO_3)$.

 

The normal components $f^-_{NH_4}$ and $f^+_{NH_4}$ decide the local behaviour: $\Sigma$ is attracting when $f^-_{NH_4}>0>f^+_{NH_4}$. The Filippov sliding field is the convex combination $\mathbf{f}_s=(1-\alpha)\mathbf{f}^-+\alpha\mathbf{f}^+$ whose ammonium component vanishes, and its equivalent ammonium share is $\chi_{\mathrm{eq}}=(1-\alpha)\chi^-+\alpha\chi^+$. The same formula defines the smooth extension of $\mathbf{f}_s$, with $\alpha$ outside $[0,1]$, wherever the two normal components differ; the extension is needed for RK4 stages evaluated near the edge of the sliding region.

Returns
-------
return sliding_result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ammonium_sliding_field(
    time: float,
    state: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
) -> "np.ndarray":
    """Return the one-sided normal rates, sliding coefficient, share, and field.
 
    The ammonium entry of state is ignored and replaced by the critical
    concentration c = 0.7; every other entry is used as supplied.
 
    Parameters:
        time: Finite time in days, used for the Step-5 forcing.
        state: Seven finite nonnegative values [O2, Phy, POC_L, POC_S, DOC, NH4, NO3].
        q10_values: Six finite positive Q10 values in the declared process order.
        parameters: The 22-value finite positive parameter vector.
        forcing_parameters: Nine finite forcing values in the Step-5 order.
 
    Returns:
        float64 array of length 11, [f_minus_NH4, f_plus_NH4, alpha, chi_eq,
        f_s[0], ..., f_s[6]], where f_s is the (possibly extended) sliding
        field in the declared state order with an ammonium component of
        exactly zero. alpha lies in (0, 1) exactly when the surface is
        attracting.
 
    Raises:
        ValueError: If an input contract fails, or if f_minus_NH4 equals
            f_plus_NH4 so that no sliding combination exists.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
 
def _critical_ammonium():
    return 0.7
 
 
def _environment(time, forcing_parameters):
    phase = 2.0 * np.pi * time / forcing_parameters[2]
    temperature = forcing_parameters[0] + forcing_parameters[1] * np.sin(phase)
    light = forcing_parameters[3] + forcing_parameters[4] * np.sin(phase - 0.4)
    return temperature, light
 
 
def _oracle_ammonium_sliding_field(
    time: float,
    state: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
) -> "np.ndarray":
    surface_state = np.array(state, dtype=np.float64)
    if surface_state.shape != (7,):
        raise ValueError("state must contain seven values")
    surface_state[5] = _critical_ammonium()
    forcing_parameters = np.asarray(forcing_parameters, dtype=np.float64)
    # Step 5 validates every contract; at NH4 = c it evaluates the chi = 1 branch.
    f_plus = _oracle_oxypom_reduced_rhs(
        time, surface_state, q10_values, parameters, forcing_parameters
    )[0]
    temperature, light = _environment(time, forcing_parameters)
    rates = _oracle_phytoplankton_process_rates(
        temperature,
        light,
        surface_state[1],
        surface_state[5],
        surface_state[6],
        q10_values,
        parameters,
    )
    total_uptake = rates[4] + rates[5]
    chi_minus = _critical_ammonium() / (_critical_ammonium() + surface_state[6])
    f_minus = f_plus.copy()
    f_minus[5] += (1.0 - chi_minus) * total_uptake
    f_minus[6] -= (1.0 - chi_minus) * total_uptake
    gap = f_minus[5] - f_plus[5]
    if gap == 0.0:
        raise ValueError("the one-sided ammonium rates coincide; no sliding combination exists")
    alpha = f_minus[5] / gap
    sliding = (1.0 - alpha) * f_minus + alpha * f_plus
    sliding[5] = 0.0
    chi_eq = (1.0 - alpha) * chi_minus + alpha
    return np.concatenate([[f_minus[5], f_plus[5], alpha, chi_eq], sliding]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return attracting, non-attracting, and documented error cases."""
    base = (
        "import numpy as np\n"
        "q=np.array([1.4,4.4,1.1,1.2,1.12,1.04])\n"
        "parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n"
        "forcing=np.array([23.,2.,4.,90.,60.,.2,.05,340.,4.])\n"
    )
    status = (
        "def status(fn):\n"
        "    try: fn(); return 0\n"
        "    except ValueError: return 1\n"
        "    except RuntimeError: return 2\n"
        "    except Exception: return 3\n"
    )
    call = "ammonium_sliding_field(t,state.copy(),q,parameters,forcing).tolist()"
    gold = "_oracle_ammonium_sliding_field(t,state.copy(),q,parameters,forcing).tolist()"
    return [
        # Attracting surface in mid-cycle.
        {"setup": base + "t=0.5;state=np.array([270.,58.,160.,20.,90.,.7,7.5])", "call": call, "gold_call": gold},
        # Ammonium entry ignored (supplied below c); still evaluated on the surface.
        {"setup": base + "t=3.1;state=np.array([262.,41.,120.,35.,70.,.33,15.])", "call": call, "gold_call": gold},
        # Nitrate nearly exhausted: both one-sided rates negative, alpha < 0 (the solution leaves downward).
        {"setup": base + "t=3.9;state=np.array([279.97,70.33,105.02,78.57,71.36,.7,.069])", "call": call, "gold_call": gold},
        # Weak uptake: both one-sided rates positive, alpha > 1 (the solution crosses upward).
        {"setup": base + "t=0.2;state=np.array([240.,6.,80.,40.,60.,.7,25.])", "call": call, "gold_call": gold},
        # No uptake (Phy = 0): one-sided rates coincide, no sliding combination.
        {
            "setup": base + "t=0.5;state=np.array([270.,0.,160.,20.,90.,.7,7.5])\n" + status,
            "call": "status(lambda: ammonium_sliding_field(t,state.copy(),q,parameters,forcing))",
            "gold_call": "status(lambda: _oracle_ammonium_sliding_field(t,state.copy(),q,parameters,forcing))",
        },
        {
            "setup": base + "t=0.5;state=np.array([270.,58.,160.,20.,90.,.7,-1.])\n" + status,
            "call": "status(lambda: ammonium_sliding_field(t,state.copy(),q,parameters,forcing))",
            "gold_call": "status(lambda: _oracle_ammonium_sliding_field(t,state.copy(),q,parameters,forcing))",
        },
    ]
