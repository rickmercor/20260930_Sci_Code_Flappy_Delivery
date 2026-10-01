"""
Assemble the seven-state reduced OxyPOM derivative and four diagnostic fluxes from the explicit forcing functions and mass-balance equations below.

Use the state order $[O_2,Phy,POC_L,POC_S,DOC,NH_4,NO_3]$ and forcing order $[T_0,A_T,\mathcal{P},I_0,A_I,k_0,A_k,Sat_0,\beta]$. For finite time $t$, set $\theta=2\pi t/\mathcal{P}$, $T=T_0+A_T\sin\theta$, $I=I_0+A_I\sin(\theta-0.4)$, $k_{\mathrm{air}}=k_0+A_k\cos(\theta+0.2)$, and $Sat=Sat_0-\beta T$. Raise ValueError when $\mathcal{P}\le0$ or when forcing makes $I$, $k_{\mathrm{air}}$, or $Sat$ negative. Reaeration is $A=k_{\mathrm{air}}(Sat-O_2)$.

From Step 2 obtain $[\mu,m,r,L,U_{NH_4},U_{NO_3},P,R]$, where $L=(m+\lambda)Phy$. From Step 3 obtain $[M_L,M_S,M_D,d_{L\to S},d_{L\to D},d_{S\to D},M_{\mathrm{tot}}]$. From Step 4 obtain $[N,f_{\mathrm{den}},D,N_{O_2},M_{O_2}]$. Use $a_N=parameters[0]$, $f=parameters[5]$, and $\lambda=parameters[21]$.

Assemble $dO_2/dt=A+P-R-N_{O_2}-M_{O_2}$; $dPhy/dt=Phy(\mu-m-r-\lambda)$; $dPOC_L/dt=(1-f)L-d_{L\to S}-d_{L\to D}-M_L$; $dPOC_S/dt=d_{L\to S}-d_{S\to D}-M_S$; $dDOC/dt=d_{L\to D}+d_{S\to D}-M_D$; $dNH_4/dt=a_NfL+a_NM_{\mathrm{tot}}-U_{NH_4}-N$; and $dNO_3/dt=N-D-U_{NO_3}$.

Return the derivative in the declared state order and diagnostic fluxes $[P,R,N_{O_2},M_{O_2}]$ in exactly this order.

Returns
-------
return derivative, diagnostic_fluxes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def oxypom_reduced_rhs(
    time: float,
    state: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    """Assemble the seven-state derivative and four diagnostic fluxes.

    State order:
        [O2, Phy, POC_L, POC_S, DOC, NH4, NO3]
    Forcing order:
        [T0, A_T, period, I0, A_I, k0, A_k, Sat0, beta]

    Required forcing:
        theta = 2*pi*time/period
        T = T0 + A_T*sin(theta)
        I = I0 + A_I*sin(theta - 0.4)
        k_air = k0 + A_k*cos(theta + 0.2)
        Sat = Sat0 - beta*T
        A = k_air*(Sat - O2)

    Required balances, using the ordered outputs of Steps 2-4:
        dO2 = A + P - R - N_O2 - M_O2
        dPhy = Phy*(mu - m - r - parameters[21])
        dPOC_L = (1 - parameters[5])*L - d_L_to_S - d_L_to_D - M_L
        dPOC_S = d_L_to_S - d_S_to_D - M_S
        dDOC = d_L_to_D + d_S_to_D - M_D
        dNH4 = parameters[0]*parameters[5]*L
            + parameters[0]*M_tot - U_NH4 - N
        dNO3 = N - D - U_NO3
        L from Step 2 is (m + parameters[21])*Phy.

    Parameters:
        time: Finite time in days.
        state: Seven finite nonnegative states in the declared order.
        q10_values: Six finite positive process-specific Q10 values.
        parameters: The 22-value finite positive parameter vector.
        forcing_parameters: Nine finite forcing values in the declared order.

    Returns:
        The seven-state derivative and [P, R, N_O2, M_O2].

    Raises:
        ValueError: If a contract fails, period is nonpositive, or forcing
            produces negative light, air exchange, or oxygen saturation.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_oxypom_reduced_rhs(
    time: float,
    state: "np.ndarray",
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
    forcing_parameters: "np.ndarray",
) -> "tuple[np.ndarray, np.ndarray]":
    state = np.asarray(state, dtype=np.float64)
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    forcing_parameters = np.asarray(forcing_parameters, dtype=np.float64)
    if not np.isfinite(time):
        raise ValueError("time must be finite")
    if state.shape != (7,) or not np.all(np.isfinite(state)) or np.any(state < 0.0):
        raise ValueError("state must contain seven finite nonnegative values")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("parameters must contain 22 finite positive values")
    if forcing_parameters.shape != (9,) or not np.all(np.isfinite(forcing_parameters)):
        raise ValueError("forcing_parameters must contain nine finite values")
    if forcing_parameters[2] <= 0.0:
        raise ValueError("forcing period must be positive")

    phase = 2.0 * np.pi * time / forcing_parameters[2]
    temperature = forcing_parameters[0] + forcing_parameters[1] * np.sin(phase)
    light = forcing_parameters[3] + forcing_parameters[4] * np.sin(phase - 0.4)
    k_air = forcing_parameters[5] + forcing_parameters[6] * np.cos(phase + 0.2)
    oxygen_saturation = forcing_parameters[7] - forcing_parameters[8] * temperature
    if light < 0.0 or k_air < 0.0 or oxygen_saturation < 0.0:
        raise ValueError("forcing produces a negative light, exchange, or saturation value")

    oxygen, phyto, poc_labile, poc_semilabile, doc, ammonium, nitrate = state
    phyto_rates = _oracle_phytoplankton_process_rates(
        temperature, light, phyto, ammonium, nitrate, q10_values, parameters
    )
    organic_rates = _oracle_organic_matter_transformations(
        temperature, poc_labile, poc_semilabile, doc, q10_values, parameters
    )
    nitrogen_rates = _oracle_nitrogen_oxygen_fluxes(
        temperature,
        oxygen,
        ammonium,
        nitrate,
        organic_rates[6],
        q10_values,
        parameters,
    )
    gross_rate, mortality_rate, respiration_rate = phyto_rates[:3]
    loss_flux, uptake_ammonium, uptake_nitrate = phyto_rates[3:6]
    photo_flux, respiration_flux = phyto_rates[6:8]
    mineral_labile, mineral_semilabile, mineral_doc = organic_rates[:3]
    d_labile_semilabile, d_labile_doc, d_semilabile_doc = organic_rates[3:6]
    nitrification, _, denitrification = nitrogen_rates[:3]
    nitrification_oxygen, mineralization_oxygen = nitrogen_rates[3:5]
    reaeration = k_air * (oxygen_saturation - oxygen)
    release_fraction = parameters[5]
    derivative = np.array(
        [
            reaeration + photo_flux - respiration_flux
            - nitrification_oxygen - mineralization_oxygen,
            phyto * (gross_rate - mortality_rate - respiration_rate - parameters[21]),
            (1.0 - release_fraction) * loss_flux
            - d_labile_semilabile - d_labile_doc - mineral_labile,
            d_labile_semilabile - d_semilabile_doc - mineral_semilabile,
            d_labile_doc + d_semilabile_doc - mineral_doc,
            parameters[0] * release_fraction * loss_flux
            + parameters[0] * organic_rates[6]
            - uptake_ammonium - nitrification,
            nitrification - denitrification - uptake_nitrate,
        ],
        dtype=np.float64,
    )
    diagnostic_fluxes = np.array(
        [photo_flux, respiration_flux, nitrification_oxygen, mineralization_oxygen],
        dtype=np.float64,
    )
    return derivative, diagnostic_fluxes

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return three valid and two documented invalid-input cases."""
    return [{'setup': 'import numpy as np\n'
           'initial_state=np.array([220.,.2,25.,40.,30.,.4,40.])\n'
           'reference_q10=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           'forcing_parameters=np.array([23.,2.,4.,90.,60.,.2,.05,340.,4.])\n'
           'duration=8.\n'
           'time_step=.04\n'
           'analysis_start=4.\n'
           'perturbation_fraction=.2\n'
           't=0.;state=initial_state.copy()',
  'call': '(lambda '
          'r:np.concatenate([np.asarray(r[0]).ravel(),np.asarray(r[1]).ravel()]))(oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))',
  'gold_call': '(lambda '
               'r:np.concatenate([np.asarray(r[0]).ravel(),np.asarray(r[1]).ravel()]))(_oracle_oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))'},
 {'setup': 'import numpy as np\n'
           'initial_state=np.array([220.,.2,25.,40.,30.,.4,40.])\n'
           'reference_q10=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           'forcing_parameters=np.array([23.,2.,4.,90.,60.,.2,.05,340.,4.])\n'
           'duration=8.\n'
           'time_step=.04\n'
           'analysis_start=4.\n'
           'perturbation_fraction=.2\n'
           't=1.;state=initial_state.copy();state[0]=250.',
  'call': '(lambda '
          'r:np.concatenate([np.asarray(r[0]).ravel(),np.asarray(r[1]).ravel()]))(oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))',
  'gold_call': '(lambda '
               'r:np.concatenate([np.asarray(r[0]).ravel(),np.asarray(r[1]).ravel()]))(_oracle_oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))'},
 {'setup': 'import numpy as np\n'
           'initial_state=np.array([220.,.2,25.,40.,30.,.4,40.])\n'
           'reference_q10=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           'forcing_parameters=np.array([23.,2.,4.,90.,60.,.2,.05,340.,4.])\n'
           'duration=8.\n'
           'time_step=.04\n'
           'analysis_start=4.\n'
           'perturbation_fraction=.2\n'
           't=2.5;state=initial_state.copy();state[2:5]=[0.,0.,0.]',
  'call': '(lambda '
          'r:np.concatenate([np.asarray(r[0]).ravel(),np.asarray(r[1]).ravel()]))(oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))',
  'gold_call': '(lambda '
               'r:np.concatenate([np.asarray(r[0]).ravel(),np.asarray(r[1]).ravel()]))(_oracle_oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))'},
 {'setup': 'import numpy as np\n'
           'initial_state=np.array([220.,.2,25.,40.,30.,.4,40.])\n'
           'reference_q10=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           'forcing_parameters=np.array([23.,2.,4.,90.,60.,.2,.05,340.,4.])\n'
           'duration=8.\n'
           'time_step=.04\n'
           'analysis_start=4.\n'
           'perturbation_fraction=.2\n'
           't=0.;state=initial_state[:-1]\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))',
  'gold_call': 'status(lambda: '
               '_oracle_oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))'},
 {'setup': 'import numpy as np\n'
           'initial_state=np.array([220.,.2,25.,40.,30.,.4,40.])\n'
           'reference_q10=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           'forcing_parameters=np.array([23.,2.,4.,90.,60.,.2,.05,340.,4.])\n'
           'duration=8.\n'
           'time_step=.04\n'
           'analysis_start=4.\n'
           'perturbation_fraction=.2\n'
           't=0.;state=initial_state.copy();forcing_parameters[2]=0.\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))',
  'gold_call': 'status(lambda: '
               '_oracle_oxypom_reduced_rhs(t,state,reference_q10,parameters,forcing_parameters))'}]
