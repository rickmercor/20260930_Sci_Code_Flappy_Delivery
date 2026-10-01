"""
Compute five ordered nitrification, denitrification, and oxygen-consumption values from the explicit redox rate laws below.

Use $\tau_{\mathrm{Min}}(T)$ and $\tau_{\mathrm{Nit}}(T)$ from Step 1. Map $parameters[15]=k_{\mathrm{Nit}}$, $[16]=K_{\mathrm{NH_4,Nit}}$, $[17]=K_{\mathrm{O_2,Nit}}$, $[18]=K_{\mathrm{NO_3}}$, $[19]=K_{\mathrm{O_2,den}}$, and $[20]=K'_{\mathrm{O_2,min}}$.

The nitrification flux is $N=k_{\mathrm{Nit}}\tau_{\mathrm{Nit}}(T)\,[NH_4/(NH_4+K_{\mathrm{NH_4,Nit}})]\,[O_2/(O_2+K_{\mathrm{O_2,Nit}})]$.

Calculate $\phi_N=[NO_3/(NO_3+K_{\mathrm{NO_3}})]\,[1-O_2/(O_2+K_{\mathrm{O_2,den}})]\,\tau_{\mathrm{Min}}(T)$ and $\phi_{O_2}=[O_2/(O_2+K'_{\mathrm{O_2,min}})]\,\tau_{\mathrm{Min}}(T)$. Then $f_{\mathrm{den}}=\phi_N/(\phi_N+\phi_{O_2})$ and $D=f_{\mathrm{den}}M_{\mathrm{tot}}$.

Nitrification consumes two oxygen units per ammonium unit: $N_{O_2}=2N$. The mineralization oxygen sink is $M_{O_2}=(1-f_{\mathrm{den}})M_{\mathrm{tot}}$.

Return $[N,f_{\mathrm{den}},D,N_{O_2},M_{O_2}]$ as float64 in exactly this order. The supplied valid configurations keep $\phi_N+\phi_{O_2}>0$.

Returns
-------
return fluxes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nitrogen_oxygen_fluxes(
    temperature: float,
    oxygen: float,
    ammonium: float,
    nitrate: float,
    total_mineralization: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    """Compute nitrification, denitrification, and oxygen sinks.

    Required method:
        tau = temperature_response_factors(temperature, q10_values)
        N = parameters[15] * tau[4]
            * ammonium / (ammonium + parameters[16])
            * oxygen / (oxygen + parameters[17])
        phi_N = nitrate / (nitrate + parameters[18])
            * (1 - oxygen / (oxygen + parameters[19])) * tau[3]
        phi_O2 = oxygen / (oxygen + parameters[20]) * tau[3]
        f_den = phi_N / (phi_N + phi_O2)
        D = f_den * total_mineralization
        N_O2 = 2 * N
        M_O2 = (1 - f_den) * total_mineralization

    Parameters:
        temperature: Finite water temperature in degrees Celsius.
        oxygen: Nonnegative dissolved oxygen.
        ammonium: Nonnegative ammonium.
        nitrate: Nonnegative nitrate.
        total_mineralization: Nonnegative total carbon mineralization.
        q10_values: Six finite positive process-specific Q10 values.
        parameters: The 22-value finite positive parameter vector.

    Returns:
        float64 [N, f_den, D, N_O2, M_O2].

    Raises:
        ValueError: If shapes, finiteness, or positivity contracts fail.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_nitrogen_oxygen_fluxes(
    temperature: float,
    oxygen: float,
    ammonium: float,
    nitrate: float,
    total_mineralization: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    values = np.asarray(
        [temperature, oxygen, ammonium, nitrate, total_mineralization],
        dtype=np.float64,
    )
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values[1:] < 0.0):
        raise ValueError("temperature must be finite and pool/flux inputs nonnegative")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("parameters must contain 22 finite positive values")
    tau = _oracle_temperature_response_factors(temperature, q10_values)
    nitrification = (
        parameters[15]
        * tau[4]
        * ammonium
        / (ammonium + parameters[16])
        * oxygen
        / (oxygen + parameters[17])
    )
    phi_nitrate = (
        nitrate
        / (nitrate + parameters[18])
        * (1.0 - oxygen / (oxygen + parameters[19]))
        * tau[3]
    )
    phi_oxygen = oxygen / (oxygen + parameters[20]) * tau[3]
    denitrification_fraction = phi_nitrate / (phi_nitrate + phi_oxygen)
    denitrification_flux = denitrification_fraction * total_mineralization
    nitrification_oxygen_sink = 2.0 * nitrification
    mineralization_oxygen_sink = (
        1.0 - denitrification_fraction
    ) * total_mineralization
    fluxes = np.array(
        [
            nitrification,
            denitrification_fraction,
            denitrification_flux,
            nitrification_oxygen_sink,
            mineralization_oxygen_sink,
        ],
        dtype=np.float64,
    )
    return fluxes

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return three valid and two documented invalid-input cases."""
    return [{'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=23.;o2=220.;nh4=.4;no3=40.;total=3.',
  'call': 'nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters).tolist()',
  'gold_call': '_oracle_nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=20.;o2=0.;nh4=2.;no3=5.;total=4.',
  'call': 'nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters).tolist()',
  'gold_call': '_oracle_nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=10.;o2=50.;nh4=0.;no3=10.;total=0.',
  'call': 'nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters).tolist()',
  'gold_call': '_oracle_nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=20.;o2=-1.;nh4=1.;no3=1.;total=1.\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters))',
  'gold_call': 'status(lambda: _oracle_nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters))'},
 {'setup': 'import numpy as np\n'
           'q=np.ones(5)\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=20.;o2=1.;nh4=1.;no3=1.;total=1.\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters))',
  'gold_call': 'status(lambda: _oracle_nitrogen_oxygen_fluxes(t,o2,nh4,no3,total,q,parameters))'}]
