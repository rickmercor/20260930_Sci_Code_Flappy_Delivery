"""
Compute eight ordered phytoplankton growth, mortality, respiration, loss, nutrient-uptake, and oxygen-budget values from the explicit formulas below.

This step is self-contained. Use the Q10 order
$[Q_{10,\mu},Q_{10,m},Q_{10,r},Q_{10,\mathrm{Min}},Q_{10,\mathrm{Nit}},Q_{10,I}]$ and obtain the six temperature factors $\tau$ from Step 1. Map $parameters[0]=a_N$, $[1]=\mu^*$, $[2]=m^*$, $[3]=I^*$, $[4]=K_N$, $[12]=r^*$, $[13]=c$, $[14]=\pi$, and $[21]=\lambda$.

Use $I_T^*=I^*\tau_I(T)$ and the saturating light limitation $f_I=1-\exp(-I/I_T^*)$. Let $DIN=NH_4+NO_3$ and raise ValueError when $DIN\le 0$. Then $f_N=DIN/(DIN+K_N)$ and $\mu=f_I f_N\tau_\mu(T)\mu^*$.

Use the exact mortality bands $m=m^*\tau_m(T)$ for $T\ge20$, $m=m^*$ for $5<T<20$, and $m=0.33m^*$ for $T\le5$. Respiration is $r=\pi\mu+(1-\pi)r^*\tau_r(T)$. The complete mortality-plus-lysis loss is $L=(m+\lambda)Phy$.

Total nitrogen uptake is $U_N=a_N(\mu-r)Phy$. Set $\chi_{NH_4}=1$ when $NH_4\ge0.7$ and $\chi_{NH_4}=NH_4/DIN$ otherwise. Then $U_{NH_4}=\chi_{NH_4}U_N$ and $U_{NO_3}=(1-\chi_{NH_4})U_N$.

The oxygen-budget fluxes are $P=\mu Phy$ and $R=(r+cm)Phy$. Return $[\mu,m,r,L,U_{NH_4},U_{NO_3},P,R]$ as float64 in exactly this order. Do not substitute Steele light limitation or another mortality, respiration, or ammonium-preference convention.

Returns
-------
return rates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def phytoplankton_process_rates(
    temperature: float,
    light: float,
    phytoplankton: float,
    ammonium: float,
    nitrate: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    """Compute the eight reduced phytoplankton process values.

    Required method:
        tau = temperature_response_factors(temperature, q10_values)
        I_star_T = parameters[3] * tau[5]
        f_light = 1 - exp(-light / I_star_T)
        DIN = ammonium + nitrate; DIN must be positive
        f_nutrient = DIN / (DIN + parameters[4])
        mu = f_light * f_nutrient * tau[0] * parameters[1]
        m = parameters[2] * tau[1] for T >= 20,
            parameters[2] for 5 < T < 20, and
            0.33 * parameters[2] for T <= 5
        r = parameters[14] * mu + (1 - parameters[14]) * parameters[12] * tau[2]
        L = (m + parameters[21]) * phytoplankton
        U_N = parameters[0] * (mu - r) * phytoplankton
        chi_NH4 = 1 if ammonium >= 0.7 else ammonium / DIN
        U_NH4 = chi_NH4 * U_N; U_NO3 = (1 - chi_NH4) * U_N
        P = mu * phytoplankton
        R = (r + parameters[13] * m) * phytoplankton

    Parameters:
        temperature: Finite water temperature in degrees Celsius.
        light: Nonnegative incident light.
        phytoplankton: Nonnegative phytoplankton carbon.
        ammonium: Nonnegative ammonium.
        nitrate: Nonnegative nitrate.
        q10_values: Six finite positive values in growth, mortality,
            respiration, mineralization, nitrification, optimal-light order.
        parameters: The 22-value finite positive parameter vector.

    Returns:
        float64 [mu, m, r, L, U_NH4, U_NO3, P, R].

    Raises:
        ValueError: If shapes, finiteness, or positivity contracts fail,
            or ammonium plus nitrate is zero.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_phytoplankton_process_rates(
    temperature: float,
    light: float,
    phytoplankton: float,
    ammonium: float,
    nitrate: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    values = np.asarray(
        [temperature, light, phytoplankton, ammonium, nitrate], dtype=np.float64
    )
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values[1:] < 0.0):
        raise ValueError("temperature must be finite and other scalar inputs nonnegative")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)):
        raise ValueError("parameters must contain 22 finite values")
    if np.any(parameters <= 0.0):
        raise ValueError("parameters must be positive")

    a_n, mu_star, mortality_star, light_star, k_n = parameters[:5]
    respiration_star, respiration_factor, growth_respiration_fraction = parameters[12:15]
    lysis_rate = parameters[21]
    tau = _oracle_temperature_response_factors(temperature, q10_values)
    light_star_t = light_star * tau[5]
    f_light = 1.0 - np.exp(-light / light_star_t)
    din = ammonium + nitrate
    if din <= 0.0:
        raise ValueError("ammonium plus nitrate must be positive")
    f_nutrient = din / (din + k_n)
    gross_rate = f_light * f_nutrient * tau[0] * mu_star
    if temperature >= 20.0:
        mortality_rate = mortality_star * tau[1]
    elif temperature > 5.0:
        mortality_rate = mortality_star
    else:
        mortality_rate = mortality_star * 0.33
    respiration_rate = (
        growth_respiration_fraction * gross_rate
        + (1.0 - growth_respiration_fraction) * respiration_star * tau[2]
    )
    loss_flux = (mortality_rate + lysis_rate) * phytoplankton
    uptake_total = a_n * (gross_rate - respiration_rate) * phytoplankton
    fraction_ammonium = 1.0 if ammonium >= 0.7 else ammonium / din
    ammonium_uptake = fraction_ammonium * uptake_total
    nitrate_uptake = (1.0 - fraction_ammonium) * uptake_total
    photosynthesis_flux = gross_rate * phytoplankton
    respiration_flux = (
        respiration_rate + respiration_factor * mortality_rate
    ) * phytoplankton
    rates = np.array(
        [
            gross_rate,
            mortality_rate,
            respiration_rate,
            loss_flux,
            ammonium_uptake,
            nitrate_uptake,
            photosynthesis_flux,
            respiration_flux,
        ],
        dtype=np.float64,
    )
    return rates

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return three valid and two documented invalid-input cases."""
    return [{'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=23.;I=90.;phy=.2;nh4=.4;no3=40.',
  'call': 'phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters).tolist()',
  'gold_call': '_oracle_phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=20.;I=0.;phy=2.;nh4=1.;no3=4.',
  'call': 'phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters).tolist()',
  'gold_call': '_oracle_phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=4.;I=150.;phy=1.;nh4=.2;no3=3.',
  'call': 'phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters).tolist()',
  'gold_call': '_oracle_phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=20.;I=90.;phy=1.;nh4=0.;no3=0.\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters))',
  'gold_call': 'status(lambda: _oracle_phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters))'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=20.;I=90.;phy=-1.;nh4=1.;no3=3.\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters))',
  'gold_call': 'status(lambda: _oracle_phytoplankton_process_rates(t,I,phy,nh4,no3,q,parameters))'}]
