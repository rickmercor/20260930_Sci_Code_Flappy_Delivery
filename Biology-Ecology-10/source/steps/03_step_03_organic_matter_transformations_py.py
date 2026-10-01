"""
Compute seven ordered mineralization and transformation values for labile POC, semilabile POC, and DOC from the explicit rate laws below.

Use the mineralization temperature factor $\tau_{\mathrm{Min}}(T)$ from Step 1. Map $parameters[6]=k_{\mathrm{Min,L}}$, $[7]=k_{\mathrm{Min,S}}$, $[8]=k_{\mathrm{Min,D}}$, $[9]=\kappa_{L\to S}$, $[10]=\kappa_{L\to D}$, and $[11]=\kappa_{S\to D}$.

Calculate $M_L=k_{\mathrm{Min,L}}\tau_{\mathrm{Min}}(T)POC_L$, $M_S=k_{\mathrm{Min,S}}\tau_{\mathrm{Min}}(T)POC_S$, and $M_D=k_{\mathrm{Min,D}}\tau_{\mathrm{Min}}(T)DOC$.

The transformation fluxes are fractions of the originating mineralization flux: $d_{L\to S}=\kappa_{L\to S}M_L$, $d_{L\to D}=\kappa_{L\to D}M_L$, and $d_{S\to D}=\kappa_{S\to D}M_S$. Let $M_{\mathrm{tot}}=M_L+M_S+M_D$.

Return $[M_L,M_S,M_D,d_{L\to S},d_{L\to D},d_{S\to D},M_{\mathrm{tot}}]$ as float64 in exactly this order.

Returns
-------
return transformations
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def organic_matter_transformations(
    temperature: float,
    poc_labile: float,
    poc_semilabile: float,
    doc: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    """Compute mineralization and transformation fluxes for three pools.

    Required method:
        tau_min = temperature_response_factors(temperature, q10_values)[3]
        M_L = parameters[6] * tau_min * poc_labile
        M_S = parameters[7] * tau_min * poc_semilabile
        M_D = parameters[8] * tau_min * doc
        d_L_to_S = parameters[9] * M_L
        d_L_to_D = parameters[10] * M_L
        d_S_to_D = parameters[11] * M_S
        M_tot = M_L + M_S + M_D

    Parameters:
        temperature: Finite water temperature in degrees Celsius.
        poc_labile: Nonnegative labile POC.
        poc_semilabile: Nonnegative semilabile POC.
        doc: Nonnegative DOC.
        q10_values: Six finite positive process-specific Q10 values.
        parameters: The 22-value finite positive parameter vector.

    Returns:
        float64 [M_L, M_S, M_D, d_L_to_S, d_L_to_D, d_S_to_D, M_tot].

    Raises:
        ValueError: If shapes, finiteness, or positivity contracts fail.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_organic_matter_transformations(
    temperature: float,
    poc_labile: float,
    poc_semilabile: float,
    doc: float,
    q10_values: "np.ndarray",
    parameters: "np.ndarray",
) -> "np.ndarray":
    pools = np.asarray([poc_labile, poc_semilabile, doc], dtype=np.float64)
    q10_values = np.asarray(q10_values, dtype=np.float64)
    parameters = np.asarray(parameters, dtype=np.float64)
    if not np.isfinite(temperature) or not np.all(np.isfinite(pools)) or np.any(pools < 0.0):
        raise ValueError("temperature must be finite and organic pools nonnegative")
    if q10_values.shape != (6,) or not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must contain six finite positive values")
    if parameters.shape != (22,) or not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("parameters must contain 22 finite positive values")
    tau_min = _oracle_temperature_response_factors(temperature, q10_values)[3]
    mineralization = parameters[6:9] * tau_min * pools
    d_labile_semilabile = parameters[9] * mineralization[0]
    d_labile_doc = parameters[10] * mineralization[0]
    d_semilabile_doc = parameters[11] * mineralization[1]
    transformations = np.array(
        [
            mineralization[0],
            mineralization[1],
            mineralization[2],
            d_labile_semilabile,
            d_labile_doc,
            d_semilabile_doc,
            np.sum(mineralization),
        ],
        dtype=np.float64,
    )
    return transformations

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return three valid and two documented invalid-input cases."""
    return [{'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=23.;pl=25.;ps=40.;doc=30.',
  'call': 'organic_matter_transformations(t,pl,ps,doc,q,parameters).tolist()',
  'gold_call': '_oracle_organic_matter_transformations(t,pl,ps,doc,q,parameters).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=20.;pl=0.;ps=0.;doc=0.',
  'call': 'organic_matter_transformations(t,pl,ps,doc,q,parameters).tolist()',
  'gold_call': '_oracle_organic_matter_transformations(t,pl,ps,doc,q,parameters).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=10.;pl=1.;ps=3.;doc=7.',
  'call': 'organic_matter_transformations(t,pl,ps,doc,q,parameters).tolist()',
  'gold_call': '_oracle_organic_matter_transformations(t,pl,ps,doc,q,parameters).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           't=20.;pl=-1.;ps=2.;doc=3.\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: organic_matter_transformations(t,pl,ps,doc,q,parameters))',
  'gold_call': 'status(lambda: _oracle_organic_matter_transformations(t,pl,ps,doc,q,parameters))'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,4.4,1.1,1.2,1.12,1.04])\n'
           'parameters=np.array([.151,1.2,.5,30.,.36,.5,.1,.01,.03,.5,.5,1.,.045,.7,.065,28.6,36.,31.3,36.,94.,63.,.05])\n'
           'parameters=parameters[:-1]\n'
           't=20.;pl=1.;ps=2.;doc=3.\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: organic_matter_transformations(t,pl,ps,doc,q,parameters))',
  'gold_call': 'status(lambda: _oracle_organic_matter_transformations(t,pl,ps,doc,q,parameters))'}]
