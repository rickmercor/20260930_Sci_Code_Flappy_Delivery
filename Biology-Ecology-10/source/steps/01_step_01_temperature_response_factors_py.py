"""
Compute process-specific temperature multipliers from Q10 coefficients.

Equation (1) uses tau_K(T) = Q10,K^((T - T*)/10) with T* = 20 degrees Celsius. The declared process order keeps each multiplier attached to the intended rate.

Returns
-------
return factors
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def temperature_response_factors(
    temperature: float,
    q10_values: "np.ndarray",
    reference_temperature: float = 20.0,
) -> "np.ndarray":
    """Compute process-specific temperature multipliers from Q10 coefficients.

    Parameters:
        temperature: Finite water temperature in degrees Celsius.
        q10_values: Six finite positive process-specific Q10 values, except Step 1 accepts any nonempty length.
        reference_temperature: Finite reference temperature in degrees Celsius.

    Returns:
        A float64 multiplier vector in the input Q10 order.

    Raises:
        ValueError: If temperatures or Q10 values violate shape, finiteness, or positivity contracts.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_temperature_response_factors(
    temperature: float,
    q10_values: "np.ndarray",
    reference_temperature: float = 20.0,
) -> "np.ndarray":
    temperature = float(temperature)
    q10_values = np.asarray(q10_values, dtype=np.float64)
    reference_temperature = float(reference_temperature)
    if not np.isfinite(temperature) or not np.isfinite(reference_temperature):
        raise ValueError("temperatures must be finite")
    if q10_values.ndim != 1 or q10_values.size == 0:
        raise ValueError("q10_values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(q10_values)) or np.any(q10_values <= 0.0):
        raise ValueError("q10_values must be finite and positive")
    factors = q10_values ** ((temperature - reference_temperature) / 10.0)
    return factors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return three valid and two documented invalid-input cases."""
    return [{'setup': 'import numpy as np\nt=25.;q=np.array([1.2,4.4,1.1])',
  'call': 'temperature_response_factors(t,q).tolist()',
  'gold_call': '_oracle_temperature_response_factors(t,q).tolist()'},
 {'setup': 'import numpy as np\nt=20.;q=np.array([1.2,4.4,1.1,1.04])',
  'call': 'temperature_response_factors(t,q).tolist()',
  'gold_call': '_oracle_temperature_response_factors(t,q).tolist()'},
 {'setup': 'import numpy as np\nt=10.;q=np.array([.8,1.,2.5]);r=18.',
  'call': 'temperature_response_factors(t,q,r).tolist()',
  'gold_call': '_oracle_temperature_response_factors(t,q,r).tolist()'},
 {'setup': 'import numpy as np\n'
           'q=np.array([1.2,0.])\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: temperature_response_factors(20.,q))',
  'gold_call': 'status(lambda: _oracle_temperature_response_factors(20.,q))'},
 {'setup': 'import numpy as np\n'
           'q=np.ones((2,2))\n'
           'def status(fn):\n'
           '    try: fn(); return 0\n'
           '    except ValueError: return 1\n'
           '    except RuntimeError: return 2\n'
           '    except Exception: return 3\n',
  'call': 'status(lambda: temperature_response_factors(20.,q))',
  'gold_call': 'status(lambda: _oracle_temperature_response_factors(20.,q))'}]
