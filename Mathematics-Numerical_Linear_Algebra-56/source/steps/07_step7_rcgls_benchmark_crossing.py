"""
Identify the first RCGLS attainment of a benchmark.

Normalized residual quality provides a common scale for comparing an iterative trajectory with an external sketching benchmark. The earliest attainment records when RCGLS first reaches that accuracy standard rather than merely whether a later iterate eventually does so.

Returns
-------
tuple[np.ndarray,int,float,float], the inflation history, one-based first crossing k_star, crossing inflation, and margin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rcgls_benchmark_crossing(
    residual_sq: np.ndarray, optimal_residual_sq: float, benchmark: float
) -> tuple[np.ndarray, int, float, float]:
    """Identify the first RCGLS attainment of a benchmark.

    Parameters
    ----------
    residual_sq : np.ndarray
        Nonempty one-dimensional finite array of nonnegative RCGLS squared
        residuals in update order.
    optimal_residual_sq : float
        Finite strictly positive minimum unsketched squared residual.
    benchmark : float
        Finite strictly positive residual-inflation threshold.

    Returns
    -------
    inflation : np.ndarray
        Float64 residual-inflation history ``residual_sq / optimal_residual_sq``.
    k_star : int
        One-based index of the first inflation not exceeding ``benchmark``.
    crossing_inflation : float
        Inflation factor at ``k_star``.
    margin : float
        Ratio ``crossing_inflation / benchmark``.

    Raises
    ------
    ValueError
        If inputs are invalid or if the RCGLS history never reaches the
        benchmark.
    """
    return inflation, k_star, crossing_inflation, margin

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rcgls_benchmark_crossing(residual_sq,optimal_residual_sq,benchmark):
    """Reference implementation for normalized first-crossing certification."""
    try: arr=np.asarray(residual_sq,dtype=np.float64); opt=float(optimal_residual_sq); bench=float(benchmark)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError("invalid inputs") from exc
    if arr.ndim!=1 or arr.size<1 or not np.all(np.isfinite(arr)) or np.any(arr<0): raise ValueError("invalid residual history")
    if not np.isfinite(opt) or opt<=0.0: raise ValueError("optimal residual must be positive finite")
    if not np.isfinite(bench) or bench<=0.0: raise ValueError("benchmark must be positive finite")
    inflation=arr/opt
    hits=np.flatnonzero(inflation<=bench)
    if hits.size==0: raise ValueError("benchmark is not reached")
    j=int(hits[0]); val=float(inflation[j]); margin=float(val/bench)
    return np.asarray(inflation,dtype=np.float64),j+1,val,margin

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
r=np.array([8.,6.8,5.6,4.8])""",'call':'rcgls_benchmark_crossing(r,4.0,1.5)','gold_call':'_oracle_rcgls_benchmark_crossing(r,4.0,1.5)'},
        {'setup':"""import numpy as np
r=np.array([3.,2.8,2.])""",'call':'rcgls_benchmark_crossing(r,2.0,1.5)','gold_call':'_oracle_rcgls_benchmark_crossing(r,2.0,1.5)'},
        {'setup':"""import numpy as np
r=np.array([9.,4.5,3.])""",'call':'rcgls_benchmark_crossing(r,3.0,1.0)','gold_call':'_oracle_rcgls_benchmark_crossing(r,3.0,1.0)'},
        {'setup':"""import numpy as np
r=np.array([8.,7.2])
def run_model():
    try: rcgls_benchmark_crossing(r,4.,1.5); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_benchmark_crossing(r,4.,1.5); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
r=np.array([1.,np.nan])
def run_model():
    try: rcgls_benchmark_crossing(r,1.,1.5); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_benchmark_crossing(r,1.,1.5); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
r=np.array([1.])
def run_model():
    try: rcgls_benchmark_crossing(r,0.,1.5); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_benchmark_crossing(r,0.,1.5); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
