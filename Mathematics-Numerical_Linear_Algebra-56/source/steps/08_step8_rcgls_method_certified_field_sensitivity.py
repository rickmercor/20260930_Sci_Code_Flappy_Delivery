"""
Combine a scale-certified RCGLS trajectory with two field benchmarks.

The final stage combines the validated RCGLS trajectory with the field-dependent sketching benchmarks to summarize when the same residual path first satisfies each accuracy standard. It uses only trajectory-level quantities that remain meaningful after the earlier consistency checks.

Returns
-------
float, the real/complex first-crossing margin ratio.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def rcgls_method_certified_field_sensitivity(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    indices: tuple[int, ...] | list[int],
    ell: int,
) -> float:
    """Combine a scale-certified RCGLS trajectory with two field benchmarks.

    Parameters
    ----------
    A : np.ndarray
        Finite real matrix with shape ``(n, d)``.
    b : np.ndarray
        Finite real vector with shape ``(n,)``.
    x0 : np.ndarray
        Finite real initial iterate with shape ``(d,)``.
    indices : tuple[int, ...] or list[int]
        Realized one-based coordinate stream.
    ell : int
        Random-orthonormal sketch-and-solve embedding dimension.

    Returns
    -------
    value : float
        Ratio of the real and complex first-crossing margins for the selected
        RCGLS trajectory.

    Raises
    ------
    ValueError
        If any component input is invalid, the unsketched optimum has zero
        residual, either benchmark theorem domain is violated, RCGLS breaks
        down, either benchmark is not reached, or the final scalar is nonfinite.
    """
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rcgls_method_certified_field_sensitivity(A,b,x0,indices,ell):
    """Reference orchestrator for scale-certified method field sensitivity."""
    A64,b64,x064=_oracle_prepare_least_squares_problem(A,b,x0)
    if not isinstance(indices,(tuple,list)) or len(indices)<1:
        raise ValueError("indices must be nonempty tuple/list")
    pattern=np.asarray([2.0,0.5,4.0,0.25,8.0,0.125],dtype=np.float64)
    scales=np.resize(pattern,len(indices))
    residual_sq,_,_,_,_,iterate_defects,residual_defects=_oracle_rcgls_residual_history(A64,b64,x064,indices,scales)
    if float(np.max(iterate_defects))>5e-10 or float(np.max(residual_defects))>5e-10:
        raise ValueError("equivalent sketch scaling changed RCGLS trajectory")
    opt,_,bench_real=_oracle_least_squares_orthonormal_reference(A64,b64,ell,"real")
    opt_c,_,bench_complex=_oracle_least_squares_orthonormal_reference(A64,b64,ell,"complex")
    if not np.isclose(opt,opt_c,rtol=0.0,atol=0.0):
        raise ValueError("inconsistent unsketched reference")
    _,_,_,margin_real=_oracle_rcgls_benchmark_crossing(residual_sq,opt,bench_real)
    _,_,_,margin_complex=_oracle_rcgls_benchmark_crossing(residual_sq,opt,bench_complex)
    value=float(margin_real/margin_complex)
    if not np.isfinite(value): raise ValueError("nonfinite field sensitivity")
    return value

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {'setup':"""import numpy as np
A=np.array([[1.,0.],[0.,1.],[1.,1.],[2.,-1.],[1.,2.]])
b=np.array([1.,-1.,.2,2.,.5]); x0=np.zeros(2); indices=[2,1,2]""",'call':'rcgls_method_certified_field_sensitivity(A,b,x0,indices,4)','gold_call':'_oracle_rcgls_method_certified_field_sensitivity(A,b,x0,indices,4)'},
        {'setup':"""import numpy as np
A=np.array([[2.,1.],[1.,-1.],[0.,2.],[3.,1.],[1.,.5],[2.,-2.]])
b=np.array([1.,2.,-1.,0.,.5,1.]); x0=np.array([.1,-.2]); indices=[2,1,2,1]""",'call':'rcgls_method_certified_field_sensitivity(A,b,x0,indices,5)','gold_call':'_oracle_rcgls_method_certified_field_sensitivity(A,b,x0,indices,5)'},
        {'setup':"""import numpy as np
A=np.array([[-.98912135,-.36778665,1.28792526,.19397442],[.92023090,.57710379,-.63646365,.54195222],[-.31659545,-.32238912,.09716732,-1.52593041],[1.19216610,-.67108968,1.00026942,.13632112],[1.53203308,-.65996941,-.31179486,.33776913],[-2.20747110,.82792144,1.54163039,1.12680679],[.75476964,-.14597789,1.28190223,1.07403062],[.39262084,.00511431,-.36176687,-1.23023220]])
b=np.array([1.22622929,-2.17204389,-.37014735,.16438007,.85988118,1.76166124,.99332378,-.29152143]); x0=np.zeros(4); indices=[3,1,2,3,4,1]""",'call':'rcgls_method_certified_field_sensitivity(A,b,x0,indices,7)','gold_call':'_oracle_rcgls_method_certified_field_sensitivity(A,b,x0,indices,7)'},
        {'setup':"""import numpy as np
A=np.eye(2); b=np.zeros(2); x0=np.zeros(2); indices=[1,2]
def run_model():
    try: rcgls_method_certified_field_sensitivity(A,b,x0,indices,2); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_method_certified_field_sensitivity(A,b,x0,indices,2); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.eye(2); b=np.ones(2); x0=np.zeros(2); indices=[]
def run_model():
    try: rcgls_method_certified_field_sensitivity(A,b,x0,indices,2); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_method_certified_field_sensitivity(A,b,x0,indices,2); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
        {'setup':"""import numpy as np
A=np.eye(2); b=np.array([1.,2.]); x0=np.zeros(2); indices=[1,2]
def run_model():
    try: rcgls_method_certified_field_sensitivity(A,b,x0,indices,3); return 0
    except ValueError: return 1
    except Exception: return 2
def run_gold():
    try: _oracle_rcgls_method_certified_field_sensitivity(A,b,x0,indices,3); return 0
    except ValueError: return 1
    except Exception: return 2""",'call':'run_model()','gold_call':'run_gold()'},
    ]
