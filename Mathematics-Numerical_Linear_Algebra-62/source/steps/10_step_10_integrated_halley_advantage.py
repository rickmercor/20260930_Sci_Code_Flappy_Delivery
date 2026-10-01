"""
Run the complete pipeline and return the final comparison diagnostics.

The final comparison joins two correction mechanisms that share the same lower-precision starting approximation but follow different structured linearizations. End-to-end correctness therefore requires consistency of precision transitions, structured solves, higher-order diagnostics, and residual evaluation across all preceding stages.

Returns
-------
tuple[float, float, float, float], Halley advantage, directional Schwarzian norm, polarized Schwarzian norm, and Halley residual norm
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def integrated_halley_advantage(A: np.ndarray, T: np.ndarray, Q: np.ndarray) -> tuple[float,float,float,float]:
    """Run the complete pipeline and return the final comparison diagnostics.

    Parameters
    ----------
    A : np.ndarray
        Original finite nonempty square matrix.
    T : np.ndarray
        Supplied compatible binary32 upper-triangular Schur factor.
    Q : np.ndarray
        Supplied compatible binary32 Schur-vector factor.

    Returns
    -------
    advantage : float
        Ratio of the mixed-precision refinement residual to the matrix-Halley residual.
    schwarzian_norm : float
        Repeated-direction matrix-Schwarzian norm.
    polarized_norm : float
        Mixed-direction polarized matrix-Schwarzian norm.
    halley_residual : float
        Residual norm after the Halley update.

    Raises
    ------
    ValueError
        If supplied data violate a required domain condition, a required Fréchet operator is singular, or the final ratio is undefined."""
    return advantage, schwarzian_norm, polarized_norm, halley_residual

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_integrated_halley_advantage(A: np.ndarray, T: np.ndarray, Q: np.ndarray) -> tuple[float, float, float, float]:
    A = np.asarray(A)
    T = np.asarray(T)
    Q = np.asarray(Q)
    S = _oracle_triangular_principal_sqrt(T)
    X0 = _oracle_reconstruct_initial_sqrt(Q, S)
    R0, _ = _oracle_working_residual_backward_error(A, X0)
    Rhat, _ = _oracle_schur_residual_transform(Q, R0)
    E = _oracle_block_recursive_sylvester(S, S, Rhat, 1)
    D, _ = _oracle_lift_validate_correction(Q, X0, E, R0)
    _, _, mixed_r = _oracle_working_precision_update(A, X0, D)
    H = _oracle_halley_newton_direction(A, X0)
    _, halley_r, sn, pn = _oracle_matrix_halley_schwarzian(A, X0, H)
    if halley_r == 0.0:
        raise ValueError('zero Halley residual makes advantage undefined')
    return (float(np.float64(mixed_r / halley_r)), float(sn), float(pn), float(halley_r))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three normal and three explicit ValueError test case specifications."""
    return [
        {
            "setup": """import numpy as np; A=np.array([[734.3125,-562.1875,-729.6875,557.8125],[320.3125,-448.1875,-319.6875,447.8125],[-809.6875,917.8125,814.3125,-922.1875],[-199.6875,7.8125,200.3125,-8.1875]],dtype=np.float64); T=np.array([[1024.0001220703125,-225.39634704589844,123.67469787597656,-1950.7784423828125],[0.,64.0001449584961,-45.988956451416016,683.4092407226562],[0.,0.,3.9994139671325684,-56.81679916381836],[0.,0.,0.,0.25045859813690186]],dtype=np.float32); Q=np.array([[0.6966455578804016,0.4486113488674164,0.5052264332771301,-0.24120382964611053],[0.21158075332641602,-0.4446275234222412,0.4567680060863495,0.7408797144889832],[-0.6356939673423767,0.5600026845932007,0.48396551609039307,0.21924364566802979],[-0.2565384805202484,-0.5361445546150208,0.5494421124458313,-0.5872394442558289]],dtype=np.float32)""",
            "call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(integrated_halley_advantage(A,T,Q))""",
            "gold_call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(_oracle_integrated_halley_advantage(A,T,Q))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[4.0001]],dtype=np.float64); T=np.array([[4.]],dtype=np.float32); Q=np.array([[1.]],dtype=np.float32)""",
            "call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(integrated_halley_advantage(A,T,Q))""",
            "gold_call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(_oracle_integrated_halley_advantage(A,T,Q))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[9.001]],dtype=np.float64); T=np.array([[9.]],dtype=np.float32); Q=np.array([[1.]],dtype=np.float32)""",
            "call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(integrated_halley_advantage(A,T,Q))""",
            "gold_call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(_oracle_integrated_halley_advantage(A,T,Q))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[10.120201356005715, 7.8399007025400955], [4.590150096559521, 9.879750144839287]],dtype=np.float64); T=np.array([[16.0, 3.25], [0.0, 4.0]],dtype=np.float32); Q=np.array([[0.800000011920929, -0.6000000238418579], [0.6000000238418579, 0.800000011920929]],dtype=np.float32)""",
            "call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(integrated_halley_advantage(A,T,Q))""",
            "gold_call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(_oracle_integrated_halley_advantage(A,T,Q))""",
        },
        {
            "setup": """import numpy as np; A=np.array([[13.23969988794327, 8.820220171661378], [16.319830171661376, 12.760280345706942]],dtype=np.float64); T=np.array([[25.0, -7.5], [0.0, 1.0]],dtype=np.float32); Q=np.array([[0.6000000238418579, -0.800000011920929], [0.800000011920929, 0.6000000238418579]],dtype=np.float32)""",
            "call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(integrated_halley_advantage(A,T,Q))""",
            "gold_call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(_oracle_integrated_halley_advantage(A,T,Q))""",
        },
        {"setup": """import numpy as np; T=np.array([[16.,3.,-2.],[0.,4.,5.],[0.,0.,1.]],dtype=np.float32); Q=np.eye(3,dtype=np.float32); A=np.asarray(T,dtype=np.float64)+np.array([[1e-3,-2e-3,3e-3],[-4e-3,5e-3,-6e-3],[7e-3,-8e-3,9e-3]],dtype=np.float64)""", "call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(integrated_halley_advantage(A,T,Q))""", "gold_call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(_oracle_integrated_halley_advantage(A,T,Q))"""},
        {"setup": """import numpy as np; T=np.array([[64.,-20.,7.,3.],[0.,9.,12.,-5.],[0.,0.,2.25,4.],[0.,0.,0.,0.25]],dtype=np.float32); Q=np.eye(4,dtype=np.float32); A=np.asarray(T,dtype=np.float64)+np.array([[1e-4,-2e-4,3e-4,-4e-4],[5e-4,-6e-4,7e-4,-8e-4],[9e-4,-1e-3,1.1e-3,-1.2e-3],[1.3e-3,-1.4e-3,1.5e-3,-1.6e-3]],dtype=np.float64)""", "call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(integrated_halley_advantage(A,T,Q))""", "gold_call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(_oracle_integrated_halley_advantage(A,T,Q))"""},
        {"setup": """import numpy as np; T=np.array([[25.,30.,-15.],[0.,1.,8.],[0.,0.,0.04]],dtype=np.float32); Q=np.diag(np.array([1.,-1.,1.],dtype=np.float32)); A=np.asarray(Q,dtype=np.float64)@np.asarray(T,dtype=np.float64)@np.asarray(Q.T,dtype=np.float64)+np.array([[2e-5,-3e-5,5e-5],[-7e-5,11e-5,-13e-5],[17e-5,-19e-5,23e-5]],dtype=np.float64)""", "call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(integrated_halley_advantage(A,T,Q))""", "gold_call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(_oracle_integrated_halley_advantage(A,T,Q))"""},
        {"setup": """import numpy as np; T=np.array([[100.,-40.,20.,-10.,5.],[0.,36.,18.,-9.,4.],[0.,0.,9.,12.,-6.],[0.,0.,0.,1.,3.],[0.,0.,0.,0.,0.0625]],dtype=np.float32); Q=np.eye(5,dtype=np.float32); A=np.asarray(T,dtype=np.float64)+np.linspace(-2e-4,2e-4,25,dtype=np.float64).reshape(5,5)""", "call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(integrated_halley_advantage(A,T,Q))""", "gold_call": """(lambda z:(float(z[0]),float(z[1]),float(z[2]),float(z[3])))(_oracle_integrated_halley_advantage(A,T,Q))"""},
        {
            "setup": """import numpy as np; A=np.empty((0,0),dtype=np.float64); T=np.empty((0,0),dtype=np.float32); Q=np.empty((0,0),dtype=np.float32)
def run_model():
    try:
        integrated_halley_advantage(A,T,Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrated_halley_advantage(A,T,Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.ones((2,3),dtype=np.float64); T=np.ones((2,3),dtype=np.float32); Q=np.ones((2,3),dtype=np.float32)
def run_model():
    try:
        integrated_halley_advantage(A,T,Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrated_halley_advantage(A,T,Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
        {
            "setup": """import numpy as np; A=np.eye(2,dtype=np.float64); T=np.eye(2,dtype=np.float32); Q=np.array([[np.nan,0.],[0.,1.]],dtype=np.float32)
def run_model():
    try:
        integrated_halley_advantage(A,T,Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_integrated_halley_advantage(A,T,Q)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            "call": """run_model()""",
            "gold_call": """run_gold()""",
        },
    ]
