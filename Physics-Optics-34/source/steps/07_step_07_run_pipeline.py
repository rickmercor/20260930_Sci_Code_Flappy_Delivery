"""
Compute the finite-harmonic propagation target for the fixed passive rectangular optical layer. Assemble inclusion and host matrices with material_operator, form exact interval indicators with indicator_matrix, compose x1-then-x2 rectangle_operator, eliminate longitudinal fields with maxwell_operator and return spectral_abscissa. Use epsilon1=[[9,1.3,0.7],[1.3,7,-0.8],[0.7,-0.8,6]], mu1=[[1.7,0.18,-0.12],[0.18,1.3,0.15],[-0.12,0.15,1.5]], xi1=[[1.2i,0.55+0.25i,0.2-0.3i],[-0.25+0.4i,0.9i,0.45+0.15i],[0.35+0.1i,-0.3+0.2i,1.1i]], epsilon0=[[2.4,0.15,-0.05],[0.15,2.1,0.12],[-0.05,0.12,2.7]], mu0=[[1.1,0.04,0],[0.04,1.2,-0.03],[0,-0.03,1.05]] and xi0=[[0.15i,0.08,0.03i],[-0.04,0.2i,0.06],[0.02i,-0.05,0.1i]]. Both materials add loss 0.025i times identity and use xi conjugate transpose in the lower-left block. Set a2/a1=1.3, k0*a1=3.3, k1/k0=0.21, k2/k0=-0.17, interval centers 0.11 and -0.08. Inputs mx,my are nonboolean integers in [0,4] with (2mx+1)(2my+1)<=25; fractions fx,fy are real in [0,1]. Return the unrounded scalar. Invalid input or upstream failure raises ValueError.

This deterministic finite discretization uses full anisotropic constitutive tensors so both electric-magnetic normal coupling and tangential Schur corrections contribute. Its scalar is the greatest real part of the unfiltered propagation spectrum, not a homogenized index or a converged continuum eigenvalue.

Returns
-------
native float, finite-harmonic propagation target
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_pipeline(mx: int, my: int, fx: float, fy: float) -> float:
    """Return the fixed-material rectangular propagation target.

    Parameters
    ----------
    mx, my : int
        Retained harmonic orders 0 through 4; total harmonics at most 25.
    fx, fy : float
        Real inclusion interval fractions in [0,1].

    Returns
    -------
    float
        Greatest real component of k3/k0, without decimal rounding.

    Raises
    ------
    ValueError
        For invalid inputs or a violated upstream numerical contract.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_pipeline(mx: int, my: int, fx: float, fy: float) -> float:
    e1=np.array([[9.,1.3,.7],[1.3,7.,-.8],[.7,-.8,6.]])
    m1=np.array([[1.7,.18,-.12],[.18,1.3,.15],[-.12,.15,1.5]])
    x1=np.array([[1.2j,.55+.25j,.2-.3j],[-.25+.4j,.9j,.45+.15j],[.35+.1j,-.3+.2j,1.1j]])
    e0=np.array([[2.4,.15,-.05],[.15,2.1,.12],[-.05,.12,2.7]])
    m0=np.array([[1.1,.04,0],[.04,1.2,-.03],[0,-.03,1.05]])
    x0=np.array([[.15j,.08,.03j],[-.04,.2j,.06],[.02j,-.05,.1j]])
    p1=_oracle_material_operator(e1,m1,x1,.025)
    p0=_oracle_material_operator(e0,m0,x0,.025)
    tx=_oracle_indicator_matrix(mx,fx,.11);ty=_oracle_indicator_matrix(my,fy,-.08)
    p=_oracle_rectangle_operator(p1,p0,tx,ty)
    mat=_oracle_maxwell_operator(p,mx,my,1.3,3.3,.21,-.17)
    return _oracle_spectral_abscissa(mat)

def _final_answer_case():
    return {'setup':'import numpy as np','gold_call':'_oracle_run_pipeline(1,1,.43,.37)','extract':'round(result,6)'}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return explicit normal, boundary, edge, and invalid-input cases."""
    invalid_setup = '''import numpy as np
def status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    return 0
'''
    return [
        {
            'setup': 'import numpy as np',
            'call': 'run_pipeline(1,1,.43,.37)',
            'gold_call': '_oracle_run_pipeline(1,1,.43,.37)',
        },
        {
            'setup': 'import numpy as np',
            'call': 'run_pipeline(0,0,.43,.37)',
            'gold_call': '_oracle_run_pipeline(0,0,.43,.37)',
        },
        {
            'setup': 'import numpy as np',
            'call': 'run_pipeline(1,1,0,.37)',
            'gold_call': '_oracle_run_pipeline(1,1,0,.37)',
        },
        {
            'setup': 'import numpy as np',
            'call': 'run_pipeline(1,2,.3,.6)',
            'gold_call': '_oracle_run_pipeline(1,2,.3,.6)',
        },
        {
            'setup': invalid_setup,
            'call': 'status(run_pipeline,True,1,.43,.37)',
            'gold_call': 'status(_oracle_run_pipeline,True,1,.43,.37)',
        },
        {
            'setup': invalid_setup,
            'call': 'status(run_pipeline,4,4,.43,.37)',
            'gold_call': 'status(_oracle_run_pipeline,4,4,.43,.37)',
        },
        {
            'setup': invalid_setup,
            'call': 'status(run_pipeline,1,1,float("inf"),.37)',
            'gold_call': 'status(_oracle_run_pipeline,1,1,float("inf"),.37)',
        },
        {
            'setup': invalid_setup,
            'call': 'status(run_pipeline,1,1,.43,-.1)',
            'gold_call': 'status(_oracle_run_pipeline,1,1,.43,-.1)',
        },
    ]
