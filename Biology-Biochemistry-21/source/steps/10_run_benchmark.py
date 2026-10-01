"""
Run the complete biochemical design benchmark.

This final-only orchestrator evaluates every design and selects the largest

eligible cost-normalized robust yield.

Returns
-------
one finite float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_benchmark(data: dict, tie_tolerance: float) -> float:
    """Return the winning eligible design score.

    Parameters
    ----------
    data
        Mapping following the complete panel contract.
    tie_tolerance
        Finite nonnegative design-score tie tolerance.

    Returns
    -------
    float
        The winning eligible cost-normalized robust product yield.

    Raises
    ------
    ValueError
        If panel evaluation or design selection rejects its inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_benchmark(data: dict, tie_tolerance: float) -> float:
    panel=_oracle_evaluate_panel(data)
    selection=_oracle_select_design(panel,tie_tolerance)
    return float(selection[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup":"""import copy
d=_dr7_fixture(); cd=copy.deepcopy(d); gd=copy.deepcopy(d)""",
            "call":"run_benchmark(cd,1e-10)",
            "gold_call":"_oracle_run_benchmark(gd,1e-10)","tol":1e-7,
        },
        {
            "setup":"""import copy
d=_dr7_fixture(); d['residual_limit']=1.; d['thermo_limit']=1.
cd=copy.deepcopy(d); gd=copy.deepcopy(d)""",
            "call":"run_benchmark(cd,0.)","gold_call":"_oracle_run_benchmark(gd,0.)","tol":1e-7,
        },
        {
            "setup":"""import copy
d=_dr7_fixture()
for k in ('design_multiplier','restriction','design_shift','cost'): d[k]=d[k][:1]
cd=copy.deepcopy(d); gd=copy.deepcopy(d)""",
            "call":"run_benchmark(cd,1e-10)",
            "gold_call":"_oracle_run_benchmark(gd,1e-10)","tol":1e-7,
        },
        {
            "setup":"""import copy
d=_dr7_fixture(); d['residual_limit']=-1.; cd=copy.deepcopy(d); gd=copy.deepcopy(d)
def case_raises(fn,*args):
    try: fn(*args)
    except ValueError: return 1.0
    return 0.0""",
            "call":"case_raises(run_benchmark,cd,1e-10)",
            "gold_call":"case_raises(_oracle_run_benchmark,gd,1e-10)","tol":0.0,
        },
    ]
