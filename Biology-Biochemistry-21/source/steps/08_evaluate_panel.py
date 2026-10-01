"""
Evaluate every biochemical design in row order.

Returns
-------
one thirteen-value design record per design.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_panel(data: dict) -> "np.ndarray":
    """Evaluate the complete design panel.

    Parameters
    ----------
    data
        Mapping following the design-record contract, with one row per design
        in both ``design_multiplier`` and ``restriction``.

    Returns
    -------
    np.ndarray
        A float64 design-by-thirteen matrix in the original design order.

    Raises
    ------
    ValueError
        If the mapping is missing design arrays, has no designs, has
        inconsistent design-row counts, or a composed evaluation fails.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_panel(data: dict) -> "np.ndarray":
    if not isinstance(data,dict) or "design_multiplier" not in data or "restriction" not in data:
        raise ValueError("panel data must contain design arrays")
    multiplier=np.asarray(data["design_multiplier"])
    restriction=np.asarray(data["restriction"])
    if (multiplier.ndim!=2 or multiplier.shape[0]==0 or restriction.ndim!=2
            or restriction.shape[0]!=multiplier.shape[0]):
        raise ValueError("design arrays must have the same nonzero row count")
    return np.vstack([_oracle_design_record(data,design)
        for design in range(multiplier.shape[0])]).astype(np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup":"""import copy
d=_dr7_fixture(); cd=copy.deepcopy(d); gd=copy.deepcopy(d)""",
            "call":"evaluate_panel(cd)","gold_call":"_oracle_evaluate_panel(gd)","tol":1e-7,
        },
        {
            "setup":"""import copy
d=_dr7_fixture()
for k in ('design_multiplier','restriction','design_shift','cost'): d[k]=d[k][:1]
cd=copy.deepcopy(d); gd=copy.deepcopy(d)""",
            "call":"evaluate_panel(cd)","gold_call":"_oracle_evaluate_panel(gd)","tol":1e-7,
        },
        {
            "setup":"""import copy
d=_dr7_fixture(); d['residual_limit']=1.; d['thermo_limit']=1.
cd=copy.deepcopy(d); gd=copy.deepcopy(d)""",
            "call":"evaluate_panel(cd)","gold_call":"_oracle_evaluate_panel(gd)","tol":1e-7,
        },
        {
            "setup":"""import copy
d=_dr7_fixture(); d['restriction']=d['restriction'][:1]
cd=copy.deepcopy(d); gd=copy.deepcopy(d)
def case_raises(fn,*args):
    try: fn(*args)
    except ValueError: return 1.0
    return 0.0""",
            "call":"case_raises(evaluate_panel,cd)",
            "gold_call":"case_raises(_oracle_evaluate_panel,gd)","tol":0.0,
        },
    ]
