"""
Run the complete structured BPS Householder QR pipeline through a requested iteration.

This step runs the complete structured BPS QR pipeline and returns the requested Householder scaling coefficient. It first constructs the banded matrix and semiseparable generators, then computes the lookup quantities and initializes the perturbation state. It repeatedly forms the next Householder reflector, computes the transformed factor row, and updates the structured perturbation state in sequence. The recurrence advances through the requested one-based iteration because each reflector depends on the trailing state produced by all previous reflectors, allowing the target coefficient to be computed without forming the full dense matrix or dense Q and R factors.

Returns
-------
A float for the deterministic Householder scaling coefficient at a given target index.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_target_tau(
    n: int = 100000,
    target_index: int = 73129,
) -> float:
    """
    Compute a target Householder coefficient with the complete structured pipeline.

    Parameters
    ----------
    n : int
        Positive BPS matrix dimension.
    target_index : int
        One-based Householder coefficient index in the range 1 through n.

    Returns
    -------
    tau_target : float
        The deterministic Householder scaling coefficient at target_index.

    Raises
    ------
    ValueError
        If the requested matrix dimension or target coefficient index is 
        invalid or lies outside the admissible range.
    """
    return tau_target

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_target_tau(
    n: int = 100000,
    target_index: int = 73129,
) -> float:
    """Reference end-to-end implementation composed from Steps 1-6."""
    if not isinstance(n,(int,np.integer)) or int(n)<1:
        raise ValueError("n must be a positive integer")
    if not isinstance(target_index,(int,np.integer)):
        raise ValueError("target_index must be an integer")
    n=int(n); target_index=int(target_index)
    if not (1<=target_index<=n):
        raise ValueError("target_index must lie in [1,n]")
    
    bands=_oracle_construct_banded_component(n)
    U,V,W,S=_oracle_construct_semiseparable_generators(n)
    precomp,state=_oracle_precompute_bps_quantities(bands,U,V,W,S)

    tau_target=None
    for q in range(target_index):
        reflector=_oracle_form_structured_householder(
            bands,U,V,W,S,precomp,state
        )
        row_update=_oracle_compute_structured_row_update(
            bands,U,V,W,S,precomp,state,reflector
        )
        tau_target=float(reflector["tau"])

        if q<target_index-1:
            state=_oracle_update_structured_perturbation(
                U,W,state,reflector,row_update
            )

    return float(tau_target)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return whole-pipeline integration tests, including invalid cases."""
    return [
        {"setup":"import numpy as np\nn=16\ntarget_index=5",
         "call":"compute_target_tau(n,target_index)",
         "gold_call":"_oracle_compute_target_tau(n,target_index)"},
        {"setup":"import numpy as np\nn=1\ntarget_index=1",
         "call":"compute_target_tau(n,target_index)",
         "gold_call":"_oracle_compute_target_tau(n,target_index)"},
        {"setup":"import numpy as np\nn=64\ntarget_index=64",
         "call":"compute_target_tau(n,target_index)",
         "gold_call":"_oracle_compute_target_tau(n,target_index)"},
        {
            "setup":r"""import numpy as np
n=0; target_index=1
def run_model():
    try:
        compute_target_tau(n,target_index); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try:
        _oracle_compute_target_tau(n,target_index); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call":"run_model()","gold_call":"run_oracle()"
        },
        {
            "setup":r"""import numpy as np
n=8; target_index=9
def run_model():
    try:
        compute_target_tau(n,target_index); return 0
    except ValueError: return 1
    except Exception: return 2
def run_oracle():
    try:
        _oracle_compute_target_tau(n,target_index); return 0
    except ValueError: return 1
    except Exception: return 2
""",
            "call":"run_model()","gold_call":"run_oracle()"
        },
    ]
