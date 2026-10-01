"""
Assemble the complete momentum-resolved local/global calculation, verify that

all preceding sub-problems are mutually consistent, and return one scalar: the

physical momentum-sector dimension d_N=s_N-z_N at the last requested chain

size.

Check agreement of shared intermediate quantities, including injectivity

lengths, local-system nullities, local-space dimension, N_ref, telescoping

dimension, and requested periodic sizes.



For each admissible N use d_N = s_N - z_N, and return float(d_N) for the last

requested size.

Returns
-------
# float, physical sector dimension at the last requested size
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def momentum_diagnostic(A: np.ndarray, k: int, sizes: 'Sequence[int]', omega: complex, X: "np.ndarray | None" = None, max_L: int = 6, tol: float = 1e-10) -> float:
    """Return d_N=s_N-z_N for the last requested resonant chain size.

    The function must execute the full Step-01--Step-06 pipeline and reject
    mutually inconsistent intermediate results; it is not a shortcut that
    computes only the final finite-size rank.

    Parameters
    ----------
    A : np.ndarray
        Finite complex MPS tensor with shape (d, D, D).
    k : int
        Local operator range; must be 1 or 2.
    sizes : sequence of int
        Nonempty sequence of resonant periodic chain lengths; the returned
        scalar corresponds to sizes[-1].
    omega : complex
        Finite unit-modulus sector phase.
    X : np.ndarray or None
        Optional finite invertible D by D virtual similarity matrix. If None,
        the identity (no transformation) is used.
    max_L : int
        Largest product length used to certify normal factors.
    tol : float
        Finite non-negative linear-algebra tolerance.

    Returns
    -------
    result : float
        The physical momentum-sector dimension d_N=s_N-z_N at sizes[-1].

    Raises
    ------
    ValueError
        If an input is invalid/non-finite, a size is non-resonant, or the
        preceding sub-problem results are mutually inconsistent.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_momentum_diagnostic(A: np.ndarray, k: int, sizes: 'Sequence[int]', omega: complex, X: "np.ndarray | None" = None, max_L: int = 6, tol: float = 1e-10) -> float:

    num_factors, factor_dims, factor_L, algebra_dim, radical_dim = _oracle_block_structure(A, max_L=max_L, tol=tol)

    factor_L_sys, N_ref, local_matrix_ranks, raw_nullities = _oracle_twisted_local_equation_system(
        A, k, omega, max_L=max_L, tol=tol
    )

    raw_nullities_proj, factor_oe_dims, operator_dim, zero_fibre_dims, operator_basis = _oracle_twisted_operator_space(
        A, k, omega, max_L=max_L, tol=tol
    )

    (sizes_out, sector_dims, zero_sum_dims, distinct_dims, state_norms,
     local_dim_periodic, N_ref_periodic) = _oracle_periodic_momentum_spaces(
        A, k, sizes, omega, max_L=max_L, tol=tol
    )

    telescopic_dim, telescopic_basis = _oracle_twisted_telescopic_space(
        int(np.asarray(A).shape[0]), k, omega, tol=tol
    )

    (local_dim, telescopic_dim_q, quotient_dim, nonidentity_dim,
     similarity_union_dim, conjugation_union_dim, intersection_dims,
     finite_quotient_dims, excess_dims, quotient_basis) = _oracle_momentum_quotients(
        A, k, sizes, omega, X=X, max_L=max_L, tol=tol
    )

    if not np.array_equal(np.asarray(factor_L_sys, dtype=int), np.asarray(factor_L, dtype=int)):
        raise ValueError("inconsistent injectivity lengths across sub-problems")
    if not np.array_equal(np.asarray(raw_nullities_proj, dtype=int), np.asarray(raw_nullities, dtype=int)):
        raise ValueError("inconsistent raw nullities across sub-problems")
    if int(local_dim_periodic) != int(operator_dim) or int(local_dim) != int(operator_dim):
        raise ValueError("inconsistent local projected dimension")
    if int(N_ref_periodic) != int(N_ref):
        raise ValueError("inconsistent reference size")
    if int(telescopic_dim_q) != int(telescopic_dim):
        raise ValueError("inconsistent telescoping dimension")
    if not np.array_equal(np.asarray(sizes_out, dtype=int), np.asarray(sizes, dtype=int)):
        raise ValueError("periodic size order changed")

    if np.asarray(distinct_dims).size < 1:
        raise ValueError("no finite-size sector dimension was produced")
    return float(np.asarray(distinct_dims, dtype=float)[-1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
           'A=np.array([[[1,-1,1,0],[0,-1,0,0],[0,0,2,1],[0,1,0,1]],[[-1,0,3,3],[0,0,2,2],[1,-1,1,0],[-1,1,1,2]],[[1,-1,0,-2],[0,1,-1,-1],[0,-1,0,1],[0,0,0,-1]]],dtype=complex); k=2; sizes=(4,8); '
           'omega=1j; X=np.array([[1,1,0,0],[0,1,0,0],[0,0,1,1],[1,0,0,1]],dtype=complex)',
  'call': 'momentum_diagnostic(A.copy(),k,sizes,omega,X.copy())',
  'gold_call': '_oracle_momentum_diagnostic(A.copy(),k,sizes,omega,X.copy())',
  'tol': 1e-12},
 {'setup': 'import numpy as np\nA=np.array([[[-2.,3.],[-5.,6.]],[[2.,0.],[3.,-1.]]],dtype=complex); k=2; sizes=(4,8); omega=1+0j; X=np.array([[2.,1.],[1.,1.]],dtype=complex)',
  'call': 'momentum_diagnostic(A.copy(),k,sizes,omega,X.copy())',
  'gold_call': '_oracle_momentum_diagnostic(A.copy(),k,sizes,omega,X.copy())',
  'tol': 1e-12},
 {'setup': 'import numpy as np\nA=np.array([[[2.,-1.],[0.,-1.]],[[2.,0.],[-2.,0.]]],dtype=complex); k=2; sizes=(2,4,6); omega=-1+0j; X=np.array([[1.,1.],[0.,1.]],dtype=complex)',
  'call': 'momentum_diagnostic(A.copy(),k,sizes,omega,X.copy())',
  'gold_call': '_oracle_momentum_diagnostic(A.copy(),k,sizes,omega,X.copy())',
  'tol': 1e-12},
 {'setup': 'import numpy as np\nA=np.array([[[1.,0.],[0.,1.]],[[np.inf,0.],[0.,1.]]],dtype=complex)\ndef refused(f):\n    try: f(A.copy(),2,(4,8),1j); return 0\n    except ValueError: return 1',
  'call': 'refused(momentum_diagnostic)',
  'gold_call': 'refused(_oracle_momentum_diagnostic)',
  'tol': 1e-12}]
