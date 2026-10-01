"""
Compute the integrated robustness gain from extending a Pauli measurement family.

This final orchestrator applies the same phase-consistent projected-polytope construction to two nested measured families. Both sets observe the same physical mixture $\rho(t)=(1-t)\rho_0+t\rho_1$, where the endpoints are normalized Gibbs states of the supplied Hamiltonians. The reduced state is not recomputed from a Hamiltonian with deleted coefficients: only measured coordinates are restricted. Trace both convex affine robustness profiles, including optimal face transitions, and integrate their squared difference on a common partition. All quantities and the final scalar are dimensionless, and no sampling or stochastic seed is involved.

Returns
-------
Dimensionless integral $\int_0^1(R_{\mathcal M}(t)-R_{\mathcal N}(t))^2\,dt$, from phase-consistent projected polytopes and certified profiles.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_integrated_measurement_gain(
    labels: tuple[str, ...],
    coefficients_start: "np.ndarray",
    coefficients_end: "np.ndarray",
    temperatures: "np.ndarray",
    measured_indices: "np.ndarray",
) -> float:
    r"""Compute the integrated robustness gain from extending a Pauli measurement family.

    Parameters
    ----------
    labels : tuple[str, ...]
        Distinct unsigned nonidentity Pauli words on one to six qubits, with at most 32
        measured observables.
    coefficients_start : np.ndarray
        Finite real $(m,)$ coefficients of the initial dimensionless Hamiltonian.
    coefficients_end : np.ndarray
        Finite real $(m,)$ coefficients of the final dimensionless Hamiltonian, in the same
        order.
    temperatures : np.ndarray
        Finite real $(2,)$ inverse temperatures $b_0,b_1\ge0$.
    measured_indices : np.ndarray
        Nonempty one-dimensional integer array of distinct indices in $0,\ldots,m-1$,
        preserving their supplied coordinate order; specifies the smaller family. The same
        full-state marginals are restricted to these indices.

    Returns
    -------
    gain : float
        Dimensionless integral $\int_0^1(R_{\mathcal M}(t)-R_{\mathcal N}(t))^2\,dt$, from
        phase-consistent projected polytopes and certified profiles.

    Raises
    ------
    ValueError
        If any label, coefficient, temperature, subset index, or profile contract is
        violated.
    RuntimeError
        If an optimization, certificate, or profile refinement fails.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.optimize import linprog


def _projected_family_vertices(labels):
    encoded = _oracle_encode_pauli_words(labels)
    adjacency = _oracle_build_frustration_matrix(encoded)
    contexts = _oracle_enumerate_maximal_contexts(adjacency)
    relations = _oracle_derive_phase_constraints(encoded, contexts)
    return _oracle_construct_projected_vertices(contexts, relations)


def _oracle_compute_integrated_measurement_gain(
    labels: tuple[str, ...],
    coefficients_start: "np.ndarray",
    coefficients_end: "np.ndarray",
    temperatures: "np.ndarray",
    measured_indices: "np.ndarray",
) -> float:
    temperatures = np.asarray(temperatures)
    indices = np.asarray(measured_indices)
    if (
        temperatures.shape != (2,)
        or np.iscomplexobj(temperatures)
        or not np.all(np.isfinite(temperatures))
        or np.any(temperatures < 0)
    ):
        raise ValueError(
            "temperatures must contain two finite nonnegative inverse temperatures"
        )
    if (
        indices.ndim != 1
        or not indices.size
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("measured indices must be a nonempty integer vector")
    if (
        np.any(indices < 0)
        or np.any(indices >= len(labels))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("measured indices must be distinct and in range")
    full_vertices = _projected_family_vertices(labels)
    reduced_labels = tuple(labels[j] for j in indices)
    reduced_vertices = _projected_family_vertices(reduced_labels)
    start = _oracle_compute_thermal_marginals(
        labels, coefficients_start, float(temperatures[0])
    )
    end = _oracle_compute_thermal_marginals(
        labels, coefficients_end, float(temperatures[1])
    )
    full_profile = _oracle_trace_robustness_profile(full_vertices, start, end)
    reduced_profile = _oracle_trace_robustness_profile(
        reduced_vertices, start[indices], end[indices]
    )
    return _oracle_integrate_measurement_gain(full_profile, reduced_profile)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent scientific cases for this numerical contract."""
    return [
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
labels = ('X','Y','Z')
a = np.array([-1.,-1.,-1.])
b = -a
temperatures = np.array([2.,2.])
indices = np.array([0,1])
""",
            "call": "compute_integrated_measurement_gain(labels, a.copy(), b.copy(), temperatures.copy(), indices.copy())",
            "gold_call": "_oracle_compute_integrated_measurement_gain(labels, a.copy(), b.copy(), temperatures.copy(), indices.copy())",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
labels = ('XI','IX','XX','YI','IY','YY','ZI','IZ','ZZ')
a = np.array([-.7,.7,.7,-.7,.6,.8,-.9,.8,.5])
b = np.array([.4,-.2,.9,.6,-.8,.5,-.3,.7,.8])
temperatures = np.array([2.3,1.7])
indices = np.array([8,6,2,0])
""",
            "call": "compute_integrated_measurement_gain(labels, a.copy(), b.copy(), temperatures.copy(), indices.copy())",
            "gold_call": "_oracle_compute_integrated_measurement_gain(labels, a.copy(), b.copy(), temperatures.copy(), indices.copy())",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
labels = ('X','Y','Z')
a = np.zeros(3)
b = np.zeros(3)
temperatures = np.array([0.,5.])
indices = np.array([1])
""",
            "call": "compute_integrated_measurement_gain(labels, a.copy(), b.copy(), temperatures.copy(), indices.copy())",
            "gold_call": "_oracle_compute_integrated_measurement_gain(labels, a.copy(), b.copy(), temperatures.copy(), indices.copy())",
            "tol": 1e-08,
        },
        {
            "setup": """import numpy as np
from scipy.optimize import linprog
labels = ('X',)
a = np.array([1.])
b = np.array([-1.])
temperatures = np.array([1.,2.])
indices = np.array([0,0])

def _reject(fn):
    try:
        fn(labels, a.copy(), b.copy(), temperatures.copy(), indices.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_reject(compute_integrated_measurement_gain)",
            "gold_call": "_reject(_oracle_compute_integrated_measurement_gain)",
            "tol": 0.0,
        },
    ]
