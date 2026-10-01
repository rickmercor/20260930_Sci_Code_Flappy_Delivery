"""
Assemble sample-major midway transition-amplitude data.

Midway trajectories provide signed transition-amplitude observations between an ordered set of source and target basis states. Assemble those observations using the supplied basis order while preserving the source-target convention. Do not mutate the basis or trajectory arrays.

Returns
-------
Return a floating-point array with axes `(sample, target, source)` containing the signed midway transition-amplitude observations.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def midway_amplitude_samples(
    basis_states: np.ndarray,
    final_states: np.ndarray,
    path_amplitudes: np.ndarray,
) -> np.ndarray:
    """Assemble sample-major midway transition-amplitude data.

    Return an array with axes (sample, target, source) using the supplied basis
    ordering and signed path amplitudes. Preserve the source-target convention
    and do not mutate the input arrays.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_midway_amplitude_samples(
    basis_states: np.ndarray,
    final_states: np.ndarray,
    path_amplitudes: np.ndarray,
) -> np.ndarray:
    import numpy as np

    raw_basis = np.asarray(basis_states, dtype=float)
    raw_final = np.asarray(final_states, dtype=float)
    amplitudes = np.asarray(path_amplitudes, dtype=float)
    if raw_basis.ndim != 2 or len(raw_basis) == 0:
        raise ValueError("basis_states must be a nonempty matrix")
    if np.any((raw_basis != 0) & (raw_basis != 1)):
        raise ValueError("basis states must be binary")
    if amplitudes.ndim != 2 or amplitudes.shape[0] != len(raw_basis):
        raise ValueError("path_amplitudes must have shape (dimension, samples)")
    expected = (len(raw_basis), amplitudes.shape[1], raw_basis.shape[1])
    if raw_final.shape != expected:
        raise ValueError("inconsistent final-state array")
    if amplitudes.shape[1] == 0 or np.any(~np.isfinite(amplitudes)):
        raise ValueError("finite nonempty samples required")
    if np.any((raw_final != 0) & (raw_final != 1)):
        raise ValueError("final states must be binary")
    basis = raw_basis.astype(int)
    lookup = {tuple(row): index for index, row in enumerate(basis)}
    if len(lookup) != len(basis):
        raise ValueError("basis rows must be unique")

    n_samples = amplitudes.shape[1]
    matrices = np.zeros(
        (n_samples, len(basis), len(basis)), dtype=float
    )
    for source in range(len(basis)):
        for sample in range(n_samples):
            integer_state = raw_final[source, sample].astype(int)
            target = lookup.get(tuple(integer_state))
            if target is None or np.any(
                raw_final[source, sample] != integer_state
            ):
                raise ValueError("a final state is outside the supplied basis")
            matrices[sample, target, source] = amplitudes[source, sample]
    return matrices

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nb=np.array([[1,0],[0,1]]); f=np.array([[[1,0],[0,1]],[[0,1],[1,0]]]); a=np.array([[2.,-3.],[4.,5.]])",
            "call": "midway_amplitude_samples(b,f,a)",
            "gold_call": "_oracle_midway_amplitude_samples(b,f,a)",
        },
        {
            "setup": "import numpy as np\nb=np.eye(3,dtype=int); f=np.repeat(b[:,None,:],3,axis=1); a=np.arange(1.,10.).reshape(3,3)",
            "call": "midway_amplitude_samples(b,f,a)",
            "gold_call": "_oracle_midway_amplitude_samples(b,f,a)",
        },
        {
            "setup": "import numpy as np\nb=np.array([[1,0,0],[0,1,0],[0,0,1]]); f=np.array([[b[1]],[b[2]],[b[0]]]); a=np.array([[-1.],[2.],[-4.]])",
            "call": "midway_amplitude_samples(b,f,a)",
            "gold_call": "_oracle_midway_amplitude_samples(b,f,a)",
        },
    ]
