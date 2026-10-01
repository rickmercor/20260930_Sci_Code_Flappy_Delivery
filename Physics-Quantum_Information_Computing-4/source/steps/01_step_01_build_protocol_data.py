"""
Assemble the constraint observables, the matching observed-moment vector and the detector

operators of a bipartite qudit key-distribution instance whose parameter estimation uses two

mutually unbiased bases.

Alice and Bob each hold a d_local-dimensional system, so the joint space has dimension

d = d_local ** 2 and is ordered so that the basis label of the pair (a, b) is

a * d_local + b, with a on Alice's side. Two measurement settings are recorded. The first is

the computational setting, in which Alice and Bob both project onto the vectors |k> of the

computational basis. The second is the Fourier setting, in which Alice projects onto

|x_j> = (1 / sqrt(d_local)) * sum_l omega ** (j * l) |l> and Bob projects onto the complex

conjugate vectors |y_k> = (1 / sqrt(d_local)) * sum_l omega ** (-k * l) |l>, with

omega = exp(2i * pi / d_local).



The recorded data are two d_local x d_local tables of outcome probabilities, p_z for the

computational setting and p_x for the Fourier setting, each with rows indexed by Alice's

outcome and columns by Bob's outcome, each nonnegative and summing to one. Because the

projectors of one setting already resolve the identity, the pair (d_local - 1, d_local - 1)

is dropped from each table, leaving n = 2 * (d_local ** 2 - 1) observables. The returned

observables are ordered with all computational-setting entries first in row-major order of

(a, b), followed by all Fourier-setting entries in row-major order of (j, k), and the

returned moment vector uses the same order.



Alice's raw key is produced by a detector that reports her computational-basis outcome except

that, with probability epsilon, it reports a symbol drawn uniformly at random instead. The

positive-operator-valued measure describing it therefore has the d_local elements

E_a = (1 - epsilon) |a><a| + (epsilon / d_local) * identity on Alice's system. The returned

detector operators are the positive square roots of these elements, each tensored with the

identity on Bob's system, indexed by a in increasing order, so element a acts on the joint

space and the sum of their adjoint-times-self products is the joint identity.

Returns
-------
return operators, moments, detector_ops
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_protocol_data(d_local: int, p_z: "np.ndarray", p_x: "np.ndarray", epsilon: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    '''Build the constraint observables, observed moments and detector operators.

    Parameters
    ----------
    d_local : int
        Local Hilbert-space dimension of Alice and of Bob, at least 2.
    p_z : np.ndarray
        Real array of shape (d_local, d_local) holding the computational-setting outcome
        probabilities, rows indexed by Alice, columns by Bob. Entries are nonnegative and
        sum to one within 1e-9.
    p_x : np.ndarray
        Real array of shape (d_local, d_local) holding the Fourier-setting outcome
        probabilities, with the same layout and validity requirements as p_z.
    epsilon : float
        Detector randomisation probability, strictly greater than 0 and at most 1.

    Returns
    -------
    operators : np.ndarray
        Complex array of shape (n, d, d) with n = 2 * (d_local ** 2 - 1) and
        d = d_local ** 2, holding the Hermitian constraint observables in the documented
        order.
    moments : np.ndarray
        Real array of shape (n, ) holding the observed value attached to each observable,
        in the same order.
    detector_ops : np.ndarray
        Complex array of shape (d_local, d, d) holding the detector operators, indexed by
        Alice's key symbol in increasing order.

    Raises
    ------
    ValueError
        If d_local is not an integer of at least 2, if p_z or p_x is not a real array of
        shape (d_local, d_local) with finite entries, if any entry is negative, if either
        table fails to sum to one within 1e-9, or if epsilon is not a real scalar in the
        half-open interval from 0 exclusive to 1 inclusive.
    '''
    return operators, moments, detector_ops

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validate_probability_table(table: "np.ndarray", d_local: int, name: str) -> "np.ndarray":
    """Return the table as a float array after checking shape, finiteness and normalisation."""
    array = np.asarray(table)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = array.astype(float)
    if array.shape != (d_local, d_local):
        raise ValueError(f"{name} must have shape ({d_local}, {d_local})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite entries")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must have nonnegative entries")
    if abs(float(array.sum()) - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to one within 1e-9")
    return array


def _hermitian_square_root(matrix: "np.ndarray") -> "np.ndarray":
    """Return the positive square root of a Hermitian positive semidefinite operator."""
    spectrum, vectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    return (vectors * np.sqrt(np.clip(spectrum, 0.0, None))) @ vectors.conj().T


def _oracle_build_protocol_data(d_local: int, p_z: "np.ndarray", p_x: "np.ndarray", epsilon: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    if isinstance(d_local, bool) or not isinstance(d_local, (int, np.integer)):
        raise ValueError("d_local must be an integer")
    d_local = int(d_local)
    if d_local < 2:
        raise ValueError("d_local must be at least 2")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, np.integer, np.floating)):
        raise ValueError("epsilon must be a real scalar")
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon > 1.0:
        raise ValueError("epsilon must satisfy 0 < epsilon <= 1")

    table_z = _validate_probability_table(p_z, d_local, "p_z")
    table_x = _validate_probability_table(p_x, d_local, "p_x")

    omega = np.exp(2j * np.pi / d_local)
    powers = np.outer(np.arange(d_local), np.arange(d_local))
    fourier = omega ** powers / np.sqrt(d_local)      # column j is |x_j>, row index is l
    computational = np.eye(d_local, dtype=complex)

    settings = ((computational, computational, table_z), (fourier, fourier.conj(), table_x))
    operators = []
    moments = []
    for basis_a, basis_b, table in settings:
        for a in range(d_local):
            for b in range(d_local):
                if a == d_local - 1 and b == d_local - 1:
                    continue
                vec_a = basis_a[:, a].reshape(-1, 1)
                vec_b = basis_b[:, b].reshape(-1, 1)
                operators.append(np.kron(vec_a @ vec_a.conj().T, vec_b @ vec_b.conj().T))
                moments.append(table[a, b])

    identity_bob = np.eye(d_local, dtype=complex)
    detector = []
    for a in range(d_local):
        element = (epsilon / d_local) * np.eye(d_local, dtype=complex)
        element[a, a] += 1.0 - epsilon
        detector.append(np.kron(_hermitian_square_root(element), identity_bob))

    return (np.array(operators, dtype=complex), np.array(moments, dtype=float),
            np.array(detector, dtype=complex))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    header = "import numpy as np\n"

    qutrit_data = header + """
p_z = np.array([[0.297, 0.021, 0.014],
                [0.018, 0.284, 0.026],
                [0.011, 0.023, 0.306]])
p_x = np.array([[0.281, 0.029, 0.019],
                [0.024, 0.292, 0.031],
                [0.017, 0.026, 0.281]])
"""

    qubit_data = header + """
p_z = np.array([[0.46, 0.04],
                [0.03, 0.47]])
p_x = np.array([[0.44, 0.06],
                [0.05, 0.45]])
"""

    quartit_data = header + """
p_z = np.full((4, 4), 0.01)
np.fill_diagonal(p_z, 0.22)
p_z[3, 3] = 0.22 + (1.0 - p_z.sum())
p_x = np.full((4, 4), 0.012)
np.fill_diagonal(p_x, 0.214)
p_x[3, 3] = 0.214 + (1.0 - p_x.sum())
"""

    structure_probe = """
def structure_probe(fn, d_local, p_z, p_x, epsilon):
    operators, moments, detector = fn(d_local, p_z, p_x, epsilon)
    d = d_local ** 2
    resolution = sum(a.conj().T @ a for a in detector) - np.eye(d, dtype=complex)
    hermitian = max(float(np.abs(operators[i] - operators[i].conj().T).max())
                    for i in range(operators.shape[0]))
    idempotent = max(float(np.abs(operators[i] @ operators[i] - operators[i]).max())
                     for i in range(operators.shape[0]))
    return np.array([float(operators.shape[0]), float(d), float(detector.shape[0]),
                     hermitian, idempotent, float(np.abs(resolution).max()),
                     float(moments.sum())])
"""

    completeness_probe = """
def completeness_probe(fn, d_local, p_z, p_x, epsilon):
    operators, moments, detector = fn(d_local, p_z, p_x, epsilon)
    half = d_local ** 2 - 1
    d = d_local ** 2
    dropped_z = np.eye(d, dtype=complex) - operators[:half].sum(axis=0)
    dropped_x = np.eye(d, dtype=complex) - operators[half:].sum(axis=0)
    return np.array([float(np.trace(dropped_z).real), float(np.trace(dropped_x).real),
                     float(np.abs(dropped_z @ dropped_z - dropped_z).max()),
                     float(np.abs(dropped_x @ dropped_x - dropped_x).max())])
"""

    invalid = """
def probe(fn, args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1
"""

    return [
        # --- Normal: the qutrit benchmark instance at the benchmark detector noise ---
        {
            "setup": qutrit_data,
            "call": "build_protocol_data(3, p_z.copy(), p_x.copy(), 0.04)",
            "gold_call": "_oracle_build_protocol_data(3, p_z, p_x, 0.04)",
        },
        # --- Normal: structural invariants of the observable stack and detector resolution ---
        {
            "setup": qutrit_data + structure_probe,
            "call": "structure_probe(build_protocol_data, 3, p_z.copy(), p_x.copy(), 0.04)",
            "gold_call": "structure_probe(_oracle_build_protocol_data, 3, p_z, p_x, 0.04)",
        },
        # --- Boundary: smallest admissible local dimension ---
        {
            "setup": qubit_data,
            "call": "build_protocol_data(2, p_z.copy(), p_x.copy(), 0.1)",
            "gold_call": "_oracle_build_protocol_data(2, p_z, p_x, 0.1)",
        },
        # --- Boundary: fully randomising detector, where every element is the scaled identity ---
        {
            "setup": qutrit_data,
            "call": "build_protocol_data(3, p_z.copy(), p_x.copy(), 1.0)",
            "gold_call": "_oracle_build_protocol_data(3, p_z, p_x, 1.0)",
        },
        # --- Edge: the dropped projector of each setting must complete the identity ---
        {
            "setup": quartit_data + completeness_probe,
            "call": "completeness_probe(build_protocol_data, 4, p_z.copy(), p_x.copy(), 0.03)",
            "gold_call": "completeness_probe(_oracle_build_protocol_data, 4, p_z, p_x, 0.03)",
        },
        # --- Edge: a table with an exactly vanishing entry and a very quiet detector ---
        {
            "setup": header + """
p_z = np.array([[0.5, 0.0],
                [0.1, 0.4]])
p_x = np.array([[0.25, 0.25],
                [0.25, 0.25]])
""",
            "call": "build_protocol_data(2, p_z.copy(), p_x.copy(), 1e-9)",
            "gold_call": "_oracle_build_protocol_data(2, p_z, p_x, 1e-9)",
        },
        # --- Invalid: local dimension below two ---
        {
            "setup": header + invalid + """
p_z = np.array([[1.0]])
p_x = np.array([[1.0]])
""",
            "call": "probe(build_protocol_data, (1, p_z.copy(), p_x.copy(), 0.04))",
            "gold_call": "probe(_oracle_build_protocol_data, (1, p_z, p_x, 0.04))",
        },
        # --- Invalid: detector randomisation probability at zero ---
        {
            "setup": qutrit_data + invalid,
            "call": "probe(build_protocol_data, (3, p_z.copy(), p_x.copy(), 0.0))",
            "gold_call": "probe(_oracle_build_protocol_data, (3, p_z, p_x, 0.0))",
        },
        # --- Invalid: detector randomisation probability above one ---
        {
            "setup": qutrit_data + invalid,
            "call": "probe(build_protocol_data, (3, p_z.copy(), p_x.copy(), 1.2))",
            "gold_call": "probe(_oracle_build_protocol_data, (3, p_z, p_x, 1.2))",
        },
        # --- Invalid: unnormalised table ---
        {
            "setup": header + invalid + """
p_z = np.array([[0.5, 0.1],
                [0.1, 0.5]])
p_x = np.array([[0.25, 0.25],
                [0.25, 0.25]])
""",
            "call": "probe(build_protocol_data, (2, p_z.copy(), p_x.copy(), 0.05))",
            "gold_call": "probe(_oracle_build_protocol_data, (2, p_z, p_x, 0.05))",
        },
    ]
