"""
Assemble the QSVT singlet Green and field-amplitude digitization.

The source paper implements the nonlocal term of Equation (8) by a QSVT approximation of K^{-1} with query access to K, and it digitizes each physical amplitude in the field-amplitude basis of Equations (9) and (24)-(25). Return those two paper objects. For the QSVT Green, let B_i be the dimensionless incidence rows in atlas family i (entries -1 and +1), and form the physical kernel `K=sum_i (B_i.T@B_i)/a**2`, where $a=lattice_spacing$. Reduce this K to the singlet with the Helmert basis W of shape (V,V-1): for column j=0,...,V-2, W[:j+1,j]=1/sqrt((j+1)*(j+2)) and W[j+1,j]=-(j+1)/sqrt((j+1)*(j+2)). Let A=W.T@K@W, and let alpha and beta be its smallest and largest eigenvalues. Build the degree-d Gauss-Chebyshev projection of 1/x on [alpha,beta]. With N=2*(d+1)+1, nodes theta_j=pi*(j+0.5)/N and y_j=cos(theta_j), center=(alpha+beta)/2, radius=(beta-alpha)/2, the coefficients are c_k=(2/N)*sum_j cos(k*theta_j)/(center+radius*y_j) for k=0,...,d, then halve c_0. Map the singlet kernel by X=(2A-(alpha+beta)I)/(beta-alpha), run T_0=I, T_1=X, T_(k+1)=2 X T_k - T_(k-1), and form G_red=sum_k c_k T_k. Lift G=W@G_red@W.T and symmetrize once. This is not the Moore-Penrose inverse, not a bordered saddle, and not a polynomial applied to the full kernel including its zero mode. For the digitization, the computational basis is the unsigned label lambda=0,...,2**K-1 with bit J the coefficient of 2**J. Equation (9) places evenly spaced eigenvalues from -a_max to a_max inclusive, with the paper's delta_A=2*a_max/(2**K-1). Equation (24) writes A as a linear combination of Pauli Z on those K qubits. Equation (25) obtains Pi by the unitary DFT similarity F^dagger A F scaled by delta_Pi/delta_A, realising delta_Pi ~ 1/a_max as delta_Pi=1/a_max. Use F[lambda,mu]=2**(-K/2)*exp(2 pi i lambda mu / 2**K). Return the Pauli-Z weights in increasing J, the real diagonal A, and the real and imaginary parts of Pi stacked as a real (3,2**K,2**K) array. Do not bit-reverse, do not use an unnormalized DFT, and do not replace Pi by a sine transform.

Returns
-------
green : np.ndarray, shape (L**3, L**3), float Lifted degree-d QSVT singlet Green. z_weights : np.ndarray, shape (n_qubits,), float Pauli-Z coefficients of A, qubit J the 2**J bit of lambda. operators : np.ndarray, shape (3, 2**n_qubits, 2**n_qubits), float Real A, real part of Pi, and imaginary part of Pi.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def assemble_qsvt_field_amplitude_objects(
    side_length,
    lattice_spacing,
    polynomial_degree,
    n_qubits,
    a_max,
    tree_atlas,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''QSVT singlet Green and field-amplitude A, Pi digitization.

    Parameters
    ----------
    side_length : int
        Number L of sites per cubic direction, 2 <= L <= 5.
    lattice_spacing : float
        Positive lattice spacing a.
    polynomial_degree : int
        Chebyshev degree d, 3 <= d <= 16.
    n_qubits : int
        Number K of field-amplitude qubits, 3 <= K <= 6.
    a_max : float
        Positive field-amplitude cutoff Amax.
    tree_atlas : array-like, shape (L**3 - 1, 6 + 2*L**3)
        Rooted maximal-tree atlas returned by Step 01.

    Returns
    -------
    green : np.ndarray, shape (L**3, L**3), float
        Lifted degree-d QSVT singlet Green.
    z_weights : np.ndarray, shape (n_qubits,), float
        Pauli-Z coefficients of A, qubit J the 2**J bit of lambda.
    operators : np.ndarray, shape (3, 2**n_qubits, 2**n_qubits), float
        Real A, real part of Pi, and imaginary part of Pi.

    Raises
    ------
    ValueError
        If any integer argument is not a Python int or NumPy integer in its
        stated range (bool excluded); if lattice_spacing or a_max is not
        finite and positive; or if the atlas has the wrong shape or
        nonfinite entries.
    '''
    return np.empty((0, 0)), np.empty(0), np.empty((3, 0, 0))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_assemble_qsvt_field_amplitude_objects(
    side_length,
    lattice_spacing,
    polynomial_degree,
    n_qubits,
    a_max,
    tree_atlas,
):
    import numpy as np

    def require_int(value, name):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError("invalid " + name)
        return int(value)

    L = require_int(side_length, "side_length")
    degree = require_int(polynomial_degree, "polynomial_degree")
    K = require_int(n_qubits, "n_qubits")
    try:
        a = float(lattice_spacing)
        cutoff = float(a_max)
    except Exception as exc:
        raise ValueError("invalid scale") from exc
    if (
        L < 2 or L > 5
        or degree < 3 or degree > 16
        or K < 3 or K > 6
        or not np.isfinite(a) or a <= 0.0
        or not np.isfinite(cutoff) or cutoff <= 0.0
    ):
        raise ValueError("invalid lattice or digitization")
    V = L ** 3
    try:
        raw = np.asarray(tree_atlas)
        if np.iscomplexobj(raw):
            if not np.all(np.isfinite(raw)) or np.any(raw.imag != 0.0):
                raise ValueError("invalid tree atlas")
            raw = raw.real
        atlas = np.asarray(raw, dtype=float)
    except Exception as exc:
        raise ValueError("invalid tree atlas") from exc
    if atlas.shape != (V - 1, 6 + 2 * V) or not np.all(np.isfinite(atlas)):
        raise ValueError("invalid tree atlas")

    B = atlas[:, 6 : 6 + V]
    families = atlas[:, 0]
    kernel = np.zeros((V, V), dtype=float)
    for code in (3.0, 2.0, 1.0):
        block = B[np.isclose(families, code)]
        kernel = kernel + block.T @ block / (a * a)

    W = np.zeros((V, V - 1), dtype=float)
    for j in range(V - 1):
        scale = np.sqrt((j + 1.0) * (j + 2.0))
        W[: j + 1, j] = 1.0 / scale
        W[j + 1, j] = -(j + 1.0) / scale
    reduced = W.T @ kernel @ W
    spectrum = np.linalg.eigvalsh(reduced)
    alpha = float(spectrum[0])
    beta = float(spectrum[-1])
    if not (alpha > 0.0 and beta > alpha):
        raise ValueError("invalid singlet interval")
    n_nodes = 2 * (degree + 1) + 1
    theta = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    y = np.cos(theta)
    center = 0.5 * (alpha + beta)
    radius = 0.5 * (beta - alpha)
    coeffs = np.empty(degree + 1, dtype=float)
    for k in range(degree + 1):
        coeffs[k] = (2.0 / n_nodes) * np.sum(
            np.cos(k * theta) / (center + radius * y)
        )
    coeffs[0] *= 0.5
    mapped = (2.0 * reduced - (alpha + beta) * np.eye(V - 1)) / (beta - alpha)
    t_prev = np.eye(V - 1, dtype=float)
    t_curr = mapped.copy()
    green_red = coeffs[0] * t_prev + coeffs[1] * t_curr
    for k in range(2, degree + 1):
        t_next = 2.0 * mapped @ t_curr - t_prev
        green_red = green_red + coeffs[k] * t_next
        t_prev, t_curr = t_curr, t_next
    green = W @ green_red @ W.T
    green = 0.5 * (green + green.T)

    n_states = 2 ** K
    labels = np.arange(n_states)
    delta_a = 2.0 * cutoff / (n_states - 1)
    a_vals = -cutoff + labels * delta_a
    a_op = np.diag(a_vals)
    z_weights = np.empty(K, dtype=float)
    reconstructed = np.zeros((n_states, n_states), dtype=float)
    for bit in range(K):
        pauli_z = np.diag(1.0 - 2.0 * ((labels >> bit) & 1).astype(float))
        z_weights[bit] = -delta_a * (2.0 ** (bit - 1))
        reconstructed = reconstructed + z_weights[bit] * pauli_z
    fourier = np.exp(
        2.0j * np.pi * np.outer(labels, labels) / n_states
    ) / np.sqrt(n_states)
    delta_pi = 1.0 / cutoff
    pi_op = (delta_pi / delta_a) * (fourier.conj().T @ a_op @ fourier)
    operators = np.stack((a_op, np.real(pi_op), np.imag(pi_op)))
    if not (
        np.allclose(a_op, reconstructed, rtol=1e-12, atol=1e-12)
        and np.all(np.isfinite(green))
        and np.all(np.isfinite(z_weights))
        and np.all(np.isfinite(operators))
    ):
        raise ValueError("nonfinite digitization or Green")
    return green, z_weights, operators

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "# case: boundary\n"
            "L=2\na=1.0\nd=3\nK=3\nAmax=1.0\n"
            "T=build_maximal_tree_reduction_atlas(L)\n",
            "call": "assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)",
            "gold_call": "_oracle_assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)",
        },
        {
            "setup": "# case: normal\n"
            "L=3\na=0.41\nd=7\nK=4\nAmax=1.7\n"
            "T=build_maximal_tree_reduction_atlas(L)\n",
            "call": "assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)[0]",
            "gold_call": "_oracle_assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)[0]",
        },
        {
            "setup": "# case: normal\n"
            "L=2\na=0.5\nd=5\nK=5\nAmax=0.8\n"
            "T=build_maximal_tree_reduction_atlas(L)\n",
            "call": "assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)[1]",
            "gold_call": "_oracle_assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)[1]",
        },
        {
            "setup": "# case: edge\n"
            "L=3\na=1.0\nd=4\nK=4\nAmax=1.0\n"
            "T=build_maximal_tree_reduction_atlas(L)\n",
            "call": "assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)[2][0]",
            "gold_call": "_oracle_assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)[2][0]",
        },
        {
            "setup": "# case: edge\n"
            "L=2\na=1.0\nd=11\nK=3\nAmax=1.2\n"
            "T=build_maximal_tree_reduction_atlas(L)\n",
            "call": "assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)[2][1:]",
            "gold_call": "_oracle_assemble_qsvt_field_amplitude_objects(L,a,d,K,Amax,T)[2][1:]",
        },
        {
            "setup": "# case: edge\n"
            "import numpy as np\n"
            "L=3\n"
            "T=build_maximal_tree_reduction_atlas(L)\n"
            "def status(fn):\n"
            "    try: fn(); return 0\n"
            "    except ValueError: return 1\n"
            "    except Exception: return 2\n",
            "call": "tuple((status(fn) for fn in (lambda: assemble_qsvt_field_amplitude_objects(True,0.4,5,4,1.0,T), "
            "lambda: assemble_qsvt_field_amplitude_objects(1,0.4,5,4,1.0,T), lambda: "
            "assemble_qsvt_field_amplitude_objects(3,0.0,5,4,1.0,T), lambda: "
            "assemble_qsvt_field_amplitude_objects(3,0.4,True,4,1.0,T), lambda: "
            "assemble_qsvt_field_amplitude_objects(3,0.4,2,4,1.0,T), lambda: "
            "assemble_qsvt_field_amplitude_objects(3,0.4,5,2,1.0,T), lambda: "
            "assemble_qsvt_field_amplitude_objects(3,0.4,5,4,0.0,T), lambda: "
            "assemble_qsvt_field_amplitude_objects(3,0.4,5,4,1.0,T[:1]))))",
            "gold_call": "tuple((status(fn) for fn in (lambda: _oracle_assemble_qsvt_field_amplitude_objects(True,0.4,5,4,1.0,T), "
            "lambda: _oracle_assemble_qsvt_field_amplitude_objects(1,0.4,5,4,1.0,T), lambda: "
            "_oracle_assemble_qsvt_field_amplitude_objects(3,0.0,5,4,1.0,T), lambda: "
            "_oracle_assemble_qsvt_field_amplitude_objects(3,0.4,True,4,1.0,T), lambda: "
            "_oracle_assemble_qsvt_field_amplitude_objects(3,0.4,2,4,1.0,T), lambda: "
            "_oracle_assemble_qsvt_field_amplitude_objects(3,0.4,5,2,1.0,T), lambda: "
            "_oracle_assemble_qsvt_field_amplitude_objects(3,0.4,5,4,0.0,T), lambda: "
            "_oracle_assemble_qsvt_field_amplitude_objects(3,0.4,5,4,1.0,T[:1]))))",
        },
    ]
