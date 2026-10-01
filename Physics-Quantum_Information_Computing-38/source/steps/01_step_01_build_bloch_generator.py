"""
Build the real affine generator that propagates the Bloch vector of a qubit whose ensemble-averaged state follows a Lindblad master equation.

Averaging a diffusive stochastic master equation over its measurement noise removes the innovation term, so the unconditional state of a monitored qubit obeys a deterministic Lindblad equation that is linear and trace preserving on the Bloch vector.

Returns
-------
np.ndarray: real (4, 4) generator G with d(1, x, y, z)/dt = G (1, x, y, z).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_bloch_generator(hamiltonian: 'np.ndarray', jump_operators: 'np.ndarray') -> 'np.ndarray':
    """Return the affine generator of the Bloch-vector dynamics of a Lindblad qubit.

    The qubit state ``rho`` obeys (hbar = 1)
    ``d rho / d t = -i [H, rho] + sum_k (L_k rho L_k^dag - {L_k^dag L_k, rho} / 2)``
    with ``H = hamiltonian`` and ``L_k = jump_operators[k]``. Write
    ``u = (1, x, y, z)`` with ``x = Tr(rho sigma_x)``, ``y = Tr(rho sigma_y)``,
    ``z = Tr(rho sigma_z)`` and the Pauli matrices
    ``sigma_x = [[0, 1], [1, 0]]``, ``sigma_y = [[0, -1j], [1j, 0]]``,
    ``sigma_z = [[1, 0], [0, -1]]``. Return the real matrix ``G`` for which
    ``d u / d t = G u`` holds for every state; its first row is zero.

    Parameters
    ----------
    hamiltonian : np.ndarray
        Complex ``(2, 2)`` Hermitian matrix, with each entry's modulus at
        most 10 in the task's units.
    jump_operators : np.ndarray
        Complex array of shape ``(m, 2, 2)`` holding ``0 <= m <= 16`` jump
        operators, with each entry's modulus at most 10. A sequence of
        ``(2, 2)`` matrices is also accepted.

    Returns
    -------
    np.ndarray
        Real array of shape ``(4, 4)``.

    Raises
    ------
    ValueError
        If ``hamiltonian`` is not a finite ``(2, 2)`` matrix, if it differs
        from its conjugate transpose by more than ``1e-12`` in any entry, or
        if ``jump_operators`` cannot be read as a finite array of shape
        ``(m, 2, 2)``, either operator array exceeds the supported entry
        bound of 10, there are more than 16 jump operators, or numerical
        evaluation produces a nonfinite intermediate or result.
    """
    return generator

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_build_bloch_generator(hamiltonian: 'np.ndarray', jump_operators: 'np.ndarray') -> 'np.ndarray':
    """Reference implementation (Pauli projection of the Lindblad superoperator)."""
    import numpy as np

    def _as_complex(value):
        try:
            return np.asarray(value, dtype=complex)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("operators must be numeric arrays") from None

    ham = _as_complex(hamiltonian)
    if ham.shape != (2, 2) or not np.all(np.isfinite(ham)):
        raise ValueError("hamiltonian must be a finite (2, 2) matrix")
    if np.any(np.abs(ham) > 10.0):
        raise ValueError("hamiltonian entries must have modulus <= 10")
    if np.max(np.abs(ham - ham.conj().T)) > 1e-12:
        raise ValueError("hamiltonian must be Hermitian")
    jumps = _as_complex(jump_operators)
    if jumps.size == 0:
        jumps = np.zeros((0, 2, 2), dtype=complex)
    if jumps.ndim != 3 or jumps.shape[1:] != (2, 2):
        raise ValueError("jump_operators must have shape (m, 2, 2)")
    if not np.all(np.isfinite(jumps)):
        raise ValueError("jump_operators must be finite")
    if jumps.shape[0] > 16 or np.any(np.abs(jumps) > 10.0):
        raise ValueError("at most 16 jump operators with entry modulus <= 10 are supported")

    basis = [
        np.eye(2, dtype=complex),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
    ]

    def _lindblad(rho):
        out = -1.0j * (ham @ rho - rho @ ham)
        for jump in jumps:
            dag = jump.conj().T
            out = out + jump @ rho @ dag - 0.5 * (dag @ jump @ rho + rho @ dag @ jump)
        return out

    # rho = (u_0 I + x sigma_x + y sigma_y + z sigma_z) / 2 with u_0 = 1, and
    # d u_a / d t = Tr(sigma_a L(rho)), so column b holds Tr(sigma_a L(P_b)) / 2.
    generator = np.zeros((4, 4))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for col, element in enumerate(basis):
                image = _lindblad(element)
                if not np.all(np.isfinite(image)):
                    raise ValueError("nonfinite Lindblad image")
                for row in range(1, 4):
                    generator[row, col] = 0.5 * np.trace(basis[row] @ image).real
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError("nonfinite generator evaluation") from exc
    if not np.all(np.isfinite(generator)):
        raise ValueError("nonfinite Bloch generator")
    return generator

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "SX = np.array([[0, 1], [1, 0]], dtype=complex)\n"
        "SY = np.array([[0, -1j], [1j, 0]], dtype=complex)\n"
        "SZ = np.array([[1, 0], [0, -1]], dtype=complex)\n"
        "def _gsig(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (4, 4):\n"
        "        return -1.0\n"
        "    flat = a.ravel()\n"
        "    weights = np.cos(np.arange(1, flat.size + 1, dtype=float))\n"
        "    return float(np.sum(np.abs(flat)) + np.sum(flat * weights))\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": helpers,
            "call": "_gsig(build_bloch_generator(-0.65 * SX, np.array([0.7 * SZ])))",
            "gold_call": "_gsig(_oracle_build_bloch_generator(-0.65 * SX, np.array([0.7 * SZ])))",
        },
        {
            "setup": helpers,
            "call": "float(build_bloch_generator(0.9 * SX, np.array([0.4 * SZ]))[2, 3])",
            "gold_call": "float(_oracle_build_bloch_generator(0.9 * SX, np.array([0.4 * SZ]))[2, 3])",
        },
        {
            "setup": helpers + (
                "L = np.sqrt(0.8) * np.array([[0, 1], [0, 0]], dtype=complex)\n"
                "H = 0.35 * SZ + 0.2 * SX\n"
            ),
            "call": "_gsig(build_bloch_generator(H, np.array([L])))",
            "gold_call": "_gsig(_oracle_build_bloch_generator(H, np.array([L])))",
        },
        {
            "setup": helpers + (
                "H = np.array([[0.4, 0.3 - 0.7j], [0.3 + 0.7j, -1.1]])\n"
                "L1 = np.array([[0.2 + 0.1j, 0.9], [-0.3j, 0.5]])\n"
                "L2 = np.array([[0.0, 0.25], [0.6, -0.4j]])\n"
            ),
            "call": "_gsig(build_bloch_generator(H, np.array([L1, L2])))",
            "gold_call": "_gsig(_oracle_build_bloch_generator(H, np.array([L1, L2])))",
        },
        {
            "setup": helpers + (
                "L1 = np.array([[0.2 + 0.1j, 0.9], [-0.3j, 0.5]])\n"
            ),
            "call": "float(build_bloch_generator(np.zeros((2, 2)), [L1])[3, 0])",
            "gold_call": "float(_oracle_build_bloch_generator(np.zeros((2, 2)), [L1])[3, 0])",
        },
        {
            "setup": helpers,
            "call": "_gsig(build_bloch_generator(0.3 * SX - 0.5 * SY + 0.9 * SZ, np.zeros((0, 2, 2))))",
            "gold_call": "_gsig(_oracle_build_bloch_generator(0.3 * SX - 0.5 * SY + 0.9 * SZ, np.zeros((0, 2, 2))))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_bloch_generator(np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros((0, 2, 2))))",
            "gold_call": "_status(lambda: _oracle_build_bloch_generator(np.array([[0.0, 1.0], [0.0, 0.0]]), np.zeros((0, 2, 2))))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_bloch_generator(SZ, np.ones((1, 2, 3))))",
            "gold_call": "_status(lambda: _oracle_build_bloch_generator(SZ, np.ones((1, 2, 3))))",
        },
        {
            "setup": helpers,
            "call": "_gsig(build_bloch_generator(10 * SZ, np.tile((10 * SX)[None], (16, 1, 1)))) / 1e5",
            "gold_call": "_gsig(_oracle_build_bloch_generator(10 * SZ, np.tile((10 * SX)[None], (16, 1, 1)))) / 1e5",
        },
        {
            "setup": helpers,
            "call": "_gsig(build_bloch_generator(10 * SY, []))",
            "gold_call": "_gsig(_oracle_build_bloch_generator(10 * SY, []))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_bloch_generator(np.diag([1e308, 0]), []))",
            "gold_call": "_status(lambda: _oracle_build_bloch_generator(np.diag([1e308, 0]), []))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_bloch_generator(np.nextafter(10., np.inf) * SZ, []))",
            "gold_call": "_status(lambda: _oracle_build_bloch_generator(np.nextafter(10., np.inf) * SZ, []))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_bloch_generator(SZ, [np.nextafter(10., np.inf) * SX]))",
            "gold_call": "_status(lambda: _oracle_build_bloch_generator(SZ, [np.nextafter(10., np.inf) * SX]))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_bloch_generator(SZ, np.zeros((17, 2, 2))))",
            "gold_call": "_status(lambda: _oracle_build_bloch_generator(SZ, np.zeros((17, 2, 2))))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_bloch_generator([[10**400, 0], [0, 0]], []))",
            "gold_call": "_status(lambda: _oracle_build_bloch_generator([[10**400, 0], [0, 0]], []))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_bloch_generator(SZ, [np.array([[np.nan, 0], [0, 0]])]))",
            "gold_call": "_status(lambda: _oracle_build_bloch_generator(SZ, [np.array([[np.nan, 0], [0, 0]])]))",
        },
    ]
