"""
Stabilizer partition function of a pure state's Pauli spectrum.

Definition 3 of the source paper is the stabilizer partition function of the signed Pauli spectrum at inverse temperature beta, equivalently one-half the trace of the Boltzmann factor of the paper's fictitious Hamiltonian built from that spectrum. The identity contribution is included. This is the pure-state object; the paper's mixed-state extension (Definition 8) is built on top of it in later steps and is not evaluated here.

This step evaluates the stated generating-function expression on the supplied numerical spectrum. Test arrays may be synthetic and need not represent a physical state, but every entry must satisfy |x| <= 1 and beta must be finite and nonnegative. The function must return a finite value for every admissible beta, including beta >> 1 where a naive cosh evaluation overflows.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def stabilizer_partition_function(spectrum: np.ndarray, beta: float) -> float:
    """Stabilizer partition function of a pure state's Pauli spectrum.

    Evaluate Definition 3 of the source paper on the supplied signed
    Pauli spectrum at inverse temperature ``beta``. The identity
    contribution is included. Recover the paper's generating function.
    This is the pure-state quantity; do not apply any mixed-state
    extension here.

    Supported domain: ``spectrum`` is a 1-D real array of length 4**n
    (n >= 1) with every entry in [-1, 1]; ``beta`` is finite and >= 0.
    The result must be finite for every admissible ``beta``, including
    large ``beta`` where a direct ``cosh`` overflows. Raise ValueError
    for inputs outside this domain.

    Parameters
    ----------
    spectrum : np.ndarray
        1-D real array of length 4**n with entries in [-1, 1].
    beta : float
        Inverse-temperature parameter, finite and >= 0.

    Returns
    -------
    z : float
        Stabilizer partition function Z_beta.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_stabilizer_partition_function(spectrum: np.ndarray, beta: float) -> float:
    """Reference oracle: Z_beta = e^{-beta} sum_x cosh(beta x), overflow-free form."""
    import numpy as np

    spec = np.asarray(spectrum, dtype=float).reshape(-1)
    if spec.size == 0 or not np.all(np.isfinite(spec)):
        raise ValueError("spectrum must be a nonempty finite 1-D array.")
    size = int(spec.size)
    tmp = size
    n_ok = 0
    while tmp > 1:
        if tmp % 4 != 0:
            raise ValueError("spectrum length must equal 4**n for integer n >= 1.")
        tmp //= 4
        n_ok += 1
    if n_ok < 1:
        raise ValueError("spectrum length must equal 4**n for integer n >= 1.")
    if float(np.max(np.abs(spec))) > 1.0 + 1e-12:
        raise ValueError("Pauli expectation values must satisfy |x| <= 1.")
    b = float(beta)
    if not np.isfinite(b) or b < 0.0:
        raise ValueError("beta must be finite and >= 0.")
    a = np.abs(spec)
    # e^{-b} cosh(b a) = 0.5 (e^{-b(1-a)} + e^{-b(1+a)}); both exponents <= 0 for |a| <= 1.
    return float(0.5 * np.sum(np.exp(-b * (1.0 - a)) + np.exp(-b * (1.0 + a))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pair = "float(np.round({fn}(spec, beta), 12))"
    return [
        {
            # Stabilizer state |+> at beta = 2: e^{-2}(2 cosh 2 + 2) = 1.288986205.
            "setup": "import numpy as np\nspec = np.array([1.0, 1.0, 0.0, 0.0])\nbeta = 2.0\n",
            "call": pair.format(fn="stabilizer_partition_function"),
            "gold_call": pair.format(fn="_oracle_stabilizer_partition_function"),
        },
        {
            # Signed entry: |1> at beta = 1.
            "setup": "import numpy as np\nspec = np.array([1.0, 0.0, 0.0, -1.0])\nbeta = 1.0\n",
            "call": pair.format(fn="stabilizer_partition_function"),
            "gold_call": pair.format(fn="_oracle_stabilizer_partition_function"),
        },
        {
            # T state (I + (X + Y)/sqrt 2)/2 at beta = 2: 1.234063280.
            "setup": "import numpy as np\nspec = np.array([1.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0])\nbeta = 2.0\n",
            "call": pair.format(fn="stabilizer_partition_function"),
            "gold_call": pair.format(fn="_oracle_stabilizer_partition_function"),
        },
        {
            # beta = 0 boundary: Z_0 = 4**n.
            "setup": "import numpy as np\nspec = np.array([1.0, 1.0, 0.0, 0.0])\nbeta = 0.0\n",
            "call": pair.format(fn="stabilizer_partition_function"),
            "gold_call": pair.format(fn="_oracle_stabilizer_partition_function"),
        },
        {
            # Large beta where cosh(beta) alone overflows a naive evaluation.
            "setup": "import numpy as np\nspec = np.array([1.0, 0.5, 0.5, 0.0])\nbeta = 800.0\n",
            "call": pair.format(fn="stabilizer_partition_function"),
            "gold_call": pair.format(fn="_oracle_stabilizer_partition_function"),
        },
    ]
