"""
Step 07 - Configuration weights of a coupled-cluster bra-ket pair.

Configuration weights of a (time-dependent) coupled-cluster state.

A truncated coupled-cluster state has no normalised wave function whose squared
coefficients could serve as populations of the Slater determinants. Its state
is the pair of the ket |Psi> = exp(T)|Phi0> and the bra
<~Psi| = <Phi0|(1 + Lambda) exp(-T), with <~Psi|Psi> = 1, and populations are
defined from that pair. For a determinant |Phi_mu> take its coefficient in the
ket, c_mu = <Phi_mu|Psi>, and its coefficient in the bra, ~c_mu = <~Psi|Phi_mu>.
The configuration weight of the determinant is the real part of their product,

W_mu = Re( ~c_mu c_mu ).

The operator conventions are those of the ground-state step: the reference
|Phi0> fills spin orbitals 0 .. n_electrons - 1, |Phi_i^a> = X_i^a |Phi0> with
X_i^a = a_a^+ a_i, and |Phi_ij^ab> = X_ij^ab |Phi0> with
X_ij^ab = a_a^+ a_b^+ a_j a_i, virtual indices counted from 0 over the virtual
block. The phase amplitude of the unit operator is not part of the packed
vector; it would multiply every c_mu by one number and every ~c_mu by its
inverse, so the weights do not depend on it and the ket coefficient of the
reference is taken as one. Express the coefficients of the reference, of every
single and of every double excitation through the packed amplitudes by
expanding the exponentials; the expressions are not restated here.

The weights are real numbers but are not guaranteed to lie between zero and
one; report them as they come, without clipping or renormalising.

Return the weights as one real vector: first the reference weight, then the
single-excitation weights W[i, a] flattened in C order (o v entries), then the
double-excitation weights for i < j and a < b, with the occupied pair (i, j) in
lexicographic order as the outer loop and the virtual pair (a, b) in
lexicographic order as the inner loop.

Returns
-------
numpy.ndarray of length 1 + o v + (o(o-1)/2)(v(v-1)/2): reference, single and double configuration weights
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def configuration_weights(amplitudes: np.ndarray, n_electrons: int, n_spin_orbitals: int) -> np.ndarray:
    '''Reference, single and double configuration weights of a CCSD bra-ket pair.

    Parameters
    ----------
    amplitudes : np.ndarray
        Packed vector [t1, t2, l1, l2], real or complex, of length
        2 (o v + o^2 v^2) with o = n_electrons and v = n_spin_orbitals - o;
        the doubles arrays must be antisymmetric in (i, j) and in (a, b).
    n_electrons : int
        Number of occupied spin orbitals o, 1 <= o < n_spin_orbitals.
    n_spin_orbitals : int
        Total number of spin orbitals, at least 2.

    Returns
    -------
    weights : np.ndarray
        Real vector of length 1 + o v + (o (o - 1) / 2) (v (v - 1) / 2):
        [W_0, W_i^a in C order, W_ij^ab for i < j, a < b].

    Raises
    ------
    ValueError
        If n_spin_orbitals is not an integer of at least 2, if n_electrons is
        not an integer in [1, n_spin_orbitals - 1], if amplitudes is not a
        finite 1D vector of the stated length, or if either doubles array is
        not antisymmetric in (i, j) and in (a, b) within 1e-10.
    '''
    return weights

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _cw_coefficients(t1, t2, l1, l2):
    """Ket coefficients (c1, c2) and bra coefficients (~c0, ~c1, ~c2); the ket reference coefficient is one."""
    tt = np.einsum('ia,jb->ijab', t1, t1)
    c1 = t1
    c2 = t2 + tt - tt.transpose(0, 1, 3, 2)
    ct0 = (1.0 - np.einsum('ia,ia->', l1, t1) - 0.25 * np.einsum('ijab,ijab->', l2, t2)
           + 0.5 * np.einsum('ijab,ia,jb->', l2, t1, t1))
    ct1 = l1 - np.einsum('ijab,jb->ia', l2, t1)
    return c1, c2, ct0, ct1, l2


def _cw_weights(t1, t2, l1, l2):
    o, v = t1.shape
    c1, c2, ct0, ct1, ct2 = _cw_coefficients(t1, t2, l1, l2)
    occ_pairs = [(i, j) for i in range(o) for j in range(i + 1, o)]
    vir_pairs = [(a, b) for a in range(v) for b in range(a + 1, v)]
    w2 = [np.real(ct2[i, j, a, b] * c2[i, j, a, b]) for (i, j) in occ_pairs for (a, b) in vir_pairs]
    return np.concatenate([[np.real(ct0)], np.real(ct1 * c1).ravel(), np.asarray(w2, dtype=float)])


def _oracle_configuration_weights(amplitudes: np.ndarray, n_electrons: int, n_spin_orbitals: int) -> np.ndarray:
    if isinstance(n_spin_orbitals, bool) or not isinstance(n_spin_orbitals, (int, np.integer)) or n_spin_orbitals < 2:
        raise ValueError("n_spin_orbitals must be an integer of at least 2")
    if isinstance(n_electrons, bool) or not isinstance(n_electrons, (int, np.integer)) or not 1 <= n_electrons <= n_spin_orbitals - 1:
        raise ValueError("n_electrons must be an integer between 1 and n_spin_orbitals - 1")
    o, v = int(n_electrons), int(n_spin_orbitals) - int(n_electrons)
    y = np.asarray(amplitudes, dtype=complex)
    if y.ndim != 1 or y.size != 2 * (o * v + o * o * v * v) or not np.all(np.isfinite(y)):
        raise ValueError("amplitudes must be a finite 1D vector of length 2 (o v + o^2 v^2)")
    t1, t2, l1, l2 = _cc_unpack(y, o, v)
    for x in (t2, l2):
        if np.max(np.abs(x + x.transpose(1, 0, 2, 3)), initial=0.0) > 1e-10 or np.max(np.abs(x + x.transpose(0, 1, 3, 2)), initial=0.0) > 1e-10:
            raise ValueError("doubles amplitudes must be antisymmetric in (i, j) and in (a, b)")
    return _cw_weights(t1, t2, l1, l2)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: four electrons in eight spin orbitals, random complex bra and ket amplitudes ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(701)
o, v = 4, 4
def amp(shape, s):
    return rng.normal(scale=s, size=shape) + 1j * rng.normal(scale=s, size=shape)
def asym(x):
    x = x - x.transpose(1, 0, 2, 3)
    return x - x.transpose(0, 1, 3, 2)
amplitudes = np.concatenate([amp((o, v), 0.2).ravel(), asym(amp((o, o, v, v), 0.1)).ravel(),
                             amp((o, v), 0.2).ravel(), asym(amp((o, o, v, v), 0.1)).ravel()])
""",
            "call": "configuration_weights(copy.deepcopy(amplitudes), 4, 8)",
            "gold_call": "_oracle_configuration_weights(copy.deepcopy(amplitudes), 4, 8)",
            "tol": 1e-12,
        },
        # --- Boundary: all amplitudes zero, the bare reference determinant ---
        {
            "setup": """import numpy as np
import copy
amplitudes = np.zeros(2 * (2 * 3 + 4 * 9))
""",
            "call": "configuration_weights(copy.deepcopy(amplitudes), 2, 5)",
            "gold_call": "_oracle_configuration_weights(copy.deepcopy(amplitudes), 2, 5)",
            "tol": 1e-12,
        },
        # --- Edge: large amplitudes that drive some weights negative, three electrons in seven spin orbitals ---
        {
            "setup": """import numpy as np
import copy
rng = np.random.default_rng(703)
o, v = 3, 4
def amp(shape, s):
    return rng.normal(scale=s, size=shape) + 1j * rng.normal(scale=s, size=shape)
def asym(x):
    x = x - x.transpose(1, 0, 2, 3)
    return x - x.transpose(0, 1, 3, 2)
amplitudes = np.concatenate([amp((o, v), 0.8).ravel(), asym(amp((o, o, v, v), 0.6)).ravel(),
                             amp((o, v), 0.8).ravel(), asym(amp((o, o, v, v), 0.6)).ravel()])
""",
            "call": "configuration_weights(copy.deepcopy(amplitudes), 3, 7)",
            "gold_call": "_oracle_configuration_weights(copy.deepcopy(amplitudes), 3, 7)",
            "tol": 1e-12,
        },
        # --- Invalid: cluster doubles that are not antisymmetric ---
        {
            "setup": """import numpy as np
import copy
o, v = 2, 2
t2 = np.zeros((o, o, v, v))
t2[0, 1, 0, 1] = 0.1
amplitudes = np.concatenate([np.zeros(o * v), t2.ravel(), np.zeros(o * v), np.zeros(o * o * v * v)])
def run_model():
    try:
        configuration_weights(copy.deepcopy(amplitudes), 2, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_configuration_weights(copy.deepcopy(amplitudes), 2, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]
