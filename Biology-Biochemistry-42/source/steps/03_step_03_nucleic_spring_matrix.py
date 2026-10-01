"""
Turn the contact class codes into the edENM-NA spring constant matrix. Each class carries either a fixed strength or the distance-decaying law of equation (2) of the source, with the strength and decay parameters listed for that class in Table 1. One class additionally distinguishes two chemical sub-classes of nucleotide, which the source assigns different fixed strengths. Entries with no contact stay zero and the result is symmetric.

Fitting the apparent force constants of atomistic simulations against inter-bead distance gives, per contact class, either a flat relationship or a decaying one. The source encodes the decaying classes with a single two-parameter expression, equation (2), whose parameters were first estimated by regression against the simulation force constants and then refined by a random search that maximised the agreement between the resulting normal modes and the principal components of the training ensembles. The numerical values that came out of that refinement are the ones tabulated in the source.

Returns
-------
np.ndarray of shape (N, N), float64: symmetric spring constants in kcal/(mol angstrom^2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def nucleic_spring_matrix(seq_a: str, coords: "np.ndarray",
                          class_matrix: "np.ndarray") -> "np.ndarray":
    """Turn the contact class codes into the edENM-NA spring constant matrix. Each class
    carries either a fixed strength or the distance-decaying law of equation (2) of the
    source, with the strength and decay parameters listed for that class in Table 1. One
    class additionally distinguishes two chemical sub-classes of nucleotide, which the
    source assigns different fixed strengths. Entries with no contact stay zero and the
    result is symmetric.

    Parameters
    ----------
    seq_a : str
        the strand A sequence the network was built from; it fixes the bead
        table the row and column order refers to.
    coords : np.ndarray of shape (N, 3)
        bead coordinates in angstrom, in the same bead order.
    class_matrix : np.ndarray of shape (N, N), integer
        the contact class codes of the previous step, with 0 for no contact.

    Returns
    -------
    np.ndarray of shape (N, N), float64: symmetric spring constants in kcal/(mol angstrom^2)

    Raises
    ------
    ValueError
        if class_matrix is not square or does not match coords, if the bead
        count does not match the bead table implied by seq_a, if class_matrix is not
        symmetric, if any entry of class_matrix, its diagonal included, carries a code
        outside the defined contact classes, or if a diagonal entry carries a contact
        code.
    """
    return springs

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _bead_table(seq_a):
    """Bead bookkeeping for the duplex: (labels, bases, strands, nt_index).

    Bead order is P, C1, C2 per nucleotide; strand A first in its 5->3 direction,
    then strand B in its own 5->3 direction.
    """
    complement = {"A": "U", "U": "A", "T": "A", "G": "C", "C": "G"}
    seq_b = "".join(complement[c] for c in reversed(seq_a))
    labels, bases, strands, nt = [], [], [], []
    for s, seq in ((0, seq_a), (1, seq_b)):
        for j, base in enumerate(seq):
            for lab in ("P", "C1", "C2"):
                labels.append(lab)
                bases.append(base)
                strands.append(s)
                nt.append(j)
    return labels, bases, strands, nt


def _distances(coords):
    a = np.asarray(coords, dtype=float)
    diff = a[:, None, :] - a[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def _oracle_nucleic_spring_matrix(seq_a: str, coords: "np.ndarray",
                          class_matrix: "np.ndarray") -> "np.ndarray":
    coords = np.asarray(coords, dtype=float)
    class_matrix = np.asarray(class_matrix, dtype=np.int64)
    if class_matrix.shape != (len(coords), len(coords)):
        raise ValueError("class_matrix must be square and match coords")
    labels, bases, strands, nt = _bead_table(seq_a)
    if len(labels) != len(coords):
        raise ValueError("coords does not match the bead table implied by seq_a")
    CLASS_NONE, CLASS_INTRA, CLASS_BACKBONE, CLASS_BASEBASE, CLASS_VDW = 0, 1, 2, 3, 4
    if not np.all(np.isin(class_matrix, (CLASS_NONE, CLASS_INTRA, CLASS_BACKBONE, CLASS_BASEBASE, CLASS_VDW))):
        raise ValueError("class_matrix carries a code outside the defined contact classes")
    if np.any(np.diag(class_matrix) != CLASS_NONE):
        raise ValueError("a bead cannot carry a contact with itself")
    if not np.array_equal(class_matrix, class_matrix.T):
        raise ValueError("class_matrix must be symmetric")
    pyrimidines = frozenset("CUT")
    D = _distances(coords)
    n = len(coords)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            code = int(class_matrix[i, j])
            if code == CLASS_NONE:
                continue
            d = D[i, j]
            if code == CLASS_INTRA:
                k = 290.0 if bases[i] in pyrimidines else 120.0
            elif code == CLASS_BACKBONE:
                k = (20.0 / d) ** 2.8
            elif code == CLASS_BASEBASE:
                k = 65.0
            else:
                k = (25.0 / d) ** 1.0
            K[i, j] = K[j, i] = k
    return K

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
seq = "GAGCGCUCAG"
coords = _oracle_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _oracle_contact_class_matrix(seq, coords, 11.0)
""",
            "call": "nucleic_spring_matrix(seq, coords, classes)",
            "gold_call": "_oracle_nucleic_spring_matrix(seq, coords, classes)",
        },
        {
            "setup": """import numpy as np
seq = "GC"
coords = _oracle_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _oracle_contact_class_matrix(seq, coords, 11.0)
""",
            "call": "nucleic_spring_matrix(seq, coords, classes)",
            "gold_call": "_oracle_nucleic_spring_matrix(seq, coords, classes)",
        },
        {
            "setup": """import numpy as np
seq = "UUUUAAAA"
coords = _oracle_bead_network(seq, 28.0, 3.60, 100.0, 8.20, 6.40, 3.000)
classes = _oracle_contact_class_matrix(seq, coords, 11.0)
""",
            "call": "nucleic_spring_matrix(seq, coords, classes)",
            "gold_call": "_oracle_nucleic_spring_matrix(seq, coords, classes)",
        },
        {
            "setup": """import numpy as np
seq = "GAGC"
def _fx_bead_network(seq_a, twist_deg, rise, phase_deg, r_p, r_c1, r_c2):
    if not seq_a or any(c not in "ACGU" for c in seq_a):
        raise ValueError("seq_a must be a non-empty sequence over A, C, G, U")
    if rise <= 0 or min(r_p, r_c1, r_c2) <= 0:
        raise ValueError("rise and the three bead radii must be positive")
    complement = {"A": "U", "U": "A", "T": "A", "G": "C", "C": "G"}
    length = len(seq_a)
    seq_b = "".join(complement[c] for c in reversed(seq_a))
    out = []
    for s, seq in ((0, seq_a), (1, seq_b)):
        for j in range(len(seq)):
            idx = j if s == 0 else length - 1 - j
            theta = np.deg2rad(idx * twist_deg) + (np.deg2rad(phase_deg) if s else 0.0)
            z = idx * rise
            for r in (r_p, r_c1, r_c2):
                out.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.asarray(out, dtype=float)
coords = _fx_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = np.zeros((3, 3), dtype=np.int64)
def run_model():
    try:
        nucleic_spring_matrix(seq, coords, classes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_nucleic_spring_matrix(seq, coords, classes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
seq = "GAGC"
def _fx_bead_network(seq_a, twist_deg, rise, phase_deg, r_p, r_c1, r_c2):
    if not seq_a or any(c not in "ACGU" for c in seq_a):
        raise ValueError("seq_a must be a non-empty sequence over A, C, G, U")
    if rise <= 0 or min(r_p, r_c1, r_c2) <= 0:
        raise ValueError("rise and the three bead radii must be positive")
    complement = {"A": "U", "U": "A", "T": "A", "G": "C", "C": "G"}
    length = len(seq_a)
    seq_b = "".join(complement[c] for c in reversed(seq_a))
    out = []
    for s, seq in ((0, seq_a), (1, seq_b)):
        for j in range(len(seq)):
            idx = j if s == 0 else length - 1 - j
            theta = np.deg2rad(idx * twist_deg) + (np.deg2rad(phase_deg) if s else 0.0)
            z = idx * rise
            for r in (r_p, r_c1, r_c2):
                out.append([r * np.cos(theta), r * np.sin(theta), z])
    return np.asarray(out, dtype=float)
def _fx_bead_table(seq_a):
    complement = {"A": "U", "U": "A", "T": "A", "G": "C", "C": "G"}
    seq_b = "".join(complement[c] for c in reversed(seq_a))
    labels, bases, strands, nt = [], [], [], []
    for s, seq in ((0, seq_a), (1, seq_b)):
        for j, base in enumerate(seq):
            for lab in ("P", "C1", "C2"):
                labels.append(lab)
                bases.append(base)
                strands.append(s)
                nt.append(j)
    return labels, bases, strands, nt
def _fx_distances(coords):
    a = np.asarray(coords, dtype=float)
    diff = a[:, None, :] - a[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))
def _fx_contact_class_matrix(seq_a, coords, cutoff_na):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    labels, bases, strands, nt = _fx_bead_table(seq_a)
    if len(labels) != len(coords):
        raise ValueError("coords does not match the bead table implied by seq_a")
    if cutoff_na <= 0:
        raise ValueError("cutoff_na must be positive")
    CLASS_NONE, CLASS_INTRA, CLASS_BACKBONE, CLASS_BASEBASE, CLASS_VDW = 0, 1, 2, 3, 4
    D = _fx_distances(coords)
    n = len(coords)
    out = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            pair = {labels[i], labels[j]}
            same_nt = strands[i] == strands[j] and nt[i] == nt[j]
            same_strand = strands[i] == strands[j]
            code = CLASS_NONE
            is_cg = {bases[i], bases[j]} == {"C", "G"}
            is_gg = bases[i] == "G" and bases[j] == "G"
            if pair == {"C2", "C1"} and same_nt:
                code = CLASS_INTRA
            elif pair == {"C2"} and ((is_cg and 4.1 <= d <= 4.4) or (is_gg and 6.7 <= d <= 7.0)):
                code = CLASS_BASEBASE
            elif d <= cutoff_na:
                if (pair == {"P", "C1"} or pair == {"P"}) and same_strand:
                    code = CLASS_BACKBONE
                else:
                    code = CLASS_VDW
            out[i, j] = out[j, i] = code
    return out
coords = _fx_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _fx_contact_class_matrix(seq, coords, 11.0)
classes[3, 3] = 4
def run_model():
    try:
        nucleic_spring_matrix(seq, coords, classes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_nucleic_spring_matrix(seq, coords, classes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]
