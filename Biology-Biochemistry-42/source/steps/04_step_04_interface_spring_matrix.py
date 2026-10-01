"""
Extend the nucleic-acid network to the protein-nucleic acid complex by adding the interface springs of the source's edENM-PNA, and return the combined matrix with the nucleic-acid block in the leading positions and the protein beads appended in the order given. The source splits interface contacts into two types according to the chemical class of the protein residue and which nucleic-acid bead it touches, and gives each type its own parameters in Table 1 together with the interface cutoff; that assignment and those parameters are the source's. The source also departs from standard elastic network practice in the topology it builds at the interface, and that rule is the source's too. The residue chemical classes used here are the four stated in the problem statement.

For complexes the source parametrises the protein-nucleic acid contacts separately from the intramolecular ones, again from apparent force constants, and finds that the interface interactions split into a strongly distance-dependent group and a weakly dependent one depending on the chemistry of the residue and the bead it faces. The interface reaches a shorter distance than the intramolecular network. Protein-protein springs belong to the companion protein model and are outside this task: only the interface block and the nucleic-acid block are built here.

Returns
-------
np.ndarray of shape (N + P, N + P), float64: symmetric spring constants for the complex
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def interface_spring_matrix(seq_a: str, coords: "np.ndarray", k_na: "np.ndarray",
                            prot_coords: "np.ndarray", prot_residues: str,
                            cutoff_pna: float) -> "np.ndarray":
    """Extend the nucleic-acid network to the protein-nucleic acid complex by adding the
    interface springs of the source's edENM-PNA, and return the combined matrix with the
    nucleic-acid block in the leading positions and the protein beads appended in the
    order given. The source splits interface contacts into two types according to the
    chemical class of the protein residue and which nucleic-acid bead it touches, and
    gives each type its own parameters in Table 1 together with the interface cutoff;
    that assignment and those parameters are the source's. The source also departs from
    standard elastic network practice in the topology it builds at the interface, and
    that rule is the source's too. The residue chemical classes used here are the four
    stated in the problem statement.

    Parameters
    ----------
    seq_a : str
        the strand A sequence the nucleic-acid network was built from.
    coords : np.ndarray of shape (N, 3)
        nucleic-acid bead coordinates in angstrom, in the same bead order.
    k_na : np.ndarray of shape (N, N)
        the nucleic-acid spring constant matrix of the previous step.
    prot_coords : np.ndarray of shape (P, 3)
        protein Calpha coordinates in angstrom, appended after the nucleic-acid
        beads in the order given.
    prot_residues : str
        one one-letter residue code per protein bead, in the same order as
        prot_coords.
    cutoff_pna : float
        interface interaction cutoff in angstrom, applied to protein-nucleic
        acid contacts.

    Returns
    -------
    np.ndarray of shape (N + P, N + P), float64: symmetric spring constants for the complex

    Raises
    ------
    ValueError
        if prot_coords does not have shape (P, 3), if prot_residues does not
        carry one letter per protein bead, if cutoff_pna is not positive, if
        k_na is not square or does not match coords, or if a residue letter
        falls outside the chemical classes listed in the problem statement.
    """
    return springs_complex

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


def _oracle_interface_spring_matrix(seq_a: str, coords: "np.ndarray", k_na: "np.ndarray",
                            prot_coords: "np.ndarray", prot_residues: str,
                            cutoff_pna: float) -> "np.ndarray":
    coords = np.asarray(coords, dtype=float)
    prot_coords = np.asarray(prot_coords, dtype=float)
    k_na = np.asarray(k_na, dtype=float)
    if prot_coords.ndim != 2 or prot_coords.shape[1] != 3:
        raise ValueError("prot_coords must have shape (P, 3)")
    if len(prot_residues) != len(prot_coords):
        raise ValueError("prot_residues must carry one letter per protein bead")
    if cutoff_pna <= 0:
        raise ValueError("cutoff_pna must be positive")
    acidic, basic = frozenset("DE"), frozenset("KR")
    polar, hydrophobic = frozenset("STNQ"), frozenset("AVLIF")
    labels, bases, strands, nt = _bead_table(seq_a)
    n, p = len(coords), len(prot_coords)
    if k_na.shape != (n, n):
        raise ValueError("k_na must be square and match coords")
    K = np.zeros((n + p, n + p), dtype=float)
    K[:n, :n] = k_na
    known = acidic | basic | polar | hydrophobic
    nucleotides = {}
    for b in range(n):
        nucleotides.setdefault((strands[b], nt[b]), []).append(b)
    for a in range(p):
        res = prot_residues[a]
        if res not in known:
            raise ValueError("residue letter is not in a declared chemical class")
        for beads in nucleotides.values():
            best, best_d = -1, float("inf")
            for b in beads:
                d = float(np.sqrt(((prot_coords[a] - coords[b]) ** 2).sum()))
                if d < best_d:
                    best, best_d = b, d
            if best_d > cutoff_pna:
                continue
            type1 = ((labels[best] == "P" and (res in polar or res in basic))
                     or (labels[best] == "C2" and res in acidic))
            k = (25.0 / best_d) ** 2.2 if type1 else (30.0 / best_d) ** 1.0
            K[n + a, best] = K[best, n + a] = k
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
k_na = _oracle_nucleic_spring_matrix(seq, coords, classes)
prot = np.array([[-2.588, 9.659, 3.500], [1.531, 3.696, -3.000],
                 [3.827, -9.239, 20.500]])
""",
            "call": "interface_spring_matrix(seq, coords, k_na, prot, \"KDL\", 8.0)",
            "gold_call": "_oracle_interface_spring_matrix(seq, coords, k_na, prot, \"KDL\", 8.0)",
        },
        {
            "setup": """import numpy as np
seq = "GAGCGCUCAG"
coords = _oracle_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _oracle_contact_class_matrix(seq, coords, 11.0)
k_na = _oracle_nucleic_spring_matrix(seq, coords, classes)
prot = np.array([[60.0, 0.0, 0.0]])
""",
            "call": "interface_spring_matrix(seq, coords, k_na, prot, \"E\", 8.0)",
            "gold_call": "_oracle_interface_spring_matrix(seq, coords, k_na, prot, \"E\", 8.0)",
        },
        {
            "setup": """import numpy as np
seq = "GAGCGCUCAG"
coords = _oracle_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _oracle_contact_class_matrix(seq, coords, 11.0)
k_na = _oracle_nucleic_spring_matrix(seq, coords, classes)
prot = np.array([[-2.588, 9.659, 3.500], [-8.660, -5.000, 14.000],
                 [1.531, 3.696, -3.000], [3.222, -0.424, 31.000],
                 [-9.093, 5.250, 9.500], [-11.897, -1.566, 12.000],
                 [3.827, -9.239, 20.500], [-4.401, -10.625, 18.000]])
""",
            "call": "interface_spring_matrix(seq, coords, k_na, prot, \"KRDESTLA\", 25.0)",
            "gold_call": "_oracle_interface_spring_matrix(seq, coords, k_na, prot, \"KRDESTLA\", 25.0)",
        },
        {
            "setup": """import numpy as np
seq = "GAGCGCUCAG"
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
def _fx_nucleic_spring_matrix(seq_a, coords, class_matrix):
    coords = np.asarray(coords, dtype=float)
    class_matrix = np.asarray(class_matrix, dtype=np.int64)
    if class_matrix.shape != (len(coords), len(coords)):
        raise ValueError("class_matrix must be square and match coords")
    labels, bases, strands, nt = _fx_bead_table(seq_a)
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
    D = _fx_distances(coords)
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
coords = _fx_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _fx_contact_class_matrix(seq, coords, 11.0)
k_na = _fx_nucleic_spring_matrix(seq, coords, classes)
prot = np.array([[-2.588, 9.659, 3.500]])
def run_model():
    try:
        interface_spring_matrix(seq, coords, k_na, prot, "KD", 8.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_interface_spring_matrix(seq, coords, k_na, prot, "KD", 8.0)
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
