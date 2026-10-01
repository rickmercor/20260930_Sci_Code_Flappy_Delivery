"""
Assign every bead pair of the duplex to one of the four edENM-NA contact classes defined in Table 1 of the source, or to no contact at all. Return an integer code matrix using 0 for no contact, 1 for the intra-nucleoside class, 2 for the backbone class, 3 for the base-base class and 4 for the van der Waals class. The class definitions, including which bead pairs belong to each class, which sequence separations qualify and the distance criteria that single out specific base-base contacts, are those of the source; the long-range interaction cutoff for the distance-decaying classes is supplied as an argument.

The source replaces the uniform-spring network of its reference with classes of contact that were identified from the apparent force constants of atomistic simulations: the plots behind Table 1 show separate populations for the sugar-base link inside one nucleotide, for the phosphodiester backbone, for specific nucleobase interactions, and for everything else. Two of those populations are essentially independent of distance, so the source treats them as fixed-strength contacts; the other two decay with distance. Classifying the pairs is therefore the step that decides which spring law applies.

Returns
-------
np.ndarray of shape (N, N), int64: symmetric matrix of contact class codes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def contact_class_matrix(seq_a: str, coords: "np.ndarray", cutoff_na: float) -> "np.ndarray":
    """Assign every bead pair of the duplex to one of the four edENM-NA contact classes
    defined in Table 1 of the source, or to no contact at all. Return an integer code
    matrix using 0 for no contact, 1 for the intra-nucleoside class, 2 for the backbone
    class, 3 for the base-base class and 4 for the van der Waals class. The class
    definitions, including which bead pairs belong to each class, which sequence
    separations qualify and the distance criteria that single out specific base-base
    contacts, are those of the source; the long-range interaction cutoff for the
    distance-decaying classes is supplied as an argument.

    Parameters
    ----------
    seq_a : str
        the strand A sequence the network was built from; it fixes the bead
        table the row and column order refers to.
    coords : np.ndarray of shape (N, 3)
        bead coordinates in angstrom, in the bead order of the previous step.
    cutoff_na : float
        intramolecular interaction cutoff in angstrom, applied to the distance-
        decaying classes.

    Returns
    -------
    np.ndarray of shape (N, N), int64: symmetric matrix of contact class codes

    Raises
    ------
    ValueError
        if coords does not have shape (N, 3), if its bead count does not match
        the bead table implied by seq_a, or if cutoff_na is not positive.
    """
    return classes

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


def _oracle_contact_class_matrix(seq_a: str, coords: "np.ndarray", cutoff_na: float) -> "np.ndarray":
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    labels, bases, strands, nt = _bead_table(seq_a)
    if len(labels) != len(coords):
        raise ValueError("coords does not match the bead table implied by seq_a")
    if cutoff_na <= 0:
        raise ValueError("cutoff_na must be positive")
    CLASS_NONE, CLASS_INTRA, CLASS_BACKBONE, CLASS_BASEBASE, CLASS_VDW = 0, 1, 2, 3, 4
    D = _distances(coords)
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
                # intra-nucleoside: the sugar-base link inside one nucleotide, no cutoff
                code = CLASS_INTRA
            elif pair == {"C2"} and ((is_cg and 4.1 <= d <= 4.4) or (is_gg and 6.7 <= d <= 7.0)):
                # base-base: identity-and-distance rule of the source, no pairing test; the
                # two windows bound the distance themselves, so the cutoff does not apply
                code = CLASS_BASEBASE
            elif d <= cutoff_na:
                if (pair == {"P", "C1"} or pair == {"P"}) and same_strand:
                    # backbone: P-C1' and P-P pairs along ONE strand only; the same bead
                    # pairs across strands are ordinary Van der Waals contacts
                    code = CLASS_BACKBONE
                else:
                    code = CLASS_VDW
            out[i, j] = out[j, i] = code
    return out

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
""",
            "call": "contact_class_matrix(seq, coords, 11.0)",
            "gold_call": "_oracle_contact_class_matrix(seq, coords, 11.0)",
        },
        {
            "setup": """import numpy as np
seq = "GC"
coords = _oracle_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
""",
            "call": "contact_class_matrix(seq, coords, 4.2)",
            "gold_call": "_oracle_contact_class_matrix(seq, coords, 4.2)",
        },
        {
            "setup": """import numpy as np
seq = "GGGGCCCC"
coords = _oracle_bead_network(seq, 36.0, 2.60, 140.0, 9.40, 5.20, 2.100)
""",
            "call": "contact_class_matrix(seq, coords, 30.0)",
            "gold_call": "_oracle_contact_class_matrix(seq, coords, 30.0)",
        },
        {
            "setup": """import numpy as np
seq = "GAGC"
coords = np.array([[8.91, 0.0, 0.0], [5.84, 0.0, 0.0], [2.444, 0.0, 0.0], [7.4978610671326775, 4.813541255455906, 3.1], [4.914422966560588, 3.1550034715895054, 3.1]])
def run_model():
    try:
        contact_class_matrix(seq, coords, 11.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_contact_class_matrix(seq, coords, 11.0)
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
