"""
Place the three-bead sugar-base-phosphate representation of the idealised antiparallel duplex described in the problem statement. Each nucleotide contributes one phosphate bead, one sugar bead and one base bead, in that order; strand A is laid down first along its 5'->3' direction and strand B second along its own 5'->3' direction, so that the two strands run antiparallel and paired nucleotides share a helical rise level. The helical parameters, the bead radii and the strand phase offset are all given in the problem statement.

Coarse-grained elastic network models of nucleic acids replace each nucleotide by a small number of pseudo-atoms. The source adopts the three-bead sugar-base-phosphate topology of its reference [39] rather than a single bead per nucleotide, because the extra beads let the sugar, the base and the phosphate carry different stiffnesses. This step only builds the topology; no spring is assigned here.

Returns
-------
np.ndarray of shape (6 * len(seq_a), 3), float64: bead coordinates in angstrom
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def bead_network(seq_a: str, twist_deg: float, rise: float, phase_deg: float,
                 r_p: float, r_c1: float, r_c2: float) -> "np.ndarray":
    """Place the three-bead sugar-base-phosphate representation of the idealised
    antiparallel duplex described in the problem statement. Each nucleotide contributes
    one phosphate bead, one sugar bead and one base bead, in that order; strand A is
    laid down first along its 5'->3' direction and strand B second along its own 5'->3'
    direction, so that the two strands run antiparallel and paired nucleotides share a
    helical rise level. The helical parameters, the bead radii and the strand phase
    offset are all given in the problem statement.

    Parameters
    ----------
    seq_a : str
        strand A written 5'->3' over the letters A, C, G and U; strand B is its
        Watson-Crick complement and is generated from it.
    twist_deg : float
        helical twist per nucleotide in degrees, applied to the rise-level index;
        nucleotide j of strand A and its Watson-Crick partner share rise level j.
    rise : float
        helical rise per nucleotide in angstrom.
    phase_deg : float
        additional angular offset carried by strand B relative to strand A, in
        degrees.
    r_p : float
        radial distance of the phosphate bead from the helical axis, in
        angstrom.
    r_c1 : float
        radial distance of the sugar bead from the helical axis, in angstrom.
    r_c2 : float
        radial distance of the base bead from the helical axis, in angstrom.

    Returns
    -------
    np.ndarray of shape (6 * len(seq_a), 3), float64: bead coordinates in angstrom

    Raises
    ------
    ValueError
        if seq_a is empty or carries a letter outside A, C, G and U, or if rise
        or any of the three bead radii is not positive.
    """
    return coords

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_bead_network(seq_a: str, twist_deg: float, rise: float, phase_deg: float,
                 r_p: float, r_c1: float, r_c2: float) -> "np.ndarray":
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

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
seq = "GAGCGCUCAG"
""",
            "call": "bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)",
            "gold_call": "_oracle_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)",
        },
        {
            "setup": """import numpy as np
seq = "G"
""",
            "call": "bead_network(seq, 0.0, 1.0, 180.0, 9.0, 6.0, 2.5)",
            "gold_call": "_oracle_bead_network(seq, 0.0, 1.0, 180.0, 9.0, 6.0, 2.5)",
        },
        {
            "setup": """import numpy as np
seq = "GCGCGCGCGCGC"
""",
            "call": "bead_network(seq, 359.5, 0.001, 0.0, 30.0, 0.5, 0.25)",
            "gold_call": "_oracle_bead_network(seq, 359.5, 0.001, 0.0, 30.0, 0.5, 0.25)",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        bead_network("GATC", 32.7, 3.1, 121.0, 8.91, 5.84, 2.444)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_bead_network("GATC", 32.7, 3.1, 121.0, 8.91, 5.84, 2.444)
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
