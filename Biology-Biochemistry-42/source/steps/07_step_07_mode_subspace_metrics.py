"""
For each of the first n_modes vibrational modes of the network, return three numbers in this column order: the collectivity degree of the mode as defined in equation (9) of the source, with each bead weighted by the Euclidean magnitude of its displacement vector in the mode (not by its square), the overlap of the mode with the first reference vector as defined in equation (6), and the largest overlap of that mode over the supplied reference vectors. The reference vectors are supplied as columns and are not assumed orthonormal. The same modes are excluded here as in the spectrum step. The input must be the Hessian of a connected three-dimensional network, which has exactly six such zero-frequency modes; an eigenvalue counts as zero-frequency when its magnitude is at most 1e-8 times the largest eigenvalue magnitude, and a Hessian with any other number of zero-frequency modes is rejected.

Two questions are asked of a computed mode: does it point along a motion seen experimentally, and is that motion global or localised. The first is answered by a normalised alignment between the mode and a reference direction, which is one for parallel vectors and zero for orthogonal ones and is insensitive to the arbitrary sign of an eigenvector. The second is answered by a collectivity measure built from the distribution of displacement magnitude over the beads: it reaches its maximum when every bead moves equally and its minimum when only one bead moves. The source gives the exact definition it uses for each.

Returns
-------
np.ndarray of shape (n_modes, 3), float64: per-mode collectivity, overlap with the first reference vector, and maximum overlap over the reference vectors
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mode_subspace_metrics(hessian: "np.ndarray", reference: "np.ndarray",
                          n_modes: int) -> "np.ndarray":
    """For each of the first n_modes vibrational modes of the network, return three numbers
    in this column order: the collectivity degree of the mode as defined in equation (9)
    of the source, with each bead weighted by the Euclidean magnitude of its displacement
    vector in the mode (not by its square), the overlap of the mode with the
    first reference vector as defined in equation (6), and the largest overlap of that
    mode over the supplied reference vectors. The reference
    vectors are supplied as columns and are not assumed orthonormal. The same modes are
    excluded here as in the spectrum step. The input must be the Hessian of a connected
    three-dimensional network, which has exactly six such zero-frequency modes; an
    eigenvalue counts as zero-frequency when its magnitude is at most 1e-8 times the
    largest eigenvalue magnitude, and a Hessian with any other number of zero-frequency
    modes is rejected.

    Parameters
    ----------
    hessian : np.ndarray of shape (3M, 3M)
        the symmetric Hessian of the network.
    reference : np.ndarray of shape (3M, n_ref)
        reference deformation fields as columns, each flattened bead-major and
        not assumed normalised; column 0 is the field the second output column
        is measured against.
    n_modes : int
        how many of the lowest vibrational modes to report metrics for.

    Returns
    -------
    np.ndarray of shape (n_modes, 3), float64: per-mode collectivity, overlap with the first reference vector, and maximum overlap over the reference vectors

    Raises
    ------
    ValueError
        if hessian is not square, if reference does not have one row per
        coordinate of the network, if n_modes is not a positive integer or exceeds
        the number of vibrational modes the network has, if the number of
        zero-frequency eigenvalues, counted with the stated tolerance, is not six,
        if a mode has zero displacement on every bead, or if a reference column is
        the zero vector.
    """
    return metrics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _vibrational_modes(hessian):
    """Eigen-decompose a symmetric Hessian and strip exactly six zero-frequency modes.

    An eigenvalue counts as zero when its magnitude is at most 1e-8 times the largest
    eigenvalue magnitude. A network floating in space has six such modes (three
    translations, three rotations); any other count means the input is not the Hessian
    of a connected three-dimensional network and is rejected.
    """
    H = np.asarray(hessian, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] % 3 != 0:
        raise ValueError("hessian must be square with 3M rows")
    w, V = np.linalg.eigh(H)
    order = np.argsort(w)
    w, V = w[order], V[:, order]
    scale = float(np.abs(w).max())
    n_zero = int((np.abs(w) <= 1e-8 * scale).sum()) if scale > 0.0 else len(w)
    if n_zero != 6:
        raise ValueError("the network must have exactly six zero-frequency modes")
    return w[6:], V[:, 6:]


def _reference_fields(coords: "np.ndarray", n_ref: int) -> "np.ndarray":
    """Declared in the prompt: analytic probe fields, column-normalised.

    Columns, in order: bend-x, bend-y, stretch-z, twist, radial breathing, each
    evaluated at the bead coordinates and flattened bead-major.
    """
    a = np.asarray(coords, dtype=float)
    z = a[:, 2] - a[:, 2].mean()
    x, y = a[:, 0], a[:, 1]
    o = np.zeros_like(z)
    fields = [np.c_[z ** 2, o, o], np.c_[o, z ** 2, o], np.c_[o, o, z],
              np.c_[-y, x, o], np.c_[x, y, o]]
    cols = []
    for f in fields[:n_ref]:
        v = f.ravel()
        cols.append(v / np.linalg.norm(v))
    return np.asarray(cols, dtype=float).T


def _oracle_mode_subspace_metrics(hessian: "np.ndarray", reference: "np.ndarray",
                          n_modes: int) -> "np.ndarray":
    H = np.asarray(hessian, dtype=float)
    R = np.asarray(reference, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("hessian must be square")
    if R.ndim != 2 or R.shape[0] != H.shape[0]:
        raise ValueError("reference must have shape (3M, n_ref)")
    if isinstance(n_modes, bool) or int(n_modes) != n_modes or n_modes <= 0 or n_modes > H.shape[0] - 6:
        raise ValueError("n_modes must be a positive integer no larger than the number of vibrational modes")
    w, V = _vibrational_modes(H)
    n_ref = R.shape[1]
    out = np.zeros((n_modes, 3), dtype=float)
    O = np.zeros((n_modes, n_ref), dtype=float)
    for i in range(n_modes):
        vi = V[:, i]
        nv = np.linalg.norm(vi)
        # per-bead weight: the magnitude of the bead's displacement vector (not its square)
        delta = np.linalg.norm(vi.reshape(-1, 3), axis=1)
        total = delta.sum()
        if total <= 0.0:
            raise ValueError("a mode has zero displacement everywhere")
        p = delta / total
        p = p[p > 0.0]
        out[i, 0] = float(np.exp(-(p * np.log(p)).sum()) / len(delta))
        for j in range(n_ref):
            wj = R[:, j]
            nw = np.linalg.norm(wj)
            if nw <= 0.0:
                raise ValueError("a reference vector is the zero vector")
            O[i, j] = abs(float(vi @ wj)) / (nv * nw)
        out[i, 1] = O[i, 0]
        out[i, 2] = O[i].max()
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
def _fx_hessian_matrix(coords, springs):
    coords = np.asarray(coords, dtype=float)
    springs = np.asarray(springs, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (M, 3)")
    if springs.shape != (len(coords), len(coords)):
        raise ValueError("springs must be square and match coords")
    if not np.allclose(springs, springs.T, rtol=0.0, atol=1e-12):
        raise ValueError("springs must be symmetric")
    n = len(coords)
    H = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j or springs[i, j] == 0.0:
                continue
            dr = coords[j] - coords[i]
            d2 = float(dr @ dr)
            if d2 <= 0.0:
                raise ValueError("two connected beads share a position")
            blk = -(springs[i, j] / d2) * np.outer(dr, dr)
            H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = blk
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= blk
    return H
coords = _fx_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _fx_contact_class_matrix(seq, coords, 11.0)
k_na = _fx_nucleic_spring_matrix(seq, coords, classes)
H = _fx_hessian_matrix(coords, k_na)
R = _reference_fields(coords, 5)
""",
            "call": "mode_subspace_metrics(H, R, 10)",
            "gold_call": "_oracle_mode_subspace_metrics(H, R, 10)",
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
def _fx_hessian_matrix(coords, springs):
    coords = np.asarray(coords, dtype=float)
    springs = np.asarray(springs, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (M, 3)")
    if springs.shape != (len(coords), len(coords)):
        raise ValueError("springs must be square and match coords")
    if not np.allclose(springs, springs.T, rtol=0.0, atol=1e-12):
        raise ValueError("springs must be symmetric")
    n = len(coords)
    H = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j or springs[i, j] == 0.0:
                continue
            dr = coords[j] - coords[i]
            d2 = float(dr @ dr)
            if d2 <= 0.0:
                raise ValueError("two connected beads share a position")
            blk = -(springs[i, j] / d2) * np.outer(dr, dr)
            H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = blk
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= blk
    return H
coords = _fx_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _fx_contact_class_matrix(seq, coords, 11.0)
k_na = _fx_nucleic_spring_matrix(seq, coords, classes)
H = _fx_hessian_matrix(coords, k_na)
R = _reference_fields(coords, 1)
""",
            "call": "mode_subspace_metrics(H, R, 1)",
            "gold_call": "_oracle_mode_subspace_metrics(H, R, 1)",
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
def _fx_hessian_matrix(coords, springs):
    coords = np.asarray(coords, dtype=float)
    springs = np.asarray(springs, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (M, 3)")
    if springs.shape != (len(coords), len(coords)):
        raise ValueError("springs must be square and match coords")
    if not np.allclose(springs, springs.T, rtol=0.0, atol=1e-12):
        raise ValueError("springs must be symmetric")
    n = len(coords)
    H = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j or springs[i, j] == 0.0:
                continue
            dr = coords[j] - coords[i]
            d2 = float(dr @ dr)
            if d2 <= 0.0:
                raise ValueError("two connected beads share a position")
            blk = -(springs[i, j] / d2) * np.outer(dr, dr)
            H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = blk
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= blk
    return H
coords = _fx_bead_network(seq, 32.7, 3.10, 121.0, 8.91, 5.84, 2.444)
classes = _fx_contact_class_matrix(seq, coords, 11.0)
k_na = _fx_nucleic_spring_matrix(seq, coords, classes)
H = _fx_hessian_matrix(coords, k_na)
R = _reference_fields(coords, 5) * 1e-8
""",
            "call": "mode_subspace_metrics(H, R, 4)",
            "gold_call": "_oracle_mode_subspace_metrics(H, R, 4)",
        },
        {
            "setup": """import numpy as np
H = np.eye(30)
R = np.ones((12, 2))
def run_model():
    try:
        mode_subspace_metrics(H, R, 3)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mode_subspace_metrics(H, R, 3)
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
coords = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 2.0], [0.0, 40.0, 5.0], [2.0, 43.0, 7.0]])
springs = np.zeros((4, 4))
springs[0, 1] = springs[1, 0] = 5.0
springs[2, 3] = springs[3, 2] = 5.0
def _fx_hessian_matrix(coords, springs):
    coords = np.asarray(coords, dtype=float)
    springs = np.asarray(springs, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (M, 3)")
    if springs.shape != (len(coords), len(coords)):
        raise ValueError("springs must be square and match coords")
    if not np.allclose(springs, springs.T, rtol=0.0, atol=1e-12):
        raise ValueError("springs must be symmetric")
    n = len(coords)
    H = np.zeros((3 * n, 3 * n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j or springs[i, j] == 0.0:
                continue
            dr = coords[j] - coords[i]
            d2 = float(dr @ dr)
            if d2 <= 0.0:
                raise ValueError("two connected beads share a position")
            blk = -(springs[i, j] / d2) * np.outer(dr, dr)
            H[3 * i:3 * i + 3, 3 * j:3 * j + 3] = blk
            H[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= blk
    return H
H = _fx_hessian_matrix(coords, springs)
R = _reference_fields(coords, 2)
def run_model():
    try:
        mode_subspace_metrics(H, R, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_mode_subspace_metrics(H, R, 2)
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
