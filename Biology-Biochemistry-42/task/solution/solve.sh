#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np


def bead_network(seq_a: str, twist_deg: float, rise: float, phase_deg: float,
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


def contact_class_matrix(seq_a: str, coords: "np.ndarray", cutoff_na: float) -> "np.ndarray":
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


def nucleic_spring_matrix(seq_a: str, coords: "np.ndarray",
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


def interface_spring_matrix(seq_a: str, coords: "np.ndarray", k_na: "np.ndarray",
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

import numpy as np


def hessian_matrix(coords: "np.ndarray", springs: "np.ndarray") -> "np.ndarray":
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


def mode_spectrum(hessian: "np.ndarray", n_keep: int) -> "np.ndarray":
    H = np.asarray(hessian, dtype=float)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("hessian must be square")
    if isinstance(n_keep, bool) or int(n_keep) != n_keep or n_keep <= 0 or n_keep > H.shape[0] - 6:
        raise ValueError("n_keep must be a positive integer no larger than the number of vibrational modes")
    w, _ = _vibrational_modes(H)
    return np.asarray(w[:int(n_keep)], dtype=float)

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


def mode_subspace_metrics(hessian: "np.ndarray", reference: "np.ndarray",
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

import numpy as np


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


def binding_stiffening_index(
        seq_a: str = "GAGCGCUCAG",
        geometry: tuple = (32.7, 3.10, 121.0, 8.91, 5.84, 2.444),
        prot_coords: tuple = ((-2.588, 9.659, 3.500), (-8.660, -5.000, 14.000),
                              (1.531, 3.696, -3.000), (3.222, -0.424, 31.000),
                              (-9.093, 5.250, 9.500), (-11.897, -1.566, 12.000),
                              (3.827, -9.239, 20.500), (-4.401, -10.625, 18.000)),
        prot_residues: str = "KRDESTLA",
        n_modes: int = 10,
        kappa_min: float = 0.52) -> float:
    """Orchestrator. Chains steps 01-07 over the free duplex and the complex."""
    twist_deg, rise, phase_deg, r_p, r_c1, r_c2 = (float(v) for v in geometry)
    if not 0.0 < kappa_min < 1.0:
        raise ValueError("kappa_min must lie strictly between 0 and 1")
    if int(n_modes) <= 0:
        raise ValueError("n_modes must be positive")
    cutoff_na = 11.0
    cutoff_pna = 8.0
    coords = bead_network(seq_a, twist_deg, rise, phase_deg, r_p, r_c1, r_c2)
    classes = contact_class_matrix(seq_a, coords, cutoff_na)
    k_na = nucleic_spring_matrix(seq_a, coords, classes)
    k_complex = interface_spring_matrix(seq_a, coords, k_na, prot_coords,
                                                prot_residues, cutoff_pna)
    all_coords = np.vstack([np.asarray(coords, dtype=float),
                            np.asarray(prot_coords, dtype=float)])
    selected = []
    for crd, spr in ((coords, k_na), (all_coords, k_complex)):
        H = hessian_matrix(crd, spr)
        w = mode_spectrum(H, n_modes)
        metrics = mode_subspace_metrics(H, _reference_fields(crd, 5), n_modes)
        eligible = [i for i in range(n_modes) if metrics[i, 0] >= kappa_min]
        if not eligible:
            raise ValueError("no mode clears kappa_min in one of the two networks")
        best = max(eligible, key=lambda i: metrics[i, 1])
        selected.append(float(w[best]))
    return selected[1] / selected[0]
SCICODE_GOLD_EOF
