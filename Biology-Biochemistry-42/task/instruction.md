# Biology-Biochemistry-42

## Background

Normal mode analysis of coarse-grained elastic networks is the cheapest way to ask which large-scale deformations a biomolecule can perform: the structure is reduced to a few beads per residue joined by harmonic springs, the Hessian of that harmonic potential is diagonalised, and the lowest-frequency modes come out aligned with the functional motions seen in experimental ensembles. For proteins the approach has been refined for two decades, and the refinements that mattered were not better sampling but better springs, with stiffnesses that depend on what kind of contact each spring represents rather than a single constant everywhere.

Nucleic acids and their complexes with proteins were left behind by that refinement. The models in common use give every contact in a DNA or RNA network the same spring constant and a single interaction cutoff, which is a poor description of a molecule whose sugar-base linkage is nearly covalent, whose backbone stiffness falls off steeply with distance, and whose base pairing depends on the identity of the bases involved. Protein-nucleic acid interfaces are cruder still, since the chemistry of the contacting residue plainly matters for how tightly a protein grips a phosphate backbone or a nucleobase.

The work this task is built on closes that gap. Apparent force constants were extracted from a large set of atomistic simulations of DNA, RNA and protein-nucleic acid complexes, plotted against inter-bead distance, and used to identify distinct contact populations; each population then received its own spring law, and the parameters were refined against the principal components of experimental NMR and cryo-EM ensembles. The result is a parametrisation whose normal modes reproduce observed conformational transitions markedly better than the uniform-spring model it replaces, and which extends to complexes through a separate set of interface contact types.

The setting for this task is the question such a model is built to answer: when a protein binds a nucleic acid, how much stiffer does the molecule's functional bending motion become. That is a comparison of one mode before and after binding, which means the modes must first be matched to each other rather than compared by index, because the beads added at an interface carry local motions of their own that have no counterpart in the free molecule, wherever those motions happen to fall in the spectrum.

## Problem

Coarse-grained elastic network models describe the slow collective deformations of nucleic acids and of their complexes with proteins by replacing each residue with a few beads joined by harmonic springs, and a recent essential-dynamics-refined parametrisation for nucleic acids, together with its protein-nucleic acid extension, replaces the single uniform spring constant of earlier models with contact-class-specific stiffnesses fitted to atomistic force constants and then refined against experimental conformational ensembles. Your task is to quantify how much protein binding stiffens the bending motion of a short double-stranded RNA under that parametrisation. Build the three-bead sugar-base-phosphate network of the free duplex specified below, build the corresponding network of the protein-RNA complex by adding the protein-nucleic acid interface springs that parametrisation prescribes, run normal mode analysis on each, and compare the bending deformation before and after binding; classify every contact and give it the stiffness that parametrisation prescribes for its class, use the interaction cutoffs it specifies for intramolecular and for interface contacts, connect the interface the way it prescribes rather than the way a standard elastic network would, and evaluate the per-mode collectivity and the mode-to-reference alignment with the definitions that come with it. Report one number, the binding stiffening index defined at the end of this configuration block.

**Duplex.** Strand A is 5'-GAGCGCUCAG-3' and strand B is its Watson-Crick complement, so nucleotide j of strand A pairs with nucleotide 9 - j of strand B; three beads per nucleotide in the order phosphate, sugar, base; strand A laid down first along its 5'->3' direction, then strand B along its own 5'->3' direction.

**Geometry.** Rise level i of the duplex sits at helical angle i x 32.7 degrees and height i x 3.10 A, for i = 0 to 9. Nucleotide j of strand A occupies rise level j, and its Watson-Crick partner, nucleotide 9 - j of strand B, occupies that same rise level with an additional angular offset of 121.0 degrees, so the strands run antiparallel and both bases of a pair sit at one height. Bead radii from the helical axis are 8.91 A for the phosphate bead, 5.84 A for the sugar bead, 2.444 A for the base bead.

**Protein.** Eight Calpha beads, appended after the nucleic-acid beads in this order, at coordinates in A:

    K  ( -2.588,   9.659,   3.500)
    R  ( -8.660,  -5.000,  14.000)
    D  (  1.531,   3.696,  -3.000)
    E  (  3.222,  -0.424,  31.000)
    S  ( -9.093,   5.250,   9.500)
    T  (-11.897,  -1.566,  12.000)
    L  (  3.827,  -9.239,  20.500)
    A  ( -4.401, -10.625,  18.000)

Residue chemical classes are acidic {D, E}, basic {K, R}, polar {S, T, N, Q}, hydrophobic {A, V, L, I, F}; springs between two Calpha beads are out of scope, so only intramolecular nucleic-acid contacts and protein-RNA interface contacts are built.

**Reference deformations.** Five probe fields, evaluated at the bead coordinates of whichever network is under analysis, with z measured from the mean height of all beads of that network (protein beads included for the complex) and x, y the bead coordinates themselves: (1) (z^2, 0, 0), (2) (0, z^2, 0), (3) (0, 0, z), (4) (-y, x, 0), (5) (x, y, 0); each field is flattened bead-major into a 3M-vector and normalised, and field (1) is the bending reference.

**Binding stiffening index.** In each of the two networks, consider the ten lowest-frequency vibrational modes, keep only those whose collectivity degree is at least 0.52, then select among those the mode whose overlap with the bending reference is largest; the index is the selected eigenvalue of the complex divided by the selected eigenvalue of the free duplex. For the collectivity degree, weight each bead by the Euclidean magnitude of its displacement vector in the mode, not by its square; the functional form of the degree is the one the parametrisation comes with.

State the conventions you adopted and justify each from the source. Alongside the index, report these scalars: how many contacts fall in each contact class of your free duplex and what those classes are; how the sequence-specific class you use is made up; the spring constant your model gives to an A-U Watson-Crick pair; how many interface springs of each type you build; both interaction cutoffs; how many zero-frequency modes you discard from each network; and the selected eigenvalue of each network together with its collectivity degree.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_bead_network

Goal
----
Place the three-bead sugar-base-phosphate representation of the idealised antiparallel duplex described in the problem statement. Each nucleotide contributes one phosphate bead, one sugar bead and one base bead, in that order; strand A is laid down first along its 5'->3' direction and strand B second along its own 5'->3' direction, so that the two strands run antiparallel and paired nucleotides share a helical rise level. The helical parameters, the bead radii and the strand phase offset are all given in the problem statement.

```python
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
```

### Step 2

02_contact_class_matrix

Goal
----
Assign every bead pair of the duplex to one of the four edENM-NA contact classes defined in Table 1 of the source, or to no contact at all. Return an integer code matrix using 0 for no contact, 1 for the intra-nucleoside class, 2 for the backbone class, 3 for the base-base class and 4 for the van der Waals class. The class definitions, including which bead pairs belong to each class, which sequence separations qualify and the distance criteria that single out specific base-base contacts, are those of the source; the long-range interaction cutoff for the distance-decaying classes is supplied as an argument.

```python
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
```

### Step 3

03_nucleic_spring_matrix

Goal
----
Turn the contact class codes into the edENM-NA spring constant matrix. Each class carries either a fixed strength or the distance-decaying law of equation (2) of the source, with the strength and decay parameters listed for that class in Table 1. One class additionally distinguishes two chemical sub-classes of nucleotide, which the source assigns different fixed strengths. Entries with no contact stay zero and the result is symmetric.

```python
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
```

### Step 4

04_interface_spring_matrix

Goal
----
Extend the nucleic-acid network to the protein-nucleic acid complex by adding the interface springs of the source's edENM-PNA, and return the combined matrix with the nucleic-acid block in the leading positions and the protein beads appended in the order given. The source splits interface contacts into two types according to the chemical class of the protein residue and which nucleic-acid bead it touches, and gives each type its own parameters in Table 1 together with the interface cutoff; that assignment and those parameters are the source's. The source also departs from standard elastic network practice in the topology it builds at the interface, and that rule is the source's too. The residue chemical classes used here are the four stated in the problem statement.

```python
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
```

### Step 5

05_hessian_matrix

Goal
----
Assemble the 3M x 3M Hessian of the elastic network from the bead coordinates and the spring constant matrix, following equation (5) of the source for the off-diagonal three-by-three sub-blocks and completing each diagonal sub-block so that a rigid translation of the whole network costs no energy. Pairs with no spring contribute nothing.

```python
def hessian_matrix(coords: "np.ndarray", springs: "np.ndarray") -> "np.ndarray":
    """Assemble the 3M x 3M Hessian of the elastic network from the bead coordinates and
    the spring constant matrix, following equation (5) of the source for the off-
    diagonal three-by-three sub-blocks and completing each diagonal sub-block so that a
    rigid translation of the whole network costs no energy. Pairs with no spring
    contribute nothing.

    Parameters
    ----------
    coords : np.ndarray of shape (M, 3)
        bead coordinates in angstrom of the network under analysis.
    springs : np.ndarray of shape (M, M)
        symmetric spring constants in kcal/(mol angstrom^2); a zero entry means
        the two beads are not connected.

    Returns
    -------
    np.ndarray of shape (3M, 3M), float64: the symmetric Hessian

    Raises
    ------
    ValueError
        if coords does not have shape (M, 3), if springs is not square, does not
        match coords or is not symmetric, or if two beads joined by a spring sit
        at the same position.
    """
    return hessian
```

### Step 6

06_mode_spectrum

Goal
----
Diagonalise the Hessian and return the first n_keep eigenvalues of the vibrational modes, in ascending order. The modes that carry no vibrational information for a network floating in space are excluded, as the source states; how many of them there are follows from the rigid-body motions of a three-dimensional body. The input must be the Hessian of a connected three-dimensional network, which has exactly six such zero-frequency modes; an eigenvalue counts as zero-frequency when its magnitude is at most 1e-8 times the largest eigenvalue magnitude, and a Hessian with any other number of zero-frequency modes is rejected.

```python
def mode_spectrum(hessian: "np.ndarray", n_keep: int) -> "np.ndarray":
    """Diagonalise the Hessian and return the first n_keep eigenvalues of the vibrational
    modes, in ascending order. The modes that carry no vibrational information for a
    network floating in space are excluded, as the source states; how many of them there
    are follows from the rigid-body motions of a three-dimensional body. The input
    must be the Hessian of a connected three-dimensional network, which has exactly six
    such zero-frequency modes; an eigenvalue counts as zero-frequency when its magnitude
    is at most 1e-8 times the largest eigenvalue magnitude, and a Hessian with any other
    number of zero-frequency modes is rejected.

    Parameters
    ----------
    hessian : np.ndarray of shape (3M, 3M)
        the symmetric Hessian of the network.
    n_keep : int
        how many of the lowest vibrational eigenvalues to return.

    Returns
    -------
    np.ndarray of shape (n_keep,), float64: ascending eigenvalues of the vibrational modes

    Raises
    ------
    ValueError
        if hessian is not square, if n_keep is not a positive integer or exceeds
        the number of vibrational modes the network has, or if the number of
        zero-frequency eigenvalues, counted with the stated tolerance, is not six.
    """
    return eigenvalues
```

### Step 7

07_mode_subspace_metrics

Goal
----
For each of the first n_modes vibrational modes of the network, return three numbers in this column order: the collectivity degree of the mode as defined in equation (9) of the source, with each bead weighted by the Euclidean magnitude of its displacement vector in the mode (not by its square), the overlap of the mode with the first reference vector as defined in equation (6), and the largest overlap of that mode over the supplied reference vectors. The reference vectors are supplied as columns and are not assumed orthonormal. The same modes are excluded here as in the spectrum step. The input must be the Hessian of a connected three-dimensional network, which has exactly six such zero-frequency modes; an eigenvalue counts as zero-frequency when its magnitude is at most 1e-8 times the largest eigenvalue magnitude, and a Hessian with any other number of zero-frequency modes is rejected.

```python
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
```

### Step 8

08_binding_stiffening_index

Goal
----
Run the end-to-end audit. Build the network of the free duplex and the network of the protein-nucleic acid complex for the supplied configuration by calling the earlier steps in order, take the vibrational spectrum and the per-mode metrics of each network, keep only those modes whose collectivity reaches kappa_min, select among them the mode whose overlap with the first reference field is largest, and return the ratio of the selected eigenvalue of the complex to the selected eigenvalue of the free duplex. The two interaction cutoffs are the ones the source specifies for its two networks. The reference fields are the five probe fields (z^2, 0, 0), (0, z^2, 0), (0, 0, z), (-y, x, 0) and (x, y, 0), evaluated at the bead coordinates of the network under analysis with z measured from the mean height of all beads of that network (protein beads included for the complex) and x, y the bead coordinates themselves, each flattened bead-major into a 3M-vector and normalised; the first field is the bending reference. Raise ValueError if no mode of either network reaches kappa_min.

```python
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
    """Run the end-to-end audit. Build the network of the free duplex and the network of
    the protein-nucleic acid complex for the supplied configuration by calling the
    earlier steps in order, take the vibrational spectrum and the per-mode metrics of
    each network, keep only those modes whose collectivity reaches kappa_min, select
    among them the mode whose overlap with the first reference field is largest, and
    return the ratio of the selected eigenvalue of the complex to the selected
    eigenvalue of the free duplex. The two interaction cutoffs are the ones the source
    specifies for its two networks. The reference fields are the five probe fields
    (z^2, 0, 0), (0, z^2, 0), (0, 0, z), (-y, x, 0) and (x, y, 0), evaluated at the bead
    coordinates of the network under analysis with z measured from the mean height of
    all beads of that network (protein beads included for the complex) and x, y the bead
    coordinates themselves, each flattened bead-major into a 3M-vector and normalised;
    the first field is the bending reference. Raise ValueError if no mode of either
    network reaches kappa_min.

    Parameters
    ----------
    seq_a : str
        strand A of the duplex, written 5'->3'.
    geometry : tuple of length 6
        helical twist per nucleotide in degrees, helical rise per nucleotide in
        angstrom, angular offset of strand B in degrees, and the radial
        distances of the phosphate, sugar and base beads in angstrom, in that
        order.
    prot_coords : array-like of shape (P, 3)
        protein Calpha coordinates in angstrom, appended after the nucleic-acid
        beads in the order given.
    prot_residues : str
        one one-letter residue code per protein bead, in the same order as
        prot_coords.
    n_modes : int
        how many of the lowest vibrational modes of each network to consider.
    kappa_min : float
        the collectivity degree a mode must reach to stay in the comparison.

    Returns
    -------
    float: the ratio of the selected complex eigenvalue to the selected free-duplex eigenvalue

    Raises
    ------
    ValueError
        if kappa_min does not lie strictly between 0 and 1, if n_modes is not
        positive, or if no mode of either network reaches kappa_min.
    """
    return index
```
