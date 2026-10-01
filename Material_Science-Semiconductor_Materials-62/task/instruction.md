# Material_Science-Semiconductor_Materials-62

## Background

The eight-band Kane k.p model describes the conduction band and the light-hole, heavy-hole and split-off valence bands of a zincblende III-V compound near the zone centre, and the Bir-Pikus terms add the effect of a uniform strain to it. Folding the valence bands into the conduction band, as a self-energy taken to lowest order in perturbation theory, gives a two-component effective conduction-band model whose parameters, the effective mass, the spin-orbit coupling and the effective g-tensor, are set by the interband coupling and by the gaps to the three valence bands. Without strain this folding is a textbook result. With strain the valence bands move by different amounts, the light and heavy holes are no longer degenerate at the zone centre, and the interband coupling acquires a strain correction, so the effective parameters are renormalized and become anisotropic. The source carries out that folding with strain included and obtains closed-form strain-renormalized parameters, which it validates against the full eight-band model. The conduction-band g-factor of a narrow-gap compound such as InAs is dominated by the spin-orbit-split valence bands, which is why it sits far from the free-electron value and why it responds so strongly to strain. The quantity asked for here is the anisotropy of that effective g-tensor between the growth axis and an in-plane axis, a number that vanishes identically for an isotropic g-factor whatever the strain.

## Problem

The source folds the eight-band Bir-Pikus k.p Hamiltonian of a uniformly strained zincblende III-V semiconductor down onto the conduction band, under the assumption that the interband coupling dominates the couplings among the valence bands, and splits the resulting effective conduction-band Hamiltonian into an isotropic part and an anisotropic part tied to the (001) axis of the cell. Each part carries its own Zeeman term with its own strain-renormalized effective g-tensor, written in terms of a strain-dependent tensor that takes the place of the identity of the unstrained result and of the three strained gaps to the light-hole, heavy-hole and split-off bands.

Work in the crystal frame with z along (001), with energies in eV and lengths in nm, so that hbar^2/(2 m_e) = 0.0380998212 eV nm^2, and take the free-electron g-factor g_e = 2.0023193. Use the InAs parameters at 0 K that the source tabulates: unstrained gap Delta_g = 0.417 eV, split-off gap Delta_soff = 0.390 eV, interband coupling P = 0.905 eV nm, deformation potentials a = 6.0 eV and b = 1.8 eV, and strain correction to the interband coupling d_cv = 3.6 eV.

Take the strain tensor

```
e = [[ 0.020, 0.006, 0.004],
     [ 0.006, 0.015, 0.008],
     [ 0.004, 0.008,-0.018]]
```

and write e_s = (e_yz, e_zx, e_xy) for the vector of its shear components and ||e_s|| for the Euclidean norm of that vector. Use the following strained gaps between the conduction band and the light-hole, heavy-hole and split-off bands; they are given, so do not rederive them:

```
Delta_LHB = Delta_g - (a + b/2) tr(e) - (3b/2) e_zz - d_cv ||e_s||
Delta_HHB = Delta_g - (a - b/2) tr(e) + (3b/2) e_zz + d_cv ||e_s||
Delta_SOB = Delta_g + Delta_soff - a tr(e)
```

Adopt the source's strain-renormalized isotropic and anisotropic effective g-tensors, and the way its anisotropic Zeeman term enters the Hamiltonian, exactly as the source states them. Evaluate the source's strain-dependent tensor factor through first order in the strain, discarding all terms of second and higher order in the strain components. Combine the two Zeeman terms into a single effective conduction-band g-tensor G, defined by writing their sum as (mu_B/2) B . G . sigma, with B the magnetic field and sigma the vector of Pauli matrices acting on the conduction band. Report the strain-induced anisotropy G_zz - G_xx, which is dimensionless.

State the conventions you adopted and justify each from the source.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

shear_strain_vector

Goal
----
Return the vector of the three independent shear strains of a symmetric 3x3 strain tensor, with the components in the order the source uses. Raise ValueError if e is not 3x3, not finite, or not symmetric.

```python
def shear_strain_vector(e: "np.ndarray") -> "np.ndarray":
    """Return the vector of the three independent shear strains of a symmetric 3x3 strain tensor, with the components in the order the source uses. Raise ValueError if e is not 3x3, not finite, or not symmetric.

    Parameters
    ----------
    e : numpy.ndarray
        Symmetric strain tensor of shape (3, 3) in the crystal frame (x, y, z along the cubic axes, z the (001) growth axis), dimensionless.

    Returns
    -------
    e_s : numpy.ndarray
        Array of shape (3,) holding the three shear strains in the source's order (float64).

    Raises
    ------
    ValueError
        If e is not of shape (3, 3), contains a non-finite entry, or is not symmetric.
    """
    return None
```

### Step 2

strained_band_gaps

Goal
----
Return the three strained gaps between the conduction band and, in order, the light-hole, heavy-hole and split-off bands, using exactly the three expressions given in the problem statement, with delta_g and delta_soff the unstrained gaps, a and b the deformation potentials, d_cv the strain correction to the interband coupling, e the strain tensor and e_s its shear-strain vector. Raise ValueError if either unstrained gap is not positive, if d_cv is negative, if the shapes are wrong, or if any strained gap is not positive.

```python
def strained_band_gaps(delta_g: float, delta_soff: float, a: float, b: float, d_cv: float,
                               e: "np.ndarray", e_s: "np.ndarray") -> "np.ndarray":
    """Return the three strained gaps between the conduction band and, in order, the light-hole, heavy-hole and split-off bands, using exactly the three expressions given in the problem statement, with delta_g and delta_soff the unstrained gaps, a and b the deformation potentials, d_cv the strain correction to the interband coupling, e the strain tensor and e_s its shear-strain vector. Raise ValueError if either unstrained gap is not positive, if d_cv is negative, if the shapes are wrong, or if any strained gap is not positive.

    Parameters
    ----------
    delta_g : float
        Unstrained band gap Delta_g in eV, positive.
    delta_soff : float
        Unstrained split-off gap Delta_soff in eV, positive.
    a : float
        Hydrostatic deformation potential a in eV.
    b : float
        Uniaxial (shear) deformation potential b in eV.
    d_cv : float
        Strain correction to the interband coupling d_cv in eV, nonnegative.
    e : numpy.ndarray
        Symmetric strain tensor of shape (3, 3) in the crystal frame (x, y, z along the cubic axes, z the (001) growth axis), dimensionless.
    e_s : numpy.ndarray
        Shear-strain vector of shape (3,) from step 01.

    Returns
    -------
    gaps : numpy.ndarray
        Array of shape (3,): [gap to LHB, gap to HHB, gap to SOB] in eV (float64).

    Raises
    ------
    ValueError
        If delta_g or delta_soff is not positive, d_cv is negative, e is not (3, 3), e_s is not (3,), or any strained gap is not positive.
    """
    return None
```

### Step 3

strain_tensor_factor

Goal
----
Return the 3x3 strain-dependent tensor factor that the source's effective conduction-band parameters carry in place of the identity, evaluated through first order in the strain with all O(e^2) terms discarded, as the problem statement prescribes. It must reduce to the identity at zero strain. Raise ValueError if e is not 3x3, not finite, or not symmetric.

```python
def strain_tensor_factor(e: "np.ndarray") -> "np.ndarray":
    """Return the 3x3 strain-dependent tensor factor that the source's effective conduction-band parameters carry in place of the identity, evaluated through first order in the strain with all O(e^2) terms discarded, as the problem statement prescribes. It must reduce to the identity at zero strain. Raise ValueError if e is not 3x3, not finite, or not symmetric.

    Parameters
    ----------
    e : numpy.ndarray
        Symmetric strain tensor of shape (3, 3) in the crystal frame (x, y, z along the cubic axes, z the (001) growth axis), dimensionless.

    Returns
    -------
    adj : numpy.ndarray
        Array of shape (3, 3), symmetric, equal to the identity at zero strain (float64).

    Raises
    ------
    ValueError
        If e is not of shape (3, 3), contains a non-finite entry, or is not symmetric.
    """
    return None
```

### Step 4

isotropic_g_tensor

Goal
----
Return the isotropic strain-renormalized effective g-tensor of the conduction band as the source defines it, from the interband coupling P in eV nm, the tensor factor of the previous step, and the strained conduction-to-light-hole and conduction-to-split-off gaps in eV. Use hbar^2/(2 m_e) = 0.0380998212 eV nm^2 and the free-electron g-factor g_e = 2.0023193, both given in the problem statement. Raise ValueError if P or either gap is not positive, or if adj is not 3x3.

```python
def isotropic_g_tensor(P: float, adj: "np.ndarray", delta_lhb: float, delta_sob: float) -> "np.ndarray":
    """Return the isotropic strain-renormalized effective g-tensor of the conduction band as the source defines it, from the interband coupling P in eV nm, the tensor factor of the previous step, and the strained conduction-to-light-hole and conduction-to-split-off gaps in eV. Use hbar^2/(2 m_e) = 0.0380998212 eV nm^2 and the free-electron g-factor g_e = 2.0023193, both given in the problem statement. Raise ValueError if P or either gap is not positive, or if adj is not 3x3.

    Parameters
    ----------
    P : float
        Interband coupling P in eV nm, positive.
    adj : numpy.ndarray
        The 3x3 strain-dependent tensor factor of step 03.
    delta_lhb : float
        Strained conduction-to-light-hole gap in eV, positive.
    delta_sob : float
        Strained conduction-to-split-off gap in eV, positive.

    Returns
    -------
    g_iso : numpy.ndarray
        Array of shape (3, 3): the isotropic effective g-tensor, dimensionless (float64).

    Raises
    ------
    ValueError
        If P, delta_lhb or delta_sob is not positive, or adj is not of shape (3, 3).
    """
    return None
```

### Step 5

anisotropic_g_tensor

Goal
----
Return the anisotropic correction to the effective g-tensor that the source derives from the strain-induced splitting of the light and heavy holes, from P in eV nm, the tensor factor, and the strained conduction-to-light-hole and conduction-to-heavy-hole gaps in eV, using hbar^2/(2 m_e) = 0.0380998212 eV nm^2 as given in the problem statement. It must vanish identically when those two gaps are equal. Raise ValueError if P or either gap is not positive, or if adj is not 3x3.

```python
def anisotropic_g_tensor(P: float, adj: "np.ndarray", delta_lhb: float, delta_hhb: float) -> "np.ndarray":
    """Return the anisotropic correction to the effective g-tensor that the source derives from the strain-induced splitting of the light and heavy holes, from P in eV nm, the tensor factor, and the strained conduction-to-light-hole and conduction-to-heavy-hole gaps in eV, using hbar^2/(2 m_e) = 0.0380998212 eV nm^2 as given in the problem statement. It must vanish identically when those two gaps are equal. Raise ValueError if P or either gap is not positive, or if adj is not 3x3.

    Parameters
    ----------
    P : float
        Interband coupling P in eV nm, positive.
    adj : numpy.ndarray
        The 3x3 strain-dependent tensor factor of step 03.
    delta_lhb : float
        Strained conduction-to-light-hole gap in eV, positive.
    delta_hhb : float
        Strained conduction-to-heavy-hole gap in eV, positive.

    Returns
    -------
    g_ani : numpy.ndarray
        Array of shape (3, 3): the anisotropic g correction, dimensionless (float64).

    Raises
    ------
    ValueError
        If P, delta_lhb or delta_hhb is not positive, or adj is not of shape (3, 3).
    """
    return None
```

### Step 6

effective_g_tensor

Goal
----
Return the effective conduction-band g-tensor G obtained by rewriting the sum of the source's isotropic and anisotropic Zeeman terms as one term of the form (mu_B/2) B . G . sigma, with B the magnetic field and sigma the vector of Pauli matrices. Follow exactly which spin components the source's anisotropic term couples to; G is not required to be symmetric. Raise ValueError if either input is not 3x3.

```python
def effective_g_tensor(g_iso: "np.ndarray", g_ani: "np.ndarray") -> "np.ndarray":
    """Return the effective conduction-band g-tensor G obtained by rewriting the sum of the source's isotropic and anisotropic Zeeman terms as one term of the form (mu_B/2) B . G . sigma, with B the magnetic field and sigma the vector of Pauli matrices. Follow exactly which spin components the source's anisotropic term couples to; G is not required to be symmetric. Raise ValueError if either input is not 3x3.

    Parameters
    ----------
    g_iso : numpy.ndarray
        Isotropic effective g-tensor of shape (3, 3) from step 04.
    g_ani : numpy.ndarray
        Anisotropic g correction of shape (3, 3) from step 05.

    Returns
    -------
    g : numpy.ndarray
        Array of shape (3, 3): the effective g-tensor G, dimensionless (float64), not necessarily symmetric.

    Raises
    ------
    ValueError
        If g_iso or g_ani is not of shape (3, 3).
    """
    return None
```

### Step 7

strain_g_anisotropy

Goal
----
Orchestrator. Build the shear-strain vector, the three strained gaps, the tensor factor, the isotropic g-tensor, the anisotropic correction and the effective g-tensor, in that order, and return the strain-induced anisotropy G_zz minus G_xx of the effective tensor. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if e is not 3x3.

```python
def strain_g_anisotropy(delta_g: float, delta_soff: float, P: float, a: float, b: float, d_cv: float,
                                e: "np.ndarray") -> float:
    """Orchestrator. Build the shear-strain vector, the three strained gaps, the tensor factor, the isotropic g-tensor, the anisotropic correction and the effective g-tensor, in that order, and return the strain-induced anisotropy G_zz minus G_xx of the effective tensor. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if e is not 3x3.

    Parameters
    ----------
    delta_g : float
        Unstrained band gap Delta_g in eV, positive.
    delta_soff : float
        Unstrained split-off gap Delta_soff in eV, positive.
    P : float
        Interband coupling P in eV nm, positive.
    a : float
        Hydrostatic deformation potential a in eV.
    b : float
        Uniaxial (shear) deformation potential b in eV.
    d_cv : float
        Strain correction to the interband coupling d_cv in eV, nonnegative.
    e : numpy.ndarray
        Symmetric strain tensor of shape (3, 3) in the crystal frame (x, y, z along the cubic axes, z the (001) growth axis), dimensionless.

    Returns
    -------
    dg : float
        G_zz - G_xx of the effective conduction-band g-tensor, dimensionless.

    Raises
    ------
    ValueError
        If e is not of shape (3, 3), or any earlier step rejects its input (closed gap, non-symmetric strain, non-positive parameter).
    """
    return None
```
