# Physics-Optics-10

## Background

Bianisotropic media materials whose electric and magnetic responses are cross-coupled, as in chiral metamaterials -- support wave equations that are naturally non-Hermitian even in the complete absence of loss or gain, because the coupling terms enter the generalized Maxwell eigenvalue problem antisymmetrically. This makes such media a natural, and largely unexplored, setting for the exceptional-point phenomena that have driven much recent interest in non-Hermitian photonics (parity-time-symmetric lasers, unidirectional invisibility, enhanced sensing), phenomena usually studied instead in explicitly lossy or gain-loaded photonic and acoustic systems.

This paper develops a general analytic theory of the phase transitions undergone by the eigenmodes of a bianisotropic medium's wave operator as its constitutive tensors are varied, working directly with the exact 6x6 generalized eigenvalue problem for the electromagnetic field rather than any reduced or perturbative model. It shows that the possible coalescence loci fall into a small number of qualitatively distinct classes rather than being a single generic phenomenon as in simple two-band non-Hermitian models and derives closed-form conditions and, for one such class, closed-form wavevector solutions, for several concrete families of bianisotropic media, including uniaxial dielectrics endowed with isotropic chirality. These exact results let the theory's predictions be checked and applied without any numerical search over the medium's full non-Hermitian spectrum.

## Problem

n an ordinary non-Hermitian two-band system, an exceptional point (EP), a parameter locus where two eigenmodes coalesce into a single degenerate eigenvector rather than merely sharing an eigenvalue is generic: it appears upon tuning just one real parameter. A recent analytic theory of the full six-dimensional (E, H) wave operator of a general bianisotropic optical medium shows that this picture is incomplete in two ways. First, exceptional points of this operator come in several qualitatively distinct types, distinguished by how the medium's dispersion relation degenerates near the coalescence locus. Second, for the finite-frequency members of one particular type, the theory provides closed-form algebraic expressions for the wavevector components at which the coalescence occurs, written purely in terms of the medium's permittivity, permeability, and chirality tensors, no numerical diagonalization of the wave operator or root-search on its characteristic polynomial is needed. Your task is to evaluate this closed-form finite-frequency exceptional-point locus for one specific bianisotropic medium and frequency, and report the magnitude of the wavevector at which the coalescence occurs.

Here is the exact setup to use:

- Medium: a uniaxial dielectric with isotropic chirality, i.e. a bianisotropic medium whose (dimensionless, relative) constitutive tensors are diagonal/isotropic in the paper's own basis: relative permittivity eps = diag(4, 4, eps_z), relative permeability mu = I_3 (the identity, i.e. non-magnetic), and chirality tensor gamma = gamma0 * I_3 (isotropic chirality, gamma0 >= 0). Use eps_z = 9.0 and gamma0 = 2.5.
- The paper gives an explicit, closed-form classification of exactly which (eps_z, gamma0) combinations admit a ring of finite-frequency exceptional points in wavevector space (a one-parameter family of coalescing wavevectors related to one another by the medium's rotational symmetry about its optic axis) at all. Consult the paper's own case-by-case derivation of this classification directly; do not assume a single generic double-root or discriminant condition applies uniformly for every eps_z. Confirm from this classification that the (eps_z, gamma0) pair above does admit a finite-frequency exceptional-point ring before proceeding.
- At frequency omega = 1.0 (in the paper's own dimensionless frequency units), use the paper's closed-form finite-frequency exceptional-point solution to compute the squared axial wavevector component kz^2 and the squared transverse radius kx^2 + ky^2 on this ring. Both quantities are built from the same underlying algebraic pieces (the medium's reduced dispersion coefficients and a square-root discriminant built from gamma0), but do not assume the paper combines these pieces the same way for both components. Consult the paper's own equations for the exact combination used in each case, rather than reusing one component's combination for the other.
- A commonly tempting shortcut is to assume that this finite-frequency ring lies along the same (kz^2 : kx^2+ky^2) direction as the medium's zero-frequency degenerate cone (the omega -> 0 limit of the dispersion relation, which the paper also gives in closed form). Check directly, from the paper's own finite-frequency solution, whether this assumption actually holds for this specific medium and frequency before relying on it, rather than assuming it does.

Report k_total = sqrt(kz^2 + kx^2 + ky^2), the total wavevector magnitude on the finite-frequency exceptional-point ring at this configuration. The answer is graded to an absolute tolerance of 1e-6. In your reasoning, report kz^2 and kx^2+ky^2 individually, state which of the paper's classification regimes the given (eps_z, gamma0) pair falls into and why a finite-frequency ring exists there, and separately report the (kz^2 : kx^2+ky^2) ratio you would obtain from rescaling the zero-frequency degenerate-cone direction by omega^2, for direct comparison against the ratio you computed from the paper's actual finite-frequency solution.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 2.29, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste full derivations of the paper's classification or closed-form equations.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

reduced_dispersion_coefficients

Goal
----
Compute the two reduced dispersion coefficients that recur throughout the analysis of a uniaxial dielectric medium with isotropic magneto-electric chirality (a gyroelectric metamaterial with constitutive tensors epsilon = diag(4, 4, eps_z), mu = I_3x3, and gamma = gamma0 * I_3x3).

```python
def reduced_dispersion_coefficients(eps_z: float, gamma0: float) -> "np.ndarray":
    """Transverse and axial reduced dispersion coefficients.

    Args:
        eps_z (float): axial (out-of-plane) relative permittivity of the uniaxial
            medium. The in-plane permittivity is fixed at 4 for this model.
        gamma0 (float): isotropic chirality parameter, gamma0 >= 0.

    Raises:
        ValueError: if gamma0 < 0.

    Expected return:
        np.ndarray of shape (2,): [A_perp, A_par]. A_perp vanishes at gamma0 = 2;
        A_par vanishes at gamma0 = sqrt(eps_z).
    """
    return None
```

### Step 2

equatorial_critical_gamma0_sq

Goal
----
Compute the critical squared-chirality value at which the medium's equatorial (kz = 0) exceptional-point ring is born, for axial permittivities in the regime -12 < eps_z < -4.

```python
def equatorial_critical_gamma0_sq(eps_z: float) -> float:
    """Critical gamma0^2 for the equatorial finite-frequency EP ring.

    Args:
        eps_z (float): axial relative permittivity, must satisfy
            -12 < eps_z < -4 for this critical condition to be physically
            meaningful.

    Raises:
        ValueError: if eps_z <= -12 or eps_z >= -4.

    Expected return:
        float: critical value of gamma0^2, strictly positive.
    """
    return 0.0
```

### Step 3

classify_ep_sector

Goal
----
Classify which finite-frequency exceptional-point (EP) regime a given (eps_z, gamma0) point of the gyroelectric-chiral medium falls into.

```python
def classify_ep_sector(eps_z: float, gamma0: float) -> int:
    """Finite-frequency EP-ring existence classifier.

    Args:
        eps_z (float): axial relative permittivity of the uniaxial medium.
        gamma0 (float): isotropic chirality parameter, gamma0 >= 0.

    Raises:
        ValueError: if gamma0 < 0.

    Expected return:
        int: 2 if a finite-frequency exceptional-point ring exists for this
        parameter pair, 0 if it does not.
    """
    return 0
```

### Step 4

ep_ring_kz_squared

Goal
----
Compute the squared axial wavevector component kz^2 on the finite-frequency exceptional-point ring of the gyroelectric-chiral medium, at a prescribed frequency omega, for parameter pairs (eps_z, gamma0) that admit such a ring (see the sector classifier).

```python
def ep_ring_kz_squared(eps_z: float, gamma0: float, omega: float) -> float:
    """Squared axial wavevector on the finite-frequency EP ring.

    Args:
        eps_z (float): axial relative permittivity of the uniaxial medium.
        gamma0 (float): isotropic chirality parameter, gamma0 >= 0.
        omega (float): frequency, omega > 0.

    Raises:
        ValueError: if omega <= 0, or if gamma0^2 * (gamma0^2 - 4) < 0 (no real
            exceptional locus at this gamma0), or if the construction is
            degenerate for this (eps_z, gamma0) pair.

    Expected return:
        float: kz^2 evaluated at the finite-frequency exceptional-point ring,
        scaling as omega^2.
    """
    return 0.0
```

### Step 5

ep_ring_transverse_squared

Goal
----
Compute the squared transverse wavevector radius kx^2 + ky^2 on the finite-frequency exceptional-point ring of the gyroelectric-chiral medium, at a prescribed frequency omega, for parameter pairs (eps_z, gamma0) that admit such a ring (see the sector classifier).

```python
def ep_ring_transverse_squared(eps_z: float, gamma0: float, omega: float) -> float:
    """Squared transverse wavevector radius on the finite-frequency EP ring.

    Args:
        eps_z (float): axial relative permittivity of the uniaxial medium.
        gamma0 (float): isotropic chirality parameter, gamma0 >= 0.
        omega (float): frequency, omega > 0.

    Raises:
        ValueError: if omega <= 0, or if gamma0^2 * (gamma0^2 - 4) < 0 (no real
            exceptional locus at this gamma0), or if the construction is
            degenerate for this (eps_z, gamma0) pair.

    Expected return:
        float: kx^2 + ky^2 evaluated at the finite-frequency exceptional-point
        ring, scaling as omega^2.
    """
    return 0.0
```

### Step 6

combine_ep_wavevector

Goal
----
Assemble the total wavevector magnitude on the finite-frequency exceptional-point ring from its previously computed axial and transverse squared components.

```python
def combine_ep_wavevector(kz2: float, rho: float) -> "np.ndarray":
    """Assemble the total EP-ring wavevector magnitude from its parts.

    Args:
        kz2 (float): squared axial wavevector component, kz2 >= 0.
        rho (float): squared transverse wavevector radius (kx^2 + ky^2), rho >= 0.

    Raises:
        ValueError: if kz2 < 0 or rho < 0.

    Expected return:
        np.ndarray of shape (3,): [kz2, rho, k_total] with
        k_total = sqrt(kz2 + rho).
    """
    return None
```

### Step 7

run_metamaterial_ep_pipeline

Goal
----
Chain the six earlier steps into the full pipeline: reduce the constitutive tensors to their dispersion coefficients, use those coefficients to gate physical admissibility, classify whether a finite-frequency exceptional-point ring exists, compute its axial and transverse squared wavevector components at the prescribed frequency, and combine them into the total wavevector magnitude. The reference implementation calls the earlier public functions by name rather than reproducing their contents.

```python
def run_metamaterial_ep_pipeline(eps_z: float, gamma0: float, omega: float) -> float:
    """Chain the earlier steps and report the EP-ring wavevector magnitude.

    Args:
        eps_z (float): axial relative permittivity of the uniaxial medium.
        gamma0 (float): isotropic chirality parameter, gamma0 >= 0.
        omega (float): frequency, omega > 0.

    Raises:
        ValueError: if omega <= 0, if gamma0 < 0, if the reduced dispersion
            coefficients rule out physical admissibility of a finite-frequency
            ring, or if this (eps_z, gamma0) pair does not admit a
            finite-frequency exceptional-point ring.

    Expected return:
        float: k_total, the total wavevector magnitude on the finite-frequency
        exceptional-point ring.
    """
    return 0.0
```
