# Physics-Condensed_Matter_Physics-15

## Background

Conventional harmonic phonon calculations derive lattice vibrations from a mass-weighted force-constant matrix. Even when magnetic or other electronic order breaks time-reversal symmetry, the conventional dynamical matrix can retain artificial time-reversal and mirror symmetries. It may therefore predict degenerate phonon branches even when the symmetry-broken lattice dynamics supports modes with opposite phonon angular momentum and experiments observe separate peaks.

The adiabatic electronic ground state changes as the ions move. The geometric phase accumulated during this evolution generates a molecular Berry curvature, which acts as a velocity-dependent force on the nuclei. In the lattice equation of motion, this contribution produces a frequency-dependent term that can lift phonon degeneracies, generate modes with opposite angular momentum, and reveal symmetry breaking inherited from the electronic system.

The source paper develops a first-principles framework for incorporating this effect into lattice dynamics. For the ferromagnetic Weyl semimetal $\mathrm{Co_3Sn_2S_2}$, the framework explains the splitting of the Raman-active $E_g$ modes through molecular Berry curvature. For the infrared-active $E_u$ modes, the intrinsic molecular-Berry-curvature splitting is much smaller than the observed peak separation, and the source paper attributes the additional separation primarily to a branch-dependent Fano shift arising from interference between phonon and electronic excitations. This distinction connects electronic order, electron–phonon coupling, chiral phonons, phonon magnetism, and Hall-type lattice responses.

## Problem

A finite six-coordinate lattice-dynamics calculation uses the electronic-order-induced molecular-Berry-curvature framework to resolve a nominally twofold-degenerate optical phonon and then applies branch-resolved Fano corrections to the observable peak positions. For the deterministic instance below, calculate the absolute separation of the two corrected peaks in $\mathrm{cm}^{-1}$.

The force-constant matrix is

$$\Phi=\begin{pmatrix}59679.484666666664&0&-29804.330987819010&0&-22269.663077830097&0\\0&82329.401000000000&0&2341.968010268314&0&-34773.709425369052\\-29804.330987819010&0&120213.660000000000&0&-31606.617761896014&0\\0&2341.968010268314&0&165837.870000000000&0&-49353.209257352260\\-22269.663077830097&0&-31606.617761896014&0&37467.453333333338&0\\0&-34773.709425369052&0&-49353.209257352260&0&71333.500000000015\end{pmatrix}.$$

The coordinate masses are

$$\mathbf m=\begin{pmatrix}58.933&58.933&118.710&118.710&32.060&32.060\end{pmatrix}.$$

The valence- and conduction-state energies at the five sampled $\mathbf{k}$ points are

$$\boldsymbol{\varepsilon}_v=\begin{pmatrix}-1.20&-1.05&-1.35&-1.10&-1.28\end{pmatrix}\,\mathrm{meV},\qquad \boldsymbol{\varepsilon}_c=\begin{pmatrix}0.95&1.10&0.88&1.22&1.03\end{pmatrix}\,\mathrm{meV}.$$

The mass-normalized electron–phonon coupling matrix is

$$g_{cv}=\begin{pmatrix}0.653499585001107+0.049682995078799i&0.063874877690685-0.374064565267564i&-0.519956847935370+0.049682995078799i&0.063874877690685+0.799391867668913i&0.473268600995773-0.099365990157599i&0.063874877690685-0.109146657832457i\\0.451756613101524+0.026843836803511i&0.034511743703323-0.719329299328893i&-0.917275891991032+0.026843836803511i&0.034511743703323+0.649703205763663i&-0.029511023213302-0.053687673607021i&0.034511743703323+0.227716415849677i\\0.860830739376333-0.020675421232230i&-0.026581328285814-0.640894033485497i&-0.573393789768250-0.020675421232230i&-0.026581328285814+0.793330495659086i&0.143718474804042+0.041350842464459i&-0.026581328285814-0.152436462173589i\\0.808183469783874-0.049185792336648i&-0.063235649635050-0.508027567868974i&-0.300080939100576-0.049185792336648i&-0.063235649635050+0.600236841015477i&0.050802649110197+0.098371584673296i&-0.063235649635050-0.250299595430949i\\0.633055030255410-0.032474972798655i&-0.041751406335960-0.839177814692008i&-0.670785450785120-0.032474972798655i&-0.041751406335960+0.464662666348522i&-0.425362442727759+0.064949945597311i&-0.041751406335960+0.058334503774594i\end{pmatrix}.$$

The remaining parameters are

$$\omega_0=37.0\,\mathrm{meV},\qquad \mathcal P_{xy}=\bigl\{(0,1),(2,3),(4,5)\bigr\},\qquad \boldsymbol{\Gamma}=\begin{pmatrix}0.095&0.095\end{pmatrix}\,\mathrm{meV},\qquad \mathbf q^{-1}=\begin{pmatrix}0.775&0.000\end{pmatrix},\qquad C_{\mathrm{meV}\rightarrow\mathrm{cm}^{-1}}=8.065543937.$$

The coordinate order is $(x_1,y_1,x_2,y_2,x_3,y_3)$; the index pairs in $\mathcal P_{xy}$ use zero-based indexing; each row of $g_{cv}$ is already expressed in the mass-weighted phonon-coordinate convention and represents one of the five $\mathbf{k}$ points containing one valence and one conduction state; and the supplied force constants and masses are scaled so that the eigenvalues of the mass-weighted dynamical matrix are in $\mathrm{meV}^2$. Using the source paper’s molecular-Berry-curvature construction, verify the computed curvature matrix’s reality and antisymmetry and obtain the degenerate-subspace first-order frequency pair near $\omega_0$, carrying the first-order eigenvalue shifts $\lambda_s$ to the frequency linearly as $\omega_s=\omega_0+\lambda_s/(2\omega_0)$, including the associated normalized two-vector displacement basis $B$.

Solve the full frequency-dependent phonon equation using any algebraically equivalent companion, generalized-eigenvalue, or direct quadratic-eigenvalue formulation, retaining its positive-frequency physical branches. Projection-match the target pair using $P_B=BB^\dagger$ and the normalized weights

$$p_j=\frac{\epsilon_j^\dagger P_B\epsilon_j}{\epsilon_j^\dagger\epsilon_j},$$

selecting the two modes with the largest weights rather than assuming that adjacent sorted frequencies form the target pair.

For every pair in $\mathcal P_{xy}$, use

$$J_z^{(xy)}=\begin{pmatrix}0&-i\\i&0\end{pmatrix}$$

and evaluate the normalized chirality

$$\chi_j=\frac{\epsilon_j^\dagger J_z\epsilon_j}{\epsilon_j^\dagger\epsilon_j}.$$

Label the positive-chirality mode as the ${+}$ branch and the negative-chirality mode as the ${-}$ branch, interpret $\boldsymbol{\Gamma}$ and $\mathbf q^{-1}$ in the order $({+},{-})$, apply the branch-resolved Fano peak correction defined by the source paper, take the absolute corrected peak separation, and convert it using $C_{\mathrm{meV}\rightarrow\mathrm{cm}^{-1}}$.

Use binary64 arithmetic without intermediate rounding and round only the final result to 11 decimal places. In `<reasoning>`, give the paper-derived molecular-Berry-curvature identity and full frequency-dependent phonon equation, the mass-weighting relation, the two curvature-structure residuals, the projected first-order result, the projection-matched full frequencies and weights, the normalized chiralities, and the branch-resolved corrected peaks; equivalent algebraic formulations and eigenproblem linearizations are acceptable, and the complete unperturbed and full companion spectra should not be reported.
## Output format

Scientific reasoning wrapped in `<reasoning>...</reasoning>` tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in `<final_answer>...</final_answer>` tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
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

build_mass_weighted_dynamical_matrix

Goal
----
Construct the mass-weighted dynamical matrix.

```python
def build_mass_weighted_dynamical_matrix(
    phi: "np.ndarray",
    masses: "np.ndarray",
) -> "np.ndarray":
    """Construct the mass-weighted dynamical matrix.

    Parameters
    ----------
    phi : np.ndarray
        Real symmetric force-constant matrix with shape $(N, N)$.
    masses : np.ndarray
        One-dimensional array of $N$ strictly positive coordinate masses.

    Returns
    -------
    mass_weighted_matrix : np.ndarray
        Real symmetric array with shape $(N, N)$.

    Raises
    ------
    ValueError
        If either input cannot be represented by a real numeric array; if phi
        is not a nonempty square matrix; if masses does not contain exactly one
        entry per coordinate; if either input contains a nonfinite value; if
        any mass is not strictly positive; or if phi is not symmetric with
        rtol=0 and atol=$10^{-12}$.
    """
    return None
```

### Step 2

compute_molecular_berry_curvature

Goal
----
Compute the real antisymmetric molecular Berry-curvature matrix.

```python
def compute_molecular_berry_curvature(
    eps_v: "np.ndarray",
    eps_c: "np.ndarray",
    g_cv: "np.ndarray",
) -> "np.ndarray":
    """Compute the real antisymmetric molecular Berry-curvature matrix.

    Parameters
    ----------
    eps_v : np.ndarray
        One-dimensional real valence-energy array with shape $(N_k,)$.
    eps_c : np.ndarray
        One-dimensional real conduction-energy array with shape $(N_k,)$.
    g_cv : np.ndarray
        Complex electron-phonon couplings with shape $(N_k, N)$.

    Returns
    -------
    curvature : np.ndarray
        Real antisymmetric molecular Berry-curvature matrix with shape $(N, N)$.

    Raises
    ------
    ValueError
        If the energy arrays are not nonempty, real, finite, one-dimensional,
        and equal in length; if g_cv is not a finite two-dimensional array with
        one row per k point and at least one coordinate; or if any conduction
        energy is not strictly greater than its corresponding valence energy.
    """
    return None
```

### Step 3

analyze_unperturbed_subspace

Goal
----
Return the unperturbed spectrum and target-subspace projector.

```python
def analyze_unperturbed_subspace(
    k_tilde: "np.ndarray",
    omega0: float,
    degeneracy_tolerance: float = 1.0e-8,
) -> "np.ndarray":
    """Return the unperturbed spectrum and target-subspace projector.

    Parameters
    ----------
    k_tilde : np.ndarray
        Real symmetric positive-semidefinite matrix with shape $(N, N)$.
    omega0 : float
        Strictly positive finite target frequency.
    degeneracy_tolerance : float
        Strictly positive finite absolute tolerance for selecting frequencies
        equal to omega0.

    Returns
    -------
    analysis : np.ndarray
        Real array with shape $(N + 1, N)$. The first row contains the sorted
        unperturbed frequencies and the remaining rows contain the rank-two
        orthogonal projector onto the target degenerate subspace.

    Raises
    ------
    ValueError
        If k_tilde cannot be represented by a real numeric array; if it is not
        a nonempty finite square matrix symmetric with rtol=0 and atol=$10^{-12}$;
        if it has an eigenvalue below -$10^{-10}$; if omega0 or
        degeneracy_tolerance is not strictly positive and finite; or if
        exactly two frequencies are not within degeneracy_tolerance of omega0.
    """
    return None
```

### Step 4

compute_first_order_frequencies

Goal
----
Compute the two first-order MBC-split modes.

```python
def compute_first_order_frequencies(
    unperturbed_analysis: "np.ndarray",
    g_tilde: "np.ndarray",
    omega0: float,
) -> "np.ndarray":
    """Compute the two first-order MBC-split modes.

    Parameters
    ----------
    unperturbed_analysis : np.ndarray
        Real packed array from analyze_unperturbed_subspace with shape
        $(N + 1, N)$.
    g_tilde : np.ndarray
        Real antisymmetric curvature matrix with shape $(N, N)$.
    omega0 : float
        Strictly positive finite degenerate frequency.

    Returns
    -------
    first_order_modes : np.ndarray
        Complex array with shape $(N + 1, 2)$. The first row contains the two
        first-order frequencies omega0 + lam/(2*omega0) in ascending order, where
        lam are the eigenvalues of the projected first-order operator, and the
        remaining $N$ rows contain their associated orthonormal displacement
        vectors.

    Raises
    ------
    ValueError
        If unperturbed_analysis is not a finite real array with shape
        $(N + 1, N)$, a nonnegative sorted spectrum, and a symmetric idempotent
        rank-two projector within atol=$10^{-8}$; if g_tilde is not a finite real
        antisymmetric $(N, N)$ array within atol=$10^{-12}$; or if omega0 is not
        strictly positive and finite.
    """
    return None
```

### Step 5

solve_full_mbc_eigenproblem

Goal
----
Solve the companion linearization of the full MBC eigenproblem.

```python
def solve_full_mbc_eigenproblem(
    k_tilde: "np.ndarray",
    g_tilde: "np.ndarray",
    imaginary_tolerance: float = 1.0e-9,
) -> "np.ndarray":
    """Solve the companion linearization of the full MBC eigenproblem.

    Parameters
    ----------
    k_tilde : np.ndarray
        Real symmetric positive-definite matrix with shape $(N, N)$.
    g_tilde : np.ndarray
        Real antisymmetric matrix with shape $(N, N)$.
    imaginary_tolerance : float
        Strictly positive finite upper bound on the magnitude of an eligible
        companion root's imaginary part.

    Returns
    -------
    modes : np.ndarray
        Complex array with shape $(N + 1, N)$. The first row contains ascending
        positive frequencies and each remaining column contains the associated
        normalized displacement vector.

    Raises
    ------
    ValueError
        If k_tilde or g_tilde cannot be represented by real numeric arrays; if
        they are not equal-size, nonempty, finite square matrices; if k_tilde
        is not symmetric within atol=$10^{-12}$ or is not positive definite; if
        g_tilde is not antisymmetric within atol=$10^{-12}$; if
        imaginary_tolerance is not strictly positive and finite; if the
        companion spectrum does not contain exactly $N$ eligible positive roots;
        or if an eligible root has a zero or nonfinite displacement norm.
    """
    return None
```

### Step 6

match_target_modes

Goal
----
Select the two full modes with greatest target-subspace weight.

```python
def match_target_modes(
    first_order_modes: "np.ndarray",
    full_modes: "np.ndarray",
) -> "np.ndarray":
    """Select the two full modes with greatest target-subspace weight.

    Parameters
    ----------
    first_order_modes : np.ndarray
        Complex packed table from compute_first_order_frequencies with shape
        $(N + 1, 2)$. Its first row contains two ascending positive
        first-order frequencies, and its remaining rows contain the associated
        orthonormal displacement basis.
    full_modes : np.ndarray
        Complex modal table with shape $(N + 1, M)$, $M$ at least two. Its first
        row contains positive real frequencies and the remaining rows contain
        normalized displacement vectors.

    Returns
    -------
    matched_modes : np.ndarray
        Complex array with shape $(N + 2, 2)$. Row zero contains ascending target
        frequencies, row one contains their projection weights, and the final
        $N$ rows contain their normalized displacement vectors.

    Raises
    ------
    ValueError
        If first_order_modes is not a finite complex array with shape
        $(N + 1, 2)$; if its frequencies are not ascending, positive, and real
        within atol=$10^{-9}$; if its displacement columns are not orthonormal
        within atol=$10^{-8}$; if full_modes does not have shape $(N + 1, M)$ with $M$
        at least two; if its entries are nonfinite; if its frequencies are not
        positive and real within atol=$10^{-9}$; or if its displacement columns are
        not normalized within atol=$10^{-8}$.
    """
    return None
```

### Step 7

label_modes_by_chirality

Goal
----
Compute chiralities and order the two modes as plus and minus branches.

```python
def label_modes_by_chirality(
    matched_modes: "np.ndarray",
    xy_pairs: "np.ndarray",
) -> "np.ndarray":
    """Compute chiralities and order the two modes as plus and minus branches.

    Parameters
    ----------
    matched_modes : np.ndarray
        Complex array with shape $(N + 2, 2)$ from match_target_modes. Row zero
        contains frequencies, row one contains projection weights, and the
        remaining $N$ rows contain normalized displacement vectors.
    xy_pairs : np.ndarray
        Integer array with shape $(N/2, 2)$ that partitions all coordinate
        indices into ordered Cartesian (x, y) pairs.

    Returns
    -------
    branch_data : np.ndarray
        Real array with shape $(2, 2)$. Row zero contains frequencies and row one
        contains chiralities; columns are ordered [plus, minus].

    Raises
    ------
    ValueError
        If matched_modes is not a finite numeric array with shape $(N + 2, 2)$,
        $N$ at least two; if its frequencies are not positive and real within
        atol=$10^{-9}$; if its weights are not real and in $[0,1]$ within atol=$10^{-9}$;
        if its displacement vectors are not normalized within atol=$10^{-8}$; if
        xy_pairs is not an integer array that covers every coordinate exactly
        once; if a chirality expectation has an imaginary component larger
        than $10^{-9}$; or if the two chiralities are not respectively separable
        into one strictly positive and one strictly negative value.
    """
    return None
```

### Step 8

compute_corrected_peak_separation

Goal
----
Return the chirality-resolved, Fano-corrected peak separation.

```python
def compute_corrected_peak_separation(
    phi: "np.ndarray",
    masses: "np.ndarray",
    eps_v: "np.ndarray",
    eps_c: "np.ndarray",
    g_cv: "np.ndarray",
    omega0: float,
    xy_pairs: "np.ndarray",
    gamma: "np.ndarray",
    inv_q: "np.ndarray",
    conversion: float,
) -> float:
    r"""Return the chirality-resolved, Fano-corrected peak separation.

    Parameters
    ----------
    phi : np.ndarray
        Real symmetric force-constant matrix with shape $(N, N)$.
    masses : np.ndarray
        One-dimensional array of $N$ strictly positive coordinate masses.
    eps_v : np.ndarray
        One-dimensional valence-energy array with shape $(N_k,)$.
    eps_c : np.ndarray
        One-dimensional conduction-energy array with shape $(N_k,)$.
    g_cv : np.ndarray
        Complex coupling array with shape $(N_k, N)$.
    omega0 : float
        Strictly positive finite target frequency.
    xy_pairs : np.ndarray
        Integer Cartesian-pair array covering all $N$ coordinates exactly once.
    gamma : np.ndarray
        Two finite nonnegative linewidths ordered [plus, minus].
    inv_q : np.ndarray
        Two finite inverse Fano parameters ordered [plus, minus].
    conversion : float
        Strictly positive finite conversion factor from meV to $\mathrm{cm}^{-1}$.

    Returns
    -------
    separation : float
        Absolute corrected peak separation in $\mathrm{cm}^{-1}$, rounded at the final step
        to 11 decimal places.

    Raises
    ------
    ValueError
        Under any invalid-input condition declared by the preceding seven
        steps; if gamma or inv_q is not a finite one-dimensional array of
        length two; if either linewidth is negative; or if conversion is not
        strictly positive and finite.
    """
    return None
```
