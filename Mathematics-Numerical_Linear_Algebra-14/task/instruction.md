# Basis-sensitive scalar from a half-size skew-symmetric eigensolve

## Background

# Scientific background

Real skew-symmetric matrices have purely imaginary eigenvalues in conjugate pairs. For an even order $2n$, a real spectral decomposition writes

$$
A=Q(J_2\otimes\Sigma)Q^T,
$$

where $Q$ is orthogonal, $J_2=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$, and $\Sigma$ is diagonal and nonnegative. A general eigensolver can recover the spectrum, but it ignores the real skew structure and solves a complex problem of the original order.

A structure-preserving route starts from a skew-symmetric orthogonal polar factor, which supplies a real basis that converts the matrix to skew-Hamiltonian form. That block form corresponds to a Hermitian eigenproblem of half the order, after which a real orthogonal spectral basis can be recovered. The reduction preserves the paired spectrum and allows standard Hermitian eigensolvers to perform the smaller spectral calculation.

The basis construction passes through a sampled complex invariant subspace, so a reproducible numerical task must fix the random generator, QR phases, Hermitian eigenvector phases, and final ordering. Those choices do not change the eigenvalues. They do determine the signed basis entries used by the task's checksum, which prevents a spectrum-only calculation from producing the requested scalar.

## Problem

A dimensionless dense real skew-symmetric matrix $A\in\mathbb{R}^{12\times12}$ is given below; compute the single scalar defined at the end in IEEE 754 binary64 arithmetic, without replacing the required structure-preserving half-size reduction by a direct eigensolve of $iA$.

$$
A=\begin{bmatrix}
0&0&0.369501366543&0.001002095135&-0.365209499092&-4.082473041096&-6.386891388953&0.726970629471&0.000277337982&0.327976308674&0.002283942595&-3.174276156368\\
0&0&-1.152895361244&-0.003128345583&1.140112832343&-1.307728403802&-2.045896983827&-2.269773357963&-0.000865795092&-1.023878073143&-0.007130023364&-1.016807961598\\
-0.369501366543&1.152895361244&0&0.290798206480&-0.296197008441&-0.000389114602&0.000399057743&0&-1.208157343371&-0.323871559018&-0.835442286096&-0.000302490566\\
-0.001002095135&0.003128345583&-0.290798206480&0&0.114305011059&0.091082462027&0&0.247942194626&0&-0.086209001916&-0.002345138466&-0.117165690273\\
0.365209499092&-1.140112832343&0.296197008441&-0.114305011059&0&0&0&0.583104121962&-0.031634841782&0.607575500962&0.908875008730&0\\
4.082473041096&1.307728403802&0.000389114602&-0.091082462027&0&0&2.018986869636&-0.000197656948&0.329104912270&0&0&-0.001384034230\\
6.386891388953&2.045896983827&-0.000399057743&0&0&-2.018986869636&0&0.000202707725&0&0&0&-1.568104019066\\
-0.726970629471&2.269773357963&0&-0.247942194626&-0.583104121962&0.000197656948&-0.000202707725&0&0.585964195531&-0.637585241132&-1.644681839453&0.000153654892\\
-0.000277337982&0.000865795092&1.208157343371&0&0.031634841782&-0.329104912270&0&-0.585964195531&0&-0.023859042666&-0.000649036150&0.423350482189\\
-0.327976308674&1.023878073143&0.323871559018&0.086209001916&-0.607575500962&0&0&0.637585241132&0.023859042666&0&-0.858883609653&0\\
-0.002283942595&0.007130023364&0.835442286096&0.002345138466&-0.908875008730&0&0&1.644681839453&0.000649036150&0.858883609653&0&0\\
3.174276156368&1.016807961598&0.000302490566&0.117165690273&0&0.001384034230&1.568104019066&-0.000153654892&-0.423350482189&0&0&0
\end{bmatrix}.
$$

Treat the displayed entries as exact binary64 inputs, use $J_{12}=\begin{bmatrix}0&-I_6\\I_6&0\end{bmatrix}$, and obtain $P$, $Z=[Z_1\mid Z_2]$, $H$, $\Omega$, $U$, $Q$, and the nonnegative diagonal $\Sigma=\operatorname{diag}(\sigma_1,\ldots,\sigma_6)$ through the source's structure-preserving polar reduction to a Hermitian eigenproblem of order six and its real recovery. Run the reduction once for each of the five candidate seeds $260812153+k$, $k=0,\ldots,4$: at each seed generate the real Gaussian sampling matrix with `numpy.random.default_rng` at shape $(12,6)$, use reduced QR, and rotate each QR column so that the corresponding nonzero diagonal entry of $R$ is positive real. Score a candidate by the smallest modulus of its QR diagonal, and report the candidate of largest score, taking the earliest on a tie. For each column of the Hermitian eigenvector matrix $U$, consider the rows whose entry has modulus within a relative $10^{-12}$ of the largest in that column, and take as its pivot the first of them when that column's eigenvalue is nonnegative and the last of them when it is negative; rotate that pivot to be real and nonnegative; then use $\operatorname{sign}(0)=1$ and order recovered pairs by decreasing $\sigma_j$. With one-based indices, and with $Z_1$ and $Q$ those of the reported candidate, define

$$
C_Z=\sum_{r=1}^{12}\sum_{j=1}^{6}(13r+7j)(Z_1)_{rj},\qquad
C_Q=\sum_{r=1}^{12}\sum_{c=1}^{12}(11r+5c)Q_{rc},
$$

write $C_Q^{(k)}$ for the second checksum of candidate $k$, and report

$$
S=\sum_{j=1}^{6}j\sigma_j+10^{-3}C_Z+10^{-4}C_Q
+10^{-2}\Bigl(\max_kC_Q^{(k)}-\min_kC_Q^{(k)}\Bigr).
$$

Keep all intermediate arrays in binary64 or complex128 precision, do not round intermediate values, and show the six ordered $\sigma_j$, $C_Z$, $C_Q$ and the ensemble spread in the reasoning while placing only $S$ in the final-answer tag. In that same short reasoning, state what does and what does not change when one column of $U$ is multiplied by an arbitrary unit-modulus phase.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the scalars that
determine the final number.
You must also report, as a short list: the largest and smallest singular
values of $A$; the defects $\lVert P+P^T\rVert_F$ and $\lVert P^TP-I\rVert_F$;
the five candidate scores and which seed is reported; the six moduli $\lvert R_{jj}\rvert$ of that candidate's reduced-QR diagonal; the residuals
$\lVert Z^TZ-I\rVert_F$ and $\lVert P-ZJ_{12}Z^T\rVert_F$; the defect
$\lVert Q^TQ-I\rVert_F$ and the relative residual
$\lVert A-Q(J_2\otimes\Sigma)Q^T\rVert_F/\lVert A\rVert_F$; and the
spectrum-only term $\sum_j j\sigma_j$ together with the combined correction
$10^{-3}C_Z+10^{-4}C_Q+10^{-2}(\max_kC_Q^{(k)}-\min_kC_Q^{(k)})$.
You must also state, in one sentence each: why the nonsingularity of $A$ makes
its orthogonal polar factor unique and what structure that factor inherits
from $A$; the map that carries the sampled complex invariant subspace to the
real basis $Z$; the block form of $Z^TAZ$ and the half-size complex matrix it
corresponds to; the symmetry classes of $H$ and $\Omega$; and the signed
formula that recovers $Q$ and $\Sigma$.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

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

01_compute_skew_polar_factor

Goal
----
Compute the orthogonal polar factor of a nonsingular real skew-symmetric matrix and return the structure-preserving factor used by the later reduction.

```python
def compute_skew_polar_factor(
    A: np.ndarray,
    skew_tolerance: float = 1e-12,
    singularity_tolerance: float = 1e-12,
) -> np.ndarray:
    """Return the unique orthogonal polar factor of ``A``.

    Raises ``ValueError`` unless every one of the following holds: ``A`` is real
    rather than complex; ``A`` is a two-dimensional square array of even order at
    least two; every entry of ``A`` is finite; ``skew_tolerance`` and
    ``singularity_tolerance`` are each finite and strictly positive; the
    normalized defect ``norm(A + A.T) / max(1.0, norm(A))`` is at most
    ``skew_tolerance``; and the smallest singular value of ``A`` exceeds
    ``singularity_tolerance`` times the largest, so a numerically singular ``A``
    is rejected rather than factored.

    Parameters
    ----------
    A : np.ndarray
        Finite real square array of even order at least two.
    skew_tolerance : float, optional
        Maximum normalized Frobenius defect in ``A + A.T``.
    singularity_tolerance : float, optional
        Minimum allowed ratio of smallest to largest singular value.

    Returns
    -------
    np.ndarray
        Real orthogonal polar factor with the same shape as ``A``.
    """
    return result  # noqa: F821 - required model stub
```

### Step 2

02_sample_positive_imaginary_basis

Goal
----
Sample a deterministic orthonormal basis for the prescribed complex invariant subspace of the polar factor, together with the conditioning of that sample.

```python
def sample_positive_imaginary_basis(P: np.ndarray, seed: int) -> np.ndarray:
    """Return a canonical thin-QR basis of ``range(P + i I)``.

    Raises ``ValueError`` unless every one of the following holds: ``P`` is real
    rather than complex; ``P`` is a two-dimensional square array of even order at
    least two; every entry of ``P`` is finite; ``P`` is skew-symmetric to an
    absolute tolerance of ``1e-10``; ``P`` is orthogonal to the same tolerance, so
    a skew-symmetric factor that is not orthogonal is rejected rather than
    sampled; ``seed`` is an integer, not a bool; and no modulus of the reduced-QR
    diagonal falls to ``100`` machine epsilons, which would leave the sampled
    range numerically rank deficient.

    Parameters
    ----------
    P : np.ndarray
        Finite real skew-symmetric orthogonal array of shape ``(2n, 2n)``.
    seed : int
        Seed passed to ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(2n + 1, n)``.  Rows ``:2n`` hold the orthonormal
        reduced-QR factor of ``(P + 1j * I) @ G``, where ``G`` is
        ``numpy.random.default_rng(seed).standard_normal((2n, n))``, with each
        column multiplied by the phase of the matching diagonal entry of ``R`` so
        that every ``R[j, j]`` becomes positive real.  Row ``2n`` holds the moduli
        ``abs(R[j, j])`` as real values in a complex row.
    """
    return result  # noqa: F821 - required model stub
```

### Step 3

03_build_real_symplectic_basis

Goal
----
Convert the complex invariant-subspace basis into the real orthogonal basis used to reduce the original matrix.

```python
def build_real_symplectic_basis(V_tilde: np.ndarray) -> np.ndarray:
    """Return ``sqrt(2) [Re(V_tilde) | -Im(V_tilde)]``.

    Raises ``ValueError`` unless every one of the following holds: ``V_tilde`` is
    a two-dimensional array of shape ``(2n, n)`` with ``n`` at least one; every
    real and imaginary part is finite; and the columns of ``V_tilde`` are
    orthonormal to an absolute tolerance of ``1e-10``, so a rank-deficient block
    such as ``numpy.ones((4, 3))`` is rejected rather than realified.

    Parameters
    ----------
    V_tilde : np.ndarray
        Finite complex array of shape ``(2n, n)`` with orthonormal columns.

    Returns
    -------
    np.ndarray
        Real square array ``Z`` of shape ``(2n, 2n)``.
    """
    return result  # noqa: F821 - required model stub
```

### Step 4

04_compute_skew_hamiltonian_blocks

Goal
----
Compress the real skew-symmetric matrix into two half-size real blocks in the prescribed orthogonal basis.

```python
def compute_skew_hamiltonian_blocks(A: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Return the stacked pair ``[H, Omega]`` of half-size blocks.

    Raises ``ValueError`` unless every one of the following holds: ``A`` is a
    two-dimensional square array of even order at least two; ``Z`` has the same
    shape as ``A``; every entry of both arrays is finite; ``A`` is skew-symmetric
    to an absolute tolerance of ``1e-10``; and ``Z`` is orthogonal to the same
    tolerance, so a scaled basis such as ``2 * I`` is rejected rather than
    projected.

    Parameters
    ----------
    A : np.ndarray
        Finite real skew-symmetric array of shape ``(2n, 2n)``.
    Z : np.ndarray
        Finite real orthogonal array with the same shape as ``A``.

    Returns
    -------
    np.ndarray
        Real array of shape ``(2, n, n)``.  With ``Z1 = Z[:, :n]`` and
        ``Z2 = Z[:, n:]``, index zero is ``H = Z2.T @ A @ Z1`` replaced by its
        symmetric part and index one is ``Omega = Z1.T @ A @ Z1`` replaced by
        its skew-symmetric part.
    """
    return result  # noqa: F821 - required model stub
```

### Step 5

05_solve_half_size_hermitian_problem

Goal
----
Solve the half-size Hermitian eigenproblem and return eigenpairs with deterministic phase, exact-multiplicity residual-pivot gauges, and balanced Loewdin gauges for tight clusters.

```python
def solve_half_size_hermitian_problem(
    blocks: np.ndarray,
) -> np.ndarray:
    """Return the ordered, phase-fixed eigensystem of ``blocks[0] + 1j * blocks[1]``.

    Raises ``ValueError`` unless every one of the following holds: ``blocks`` is a
    three-dimensional array of shape ``(2, n, n)`` with ``n`` at least one; every
    entry is finite; ``blocks[0]`` is symmetric to an absolute tolerance of
    ``1e-10``; ``blocks[1]`` is skew-symmetric to the same tolerance, so a
    symmetric second block is rejected rather than diagonalized; and every
    eigenvector or clustered eigenspace carries a residual pivot of modulus
    above ``100`` machine epsilons, so its gauge is well defined.

    Parameters
    ----------
    blocks : np.ndarray
        Finite real array of shape ``(2, n, n)`` with symmetric ``blocks[0]``
        and skew-symmetric ``blocks[1]``.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(n + 1, n)``.  Row zero stores the real signed
        eigenvalues sorted by decreasing magnitude with stable ties; rows
        ``1:`` store the matching eigenvectors as the columns of ``U``.

        Consecutive eigenvalues from ``numpy.linalg.eigh`` belong to one
        numerical cluster when the cluster span is at most
        ``256 * eps * max(1, ||K||_2)``, for ``K = blocks[0] + 1j * blocks[1]``.
        Replace all values in a multiple cluster by their arithmetic mean.  To
        choose its coordinate flag, repeatedly project every as-yet-unused
        coordinate vector into the cluster and orthogonally away from accepted
        columns, use two modified-Gram--Schmidt passes, and accept the largest
        residual norm.  Residual norms within relative ``1e-12`` of the largest
        are tied; use the earliest coordinate for a nonnegative cluster and the
        latest for a negative cluster.  If the cluster span is at most
        ``32 * eps * max(1, ||K||_2)``, normalize each accepted residual and
        rotate its selected coordinate to real nonnegative, as for an exact
        multiplicity.  Otherwise form the matrix of spectral-projector columns
        at the selected coordinates and apply symmetric Loewdin
        orthogonalization: if ``C`` contains those columns, form
        ``G = C.conj().T @ C``, replace it by its Hermitian part, and use
        ``C @ G**(-1/2)``, with the inverse square root obtained from
        ``numpy.linalg.eigh``.  Finally rotate column ``j`` so its entry at
        selected coordinate ``j`` is real nonnegative.  Reject an inverse-root
        eigenvalue no larger than ``(100*eps)**2`` or an unstable phase pivot.
        A one-element cluster keeps the solver vector and applies the same
        earliest/nonnegative versus latest/negative rule to entries within
        relative ``1e-12`` of its largest modulus.
    """
    return result  # noqa: F821 - required model stub
```

### Step 6

06_recover_real_spectral_decomposition

Goal
----
Recover the nonnegative spectral values and real orthogonal spectral basis from the half-size eigenvectors.

```python
def recover_real_spectral_decomposition(
    Z: np.ndarray,
    eigensystem: np.ndarray,
) -> np.ndarray:
    """Return a packed real spectral basis and ordered nonnegative values.

    Raises ``ValueError`` unless every one of the following holds: ``Z`` is a
    two-dimensional square array of even order ``2n`` at least two;
    ``eigensystem`` has shape ``(n + 1, n)``; the packed values in row zero are
    real to an absolute tolerance of ``1e-12``; every entry of ``Z`` and of the
    packed eigensystem is finite; ``Z`` is orthogonal to an absolute tolerance of
    ``1e-10``; and rows ``1:`` form a unitary ``U`` to the same tolerance, so a
    scaled eigenvector matrix is rejected rather than recovered.

    Parameters
    ----------
    Z : np.ndarray
        Finite real orthogonal array of shape ``(2n, 2n)``.
    eigensystem : np.ndarray
        Complex array of shape ``(n + 1, n)`` with signed eigenvalues in row
        zero and a unitary eigenvector matrix in the remaining rows.

    Returns
    -------
    np.ndarray
        Real array of shape ``(2n + 1, 2n)``.  Write ``Z1 = Z[:, :n]``,
        ``Z2 = Z[:, n:]``, ``U_r`` and ``U_i`` for the real and imaginary parts
        of the eigenvectors, and ``s`` for the elementwise sign of the packed
        eigenvalues with ``sign(0) = 1``.  The first ``n`` columns of ``Q`` are
        ``Z1 @ U_i + Z2 @ U_r`` and the last ``n`` are
        ``(-Z1 @ U_r + Z2 @ U_i) * s``.  Rows ``:2n`` store ``Q``; row ``2n``
        stores ``abs(sigma_tilde)`` in its first ``n`` entries and zeros in the
        remaining entries.
    """
    return result  # noqa: F821 - required model stub
```

### Step 7

07_compute_basis_checksums

Goal
----
Reduce the two deterministic real bases to weighted checksums that preserve sensitivity to sign, phase, and ordering conventions.

```python
def compute_basis_checksums(Z: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Return ``[C_Z, C_Q]`` using the prescribed one-based weights.

    Raises ``ValueError`` unless every one of the following holds: ``Z`` is a
    two-dimensional square array of even order at least two; ``Q`` has the same
    shape as ``Z``; every entry of both arrays is finite; and ``Z`` and ``Q`` are
    each orthogonal to an absolute tolerance of ``1e-10``, so a scaled basis such
    as ``2 * I`` is rejected rather than summed.

    Parameters
    ----------
    Z : np.ndarray
        Finite real orthogonal array of shape ``(2n, 2n)``.
    Q : np.ndarray
        Finite real orthogonal array with the same shape as ``Z``.

    Returns
    -------
    np.ndarray
        Real vector ``[C_Z, C_Q]`` of length two.
    """
    return result  # noqa: F821 - required model stub
```

### Step 8

08_compute_basis_sensitive_skew_scalar

Goal
----
Run the reduction over a candidate ensemble of consecutive seeds, resolve a multiway near-maximin pool by a robust optimally matched shadow-ensemble medoid, and return the single basis-sensitive scalar endpoint.

```python
def compute_basis_sensitive_skew_scalar(
    A: np.ndarray,
    seed: int = 260812153,
    candidates: int = 5,
) -> float:
    """Return the basis-sensitive scalar from the candidate ensemble.

    Raises ``ValueError`` unless every one of the following holds: ``candidates``
    is an integer, not a bool, of at least two; ``seed`` is an integer, not a
    bool; every validation rule of the seven earlier stages passes on the arrays
    this function derives for each candidate, so a singular ``A`` such as
    ``numpy.zeros((4, 4))`` is rejected rather than reduced; and the assembled
    scalar is finite.

    Parameters
    ----------
    A : np.ndarray
        Finite nonsingular real skew-symmetric array of even order.
    seed : int, optional
        Seed of the first candidate.  Candidate ``k`` uses ``seed + k``.
    candidates : int, optional
        Number of consecutive seeds in the ensemble.

    Returns
    -------
    float
        One finite scalar.  Every candidate runs the full reduction and yields a
        score ``min(abs(R[j, j]))`` from step two together with ``C_Z``, ``C_Q``
        and ``sigma``.  Let ``s_max`` be the largest score and retain every
        candidate with score at least ``0.95*s_max``.  If the pool has at least
        three members, create five independent shadow seeds per original
        candidate by applying ``SeedSequence`` to the two 32-bit words of
        ``seed``, then ``order``, ``candidates``, ``0x4e4c4131``, and
        ``0x34534844``; call ``spawn(candidates*5)`` in candidate-major order and
        draw one ``uint64`` from each child.  Run the complete seven-stage
        reduction on every shadow seed and describe it by
        ``[min(abs(diag(R))), sum(log(abs(diag(R)))), C_Z, C_Q]``.  Normalize
        all pooled shadow rows componentwise by the median and by
        ``max(MAD, 128*eps*max(1,max(abs(column))))``.  The distance between two
        original candidates is the minimum, over all ``5!`` lane permutations,
        of the mean Euclidean matched-row distance.  Select the candidate whose
        sum of distances to the other pool members is smallest; values within
        ``256*eps*max(1,max(abs(distance_sums)))`` tie.  For one- or two-member
        pools this medoid stage is vacuous.  Resolve any remaining tie by
        maximizing ``sum(log(abs(R[j,j])))``, then the original score, with
        respective tolerances ``64*eps*max(1,abs(best_log_volume))`` and
        ``64*eps*max(1,s_max)``, and finally take the earliest index.  With that
        candidate's values the result is
        ``sum(j * sigma_j) + 1e-3 * C_Z + 1e-4 * C_Q``
        ``+ 1e-2 * (max_k C_Q_k - min_k C_Q_k)``.
    """
    return result  # noqa: F821 - required model stub
```
