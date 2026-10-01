# Physics-Quantum_Information_Computing-41

## Background

Quantum subspace methods estimate low-lying energies using a small collection of states prepared by quantum evolution, followed by classical diagonalization of measured matrix elements. Their nonorthogonal bases create ill-conditioned generalized eigenproblems, and measurement noise can make additional basis states harmful. Removing poorly resolved overlap directions stabilizes the calculation but may also discard useful ground-state information.
Mixing the two matrices by an eigenvector-preserving rotation before truncation changes which directions survive. Physical-overlap repair and noise-aware choices of basis size and rotation turn this into a data-dependent regularization method. A deterministic perturbation of the measured Hamiltonian tests a further consequence: even after the discrete choices are fixed, the surviving subspace moves. The requested curvature measures that local response of the regularized estimator, not the curvature of an exact many-body eigenvalue or a statistical average over fresh noise realizations.

## Problem

Consider the curvature of a noise-regularized quantum Krylov ground-energy estimate, with $\hbar=1$ and all Hamiltonian entries represented numerically in one fixed energy unit, for levels $e=(0.2,0.8,1.4,2.1,3.0)$, normalized spectral weights $w=(0.26,0.22,0.20,0.18,0.14)$, time spacing $\Delta t=0.6$, at most six columns $V_{aj}=\sqrt{w_a}\exp(-\mathrm{i}e_a j\Delta t)$, and four equal-size measurement batches, with $H_0=V^\dagger\operatorname{diag}(e)V$ and $S_0=V^\dagger V$. All indices start at zero; define Hermitian matrices $A_q,B_q,C_q$ by $(A_q)_{ii}=\cos[(q+1)(i+1)]$, $(B_q)_{ii}=\sin[(q+2)(i+1)]$, $(C_q)_{ii}=0$, and, for $i<j$ and $z=1+j-i$, $(A_q)_{ij}=\{\sin[(q+1)(i+j+2)]+\mathrm{i}\cos[(q+2)(j-i)]\}/z$, $(B_q)_{ij}=\{\cos[(q+2)(i+j+2)]+\mathrm{i}\sin[(q+1)(j-i)]\}/z$, $(C_q)_{ij}=\{\cos[(q+1)(i+j+2)]+\mathrm{i}\sin[(q+2)(j-i)]\}/z$, with conjugate lower triangles. The dimensionless perturbation $x$ changes only the measured Hamiltonians, $H_q(x)=H_0+0.035[\cos(x)A_q+\sin(x)B_q]$, while $S_q=S_0+0.065C_q$ is fixed; these analytic batches replace random draws, and a derivative jet contains actual zeroth, first and second derivatives, not factorial-divided coefficients.

Use the Frobenius-nearest physical-overlap repair for normalized states, the adjacent-dimension noise-aware stopping rule and the measurement-only rotation-angle heuristic of rotation-thresholded real-time quantum subspace diagonalization, with population batch statistics, noise multiplier $\gamma=1.3$, independent absolute stopping threshold $10^{-4}$ and spectral cutoff $\tau=0.08$. At $x=0$, process dimensions $k=2,3,\ldots,6$, repairing each raw principal overlap prefix separately and using unrotated thresholded batch ground energies; beginning at $k=3$, stop at the first strict adjacent-dimension stopping success and retain the newly added dimension, or retain six if none succeeds. At that selected dimension evaluate the angle heuristic over the ordered grid $(0,0.2,0.45,0.7,0.95,1.2,1.4)$ radians, choosing its global minimum with the earliest supplied angle winning an exact tie; a pencil rotates as $\mathcal A_\theta=H\cos\theta-X\sin\theta$, $\mathcal B_\theta=X\cos\theta+H\sin\theta$, keeps only eigenvectors of $\mathcal B_\theta$ with eigenvalue strictly greater than $\tau$, and takes the smallest physical energy after inverse rotation of every retained generalized eigenray.

Uniformly average the **raw** selected-dimension batch pencils, repair their aggregate overlap once, and define $E_*(x)$ by the same rotated thresholded solve on this aggregate, holding the selected dimension and angle fixed but allowing its retained spectral projector $P(x)$ to move. Compute the ordinary second derivative $E_*''(0)$, report its numerical value to ten digits after the decimal point, and justify the second-order moving-subspace and generalized-metric contributions; give the two stopping certificates, selected angle, aggregate retained rank, $D_X=\|X_{\rm aggregate}-\tfrac14\sum_qX_q\|_F$, $\|P'(0)\|_F$, and the aggregate energy and first derivative as compact checkpoints, while identifying the physical-repair, noise-test and angle-selection identities, the rotation's perturbation-strength invariant and the normalized coefficient-norm criterion for threshold-only error. Use converged physical projections to relative Frobenius residual at most $10^{-13}$ without intermediate rounding; inverse-rotation denominators of absolute value at most $10^{-12}$ are excluded, a cutoff distance or selected-root gap at most $10^{-10}$ is outside the differentiable contract, all such gaps are open in this instance, and all boundaries are spectral rather than spatial.

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
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

measurement_pencil_jets

Goal
----
Construct the deterministic noisy real-time Krylov pencil and its first two derivatives.

```python
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def measurement_pencil_jets(energies: 'np.ndarray', weights: 'np.ndarray', dt: float, dimension: int, batches: int, noise_h: float, noise_s: float) -> 'np.ndarray':
    """Construct the deterministic noisy real-time Krylov pencil and its first two derivatives.
    
    Parameters
    ----------
    energies : np.ndarray, shape (n,)
        Finite real energy levels, n >= 1, in the chosen energy unit.
    weights : np.ndarray, shape (n,)
        Finite nonnegative spectral weights with positive finite sum; normalize once.
    dt : float
        Finite nonnegative time spacing in reciprocal energy units.
    dimension : int
        Number of Krylov columns, between 2 and 32 inclusive; bool is invalid.
    batches : int
        Number of equal-size deterministic batches, between 1 and 64; bool is invalid.
    noise_h, noise_s : float
        Finite nonnegative Hamiltonian and overlap noise amplitudes.
    
    Returns
    -------
    result : np.ndarray, complex, shape (3, 2, batches, dimension, dimension)
        Axis 0 contains value, first derivative and second derivative at x=0.
        Axis 1 contains Hamiltonian and overlap, respectively.
    
    Raises
    ------
    ValueError
        If shapes, finite values, weight sum, scalar signs or integer ranges violate the stated contract.
    
    Notes
    -----
    Set V[a,j] = sqrt(w[a]) exp(-1j*energies[a]*j*dt), H=V^dagger diag(energies) V and S=V^dagger V. For batch q, the Hermitian arrays A, B, C have diagonal A[i,i]=cos((q+1)(i+1)), B[i,i]=sin((q+2)(i+1)), C[i,i]=0. For i<j, set z=1+j-i and A[i,j]=(sin((q+1)(i+j+2))+1j*cos((q+2)(j-i)))/z, B[i,j]=(cos((q+2)(i+j+2))+1j*sin((q+1)(j-i)))/z, C[i,j]=(cos((q+1)(i+j+2))+1j*sin((q+2)(j-i)))/z. The batch pencil is H_q(x)=H+noise_h*(cos(x)*A+sin(x)*B), S_q=S+noise_s*C. Return derivatives, not factorial-divided Taylor coefficients. Inputs are unchanged.
    """
    return result
```

### Step 2

nearest_physical_overlap

Goal
----
Find the nearest physical Gram matrix of normalized states.

```python
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def nearest_physical_overlap(overlap: 'np.ndarray', tolerance: float = 1e-13, max_iterations: int = 2000) -> 'np.ndarray':
    """Find the nearest physical Gram matrix of normalized states.
    
    Parameters
    ----------
    overlap : np.ndarray, shape (n, n)
        Nonempty finite Hermitian matrix; Hermiticity tolerance is 1e-10 absolute.
    tolerance : float, optional
        Convergence tolerance in (0, 1e-6], default 1e-13.
    max_iterations : int, optional
        Positive iteration budget, default 2000; bool is invalid.
    
    Returns
    -------
    result : np.ndarray, complex, shape (n, n)
        The Frobenius-nearest Hermitian positive-semidefinite matrix with unit diagonal,
        to the requested numerical tolerance.
    
    Raises
    ------
    ValueError
        If the input or convergence parameters violate the contract, or the constrained projection fails to converge within the budget.
    
    Notes
    -----
    Use the converged convex projection, preserving the input. For a bounded reproducible implementation, alternating cone and unit-diagonal projections with the cone correction retained may be used: start from the input and zero correction, project the corrected affine iterate onto the cone, update the correction, then impose unit diagonal. The stopping residual is the maximum of successive cone-iterate difference, successive affine-iterate difference and their mutual difference, divided by max(1, norm of the affine iterate). All norms are Frobenius norms. Return the affine iterate once this residual is at most tolerance. Equivalent converged convex solutions are accepted; do not add a final clipping or diagonal normalization that changes the minimizer.
    """
    return result
```

### Step 3

rotated_ground_energy

Goal
----
Recover the lowest physical energy of a rotated, thresholded Hermitian pencil.

```python
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def rotated_ground_energy(hamiltonian: 'np.ndarray', overlap: 'np.ndarray', theta: float, cutoff: float) -> float:
    """Recover the lowest physical energy of a rotated, thresholded Hermitian pencil.
    
    Parameters
    ----------
    hamiltonian, overlap : np.ndarray, shape (n, n)
        Matching nonempty finite Hermitian matrices, absolute Hermiticity tolerance 1e-10.
    theta : float
        Finite rotation angle in radians.
    cutoff : float
        Finite nonnegative absolute eigenvalue threshold.
    
    Returns
    -------
    result : float
        Smallest finite back-transformed generalized eigenvalue in the retained space.
    
    Raises
    ------
    ValueError
        If shapes, finite values, Hermiticity or scalar domains fail, if no denominator eigenvalue exceeds the cutoff, or if no finite physical root remains.
    
    Notes
    -----
    Write c=cos(theta), s=sin(theta), A=c*hamiltonian-s*overlap and B=c*overlap+s*hamiltonian. Retain the eigenvectors of B with eigenvalues strictly greater than cutoff; compress both A and B to that subspace, solve the Hermitian generalized eigenproblem there and undo the rotation of each eigenray. Exclude roots whose inverse-map denominator has absolute value <=1e-12. Select the minimum recovered physical energy, rather than relying on the ordering of rotated roots. The input overlap need not be positive definite; the retained denominator is positive definite by construction. Do not regularize discarded modes or normalize the projected Hamiltonian separately.
    """
    return 0.0
```

### Step 4

batch_energy_moments

Goal
----
Evaluate equal-batch population energy statistics.

```python
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def batch_energy_moments(energies: 'np.ndarray') -> 'np.ndarray':
    """Evaluate equal-batch population energy statistics.
    
    Parameters
    ----------
    energies : np.ndarray, shape (b,)
        Nonempty finite real vector of equal-weight batch ground energies.
    
    Returns
    -------
    result : np.ndarray, float, shape (3,)
        Population mean, population standard deviation and population variance, in that order.
    
    Raises
    ------
    ValueError
        If energies is not a nonempty finite real vector or its population variance is not representable.
    
    Notes
    -----
    Use the population divisor b, including for b=1, whose dispersion is zero. Preserve the input.
    """
    return result
```

### Step 5

dimension_convergence

Goal
----
Evaluate the noise-aware adjacent-dimension stopping certificate.

```python
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def dimension_convergence(previous: 'np.ndarray', current: 'np.ndarray', gamma: float, energy_tolerance: float) -> 'np.ndarray':
    """Evaluate the noise-aware adjacent-dimension stopping certificate.
    
    Parameters
    ----------
    previous, current : np.ndarray, shape (3,)
        Consecutive population triples (mean, standard deviation, variance).
        Entries are finite; dispersion entries are nonnegative.
    gamma : float
        Finite nonnegative multiplier of the larger adjacent standard deviation.
    energy_tolerance : float
        Finite nonnegative independent absolute energy-change threshold.
    
    Returns
    -------
    result : np.ndarray, float, shape (3,)
        Absolute mean change, effective threshold, and a 0.0/1.0 stopping indicator.
    
    Raises
    ------
    ValueError
        If moment shapes, finite values, dispersion signs or parameter domains fail, or the threshold is not representable.
    
    Notes
    -----
    The effective threshold is max(energy_tolerance, gamma*max(previous standard deviation, current standard deviation)). Stop only when the absolute mean change is strictly smaller. Equality continues growth. Moment triples are inputs; no statistical consistency between their final two entries needs to be re-estimated.
    """
    return result
```

### Step 6

select_variance_angle

Goal
----
Select a rotation from the measurement-only variance objective.

```python
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def select_variance_angle(angles: 'np.ndarray', batch_energies: 'np.ndarray') -> 'np.ndarray':
    """Select a rotation from the measurement-only variance objective.
    
    Parameters
    ----------
    angles : np.ndarray, shape (m,)
        Nonempty finite real grid in its supplied order; sorting is not implied.
    batch_energies : np.ndarray, shape (m, b)
        Finite physical ground energies, with b>=1 equally weighted batches per angle.
    
    Returns
    -------
    result : np.ndarray, float, shape (3,)
        Zero-based selected row index, its angle, and its population variance.
    
    Raises
    ------
    ValueError
        If shapes, finite values or nonempty requirements fail, or a variance is not representable.
    
    Notes
    -----
    Select the global minimum population variance, with the earliest supplied row winning an exact tie. A single batch gives zero variance for every angle. Preserve both inputs.
    """
    return result
```

### Step 7

cutoff_projector_jet

Goal
----
Differentiate the retained spectral projector through second order.

```python
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def cutoff_projector_jet(denominator_jet: 'np.ndarray', cutoff: float, gap_tolerance: float = 1e-10) -> 'np.ndarray':
    """Differentiate the retained spectral projector through second order.
    
    Parameters
    ----------
    denominator_jet : np.ndarray, complex, shape (3, n, n)
        Value, first derivative and second derivative of a Hermitian denominator at zero.
        Every slice is finite and Hermitian to absolute tolerance 1e-10.
    cutoff : float
        Finite nonnegative fixed spectral cutoff; retain eigenvalues strictly above it.
    gap_tolerance : float, optional
        Finite nonnegative exclusion distance from the cutoff, default 1e-10.
    
    Returns
    -------
    result : np.ndarray, complex, shape (3, n, n)
        P(0), P'(0), P''(0), the Hermitian orthogonal-projector derivatives in original coordinates.
    
    Raises
    ------
    ValueError
        If shapes, Hermiticity, finite values or scalar domains fail, or any base eigenvalue lies within gap_tolerance of the cutoff.
    
    Notes
    -----
    The local denominator is B(x)=B0+x*B1+(x*x/2)*B2+O(x**3); coefficients are actual derivatives. All-retained and all-discarded clusters have constant projectors and zero derivatives. Internal eigenvalue multiplicity is valid. Return the projector itself, not a choice of retained eigenvectors. The result must be independent of eigenbasis phases and of unitary changes within degenerate clusters.
    """
    return result
```

### Step 8

projected_energy_jet

Goal
----
Compute the second-order physical ground-energy response in a moving retained subspace.

```python
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def projected_energy_jet(numerator_jet: 'np.ndarray', denominator_jet: 'np.ndarray', projector_jet: 'np.ndarray', theta: float, gap_tolerance: float = 1e-10) -> 'np.ndarray':
    """Compute the second-order physical ground-energy response in a moving retained subspace.
    
    Parameters
    ----------
    numerator_jet, denominator_jet : np.ndarray, complex, shape (3, n, n)
        Rotated pencil value, first derivative and second derivative at zero.
    projector_jet : np.ndarray, complex, shape (3, n, n)
        Value, first derivative and second derivative of the orthogonal retained projector.
        Slices obey differentiated idempotency to absolute Frobenius tolerance 1e-7.
    theta : float
        Finite fixed rotation angle in radians.
    gap_tolerance : float, optional
        Finite nonnegative exclusion gap for the selected generalized eigenvalue, default 1e-10.
    
    Returns
    -------
    result : np.ndarray, float, shape (3,)
        Physical ground energy, its first derivative and its second derivative at zero.
    
    Raises
    ------
    ValueError
        If jets have unequal or invalid shapes, nonfinite or non-Hermitian slices, invalid projector identities, an empty range, a non-positive compressed denominator, an excluded inverse-map pole with no usable root, a nonisolated selected root, or invalid scalar parameters.
    
    Notes
    -----
    Hermiticity is checked at absolute tolerance 1e-10. The supplied projector jet defines the moving subspace; the range of P0 is identified by projector eigenvalues >0.5. For a smooth orthonormal frame U(x) of that range, solve the pencil U(x)^dagger A(x) U(x), U(x)^dagger B(x) U(x). Select the lowest physical energy after inverse rotation, excluding inverse-map denominators of absolute value <=1e-12, and follow its locally simple branch. Other generalized eigenvalues may be degenerate with one another. Metric normalization, frame acceleration, second-order mixing and the curvature of the inverse map must all be retained. Differentiating unnormalized coordinate components or holding the projector fixed defines a different observable.
    """
    return result
```

### Step 9

rotated_krylov_curvature

Goal
----
Evaluate the complete fixed-decision noise-curvature experiment.

```python
import math
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigh

def rotated_krylov_curvature(energies: 'np.ndarray', weights: 'np.ndarray', dt: float = 0.6, dimension: int = 6, batches: int = 4, noise_h: float = 0.035, noise_s: float = 0.065, cutoff: float = 0.08, angles: 'np.ndarray | tuple[float, ...]' = (0.0, 0.2, 0.45, 0.7, 0.95, 1.2, 1.4), gamma: float = 1.3, energy_tolerance: float = 1e-4) -> float:
    """Evaluate the complete fixed-decision noise-curvature experiment.
    
    Parameters
    ----------
    energies, weights : np.ndarray, shape (n,)
        Finite real levels and nonnegative spectral weights with positive finite total.
    dt : float, optional
        Nonnegative real-time spacing, default 0.6.
    dimension : int, optional
        Maximum number of Krylov columns in [3,32], default 6.
    batches : int, optional
        Equal-size batch count in [1,64], default 4.
    noise_h, noise_s : float, optional
        Nonnegative noise amplitudes, defaults 0.035 and 0.065.
    cutoff : float, optional
        Nonnegative absolute spectral cutoff, default 0.08.
    angles : np.ndarray, shape (m,), optional
        Nonempty ordered angle grid, default (0,0.2,0.45,0.7,0.95,1.2,1.4).
    gamma : float, optional
        Nonnegative adjacent-spread multiplier, default 1.3.
    energy_tolerance : float, optional
        Nonnegative absolute stopping threshold, default 1e-4.
    
    Returns
    -------
    result : float
        The unrounded second derivative at x=0 of the final aggregate physical ground energy.
    
    Raises
    ------
    ValueError
        If any preparation or selection parameter violates its stated domain, or any physical projection, thresholded solve, projector derivative or selected-energy derivative is undefined under the preceding contracts.
    
    Notes
    -----
    Call measurement_pencil_jets to construct the input jets. At dimensions 2,3,..., repair every raw overlap prefix with nearest_physical_overlap and compute the unrotated batch energies with rotated_ground_energy. Use batch_energy_moments and dimension_convergence on adjacent dimensions; the first success returns the current dimension, otherwise keep the maximum. Recompute the retained-dimension batch energies for all supplied angles and use select_variance_angle. Average the raw retained-dimension pencil jets uniformly, then repair the raw aggregate overlap once. Rotate the aggregate jets, obtain cutoff_projector_jet and evaluate projected_energy_jet. Return only its second derivative. All eight preceding public functions are required components. Use default repair tolerance 1e-13 and iteration budget 2000, derivative gap tolerance 1e-10 and inverse-map pole tolerance 1e-12. Every input is read-only; perform no I/O. For the derivative, freeze the selected dimension and grid angle, not the cutoff projector. Do not differentiate the discrete selection procedure.
    """
    return 0.0
```
