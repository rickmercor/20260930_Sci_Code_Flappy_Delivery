# Material_Science-Semiconductor_Materials-18

## Background

Disorder and repulsive interactions compete in the spatial coherence of confined semiconductor carriers. A particle-addition density matrix connects the response of an interacting many-particle state to finite-size localization diagnostics.

## Problem

A spin-polarized semiconductor quantum-dot ladder is designed against small interaction uncertainty. Determine the largest admissible uncertainty-penalized localization length J for the finite ensemble below, using the interacting particle-addition subtracted-density-matrix (SDM) method and its transverse modular-density-matrix (MDM) localization observable for selected-bond disorder. Report J in longitudinal lattice spacings to four decimals. In the reasoning, report the selected V/t, long-ladder mean and standard error, paired size-difference standard error, drift certificate, maximum fit residual, and (gamma,gamma'_+,gamma'_-) at separation 2 in batch 0 of the selected long ladder. Absolute tolerances are 0.0005 for lengths, 0.0001 for V/t, 0.00005 for the residual, and 0.00000005 for each response diagnostic. Identify and apply the source method's particle-addition matrix, modular averaging rule and spectral decay observable. Briefly justify the ground-cluster response, non-smooth modular response, origin of negative addition-matrix eigenvalues, and amplitude constraint.

The open ladder has sites p=2x+a, x=0,...,L-1, a=0,1, and Hamiltonian H(V)=sum_(p,q) h_pq c_p^dagger c_q+sum_(p<q) U_pq n_p n_q, with t=1 and longitudinal spacing 1. Every nearest-neighbor longitudinal and transverse hopping is -1. A selected bond b=(p,q) shares diagonal potential h_pp=h_qq=epsilon_b and has U_pq=U_qp=V. Other U entries and other off-diagonal h entries vanish. For x congruent to 0 modulo 4 the matching includes (2x,2x+2) and (2x+1,2x+3); for x congruent to 2 or 3 it includes (2x,2x+1). Order bonds by their smaller endpoint. L is 6 or 8, N=floor(2L/3), and V/t belongs to (0,0.5,1,1.5,2,2.5,3). The fabrication log records an 18 nm oxide and a 1.4 kilohm contact resistance.

Draw z=np.random.default_rng(49273).uniform(-1,1,size=(4,2,8)), indexed batch, sample, bond. Set epsilon=-1.9 for z<0 and +1.9 otherwise. Each interaction uses this array; the shorter ladder uses the first six bonds. Batch labels pair the sizes.

In sector n, let Pi_n be the spectral projector onto all eigenstates with E-E_min<=1e-9, and m_n its rank. Define C^(n)_pq=Tr[(Pi_n/m_n)c_p^dagger c_q], with increasing-site creation-operator order. C'^(n) is the derivative at alpha=0 of this density matrix for H(V)+alpha*partial_V H, following the entire same spectral cluster continuously with fixed rank m_n. The source SDM prescription defines Delta from sectors N and N-1 of this same quenched Hamiltonian; Delta' is its interaction tangent. This cluster continuation is a benchmark convention.

Within each batch and d=1,2,3,4, let M(d,alpha) be the source method's 2x2 transverse MDM evaluated on Delta+alpha*Delta'. Entry (a,b), a,b=0,1, uses sites (2x+a,2(x+d)+b) and an equally weighted mean over all samples in that batch and all x=0,...,L-d-1. Coherences with |Delta_pq|<=1e-12 are treated as exactly zero at alpha=0 and in this local linearization. Let gamma(d,alpha) be the MDM method's leading spectral decay observable, gamma(d)=gamma(d,0), and gamma'_sigma(d)=lim_(alpha downarrow 0)[gamma(d,sigma*alpha)-gamma(d,0)]/alpha for sigma=+1,-1. Eigenvalues within 1e-10*max(gamma,1e-300) of the leading spectral value form one top eigenspace for these directional derivatives.

The prescribed linearized design curve is y(d)=max(0,gamma(d)+0.1*min(0,gamma'_+(d),gamma'_-(d))). Its fitted family is y_fit(d)=A exp(-k*d)+B cos(2*pi*d/3) exp(-k_o*d), with k in {m/40:m=1,...,40}, k_o in {0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1,1.4,2,3}, k_o>=k, and A>=|B|. Globally minimize r=||y-y_fit||_2/||y||_2. Rate pairs within 1e-12 of the minimum r tie; choose smaller k, then smaller k_o. At fixed rates, minimum A^2+B^2 resolves nonunique amplitude fits. Let xi=1/k. For an identically zero y, the residual is defined as zero and the same ties apply. The response radius, cluster convention, frequency, discrete dictionaries and statistical certificate are declared extensions of the paper's finite-size diagnostic.

For each interaction, mu_s and mu_l are means of the four separately fitted batch lengths at each size; s_s and s_l are their standard errors from unbiased variances; c is the unbiased paired sample covariance divided by four. Define s_delta=sqrt(max(0,s_s^2+s_l^2-2c)), D=|mu_l-mu_s|+s_delta, J=mu_l-s_l, and r_max as the largest residual over both sizes and all batches. Eligibility is D<=1, r_max<=0.2 and J>0. Maximize J; scores within 1e-10 of the largest eligible score tie, choosing smaller V, then earlier input position. If none qualifies, J=-1 and all diagnostics are zero. Retain full precision until reporting. NumPy and SciPy are available.

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

bond_ladder_model

Goal
----
Construct the selected-bond ladder Hamiltonian and its interaction tangent.

```python
def bond_ladder_model(potentials: 'np.ndarray | list | tuple', length: int, width: int, bonds: 'np.ndarray | list | tuple', hopping: float, interaction: float) -> 'np.ndarray':
    """Parameters
    ----------
    potentials : array_like, shape (P/2,)
        Finite real bond potentials, assigned in the supplied bond-row order.
    length, width : int
        Positive open rectangular dimensions, P=length*width even and at most 18.
    bonds : array_like, shape (P/2,2)
        Integer nearest-neighbor pairs partitioning sites p=width*x+a. Row and endpoint orders are arbitrary.
    hopping : float
        Finite positive hopping magnitude.
    interaction : float
        Finite selected-bond repulsion, including negative values.
    
    Returns
    -------
    model : ndarray, shape (2,2,P,P)
        First axis is value and interaction derivative; second is h and U. At value order, every nearest-neighbor off-diagonal h entry is -hopping and each bond shares its potential on h's diagonal. U equals interaction on selected bonds symmetrically, with zero diagonal. Derivative h is zero and derivative U is one on selected bonds. Other entries are zero. Invalid domains raise ValueError."""
    return None
```

### Step 2

sector_density_pair

Goal
----
Resolve adjacent-sector densities and adiabatic ground-cluster responses.

```python
def sector_density_pair(model: 'np.ndarray | list | tuple', number: int) -> 'np.ndarray':
    """Parameters
    ----------
    model : array_like, shape (2,2,P,P)
        Finite Hamiltonian jet: first axis is value and derivative, second is h and U, for H=sum_pq h_pq c_p^dagger c_q+sum_(p<q) U_pq n_p n_q. Both orders have Hermitian h and real symmetric zero-diagonal U within absolute tolerance 1e-12; average conjugate-transposed h and transposed real U within that tolerance. 1<=P<=18. The derivative can perturb hopping, potentials and interactions.
    number : int
        Particle number in 1,...,P. Creation operators are ordered by increasing site.
    
    Returns
    -------
    pair : complex ndarray, shape (2,2,P,P)
        First axis is sector number and number-1; second is C and C'. C_pq=Tr[(Pi/m)c_p^dagger c_q], with Pi the entire eigenspace E-E_min<=1e-9 and m its rank. C' differentiates the continuously followed same rank-m spectral cluster of H+alpha*H' at alpha=0, with equal fixed weights 1/m. The vacuum has C=C'=0. Invalid domains raise ValueError."""
    return None
```

### Step 3

addition_density

Goal
----
Obtain the full particle-addition density and its response.

```python
def addition_density(pair: 'np.ndarray | list | tuple') -> 'np.ndarray':
    """Parameters
    ----------
    pair : array_like, shape (2,2,P,P)
        Finite real or complex matrices, P>=1; first axis labels N and N-1, second labels density and derivative. The algebraic operation is defined for all finite pairs of this shape.
    
    Returns
    -------
    delta : ndarray, shape (2,P,P)
        Delta and its interaction tangent Delta' under the interacting particle-addition SDM prescription, in the full complex site basis. Invalid domains raise ValueError."""
    return None
```

### Step 4

modular_batch_curves

Goal
----
Evaluate matrix-valued modular coherence and both one-sided responses.

```python
def modular_batch_curves(deltas: 'np.ndarray | list | tuple', distances: 'np.ndarray | list | tuple', trim: int, width: int) -> 'np.ndarray':
    """Parameters
    ----------
    deltas : array_like, shape (B,S,2,P,P)
        Finite real or complex addition jets, B,S,P>=1. Axis 2 contains Delta and Delta'.
    distances : array_like
        Nonempty vector of integer separations 0<=d<L-2*trim, preserving order and repetitions.
    trim : int
        Nonnegative number of complete slices removed at each end.
    width : int
        Positive slice width dividing P; L=P/width.
    
    Returns
    -------
    curves : real ndarray, shape (B,3,len(distances))
        Rows of the second axis are the source MDM decay observable gamma and its right directional derivatives gamma'_+, gamma'_-. The source MDM prescription acts on Delta+alpha*Delta'; transverse entry (a,b), 0<=a,b<width, uses sites (width*x+a,width*(x+d)+b), with an equal mean over all samples of the batch and x=trim,...,L-trim-d-1. Base entries with magnitude <=1e-12 are replaced by zero for this local linearization. Gamma'_sigma differentiates the resulting observable in direction sigma*Delta', sigma=+1,-1, at alpha=0 from positive alpha. All eigenvalues within 1e-10*max(gamma,1e-300) of the leading spectral value form the top eigenspace for this derivative. Invalid domains raise ValueError."""
    return None
```

### Step 5

fit_decay_dictionary

Goal
----
Fit the response-adjusted coherence curve to the constrained decay dictionary.

```python
def fit_decay_dictionary(distances: 'np.ndarray | list | tuple', gamma: 'np.ndarray | list | tuple', frequency: float, rates: 'np.ndarray | list | tuple', oscillation_rates: 'np.ndarray | list | tuple', radius: float) -> 'np.ndarray':
    """Parameters
    ----------
    distances : array_like
        Finite real vector of at least two nonnegative separations.
    gamma : array_like, shape (3,len(distances))
        Finite real rows gamma, gamma'_+, gamma'_-, with base gamma>=0. The fitted target is y=max(0,gamma+radius*min(0,gamma'_+,gamma'_-)) entrywise.
    frequency : float
        Finite cycles per spacing in [0,1].
    rates, oscillation_rates : array_like
        Nonempty finite positive dictionaries with at least one pair k_o>=k. Order and repetitions have no effect.
    radius : float
        Finite nonnegative linearization radius.
    
    Returns
    -------
    fit : real ndarray, shape (5,)
        [xi,xi_O,A,B,r], xi=1/k, xi_O=1/k_o. Globally minimize r=||y-A exp(-k*x)-B cos(2*pi*frequency*x) exp(-k_o*x)||_2/||y||_2 subject to k_o>=k and A>=|B|. Rate pairs within 1e-12 of the minimum r tie by smaller k then smaller k_o. Minimum A^2+B^2 resolves nonunique fixed-rate amplitudes. If y is identically zero, r=0 and the same ties apply. Invalid domains raise ValueError."""
    return None
```

### Step 6

paired_batch_statistics

Goal
----
Retain paired finite-size covariance in batch localization estimates.

```python
def paired_batch_statistics(short_fits: 'np.ndarray | list | tuple', long_fits: 'np.ndarray | list | tuple') -> 'np.ndarray':
    """Parameters
    ----------
    short_fits, long_fits : array_like, shape (B,5)
        Matching finite real fits for paired batches, B>=2. Columns are xi,xi_O,A,B_amplitude,residual; xi>0 and residual>=0.
    
    Returns
    -------
    statistics : real ndarray, shape (6,)
        [mu_short,mu_long,se_short,se_long,covariance_of_means,max_residual]. Means are arithmetic means of batch xi. Standard errors use unbiased sample variances divided by B. Covariance is unbiased paired sample covariance divided by B. Maximum residual spans both sizes. Length units are lattice spacings and covariance units are squared spacings. Invalid domains raise ValueError."""
    return None
```

### Step 7

select_localization_certificate

Goal
----
Select the largest admissible localization certificate.

```python
def select_localization_certificate(couplings: 'np.ndarray | list | tuple', statistics: 'np.ndarray | list | tuple', drift_limit: float, fit_limit: float, penalty: float) -> 'np.ndarray':
    """Parameters
    ----------
    couplings : array_like, shape (C,)
        Nonempty finite real interaction vector.
    statistics : array_like, shape (C,6)
        Finite rows [mu_short,mu_long,se_short,se_long,covariance_of_means,max_residual]. Means are positive, errors and residuals nonnegative, and |covariance|<=se_short*se_long+1e-12.
    drift_limit, fit_limit, penalty : float
        Finite nonnegative acceptance bounds and uncertainty penalty.
    
    Returns
    -------
    certificate : real ndarray, shape (7,)
        [J,V,mu_long,se_long,se_delta,D,max_residual]. Eligible rows have D<=drift_limit, residual<=fit_limit and J>0. Maximize J; scores within 1e-10 of the maximum tie by smaller V then earlier row. Empty eligibility returns [-1,0,0,0,0,0,0]. Invalid domains raise ValueError."""
    return None
```

### Step 8

localization_design

Goal
----
Evaluate the complete interaction-response localization design.

```python
def localization_design(seed: int = 49273, disorder: float = 1.9, couplings: 'np.ndarray | list | tuple' = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0), lengths: 'np.ndarray | list | tuple' = (6, 8), batches: int = 4, samples: int = 2, drift_limit: float = 1.0, fit_limit: float = 0.2, penalty: float = 1.0, radius: float = 0.1) -> 'np.ndarray':
    """Parameters
    ----------
    seed : int
        Nonnegative random seed.
    disorder : float
        Finite nonnegative binary-disorder magnitude.
    couplings : array_like
        Nonempty finite real interactions.
    lengths : array_like
        The two integer lengths (6,8).
    batches, samples : int
        At least two batches and at least one sample each. Generate a local default_rng(seed) uniform array on [-1,1) of shape (batches,samples,8); signs select +/-disorder. All couplings share it and length 6 uses its first six bonds.
    drift_limit, fit_limit, penalty, radius : float
        Finite nonnegative design bounds, uncertainty penalty and response radius.
    
    Returns
    -------
    certificate : real ndarray, shape (10,)
        [J,V,mu_long,se_long,se_delta,D,max_residual,gamma,gamma'_+,gamma'_-]. The last three entries concern batch 0, L=8, separation 2, at the selected coupling. Empty eligibility returns [-1,0,0,0,0,0,0,0,0,0]. Use the problem's ordered matching, N=floor(2L/3), two-site slices, zero trim, separations 1,...,4, frequency 1/3 and stated rate dictionaries. Call all preceding scientific functions. Defaults specify the main task. Invalid domains raise ValueError."""
    return None
```
