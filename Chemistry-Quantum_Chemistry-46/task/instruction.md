# Chemistry-Quantum_Chemistry-46

## Background

The direct random-phase approximation represents correlated electronic particle-hole motion as a quadratic quasiboson problem. Quantized cavity modes add rotating and counter-rotating electron-photon couplings and dipole self-energy. The positive-frequency invariant subspace determines a stabilizing ring amplitude with electronic, electron-photon and photon-photon sectors.
In this finite model, the amplitude defines a pure Gaussian quasiboson state. Tracing out the electronic transitions gives a mixed photon state. Its Rényi-3/2 entropy depends on the symplectic spectrum of the reduced covariance. The mixed logarithmic-coupling derivative tests how all three cavity couplings change this spectrum together. The entropy diagnostic is a benchmark extension of the cavity-QED ring/RPA construction.

## Problem

Use the cavity-QED direct-RPA/ring-CCD correspondence to calculate the sixth mixed response of the photon Rényi-3/2 entropy to three logarithmic coupling changes. Retain the electronic, electron-photon and photon-photon ring amplitudes. The finite transition parameters below define the benchmark. Electronic particle-hole coordinates are canonical bosons for its Gaussian-state diagnostic. No orbital optimization is required.

Energies are in hartree. The coupling multipliers and logarithmic perturbations u, v and w are dimensionless. The dipole columns and Coulomb factors have units of the square root of a hartree.

| Transition | Gap | Factor 1 | Factor 2 | Factor 3 | Dipole 1 | Dipole 2 | Dipole 3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.42 | 0.19 | -0.06 | 0.04 | 0.82 | -0.37 | -0.24 |
| 2 | 0.71 | 0.08 | 0.17 | -0.05 | -0.46 | 0.71 | 0.53 |
| 3 | 1.03 | -0.11 | 0.09 | 0.16 | 0.35 | 0.58 | -0.67 |
| 4 | 1.28 | 0.05 | -0.12 | 0.14 | 0.62 | -0.29 | 0.41 |

| Photon mode | Frequency | Base multiplier |
|---:|---:|---:|
| 1 | 0.53 | 0.37 |
| 2 | 0.88 | 0.46 |
| 3 | 0.67 | 0.41 |

Let F be the factor matrix, d the three dipole columns and D the diagonal gap matrix. There is no additional spin factor. Define

$$
J=FF^{\mathsf T},\qquad A_e=D+J,\qquad B_e=J,
$$

$$
\ell_1=0.37e^u,\qquad \ell_2=0.46e^v,\qquad \ell_3=0.41e^w,
$$

$$
\Delta=\sum_{k=1}^{3}\ell_k^2d_kd_k^{\mathsf T},\qquad
g_k=-\sqrt{\omega_k/2}\,\ell_kd_k.
$$

Use electronic coordinates first and photon coordinates last:

$$
A=\begin{pmatrix}A_e+\Delta&g\\g^{\mathsf T}&\Omega_c\end{pmatrix},\qquad
B=\begin{pmatrix}B_e+\Delta&g\\g^{\mathsf T}&0\end{pmatrix},\qquad
\Omega_c=\operatorname{diag}(0.53,0.88,0.67).
$$

The real symmetric ring amplitude T is the stabilizing solution

$$
B+AT+TA+TBT=0.
$$

Choose the branch connected to the positive-frequency RPA invariant subspace. Both A−B and A+B are positive definite. Keep all photon-pair amplitudes. The normalized Gaussian state is

$$
|\Psi\rangle=\det(I-T^2)^{1/4}
\exp\!\left(\tfrac12c^\dagger{}^{\mathsf T}Tc^\dagger\right)|0\rangle.
$$

Quadratures satisfy [q_i,p_j]=iδ_ij, with vacuum variance 1/2. Trace out the four electronic transition modes. Write the three-photon covariance as Vγ=diag(Q,P), in the order (q₁,q₂,q₃,p₁,p₂,p₃). The first moments and q-p covariance vanish. Let ν_j be the positive square roots of the eigenvalues of QP. At the base point all three ν_j exceed 1/2. Define

$$
S_{3/2}(u,v,w)=-2\ln\operatorname{Tr}\rho_\gamma^{3/2}
=2\sum_{j=1}^{3}\ln\!\left[(\nu_j+\tfrac12)^{3/2}-(\nu_j-\tfrac12)^{3/2}\right].
$$

Calculate the dimensionless number

$$
Z=\left.\frac{\partial^6 S_{3/2}}{\partial u^2\partial v^2\partial w^2}\right|_{u=v=w=0}.
$$

Use natural logarithms. Z is an ordinary derivative. If using Taylor series, retain the rectangular set 0≤p,q,r≤2, including total degree six, and convert the (2,2,2) coefficient by 2!2!2!=8. Any numerically equivalent method is acceptable.

At u=v=w=0, report the seven positive RPA frequencies in ascending order, the 3×3 photon block of T, and the two 3×3 photon covariance blocks Q and P. Report the ordinary mixed energy derivative

$$
E_{222}=\left.\frac{\partial^6 E_c}{\partial u^2\partial v^2\partial w^2}\right|_{0}.
$$

For the entropy, report the ordinary derivatives

$$
D_{pqr}=\left.\frac{\partial^{p+q+r}S_{3/2}}{\partial u^p\partial v^q\partial w^r}\right|_{0},
\qquad p,q,r\in\{0,1,2\},\quad p+q+r\geq3.
$$

There are 17 such derivatives. List the 16 entries other than D₂₂₂ in a table with columns (p,q,r) and D_pqr, in lexicographic order. Report D₂₂₂=Z once, inside the final-answer tags. Base correlation energies, entropy derivatives of total order below three, and a separate numerical Taylor coefficient are not required.

Explain the RPA/ring energy identity and the Gaussian covariance normalization, citing the sources used. Put the derivation, citations and diagnostics in numbered steps inside `<reasoning>...</reasoning>`. Put Z inside `<final_answer>...</final_answer>`. Absolute accuracy 2×10⁻⁷ is sufficient for every reported numerical quantity.

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

01_transition_blocks

Goal
----
Build the direct electronic RPA blocks.

```python
def transition_blocks(gaps, factors):
    """Build the direct electronic RPA blocks.

    Parameters
    ----------
    gaps : real array, shape (n,)
        Positive electronic transition gaps.
    factors : real array, shape (n,r)
        Direct Coulomb factors; n and r are positive integers.
    
    Returns
    -------
    electronic_a, electronic_b : real arrays, shape (n,n)
        electronic_b=factors@factors.T; electronic_a=diag(gaps)+electronic_b.
        No extra spin factor is applied. Do not mutate either input.
        NumPy and SciPy are available. Import dependencies inside the function.
        Tests use absolute and relative tolerances of 2e-8.
    """
    return electronic_a, electronic_b
```

### Step 2

02_cavity_jets

Goal
----
Expand the cavity RPA matrices in independently varied logarithmic couplings.

```python
def cavity_jets(electronic_a, electronic_b, dipoles, frequencies, couplings, orders):
    """Expand the cavity RPA matrices in independently varied logarithmic couplings.

    Parameters
    ----------
    electronic_a, electronic_b : real symmetric arrays, shape (n,n)
        electronic_a-electronic_b is positive definite.
    dipoles : real array, shape (n,m)
    frequencies : positive real array, shape (m,)
    couplings : real array, shape (m,)
        Signed or zero base couplings are allowed; 1 <= m <= 3.
    orders : integer sequence, length m
        Each order lies between 0 and 3. Parameter x_k changes only mode k:
        ell_k=couplings[k]*exp(x_k).
    
    Returns
    -------
    a, b : real arrays, shape (*(orders+1),n+m,n+m)
        Taylor coefficients of A=[[electronic_a+Delta,g],[g.T,diag(frequencies)]]
        and B=[[electronic_b+Delta,g],[g.T,zeros((m,m))]], where
        Delta=sum_k ell_k**2*outer(dipoles[:,k],dipoles[:,k]) and
        g[:,k]=-sqrt(frequencies[k]/2)*ell_k*dipoles[:,k].
        Electronic coordinates precede photons. Coefficients involving positive
        powers of more than one parameter vanish in a and b.
    A matrix jet has shape (*grid,n,n), with one to three leading axes.
    Each grid length is between 1 and 4. For multi-index alpha, J[alpha] is
    the coefficient of x**alpha, i.e. the ordinary derivative divided by
    prod_j factorial(alpha_j). Retain the entire rectangular grid, including
    mixed coefficients up to its total degree. Products are ordered Taylor
    convolutions without binomial factors. Inputs are finite real array-like
    objects. Return float64-compatible arrays and do not mutate inputs.
    NumPy and SciPy are available; import dependencies inside the function.
    Numerical tests use absolute and relative tolerances of 2e-8.
    """
    return a, b
```

### Step 3

03_stable_ring

Goal
----
Obtain the stabilizing ring amplitude from the positive RPA subspace.

```python
def stable_ring(a, b):
    """Obtain the stabilizing ring amplitude from the positive RPA subspace.

    Parameters
    ----------
    a, b : symmetric real arrays, shape (n,n)
        Both a-b and a+b must have minimum eigenvalue strictly above 1e-12.
    
    Returns
    -------
    omega : real array, shape (n,)
        Positive frequencies of [[a,b],[-b,-a]] in ascending order.
    t : real symmetric array, shape (n,n)
        The stable ratio Y@inv(X), where [X;Y] spans the positive invariant
        subspace. It satisfies b+a@t+t@a+t@b@t=0 and a+b@t has positive spectrum.
        Degenerate positive frequencies are allowed. Do not return the other branch.
    
    Raises
    ------
    ValueError
        If either a-b or a+b has minimum eigenvalue at most 1e-12.
    
    Inputs are finite real array-like objects. Do not mutate inputs. Import
    dependencies inside the function. NumPy and SciPy are available. Numeric
    tests use absolute and relative tolerances of 2e-8.
    """
    return omega, t
```

### Step 4

04_ring_jets

Goal
----
Differentiate the stabilizing Riccati branch on a rectangular multivariate grid.

```python
def ring_jets(a, b, t0):
    """Differentiate the stabilizing Riccati branch on a rectangular multivariate grid.

    Parameters
    ----------
    a, b : real arrays, shape (*grid,n,n)
        Symmetric matrix Taylor coefficients on identical grids.
    t0 : real symmetric array, shape (n,n)
        Stabilizing base solution. a[zero]+b[zero]@t0 has positive real spectrum.
    
    Returns
    -------
    t : real array, shape (*grid,n,n)
        t[zero]=t0. Every retained coefficient of B+A@T+T@A+T@B@T is zero.
        Use the stable analytic continuation, including mixed matrix coefficients
        and degenerate positive base frequencies. No eigenvector gauge is an output.
    A matrix jet has shape (*grid,n,n), with one to three leading axes.
    Each grid length is between 1 and 4. For multi-index alpha, J[alpha] is
    the coefficient of x**alpha, i.e. the ordinary derivative divided by
    prod_j factorial(alpha_j). Retain the entire rectangular grid, including
    mixed coefficients up to its total degree. Products are ordered Taylor
    convolutions without binomial factors. Inputs are finite real array-like
    objects. Return float64-compatible arrays and do not mutate inputs.
    NumPy and SciPy are available; import dependencies inside the function.
    Numerical tests use absolute and relative tolerances of 2e-8.
    """
    return t
```

### Step 5

05_inverse_jets

Goal
----
Invert an ordered multivariate matrix Taylor series.

```python
def inverse_jets(matrix):
    """Invert an ordered multivariate matrix Taylor series.

    Parameters
    ----------
    matrix : real array, shape (*grid,n,n)
        Its base matrix is invertible; symmetry is not required.
    
    Returns
    -------
    inverse : real array, same shape
        The ordered convolution matrix@inverse is I at the all-zero multi-index
        and zero at every other retained coefficient. A singular base lies outside
        this function's input domain. Integer-valued input arrays are valid.
    A matrix jet has shape (*grid,n,n), with one to three leading axes.
    Each grid length is between 1 and 4. For multi-index alpha, J[alpha] is
    the coefficient of x**alpha, i.e. the ordinary derivative divided by
    prod_j factorial(alpha_j). Retain the entire rectangular grid, including
    mixed coefficients up to its total degree. Products are ordered Taylor
    convolutions without binomial factors. Inputs are finite real array-like
    objects. Return float64-compatible arrays and do not mutate inputs.
    NumPy and SciPy are available; import dependencies inside the function.
    Numerical tests use absolute and relative tolerances of 2e-8.
    """
    return inverse
```

### Step 6

06_photon_covariance_jets

Goal
----
Reduce the normalized ring vacuum to its photon quadrature covariance.

```python
def photon_covariance_jets(t, electronic_count):
    """Reduce the normalized ring vacuum to its photon quadrature covariance.

    Parameters
    ----------
    t : real array, shape (*grid,n,n)
        Symmetric ring-amplitude coefficients; all eigenvalues of t[zero] are
        strictly between -1 and 1.
    electronic_count : integer
        1 <= electronic_count < n. The remaining m coordinates are photons.
    
    Returns
    -------
    covariance : real array, shape (*grid,2,m,m)
        Block 0 is the photon principal submatrix of Vq=.5*(I+T)@inv(I-T).
        Block 1 is the photon principal submatrix of Vp=.5*(I-T)@inv(I+T).
        The inverses act on all n coordinates before restriction to photons.
        q=(c+c_dagger)/sqrt(2), p=(c-c_dagger)/(i*sqrt(2)); vacuum variance .5.
        The real branch has zero q-p covariance. Earlier inverse_jets is available.
    A matrix jet has shape (*grid,n,n), with one to three leading axes.
    Each grid length is between 1 and 4. For multi-index alpha, J[alpha] is
    the coefficient of x**alpha, i.e. the ordinary derivative divided by
    prod_j factorial(alpha_j). Retain the entire rectangular grid, including
    mixed coefficients up to its total degree. Products are ordered Taylor
    convolutions without binomial factors. Inputs are finite real array-like
    objects. Return float64-compatible arrays and do not mutate inputs.
    NumPy and SciPy are available; import dependencies inside the function.
    Numerical tests use absolute and relative tolerances of 2e-8.
    """
    return covariance
```

### Step 7

07_entropy_jets

Goal
----
Evaluate the photon Renyi entropy series at index 2 or 3/2.

```python
def entropy_jets(covariance, renyi=2.0):
    """Evaluate the photon Renyi entropy series at index 2 or 3/2.

    Parameters
    ----------
    covariance : real array, shape (*grid,2,m,m)
        Symmetric q and p Taylor coefficients; no q-p block. Both base blocks
        must be positive definite. Their product need not be symmetric.
    renyi : float, default 2.0
        Supported values are exactly 2.0 and 1.5. For 1.5, every base symplectic
        eigenvalue nu=sqrt(eig(Vq@Vp)) must be strictly greater than .5+1e-12.
        The 1e-12 gap resolves roundoff at the pure-state boundary. Repeated
        eigenvalues above this bound are allowed. A generic jet touching a pure
        mode is excluded from the 1.5 branch because it need not be analytic.
    
    Returns
    -------
    entropy : real array, shape (*grid)
        Taylor coefficients of S_alpha=log(Tr rho**alpha)/(1-alpha).
        At alpha=2, use .5*logdet(2*Vq)+.5*logdet(2*Vp), including its constant.
        At alpha=1.5, S=2*sum_j log((nu_j+.5)**1.5-(nu_j-.5)**1.5).
        Use the analytic continuation from the basepoint and natural logarithms.
        The input matrices and their derivatives need not commute; do not discard
        derivatives of spectral projectors. Earlier inverse_jets is available.
    
    Raises
    ------
    ValueError
        If renyi is unsupported, either base covariance block is not positive
        definite, or the 1.5 branch has any base nu<=.5+1e-12. Check these before
        attempting a singular inverse. For alpha=2 a constant vacuum is valid.
    A matrix jet has shape (*grid,n,n), with one to three leading axes.
    Each grid length is between 1 and 4. For multi-index alpha, J[alpha] is
    the coefficient of x**alpha, i.e. the ordinary derivative divided by
    prod_j factorial(alpha_j). Retain the entire rectangular grid, including
    mixed coefficients up to its total degree. Products are ordered Taylor
    convolutions without binomial factors. Inputs are finite real array-like
    objects. Return float64-compatible arrays and do not mutate inputs.
    NumPy and SciPy are available; import dependencies inside the function.
    Numerical tests use absolute and relative tolerances of 2e-8.
    """
    return entropy
```

### Step 8

08_solve_cavity

Goal
----
Return the ordinary mixed logarithmic-coupling entropy response.

```python
def solve_cavity(gaps, factors, dipoles, frequencies, couplings, orders, renyi=2.0):
    """Return the ordinary mixed logarithmic-coupling entropy response.

    Parameters
    ----------
    gaps, factors : real arrays, shapes (n,), (n,r)
        Positive gaps, 1<=n,r<=8. D=diag(gaps), J=factors@factors.T,
        Ae=D+J, Be=J. No orbital relaxation or extra spin factor.
    dipoles, frequencies, couplings : real arrays, shapes (n,m), (m,), (m,)
        1<=m<=3, frequencies positive. Mode k uses ell_k=couplings[k]*exp(x_k).
        Construct the A and B matrices specified in cavity_jets. Electronic
        coordinates precede photons. Energies are hartree; dipoles and factors
        have sqrt(hartree) units; x_k and couplings are dimensionless.
    orders : integer sequence, length m
        Each entry is 0..3. One logarithmic parameter per photon mode.
    renyi : float, default 2.0
        Exactly 2.0 or 1.5, with the entropy_jets domain. Signed and zero base
        couplings are allowed for 2.0; 1.5 requires a strictly mixed photon base.
    
    Returns
    -------
    response : float
        Ordinary derivative d**sum(orders) S_renyi / product_k dx_k**orders[k]
        at all x_k=0. Multiply the selected Taylor coefficient by
        product_k factorial(orders[k]) exactly once. All-zero orders return
        the base entropy. Call the earlier sub-problem chain.
    
    Raises
    ------
    ValueError
        Propagate the stability condition from stable_ring and the covariance
        and Renyi-domain conditions from entropy_jets.
    A matrix jet has shape (*grid,n,n), with one to three leading axes.
    Each grid length is between 1 and 4. For multi-index alpha, J[alpha] is
    the coefficient of x**alpha, i.e. the ordinary derivative divided by
    prod_j factorial(alpha_j). Retain the entire rectangular grid, including
    mixed coefficients up to its total degree. Products are ordered Taylor
    convolutions without binomial factors. Inputs are finite real array-like
    objects. Return float64-compatible arrays and do not mutate inputs.
    NumPy and SciPy are available; import dependencies inside the function.
    Numerical tests use absolute and relative tolerances of 2e-8.
    """
    return response
```
