# Material_Science-Semiconductor_Materials-43

## Background

Excitons in low-symmetry two-dimensional semiconductors such as black phosphorus have direction-dependent masses, and a perpendicular magnetic field couples their center-of-mass and relative motions so that the separation of the two is nontrivial; an exact separation through the conserved pseudomomentum leaves anisotropy-dependent coefficients in the magnetic terms of the relative-motion Hamiltonian. The magnetoexciton energies, their diamagnetic shifts and the dispersion of a moving magnetoexciton all follow from that relative-motion problem with a nonlocally screened electron-hole interaction.

## Problem

Low-symmetry two-dimensional semiconductors such as monolayer black phosphorus carry direction-dependent effective masses, and their excitons respond to a perpendicular magnetic field through both the orbital coupling and the magnetic confinement of the relative motion. A recent source shows that the separation of the center-of-mass and relative motions of such an anisotropic magnetoexciton, which most earlier treatments performed with a factorized wave function valid only for a hole much heavier than the electron, can be done exactly through the conserved pseudomomentum, and that the exact separation leaves anisotropy-dependent coefficients in the magnetic terms of the relative-motion Hamiltonian that change the magnetoexciton energies and diamagnetic coefficients. The source evaluates its Hamiltonian at zero pseudomomentum and names finite pseudomomentum as the next step. Recover the construction from the source, validate a solver of your own against it, and take that step: a moving magnetoexciton feels its pseudomomentum as an in-plane field through the coupling that the exact separation generates, and its dispersion acquires a field-dependent mass enhancement set by the in-plane polarizabilities of the state in the field. The load-bearing choices are the source's and are not derivable from the statement below: the transformation that separates the motions and the three coefficients it produces from the electron-to-hole mass ratios; which of the two diamagnetic coefficients multiplies which coordinate; the structure of the orbital term and which reduced mass divides which coordinate in it; the form of the pseudomomentum coupling, on which the source's printed relative-motion Hamiltonian and its zero-pseudomomentum equation are not mutually consistent and which must therefore be settled from the transformation itself; how the surrounding dielectric constant enters the screened electron-hole potential; what the factorized approximation of the earlier treatments amounts to in these coefficients; and the source's atomic units.

The model is the source's. An electron and a hole with masses $m^e_x, m^e_y, m^h_x, m^h_y$ (units of the free electron mass) along the principal axes move in a plane in a perpendicular field $B$, interacting through the Rytova-Keldysh potential with screening length $r_0$ in a surrounding of dielectric constant $\kappa$. In atomic units (Hartree, Bohr radius $a_0 = 0.052917721$ nm, and the atomic unit of magnetic field $\hbar/(e a_0^2)$; $E_H = 27211.386$ meV), the exact separation gives a relative-motion Hamiltonian of the form $H = -\frac{1}{2\mu_x}\partial_x^2 - \frac{1}{2\mu_y}\partial_y^2 + H_{orb} + \frac{B^2}{8}(c_x x^2 + c_y y^2) + V(r) + H_K$, with $\mu_x, \mu_y$ the reduced masses, $H_{orb}$ the orbital term linear in $B$ and in the source's coefficient $\alpha$, $c_x$ and $c_y$ the source's diamagnetic coefficients built from $\alpha$ and the two coefficients $\beta_x, \beta_y$, and $H_K$ the coupling of the pseudomomentum $(K_x, K_y)$, linear in $B$, in $K$ and in the relative coordinate, with the center-of-mass masses $M_x, M_y$. The factorized approximation sets $\alpha = 0$ and both diamagnetic coefficients to the inverse reduced masses. The relative energy at small pseudomomentum is quadratic in $K$, so the total dispersion $K_x^2/2M_x + K_y^2/2M_y + E_{rel}(K)$ defines field-dependent masses $M^*_x, M^*_y$ through $1/M^*_x = 1/M_x - \alpha_{yy}B^2/M_x^2$ and $1/M^*_y = 1/M_y - \alpha_{xx}B^2/M_y^2$, where $\alpha_{xx}, \alpha_{yy}$ are the static in-plane polarizabilities of the lowest state in the field, $\alpha_{xx} = 2\sum_{n>0}|\langle n|x|0\rangle|^2/(E_n - E_0)$; the relative mass enhancements are $\delta_x = M^*_x/M_x - 1$ and $\delta_y = M^*_y/M_y - 1$.

Implement eight functions with these conventions: the relative-motion problem is solved in the orthonormal product basis of angular harmonics $e^{im\varphi}/\sqrt{2\pi}$, $|m| \le M$, and the radial functions $\phi^{(m)}_n(r) = \frac{1}{a}\sqrt{\frac{n!}{(n+2|m|+1)!}}\,(r/a)^{|m|} e^{-r/(2a)} L^{(2|m|+1)}_n(r/a)$, $n < N$, of common length scale $a$ (Bohr radii), ordered with index $(m+M)N + n$; every matrix element is evaluated with the $N_q$-point Gauss-Laguerre rule in the variable $r/a$ with the scaled weights $W_k = a^2 t_k w_k e^{t_k}$, which is exact for the kinetic, orbital, diamagnetic and pseudomomentum terms and defines the representation of the potential; energies come from a full Hermitian diagonalization and the polarizabilities from the sum over all eigenstates of the basis; energies are reported in meV, polarizabilities in atomic units; every quantity is deterministic.

`mx_parameters(mex, mey, mhx, mhy, r0_nm, kappa, B_T)` returns, in order, $\mu_x, \mu_y, M_x, M_y, \alpha, c_x, c_y$, the screening length in Bohr radii and the field in atomic units. `mx_keldysh(r, r0_au, kappa)` returns the potential at the given distances. `mx_radial_basis(a, N, m, r)` returns, shape $(3, N, n)$, the radial functions of one channel and their first two derivatives at the given distances. `mx_quadrature(a, Nq)` returns, shape $(2, N_q)$, the nodes and scaled weights, evaluated without overflow or underflow. `mx_hamiltonian(mex, mey, mhx, mhy, r0_au, kappa, B_au, Kx, Ky, a, N, M, Nq, factorized)` returns, shape $(2, D, D)$ with $D = (2M+1)N$, the real and imaginary parts of the Hamiltonian matrix, the exact separation for factorized = 0 and the factorized approximation for factorized = 1. `mx_spectrum(..., k)` returns the k lowest energies. `mx_magnetopolarizability(mex, mey, mhx, mhy, r0_au, kappa, B_au, a, N, M, Nq, factorized)` returns the lowest energy and the two polarizabilities of the lowest state at zero pseudomomentum. `mx_audit(mex, mey, mhx, mhy, r0_nm, kappa, B_T, a, N, M, Nq, K0)`, the orchestrator, must call the earlier functions rather than reimplementing them and returns nine values: the lowest and the second-lowest energy at field B; the magnetic shift of the lowest state, $E_0(B) - E_0(0)$, from the exact separation and from the factorized approximation; $\alpha_{xx}(B)$ and $\alpha_{yy}(B)$; $\delta_x$ and $\delta_y$; and the ratio of the directly diagonalized shift of the lowest relative energy at pseudomomentum $(K_0, 0)$ to its second-order prediction $-\tfrac12\alpha_{yy}(BK_0/M_x)^2$.

All outputs are float64, finite and deterministic: two runs on identical inputs must agree exactly. Validate inputs and raise `ValueError` on non-finite values, on a non-positive mass, screening length, dielectric constant, basis scale, number of radial functions or number of nodes, on a negative field, a non-integral angular momentum, a non-integral or negative angular cutoff, a non-integral or non-positive number of requested energies or one exceeding the basis size, a factorized flag other than 0 or 1, or a non-positive probe pseudomomentum; the orchestrator also raises at zero field, when the Hamiltonian is not Hermitian to $10^{-10}$ Hartree, when the direct and the second-order pseudomomentum shifts differ by more than 2 percent, or when either mass enhancement is not positive.

Evaluate the audit for monolayer black phosphorus, $m^e_x = 0.199$, $m^e_y = 0.753$, $m^h_x = 0.168$, $m^h_y = 5.353$, $r_0 = 2.576$ nm, on a silica substrate with vacuum above, $\kappa = 2.45$, at $B = 120$ T, with $a = 6$, $N = 48$, $M = 20$, $N_q = 150$ and $K_0 = 0.02$.

In your reasoning report the conventions you used, and justify each from the source: how the pseudomomentum separates the motions and what the three coefficients are in terms of the mass ratios; which diamagnetic coefficient multiplies which coordinate and what the factorized approximation does to them; the structure of the orbital term and which reduced mass divides which coordinate; the form of the pseudomomentum coupling and how you settled the inconsistency between the source's two printed forms; how the dielectric constant enters the screened potential; what the source's atomic unit of magnetic field is and how its numerical value follows; and how the source labels the lowest states and which state is second-lowest for this material.

Report numerically, as evidence that the chain was executed: the coefficient $\alpha$ and the two diamagnetic coefficients; the zero-field energy of the lowest state of freestanding black phosphorus and of black phosphorus in hexagonal boron nitride ($\kappa = 4.9$) from your solver, against the source's tables; the lowest and second-lowest energies at 120 T; the magnetic shift of the lowest state from the exact separation and from the factorized approximation; the two polarizabilities; the two relative mass enhancements; and the ratio of the direct to the second-order pseudomomentum shift. These are the scalars that determine the final number.

As the final answer, report the relative mass enhancement of the moving magnetoexciton along y, $\delta_y = M^*_y/M_y - 1$, in percent, to five significant figures.

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

mx_parameters

Goal
----
Derives the reduced and center-of-mass masses, the exact-separation coefficients and the atomic-unit conversions from the material and field parameters.

```python
def mx_parameters(mex, mey, mhx, mhy, r0_nm, kappa, B_T):
    r"""mex, mey, mhx, mhy: positive floats, the electron and hole effective masses along the principal axes x
    and y, in units of the free electron mass. r0_nm: positive float, the Rytova-Keldysh screening length in
    nm. kappa: positive float, the dielectric constant of the surroundings. B_T: non-negative float, the
    perpendicular magnetic field in tesla.

    Returns a numpy float64 array of shape $(9,)$: the reduced masses $\mu_x = m^e_x m^h_x/(m^e_x + m^h_x)$ and
    $\mu_y$; the center-of-mass masses $M_x = m^e_x + m^h_x$ and $M_y$; the coefficient
    $\alpha = (1 - \rho_x\rho_y)/((1+\rho_x)(1+\rho_y))$ with $\rho_x = m^e_x/m^h_x$ and $\rho_y = m^e_y/m^h_y$;
    the two diamagnetic coefficients of the exact center-of-mass separation, $c_x = (\beta_y + \alpha^2)/\mu_y$
    multiplying $x^2$ and $c_y = (\beta_x + \alpha^2)/\mu_x$ multiplying $y^2$, with
    $\beta_x = 4\rho_x/(1+\rho_x)^2$ and $\beta_y = 4\rho_y/(1+\rho_y)^2$; the screening length in Bohr radii
    ($a_0 = 0.052917721$ nm); and the magnetic field in atomic units, $B/B_0$ with
    $B_0 = \hbar/(e a_0^2) = 235051.757$ T.

    Raises:
        ValueError: on a non-positive or non-finite mass, screening length or kappa, or a negative or non-finite B_T.
    """
    return None
```

### Step 2

mx_keldysh

Goal
----
Evaluates the Rytova-Keldysh electron-hole potential of a two-dimensional semiconductor in atomic units.

```python
def mx_keldysh(r, r0_au, kappa):
    r"""r: one-dimensional array of positive electron-hole distances in Bohr radii. r0_au: positive float, the
    screening length in Bohr radii. kappa: positive float.

    Returns a numpy float64 array of shape $(n,)$: the Rytova-Keldysh potential in Hartree,
    $V(r) = -\frac{\pi}{2 r_0}\left[H_0\!\left(\frac{\kappa r}{r_0}\right) - Y_0\!\left(\frac{\kappa r}{r_0}\right)\right]$
    with $H_0$ the Struve function and $Y_0$ the Bessel function of the second kind, which is the real-space
    form of $-\frac{1}{2\pi\kappa}\int d^2q\, e^{i\mathbf{q}\cdot\mathbf{r}}/[q(1 + r_0 q/\kappa)]$.

    Raises:
        ValueError: on an invalid r, or a non-positive or non-finite r0_au or kappa.
    """
    return None
```

### Step 3

mx_radial_basis

Goal
----
Builds the orthonormal generalized-Laguerre radial basis of one angular channel with its first two derivatives.

```python
def mx_radial_basis(a, N, m, r):
    r"""a: positive float, the length scale of the basis in Bohr radii. N: positive integer, the number of radial
    functions. m: integer, the angular momentum of the channel. r: one-dimensional array of positive distances
    in Bohr radii.

    Returns a numpy float64 array of shape $(3, N, n)$: the orthonormal radial functions
    $\phi^{(m)}_n(r) = \frac{1}{a}\sqrt{\frac{n!}{(n + 2|m| + 1)!}}\,(r/a)^{|m|}\, e^{-r/(2a)}\, L^{(2|m|+1)}_n(r/a)$,
    $n = 0, \dots, N-1$, with $L^{(\alpha)}_n$ the generalized Laguerre polynomials, which satisfy
    $\int_0^\infty \phi^{(m)}_n \phi^{(m)}_{n'}\, r\, dr = \delta_{nn'}$; their first derivatives $d\phi/dr$; and their
    second derivatives $d^2\phi/dr^2$, each evaluated at the given distances.

    Raises:
        ValueError: on a non-positive or non-finite a, a non-integral or non-positive N, a non-integral m, or an
            invalid r.
    """
    return None
```

### Step 4

mx_quadrature

Goal
----
Builds the scaled Gauss-Laguerre rule that evaluates every radial matrix element of the basis.

```python
def mx_quadrature(a, Nq):
    r"""a: positive float, the basis length scale in Bohr radii. Nq: positive integer, the number of nodes.

    Returns a numpy float64 array of shape $(2, Nq)$: the nodes $r_k = a t_k$ and the weights
    $W_k = a^2\, t_k\, w_k e^{t_k}$ of the $N_q$-point Gauss-Laguerre rule $(t_k, w_k)$ for the weight $e^{-t}$,
    ordered by increasing node, so that $\int_0^\infty f(r) g(r)\, r\, dr \approx \sum_k W_k f(r_k) g(r_k)$ for
    functions carrying the factor $e^{-r/(2a)}$ each, exactly whenever $f g\, e^{r/a}$ is a polynomial in $r$ of
    degree at most $2N_q - 2$. The scaled weights $w_k e^{t_k}$ must be evaluated without overflow or underflow for
    every node.

    Raises:
        ValueError: on a non-positive or non-finite a, or a non-integral or non-positive Nq.
    """
    return None
```

### Step 5

mx_hamiltonian

Goal
----
Assembles the relative-motion Hamiltonian of the anisotropic magnetoexciton at finite pseudomomentum in the coupled angular-channel Laguerre basis.

```python
def mx_hamiltonian(mex, mey, mhx, mhy, r0_au, kappa, B_au, Kx, Ky, a, N, M, Nq, factorized):
    r"""mex, mey, mhx, mhy, kappa: as before. r0_au: positive float, the screening length in Bohr radii. B_au:
    non-negative float, the magnetic field in atomic units. Kx, Ky: floats, the pseudomomentum components in
    atomic units. a, N, Nq: as before. M: non-negative integer, the largest angular momentum kept. factorized:
    0 or 1; 1 selects the factorized approximation in which $\alpha = 0$ and both diamagnetic coefficients are
    replaced by the inverse reduced masses, $c_x = 1/\mu_y$ and $c_y = 1/\mu_x$.

    Returns a numpy float64 array of shape $(2, D, D)$ with $D = (2M+1)N$: the real and the imaginary part of
    the matrix of the relative-motion Hamiltonian, in Hartree,
    $H = -\frac{1}{2\mu_x}\partial_x^2 - \frac{1}{2\mu_y}\partial_y^2
    - \frac{i\alpha B}{2}\left(\frac{y}{\mu_x}\partial_x - \frac{x}{\mu_y}\partial_y\right)
    + \frac{B^2}{8}\left(c_x x^2 + c_y y^2\right) + V(r) + B\left(\frac{K_y}{M_y}x - \frac{K_x}{M_x}y\right)$,
    with the coefficients of the parameter step and the potential of the potential step, in the orthonormal
    product basis $\phi^{(m)}_n(r)\, e^{im\varphi}/\sqrt{2\pi}$, $m = -M, \dots, M$, $n = 0, \dots, N-1$, ordered
    with index $(m + M)N + n$. Every matrix element is evaluated with the $N_q$-point rule of the quadrature step,
    which is exact for the kinetic, magnetic and pseudomomentum terms and defines the representation of the
    potential; the matrix is Hermitian.

    Raises:
        ValueError: on invalid masses, r0_au or kappa, a negative or non-finite B_au, a non-finite Kx or Ky, an
            invalid a, N or Nq, a non-integral or negative M, or a factorized flag other than 0 or 1.
    """
    return None
```

### Step 6

mx_spectrum

Goal
----
Diagonalizes the magnetoexciton Hamiltonian and returns the lowest energies in meV.

```python
def mx_spectrum(mex, mey, mhx, mhy, r0_au, kappa, B_au, Kx, Ky, a, N, M, Nq, factorized, k):
    r"""Parameters as in the Hamiltonian step; k: positive integer not larger than $(2M+1)N$.

    Returns a numpy float64 array of shape $(k,)$: the k lowest eigenvalues of the Hamiltonian of the previous
    step in increasing order, converted to meV with $E_H = 27211.386$ meV, from a full Hermitian
    diagonalization.

    Raises:
        ValueError: whenever the Hamiltonian step would raise, or on a non-integral, non-positive or too large k.
    """
    return None
```

### Step 7

mx_magnetopolarizability

Goal
----
Computes the exact-within-basis static polarizabilities of the lowest magnetoexciton state along both principal axes.

```python
def mx_magnetopolarizability(mex, mey, mhx, mhy, r0_au, kappa, B_au, a, N, M, Nq, factorized):
    r"""Parameters as in the Hamiltonian step, at zero pseudomomentum.

    Returns a numpy float64 array of shape $(3,)$: the lowest eigenvalue in meV, and the static in-plane
    polarizabilities of the lowest state at field B along x and along y, in atomic units,
    $\alpha_{xx} = 2\sum_{n>0} |\langle n|x|0\rangle|^2/(E_n - E_0)$ and the same with y, where the sum runs over
    all other eigenstates of the Hamiltonian in the basis, which is the exact second-order response within the
    basis to a uniform in-plane field, and the operators x and y are represented in the same basis with the same rule.

    Raises:
        ValueError: whenever the Hamiltonian step would raise, or when the basis holds a single state.
    """
    return None
```

### Step 8

mx_audit

Goal
----
Runs the whole chain: the magnetoexciton energies, the exact and factorized magnetic shifts, the polarizabilities and the mass enhancements of the moving magnetoexciton, with a direct finite-pseudomomentum check.

```python
def mx_audit(mex, mey, mhx, mhy, r0_nm, kappa, B_T, a, N, M, Nq, K0):
    r"""mex, mey, mhx, mhy, r0_nm, kappa, B_T: as in the parameter step. a, N, M, Nq: as before. K0: positive
    float, a probe pseudomomentum in atomic units.

    The orchestrator. It must call the earlier functions rather than reimplementing them. Returns a numpy
    float64 array of shape $(9,)$: the lowest energy at field B (meV); the second-lowest energy at field B
    (meV); the magnetic shift of the lowest state, $E_0(B) - E_0(0)$ (meV), from the exact separation; the same
    shift from the factorized approximation; the polarizabilities $\alpha_{xx}(B)$ and $\alpha_{yy}(B)$ of the
    lowest state (atomic units); the relative mass enhancements of the moving magnetoexciton,
    $\delta_x = M^*_x/M_x - 1$ and $\delta_y = M^*_y/M_y - 1$ with $1/M^*_x = 1/M_x - \alpha_{yy}B^2/M_x^2$ and
    $1/M^*_y = 1/M_y - \alpha_{xx}B^2/M_y^2$, which follow from the second-order shift of the relative energy
    under the pseudomomentum coupling; and the ratio of the directly diagonalized shift of the lowest relative
    energy at pseudomomentum $(K_x, K_y) = (K_0, 0)$ to its second-order prediction
    $-\tfrac12\alpha_{yy}(B K_0/M_x)^2$.

    Raises:
        ValueError: whenever any of the functions it calls would raise; on a non-positive or non-finite K0; when
            the field is zero (no dispersion correction to audit); when the Hamiltonian at zero pseudomomentum
            departs from Hermiticity by more than 1e-10 Hartree; when the direct and the second-order shifts differ
            by more than 2 percent (the probe must stay in the quadratic regime); or when either mass enhancement is
            not positive.
    """
    return None
```
