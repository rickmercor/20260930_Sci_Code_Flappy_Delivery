# Physics-Optics-16

## Background

Rigorous coupled-wave analysis represents a periodic optical structure through coupled spatial Fourier harmonics. For TM polarization, changes in a material boundary alter both reciprocal-permittivity coupling and the electromagnetic field matching between layers. A far-field power measurement therefore depends on the internal coupling of propagating and evanescent harmonics, not only on the visibly transmitted diffraction orders.

The source paper's contribution is a matrix-function reformulation of layer scattering, intended to avoid an eigendecomposition bottleneck while supporting differentiation. The present task uses that reformulation on an explicitly defined, small TM system and asks for a geometry sensitivity. Its two-layer geometry, truncation, and sensitivity target are task-specific choices, not a reproduction of the paper's experimental configuration or GPU benchmarks. Power is normalized to the incident vacuum flux; the requested derivative measures an infinitesimal change in fill fraction with all other physical and numerical inputs fixed.

## Problem

Use a matrix-function reformulation of rigorous coupled-wave analysis to evaluate a fixed-update sensitivity of the paper’s rational matrix-square-root iteration. Then complete the optical calculation as a validation case for the following lossless, two-layer TM grating. Calculate these quantities rather than GPU timing or an infinite-order convergence limit.

Use precisely the seven harmonics $m=-3,\ldots,3$, in ascending order, with the dimensionless coordinates $\xi=x/\Lambda$, $\zeta=k_0z$, $k_0=2\pi/\lambda$, and the following fixed data:

$$
\lambda/\Lambda=0.76,\qquad
\kappa_m=0.23+0.76m,\qquad
K=\operatorname{diag}(\kappa_m),
\qquad
\begin{array}{c|rrrr}
j&f_j&a_j&\epsilon_{\mathrm{hi},j}&d_j\;(=k_0h_j)\\ \hline
1&0.43&0&4&3.1\\
2&0.57&0.19&2.89&2.4
\end{array}
$$

Each layer has relative permeability one and relative permittivity $\epsilon_j(\xi)=\epsilon_{\mathrm{hi},j}$ when $0\leq(\xi-a_j)\bmod1<f_j$ and $\epsilon_j(\xi)=1$ otherwise. The layers are adjacent in the order $1$ then $2$ between vacuum half-spaces, with no intervening thickness.

Define the finite-dimensional TM model by

$$
\widehat\eta_{j,p}=\int_0^1\frac{e^{-2\pi i p\xi}}{\epsilon_j(\xi)}\,d\xi,
\qquad
(C_j)_{mn}=\widehat\eta_{j,m-n},
\qquad
P_j=C_j^{-1},
\qquad
Q_j=KC_jK-I.
$$

Here $\mathbf h$ contains the $H_y$ Fourier coefficients and $\mathbf v=C_j\,d\mathbf h/d\zeta$. Both quantities are continuous across interfaces. Recover the corresponding first-order block evolution from these definitions and the source’s matrix-function construction.

For the vacuum reference medium, use $W_0=I$, $V_0=\operatorname{diag}(g_m)$, and the forward-wave convention $e^{-g_m\zeta}$, where

$$
g_m=
\begin{cases}
i\sqrt{1-\kappa_m^2},&|\kappa_m|<1,\\
\sqrt{\kappa_m^2-1},&|\kappa_m|>1.
\end{cases}
$$

Use the scattering convention

$$
\begin{bmatrix}
\mathbf b_{\mathrm L}\\
\mathbf b_{\mathrm R}
\end{bmatrix}
=
S
\begin{bmatrix}
\mathbf a_{\mathrm L}\\
\mathbf a_{\mathrm R}
\end{bmatrix},
\qquad
\mathbf a_{\mathrm L}=\mathbf e_0,
\qquad
\mathbf a_{\mathrm R}=0,
$$

where $\mathbf a$ denotes incoming amplitudes, $\mathbf b$ denotes outgoing amplitudes, and $\mathbf e_0$ is the unit coefficient in harmonic $m=0$.

For each layer let $M_j=P_jQ_j$. Use the rotated branch parameter $\phi=-\pi/4$ and initialize the rational square-root iteration by

$$
Y_0=e^{i\phi}M_j,\qquad Z_0=I,\qquad H_k=Z_kY_k.
$$

For rational approximation order $m_{\mathrm{alg}}=3$, recover from Algorithm 1 of the source paper the numerator and denominator matrix polynomials, their coefficients, and the ordered linear solve defining each correction $F_k$. Preserve the source’s left-solve orientation and written multiplication order exactly.

Use only that source-defined rational correction. Substituting another Padé approximation, matrix-square-root routine, eigendecomposition, or equivalent convergent iteration does not satisfy the task.

Update

$$
Y_{k+1}=Y_kF_k,\qquad Z_{k+1}=F_kZ_k.
$$

Apply no extra normalization. Differentiate the operations actually executed with respect to $f_1$, holding the branch and update count locally fixed. Matrix products must retain their written order, and every parameter-dependent solve $AB=C$ must be differentiated as

$$
\dot B=A^{-1}(\dot C-\dot A B).
$$

For layer 1, after exactly two completed rational updates, define

$$
R_2=I-Z_2Y_2,\qquad
\rho_2=\|R_2\|_{\mathrm F},
$$

and define the primary requested sensitivity by

$$
J=
\left.\frac{d\rho_2}{df_1}\right|_{f_1=0.43}
=
\frac{\operatorname{Re}\operatorname{tr}(R_2^\dagger\dot R_2)}
{\rho_2}.
$$

The tagged final answer must be $J$. This fixed-update residual sensitivity is specific to the prescribed rational iteration; substituting another convergent matrix-square-root algorithm does not satisfy the task.

After recording $\rho_2$ and $J$, continue the same iteration independently for each layer until the first $\|I-ZY\|_{\mathrm F}<10^{-13}$, permitting at most 50 completed updates. Form the converged root as prescribed by the rotated branch construction, recover the transformed modal matrices and the gap-referenced single-layer scattering matrices, and compose layer 1 followed by layer 2 with the Redheffer star product.

Use complex128 arithmetic throughout. Matrix exponentials, Fréchet derivatives, and linear solves are permitted. Eigendecomposition, black-box matrix square roots, input symmetrization, and finite-difference differentiation are not permitted.

Let $t_m=(S_{21})_{m0}$ and $r_m=(S_{11})_{m0}$. Recover from the source convention the vacuum-flux normalization that converts these magnetic-field amplitudes into diffraction powers. For propagating orders, write $\beta_m=\sqrt{1-\kappa_m^2}$ and use

$$
T_{+1}=\frac{\beta_{+1}}{\beta_0}|t_{+1}|^2,
\qquad
D=
2\frac{\beta_{+1}}{\beta_0}
\operatorname{Re}\left(
\overline{t_{+1}}\frac{dt_{+1}}{df_1}
\right).
$$

Use the complete optical sensitivity $D$ only as a validation of the converged pipeline. Hold every input other than $f_1$ fixed and express all derivatives per unit fill fraction rather than per percentage point.

Report $J$ to absolute error at most $10^{-8}$ and with at least ten digits after the decimal point. Report $\rho_2$ and $D$ to absolute error at most $10^{-8}$. In the reasoning, report $\rho_2$, the validation value $D$, the nominal transmitted $+1$-order power, both converged square-root update counts, and the total reflected-plus-transmitted flux check. Retain all seven harmonics internally even though only propagating orders carry far-field power.

Give a concise, self-contained account of the moving-boundary Fourier tangent, ordered TM product, two differentiated rational updates, residual-norm derivative, transformed modal multiplication order, propagation Fréchet derivative, single-layer scattering construction, physical-order Redheffer composition, and flux normalization.

## Output format

```
## Output format

Give one finite decimal value for J inside <final_answer>...</final_answer>, followed by a concise, self-contained scientific explanation inside <reasoning>...</reasoning>. Include the requested intermediate diagnostics and mathematical expressions needed to support them. Put only the number inside the final-answer tags.
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

Reciprocal-permittivity Fourier coupling

Goal
----
Construct the reciprocal-permittivity convolution matrix of the periodic binary layer specified by harmonics, fill, offset, and epsilon_high, together with its derivative with respect to this call's fill argument.

```python
def reciprocal_fourier(
    harmonics: np.ndarray,
    fill: float,
    offset: float,
    epsilon_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a reciprocal-permittivity convolution matrix and fill tangent.

    Parameters
    ----------
    harmonics : np.ndarray
        Nonempty one-dimensional integer harmonic ordering.
    fill : float
        Layer fill fraction satisfying 0 < fill < 1.
    offset : float
        Finite periodic displacement. Interpret the value modulo one.
    epsilon_high : float
        Positive relative permittivity with a finite float64 reciprocal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The complex128 coefficient matrix C and its derivative dC/df,
        both with shape (N, N).

    Raises
    ------
    ValueError
        If the harmonic array or layer parameters are invalid, or if
        1/epsilon_high is not representable as a finite float64 value.
    """
    return (None, None)
```

### Step 2

Ordered TM operators

Goal
----
Construct the coupled TM operators from the reciprocal-permittivity convolution matrix and its supplied directional derivative.



Form P = C inverse, Q = K C K - I, and the ordered product M = P Q, where K is the diagonal matrix of the supplied transverse wavevectors. Compute the directional derivatives of all three matrices while holding K fixed.



Preserve matrix multiplication order: P and Q need not commute. Evaluate inverse actions through linear solves. This step receives numerical arrays directly and does not call the first step internally.

```python
def tm_operators(
    coefficients: np.ndarray,
    tangent: np.ndarray,
    kappa: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Form the ordered TM operators and their tangents.

    Parameters
    ----------
    coefficients : np.ndarray
        Nonsingular convolution matrix C of shape (N, N).
    tangent : np.ndarray
        Directional derivative of C with shape (N, N).
    kappa : np.ndarray
        Fixed real wavevector of shape (N,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray]
        P, dP, Q, dQ, P Q, and d(P Q), each complex128 (N, N).

    Raises
    ------
    np.linalg.LinAlgError
        If the coefficient matrix is singular.
    """
    return None, None, None, None, None, None
```

### Step 3

Rotated rational root and tangent

Goal
----
Implement the rotated rational matrix-square-root iteration and its exact directional tangent using the algorithm, branch convention, stopping rule, and restrictions stated in the main problem.

Returns

-------

Return the tuple:

(G, dG, update_count, final_residual, rho_2, d_rho_2)

in exactly that order.

- G: complex128 NumPy array of shape (N, N), containing the branch-selected matrix square root.

- dG: complex128 NumPy array of shape (N, N), containing its directional derivative through the executed rational iterations.

- update_count: Python integer counting completed updates, excluding initialization.

- final_residual: Python float equal to the converged Frobenius residual ||I - Z @ Y||_F.

- rho_2: Python float equal to ||I - Z_2 @ Y_2||_F after exactly two completed updates, recorded before the third correction.

- d_rho_2: Python float equal to the directional derivative of rho_2 with respect to the supplied tangent direction.

A successful return satisfies:

- 0 <= update_count <= 50

- final_residual < 1e-13

If convergence has not occurred after 50 completed updates, raise RuntimeError instead of returning an unconverged result. A singular linear solve may raise LinAlgError.

The final_residual is the converged iteration residual ||I - Z @ Y||_F, while rho_2 and d_rho_2 are the residual norm and its directional derivative after exactly two completed updates. Do not return a derivative of final_residual or of update_count.

```python
def rotated_root(
    product: np.ndarray,
    tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]:
    """Evaluate the rotated rational root and its directional derivative.

    Parameters
    ----------
    product : np.ndarray
        Complex square matrix M. It may be nonnormal.
    tangent : np.ndarray
        Complex directional derivative dM with the same shape as product.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int, float, float, float]
        G, dG, completed update count, converged residual,
        residual after exactly two updates, and the directional derivative
        of that two-update residual.

    Raises
    ------
    RuntimeError
        If the rational iteration does not converge within 50 updates.
    np.linalg.LinAlgError
        If a required correction solve is singular.
    """
    return (None, None, None, None, None, None)
```

### Step 4

Transformed modal matrix

Goal
----
Construct the source's transformed companion modal matrix for the magnetic-field state defined in the main problem, and its directional derivative.

```python
def transformed_modal(
    coupling: np.ndarray,
    coupling_tangent: np.ndarray,
    root: np.ndarray,
    root_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct the transformed modal matrix and its tangent.

    Parameters
    ----------
    coupling : np.ndarray
        Coupling matrix Q of shape (N, N).
    coupling_tangent : np.ndarray
        Directional derivative of Q with shape (N, N).
    root : np.ndarray
        Nonsingular root matrix G of shape (N, N).
    root_tangent : np.ndarray
        Directional derivative of G with shape (N, N).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The transformed companion modal matrix and its tangent, each complex128 (N, N).

    Raises
    ------
    np.linalg.LinAlgError
        If the root matrix is singular.
    """
    return None, None
```

### Step 5

Layer propagation

Goal
----
Calculate the source's forward layer-propagation matrix for the supplied root and fixed dimensionless depth, and its directional derivative induced by tangent.

```python
def layer_propagation(
    root: np.ndarray,
    tangent: np.ndarray,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate layer propagation and its Frechet derivative.

    Parameters
    ----------
    root : np.ndarray
        Root matrix G of shape (N, N).
    tangent : np.ndarray
        Directional derivative of G with shape (N, N).
    depth : float
        Fixed finite nonnegative layer depth.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The forward propagation matrix and its tangent, each complex128 (N, N).

    Raises
    ------
    ValueError
        If depth is not real, finite, and nonnegative.
    """
    return None, None
```

### Step 6

Gap-referenced layer scattering

Goal
----
Construct the source's gap-referenced amplitude-scattering matrix for one homogeneous layer and its directional derivative from the supplied transformed modal and propagation matrices.

```python
def layer_scattering(
    modal: np.ndarray,
    modal_tangent: np.ndarray,
    propagation: np.ndarray,
    propagation_tangent: np.ndarray,
    vacuum_root: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct layer scattering and its tangent.

    Parameters
    ----------
    modal : np.ndarray
        Modal matrix V of shape (N, N).
    modal_tangent : np.ndarray
        Directional derivative of V.
    propagation : np.ndarray
        Propagation matrix X of shape (N, N).
    propagation_tangent : np.ndarray
        Directional derivative of X.
    vacuum_root : np.ndarray
        Fixed vacuum-root vector of shape (N,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        S and dS, each complex128 with shape (2, 2, N, N).

    Raises
    ------
    np.linalg.LinAlgError
        If a required matrix solve is singular.
    """
    return None, None
```

### Step 7

Ordered Redheffer composition

Goal
----
Compose the left subsystem followed by the right subsystem using the physical scattering convention of the main problem, and obtain the derivative of the complete cascade.

```python
def redheffer_compose(
    left: np.ndarray,
    left_tangent: np.ndarray,
    right: np.ndarray,
    right_tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose two ordered scattering matrices and their tangents.

    Parameters
    ----------
    left : np.ndarray
        Left scattering matrix of shape (2, 2, N, N).
    left_tangent : np.ndarray
        Directional derivative of the left matrix.
    right : np.ndarray
        Right scattering matrix of shape (2, 2, N, N).
    right_tangent : np.ndarray
        Directional derivative of the right matrix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Left-to-right cascade and tangent, each with shape (2, 2, N, N).

    Raises
    ------
    np.linalg.LinAlgError
        If the internal feedback solve is singular.
    """
    return None, None
```

### Step 8

Final sensitivity orchestrator

Goal
----
Orchestrate the fixed two-layer calculation using the seven preceding public functions and the conventions in the main problem.



Only fill_first varies; all second-layer tangents are zero. Return one float64 array ordered as



[J, rho_2, D, T_plus1, R_total, T_total,

 updates_layer1, updates_layer2, d_flux_total].



Use layer 1’s two-update residual diagnostics for J and rho_2. The remaining entries come from the converged two-layer optical calculation.

```python
def solve_sensitivity(fill_first: float = 0.43) -> np.ndarray:
    """Compute the fixed-update residual and optical sensitivities.

    Parameters
    ----------
    fill_first
        Fill fraction of the first periodic layer.

    Returns
    -------
    np.ndarray
        Float64 array ordered as
        [J, rho_2, D, T_plus1, R_total, T_total,
        updates_layer1, updates_layer2, d_flux_total].
    """
    return None
```
