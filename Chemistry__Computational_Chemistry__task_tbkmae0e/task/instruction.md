# Pruning a quadrature grid for least-squares tensor hypercontraction

## Background

Essentially every correlated electronic structure method is built on the two-electron repulsion
integrals over atomic orbitals. Because there are formally as many of these integrals as the
fourth power of the number of basis functions, both storing them and contracting them with
wavefunction amplitudes become limiting long before the underlying physics does. Density fitting
and Cholesky decomposition of the integral tensor replace it by products of three-index tensors,
which lowers the memory demand but leaves pairs of orbital indices coupled. Factorizations that
separate all four orbital indices go further: once every index is carried by its own matrix,
contractions can be reordered so that whole classes of methods drop a power of the system size.

Separable factorizations of this kind are naturally expressed on a real-space grid, where sums
over grid points stand in for integrals over space. The size of the representation is then set
by the number of grid points rather than by the number of orbital pairs, which moves the accuracy
problem onto the grid itself.

Quadrature grids in quantum chemistry are usually assembled from atom-centered radial and angular
rules, and they are designed for integrating the exchange-correlation energy density rather than
for representing products of orbitals. Such grids tend to be far larger than a separable
factorization needs, and because the cost of every subsequent contraction grows with the number
of points, carrying redundant points is expensive. The competing demands are easy to state and
hard to satisfy at once: the grid must be small, it must be accurate for the products of basis
functions actually present, and it should be obtainable automatically rather than by hand-tuning
for each chemical element and basis set.

Automatic compression of an oversized input grid has been approached both by selecting a subset
of its points and by re-deriving its weights. Comparing such procedures means comparing tensors
rather than single energies, and a fair comparison applies them to the same system, basis set and
input grid.

## Problem

Least-squares tensor hypercontraction represents the four-index electron repulsion integral tensor through quantities tabulated on a real-space grid. Two published procedures compress an oversized input grid for this factorization: an earlier one that selects a subset of the input points, and a more recent one that re-derives the grid weights. Apply both to the model system below.

The system has four centers at (0.0, 0.0, 0.0), (1.9, 0.0, 0.2), (-0.2, 2.0, 0.1) and (0.9, 1.1, 1.75) bohr, each carrying normalized s-type Gaussian primitives with exponents 0.18, 0.45, 1.125, 2.8125 and 7.03125 bohr^-2. The input grid holds every center together with the six points displaced from it by ±r along each Cartesian axis for r = 0.4, 0.9, 1.7 and 3.0 bohr, and all of its points carry equal weights. Take every integral over the basis functions from its exact analytic value.

Run the point-selecting procedure at a relative threshold of 3.8 × 10^-4, in the convention of the work that introduced it, and solve every least-squares problem exactly, with no convergence tolerance and no truncation. On each compressed grid, build the factorization of the repulsion integrals and measure its error as the root-mean-square deviation of the reconstructed integrals from the exact ones over all entries of the four-index tensor.

Report the error on the grid from the point-selecting procedure divided by the error on the grid from the reweighting procedure, as a single number to at least eight significant figures.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

ao_overlap_matrix

Goal
----
Implement ao_overlap_matrix, which returns the exact overlap matrix of a basis of normalized s-type primitive Gaussians.

```python
def ao_overlap_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray") -> "np.ndarray":
    '''Exact analytic overlap matrix of normalized s-type primitive Gaussians.

    Parameters
    ----------
    ao_centers : np.ndarray
        Array of shape (n_ao, 3) holding the center of each primitive, in bohr.
        Must contain at least one row.
    ao_exponents : np.ndarray
        Array of shape (n_ao,) holding the Gaussian exponent of each primitive.
        Every exponent must be strictly positive.

    Returns
    -------
    overlap : np.ndarray
        Symmetric array of shape (n_ao, n_ao) of dtype float whose entry (mu, nu) is the
        exact overlap integral between primitives mu and nu. The diagonal is exactly one.

    Raises
    ------
    ValueError
        If ao_centers is not a two-dimensional array with three columns and at least one
        row, if ao_exponents is not a one-dimensional array of matching length, or if any
        exponent is not strictly positive.
    '''
    return overlap
```

### Step 2

ao_eri_tensor

Goal
----
Implement ao_eri_tensor, which returns the exact four-index electron repulsion integral tensor over a basis of normalized s-type primitive Gaussians.

```python
def ao_eri_tensor(ao_centers: "np.ndarray", ao_exponents: "np.ndarray") -> "np.ndarray":
    '''Exact analytic electron repulsion integrals over normalized s-type primitives.

    Parameters
    ----------
    ao_centers : np.ndarray
        Array of shape (n_ao, 3) holding the center of each primitive, in bohr.
        Must contain at least one row.
    ao_exponents : np.ndarray
        Array of shape (n_ao,) holding the Gaussian exponent of each primitive.
        Every exponent must be strictly positive.

    Returns
    -------
    eri : np.ndarray
        Array of shape (n_ao, n_ao, n_ao, n_ao) of dtype float whose entry
        (mu, nu, lambda, sigma) is the exact integral (mu nu | lambda sigma) in chemists'
        notation, so that the first two indices label the charge distribution of electron
        one and the last two that of electron two.

    Raises
    ------
    ValueError
        If ao_centers is not a two-dimensional array with three columns and at least one
        row, if ao_exponents is not a one-dimensional array of matching length, or if any
        exponent is not strictly positive.
    '''
    return eri
```

### Step 3

atom_centered_grid

Goal
----
Implement atom_centered_grid, which lays down the input real-space quadrature grid as a union of atom-centered shells built from six octahedral directions at a set of radii.

```python
def atom_centered_grid(centers: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    '''Build the atom-centered octahedral input quadrature grid.

    For each center in order, the grid holds the nuclear position followed by, for each
    radius in order, the six points displaced from that center by the radius along the
    directions +x, -x, +y, -y, +z and -z, in that order. A center with n_radii radii
    therefore contributes 1 + 6 * n_radii points, and the point at position
    1 + 6 * i + d within a center's block lies at radius index i along direction index d.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n_centers, 3) holding the center positions in bohr.
        Must contain at least one row.
    radii : np.ndarray
        Array of shape (n_radii,) holding the shell radii in bohr. Must contain at least
        one entry and every radius must be strictly positive.

    Returns
    -------
    grid_points : np.ndarray
        Array of shape (n_centers * (1 + 6 * n_radii), 3) of dtype float holding the grid
        point coordinates in bohr, ordered center block by center block.

    Raises
    ------
    ValueError
        If centers is not a two-dimensional array with three columns and at least one row,
        if radii is not a one-dimensional array with at least one entry, or if any radius
        is not strictly positive.
    '''
    return grid_points
```

### Step 4

overlap_fit_matrix

Goal
----
Implement overlap_fit_matrix, which builds the linear map from grid quadrature weights to the quadrature estimate of the atomic-orbital overlap matrix, with the orbital-pair index flattened into a single row index.

```python
def overlap_fit_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray",
                       grid_points: "np.ndarray") -> "np.ndarray":
    '''Linear map from grid weights to the quadrature estimate of the overlap matrix.

    The basis functions are normalized s-type primitive Gaussians, one per supplied
    (center, exponent) pair.

    Parameters
    ----------
    ao_centers : np.ndarray
        Array of shape (n_ao, 3) holding the center of each primitive, in bohr.
        Must contain at least one row.
    ao_exponents : np.ndarray
        Array of shape (n_ao,) holding the Gaussian exponent of each primitive.
        Every exponent must be strictly positive.
    grid_points : np.ndarray
        Array of shape (n_grid, 3) holding the grid point coordinates in bohr.
        Must contain at least one row.

    Returns
    -------
    fit_matrix : np.ndarray
        Array of shape (n_ao * n_ao, n_grid) of dtype float. Multiplying it by a weight
        vector of shape (n_grid,) gives that grid's quadrature estimate of the overlap
        matrix, flattened row-major, so row mu * n_ao + nu carries the ordered pair
        (mu, nu) and column P carries grid point P.

    Raises
    ------
    ValueError
        If ao_centers is not a two-dimensional array with three columns and at least one
        row, if ao_exponents is not a one-dimensional array of matching length, if any
        exponent is not strictly positive, or if grid_points is not a two-dimensional
        array with three columns and at least one row.
    '''
    return fit_matrix
```

### Step 5

nnls_grid_weights

Goal
----
Implement nnls_grid_weights, which refits the quadrature weights of a grid as the non-negative least-squares solution that best reproduces the exact atomic-orbital overlap matrix.

```python
def nnls_grid_weights(fit_matrix: "np.ndarray", overlap: "np.ndarray") -> "np.ndarray":
    '''Non-negative least-squares quadrature weights reproducing the overlap matrix.

    Returns the weight vector that minimizes, over all vectors with no negative entry, the
    Euclidean norm of the residual between the fit matrix applied to it and the overlap
    matrix flattened row-major, matching the compound row index mu * n_ao + nu of the fit
    matrix. The fit is solved exactly, not stopped at a convergence threshold. The problems
    supplied by the tests have a unique minimizer.

    Parameters
    ----------
    fit_matrix : np.ndarray
        Array of shape (n_ao * n_ao, n_grid) whose entry (mu * n_ao + nu, P) is the product
        of the values of orbitals mu and nu at grid point P.
    overlap : np.ndarray
        Square array of shape (n_ao, n_ao) holding the exact overlap matrix.

    Returns
    -------
    weights : np.ndarray
        Array of shape (n_grid,) of dtype float holding the fitted weights. Every entry is
        greater than or equal to zero, and the weights of points excluded by the fit are
        exactly zero.

    Raises
    ------
    ValueError
        If fit_matrix is not a two-dimensional array, if overlap is not a square
        two-dimensional array, or if the number of rows of fit_matrix differs from the
        number of entries of overlap.
    '''
    return weights
```

### Step 6

prune_zero_weight_points

Goal
----
Implement prune_zero_weight_points, which removes from a refitted quadrature grid every point whose non-negative weight is exactly zero, keeping the surviving points and weights in order.

```python
def prune_zero_weight_points(grid_points: "np.ndarray",
                             weights: "np.ndarray") -> "tuple[np.ndarray, np.ndarray]":
    '''Discard grid points whose non-negative quadrature weight is exactly zero.

    Parameters
    ----------
    grid_points : np.ndarray
        Array of shape (n_grid, 3) holding the grid point coordinates.
    weights : np.ndarray
        Array of shape (n_grid,) holding the non-negative weight of each point.

    Returns
    -------
    retained : tuple[np.ndarray, np.ndarray]
        A tuple (points, kept_weights) in which points has shape (n_kept, 3) and
        kept_weights has shape (n_kept,), both of dtype float, holding exactly the input
        points and weights whose weight is strictly greater than zero, in their original
        order. A weight that is positive but arbitrarily small is kept. If every weight is
        zero, both arrays have zero rows.

    Raises
    ------
    ValueError
        If grid_points is not a two-dimensional array with three columns, if weights is not
        a one-dimensional array with one entry per grid point, or if any weight is negative.
    '''
    return retained
```

### Step 7

thc_collocation_matrix

Goal
----
Implement thc_collocation_matrix, which forms the weighted collocation matrix that carries the atomic orbitals in grid-based least-squares tensor hypercontraction.

```python
def thc_collocation_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray",
                           grid_points: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    '''Weighted collocation matrix of least-squares tensor hypercontraction.

    The basis functions are normalized s-type primitive Gaussians, one per supplied
    (center, exponent) pair. Each entry is the value of one orbital at one grid point,
    scaled by that point's quadrature weight raised to the fixed fractional power this
    factorization uses.

    Parameters
    ----------
    ao_centers : np.ndarray
        Array of shape (n_ao, 3) holding the center of each primitive, in bohr.
        Must contain at least one row.
    ao_exponents : np.ndarray
        Array of shape (n_ao,) holding the Gaussian exponent of each primitive.
        Every exponent must be strictly positive.
    grid_points : np.ndarray
        Array of shape (n_grid, 3) holding the grid point coordinates in bohr.
        Must contain at least one row.
    weights : np.ndarray
        Array of shape (n_grid,) holding the non-negative quadrature weight of each point.

    Returns
    -------
    collocation : np.ndarray
        Array of shape (n_ao, n_grid) of dtype float whose entry (mu, P) belongs to
        orbital mu at grid point P.

    Raises
    ------
    ValueError
        If ao_centers is not a two-dimensional array with three columns and at least one
        row, if ao_exponents is not a one-dimensional array of matching length, if any
        exponent is not strictly positive, if grid_points is not a two-dimensional array
        with three columns and at least one row, if weights is not a one-dimensional array
        with one entry per grid point, or if any weight is negative.
    '''
    return collocation
```

### Step 8

thc_grid_metric

Goal
----
Implement thc_grid_metric, which forms the grid metric matrix of least-squares tensor hypercontraction from a weighted collocation matrix.

```python
def thc_grid_metric(collocation: "np.ndarray") -> "np.ndarray":
    '''Grid metric matrix of the least-squares tensor hypercontraction fit.

    Parameters
    ----------
    collocation : np.ndarray
        Array of shape (n_ao, n_grid) holding the weighted collocation matrix X. Must have
        at least one row and at least one column.

    Returns
    -------
    metric : np.ndarray
        Symmetric positive semidefinite array of shape (n_grid, n_grid) of dtype float.

    Raises
    ------
    ValueError
        If collocation is not a two-dimensional array with at least one row and at least
        one column.
    '''
    return metric
```

### Step 9

pivoted_cholesky_points

Goal
----
Implement pivoted_cholesky_points, which prunes a grid by an incomplete pivoted Cholesky decomposition of its least-squares tensor hypercontraction metric and returns the indices of the selected points in the order they become pivots.

```python
def pivoted_cholesky_points(grid_metric: "np.ndarray", cutoff: float) -> "np.ndarray":
    '''Grid points selected by an incomplete pivoted Cholesky decomposition of the metric.

    The decomposition follows the standard greedy pivoted Cholesky rule, taking the
    not-yet-selected point with the largest remaining Schur-complement diagonal as the next
    pivot and breaking exact ties by the lowest index. It stops before the first pivot that
    falls below the relative cutoff, in the convention of the work that introduced this
    pruning, and also before any pivot whose remaining diagonal is not positive.

    Parameters
    ----------
    grid_metric : np.ndarray
        Symmetric positive semidefinite array of shape (n_grid, n_grid) with n_grid >= 1.
    cutoff : float
        Strictly positive relative pruning threshold, interpreted in the convention of the
        work that introduced this pruning.

    Returns
    -------
    pivots : np.ndarray
        One-dimensional integer array holding the indices of the selected grid points in
        the order in which they became pivots.

    Raises
    ------
    ValueError
        If grid_metric is not a square two-dimensional array with at least one row, or if
        cutoff is not finite and strictly positive.
    '''
    return pivots
```

### Step 10

thc_core_matrix

Goal
----
Implement thc_core_matrix, which solves for the central matrix of least-squares tensor hypercontraction on a given weighted grid.

```python
def thc_core_matrix(collocation: "np.ndarray", grid_metric: "np.ndarray",
                    eri_tensor: "np.ndarray", rcond: float) -> "np.ndarray":
    '''Central matrix of the least-squares tensor hypercontraction fit.

    Returns the minimum-norm matrix over pairs of grid points that minimizes the squared
    Frobenius error between the supplied integral tensor and its reconstruction from the
    supplied collocation matrix. Wherever the grid metric has to be inverted, it is
    inverted through the pseudoinverse obtained from its singular value decomposition after
    discarding every singular value smaller than rcond times the largest singular value.

    Parameters
    ----------
    collocation : np.ndarray
        Array of shape (n_ao, n_grid) holding the weighted collocation matrix X.
    grid_metric : np.ndarray
        Array of shape (n_grid, n_grid) holding the grid metric M formed from the same
        collocation matrix.
    eri_tensor : np.ndarray
        Array of shape (n_ao, n_ao, n_ao, n_ao) holding the exact repulsion integrals
        (mu nu | lambda sigma) in chemists' notation.
    rcond : float
        Non-negative relative cutoff for the singular values of the grid metric.

    Returns
    -------
    core : np.ndarray
        Array of shape (n_grid, n_grid) of dtype float holding the central matrix.

    Raises
    ------
    ValueError
        If collocation is not a two-dimensional array with at least one row and at least
        one column, if grid_metric does not have shape (n_grid, n_grid), if eri_tensor does
        not have shape (n_ao, n_ao, n_ao, n_ao), or if rcond is negative or not finite.
    '''
    return core
```

### Step 11

pruning_rmsd_ratio

Goal
----
Implement pruning_rmsd_ratio, the end-to-end comparison of two ways of pruning the same input grid for least-squares tensor hypercontraction: it returns the root-mean-square deviation of the integrals reconstructed on a pivoted-Cholesky-pruned grid divided by that on a grid refitted and pruned by non-negative least squares against the overlap matrix.

```python
def pruning_rmsd_ratio(centers: "np.ndarray", exponents: "np.ndarray", radii: "np.ndarray",
                       cutoff: float, rcond: float) -> float:
    '''Ratio of LS-THC integral RMSDs: pivoted-Cholesky-pruned grid over NNLS-refitted grid.

    Every center carries one normalized s-type primitive Gaussian per exponent, and the
    orbital index is mu = n_exponents * c + k for exponent k on center c. The input grid
    is the atom-centered octahedral grid built from the same centers and radii, and all of
    its points carry equal weights. One grid comes from the non-negative least-squares
    reweighting, solved exactly, and carries its refitted weights; the other comes from the
    pivoted-Cholesky pruning at the given relative cutoff and keeps the equal input
    weights. On each grid the least-squares tensor hypercontraction of the exact repulsion
    integrals is built, inverting the grid metric through the pseudoinverse that discards
    singular values below rcond times the largest, and the error of that factorization is
    the root-mean-square deviation of the reconstructed integrals from the exact ones over
    all entries of the four-index tensor.

    Parameters
    ----------
    centers : np.ndarray
        Array of shape (n_centers, 3) holding the center positions in bohr. Must contain
        at least one row.
    exponents : np.ndarray
        Array of shape (n_exponents,) holding the exponents shared by every center. Must
        contain at least one entry and every exponent must be strictly positive.
    radii : np.ndarray
        Array of shape (n_radii,) holding the shell radii of the input grid in bohr. Must
        contain at least one entry and every radius must be strictly positive.
    cutoff : float
        Relative pruning threshold of the pivoted-Cholesky selection, with
        0 < cutoff <= 1.
    rcond : float
        Non-negative relative singular-value cutoff for the metric pseudoinverse.

    Returns
    -------
    ratio : float
        Native Python float equal to the RMSD on the Cholesky grid divided by the RMSD on
        the NNLS grid.

    Raises
    ------
    ValueError
        If centers is not a two-dimensional array with three columns and at least one row,
        if exponents is not a one-dimensional array with at least one entry, if any
        exponent or radius is not strictly positive, if radii is not a one-dimensional array
        with at least one entry, if cutoff does not satisfy 0 < cutoff <= 1, or if rcond is
        negative or not finite.
    '''
    return ratio
```
