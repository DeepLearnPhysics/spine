"""Numba JIT compiled implementation of linear algebra routines."""

import numba as nb
import numpy as np

__all__ = ["norm", "submatrix", "contingency_table"]


@nb.njit(cache=True)
def norm(x: nb.float32[:, :], axis: nb.int32) -> nb.float32[:]:
    """Compute vector norms along specified axis.

    This is a Numba-compiled implementation of `np.linalg.norm(x, axis=axis)`
    optimized for 2D arrays with specified axis reduction.

    Parameters
    ----------
    x : ndarray of shape (n, m)
        Input array of floating-point values.
    axis : {0, 1}
        Axis along which to compute the norm:
        - 0: compute norm of each column (returns array of length m)
        - 1: compute norm of each row (returns array of length n)

    Returns
    -------
    norms : ndarray of shape (m,) or (n,)
        Array of norm values along the specified axis.

    Notes
    -----
    This function uses the Euclidean (L2) norm. The implementation is
    optimized with Numba JIT compilation for performance.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[3., 4.], [0., 5.]], dtype=np.float32)
    >>> norm(x, axis=0)  # Column norms
    array([3., 6.4031], dtype=float32)
    >>> norm(x, axis=1)  # Row norms
    array([5., 5.], dtype=float32)
    """
    assert axis == 0 or axis == 1
    xnorm = np.empty(x.shape[1 - axis], dtype=x.dtype)
    if axis == 0:
        for i in range(len(xnorm)):
            xnorm[i] = np.linalg.norm(x[:, i])
    else:
        for i in range(len(xnorm)):
            xnorm[i] = np.linalg.norm(x[i])

    return xnorm


@nb.njit(cache=True)
def submatrix(
    x: nb.float32[:, :], index1: nb.int32[:], index2: nb.int32[:]
) -> nb.float32[:, :]:
    """Extract submatrix using row and column indices.

    This function creates a submatrix by selecting specific rows and columns
    from the input matrix using the provided index arrays.

    Parameters
    ----------
    x : ndarray of shape (n, m)
        Input matrix from which to extract submatrix.
    index1 : ndarray of shape (k,)
        Row indices to select from the input matrix. Must be valid
        indices in the range [0, n).
    index2 : ndarray of shape (l,)
        Column indices to select from the input matrix. Must be valid
        indices in the range [0, m).

    Returns
    -------
    submat : ndarray of shape (k, l)
        Submatrix containing elements x[index1[i], index2[j]] for all
        combinations of i and j.

    Notes
    -----
    This function is optimized with Numba JIT compilation. The indices
    are not validated for bounds checking in the compiled version.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    >>> row_indices = np.array([0, 2], dtype=np.int32)
    >>> col_indices = np.array([1, 2], dtype=np.int32)
    >>> submatrix(x, row_indices, col_indices)
    array([[2., 3.],
           [8., 9.]], dtype=float32)
    """
    subx = np.empty((len(index1), len(index2)), dtype=x.dtype)
    for i, i1 in enumerate(index1):
        for j, i2 in enumerate(index2):
            subx[i, j] = x[i1, i2]

    return subx


@nb.njit(cache=True)
def contingency_table(
    x: nb.int32[:], y: nb.int32[:], nx: nb.int32 = None, ny: nb.int32 = None
) -> nb.int64[:, :]:
    """Build a contingency table for two sets of labels.

    A contingency table (also known as a cross-tabulation or crosstab) shows
    the frequency distribution of labels between two classification results.
    Entry (i, j) represents the number of samples with label i in the first
    array and label j in the second array.

    Parameters
    ----------
    x : ndarray of shape (n_samples,)
        First array of integer labels. Labels should be non-negative integers.
    y : ndarray of shape (n_samples,)
        Second array of integer labels. Labels should be non-negative integers.
        Must have the same length as x.
    nx : int, optional
        Maximum number of unique labels allowed in x. If not provided,
        defaults to max(x) + 1. Used to determine the number of rows
        in the output table.
    ny : int, optional
        Maximum number of unique labels allowed in y. If not provided,
        defaults to max(y) + 1. Used to determine the number of columns
        in the output table.

    Returns
    -------
    table : ndarray of shape (nx, ny)
        Contingency table where entry (i, j) contains the count of samples
        having label i in x and label j in y.

    Notes
    -----
    This function is optimized with Numba JIT compilation for high performance.
    Labels are expected to be in the range [0, nx) and [0, ny) respectively.
    No bounds checking is performed on the labels in the compiled version.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 0, 1, 1], dtype=np.int32)
    >>> y = np.array([0, 1, 1, 1], dtype=np.int32)
    >>> contingency_table(x, y)
    array([[1, 1],
           [0, 2]])

    >>> # With explicit dimensions
    >>> contingency_table(x, y, nx=3, ny=2)
    array([[1, 1],
           [0, 2],
           [0, 0]])
    """
    # If not provided, assume that the max label is the max of the label array
    if not nx:
        nx = np.max(x) + 1 if len(x) > 0 else 1
    if not ny:
        ny = np.max(y) + 1 if len(y) > 0 else 1

    # Bin the table
    table = np.zeros((nx, ny), dtype=np.int64)
    for i, j in zip(x, y):
        table[i, j] += 1

    return table
