from turtle import end_fill
import numpy as np


def dot_product(a, b):
    """Implement dot product between the two vectors: a and b.

    (optional): While you can solve this using for loops, we recommend
    that you look up `np.dot()` online and use that instead.

    When inputs are 2-D array, `np.matmul()` and `np.dot()` have same result, 
    you can also use `np.matmul()`.

    notice that `np.dot()` and `np.matmul()` need `a` with shape (x, n), `b` with shape `(n, x)
    so you need to transpose `a`, you can use syntax `a.T`.


    Args:
        a: numpy array of shape (n, x)
        b: numpy array of shape (n, x)

    Returns:
        out: numpy array of shape (x, x) (scalar if x = 1)
    """
    out = None
    ### YOUR CODE HERE
    temp = a.T
    out = np.dot(temp, b)
    ### END YOUR CODE
    return out


def complicated_matrix_function(M, a, b):
    """Implement (a^Tb) x (Ma), `a^T` is transpose of `a`, 
    (a^Tb) is matrix multiplication of a^T and b,
    (Ma) is matrix multiplication of M and a.

    You can use `np.matmul()` to do matrix multiplication.

    Args:
        M: numpy matrix of shape (x, n).
        a: numpy array of shape (n, 1).
        b: numpy array of shape (n, 1).

    Returns:
        out: numpy matrix of shape (x, 1).
    """
    out = None
    ### YOUR CODE HERE
    a_Tb = np.matmul(a.T, b)
    Ma = np.matmul(M, a)
    out = a_Tb * Ma
    ### END YOUR CODE

    return out


def eigen_decomp(M):
    """Implement eigenvalue decomposition.

    (optional): You might find the `np.linalg.eig` function useful.

    Args:
        matrix: numpy matrix of shape (m, m)

    Returns:
        w: numpy array of shape (m,) such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        v: Matrix where every column is an eigenvector.
    """
    w = None
    v = None
    ### YOUR CODE HERE
    w, v = np.linalg.eig(M)
    ### END YOUR CODE
    return w, v


def euclidean_distance_native(u, v):
    """Computes the Euclidean distance between two vectors, represented as Python
    lists.

    Args:
        u (List[float]): A vector, represented as a list of floats.
        v (List[float]): A vector, represented as a list of floats.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    # First, run some checks:
    assert isinstance(u, list)
    assert isinstance(v, list)
    assert len(u) == len(v)

    # Compute the distance!
    # Notes:
    #  1) Try breaking this problem down: first, we want to get
    #     the difference between corresponding elements in our
    #     input arrays. Then, we want to square these differences.
    #     Finally, we want to sum the squares and square root the
    #     sum.

    ### YOUR CODE HERE
    dif = []
    for x in range(0, len(u)):
        dif.append(u[x] - v[x])
    sum = dif[0] * dif[0]
    for x in range(1, len(dif)):
        sum = sum + dif[x] * dif[x]
    out = np.sqrt(sum)
    ### END YOUR CODE
    return out


def euclidean_distance_numpy(u, v):
    """Computes the Euclidean distance between two vectors, represented as NumPy
    arrays.

    Args:
        u (np.ndarray): A vector, represented as a NumPy array.
        v (np.ndarray): A vector, represented as a NumPy array.

    Returns:
        float: Euclidean distance between `u` and `v`.
    """
    # First, run some checks:
    assert isinstance(u, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert u.shape == v.shape

    # Compute the distance!
    # Note:
    #  1) You shouldn't need any loops
    #  2) Some functions you can Google that might be useful:
    #         np.sqrt(), np.sum()
    #  3) Try breaking this problem down: first, we want to get
    #     the difference between corresponding elements in our
    #     input arrays. Then, we want to square these differences.
    #     Finally, we want to sum the squares and square root the
    #     sum.

    ### YOUR CODE HERE
    dif = u - v
    dif_aq = dot_product(dif, dif)
    out = np.sqrt(dif_aq)
    ### END YOUR CODE
    return out


def get_eigen_values_and_vectors(M, k):
    """Return top k eigenvalues and eigenvectors of matrix M. By top k
    here we mean the eigenvalues with the top ABSOLUTE values (lookup
    np.argsort for a hint on how to do so.)

    (optional): Use the `eigen_decomp(M)` function you wrote above
    as a helper function

    Args:
        M: numpy matrix of shape (m, m).
        k: number of eigen values and respective vectors to return.

    Returns:
        eigenvalues: list of length k containing the top k eigenvalues
        eigenvectors: list of length k containing the top k eigenvectors
            of shape (m,)
    """
    eigenvalues = []
    eigenvectors = []
    ### YOUR CODE HERE
    w, v = eigen_decomp(M)
    arg_w = np.argsort(-w)
    i = 0
    for x in range(0, k):
        eigenvalues.append(w[arg_w[i]])
        eigenvectors.append(v[:, arg_w[i]])
        i = i + 1
    ### END YOUR CODE
    return eigenvalues, eigenvectors
