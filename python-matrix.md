# Python Matrix Operations and Linear Algebra

This document provides a comprehensive guide to all Python matrix-related functions, operations, and linear algebra capabilities with syntax and usage examples.

## Built-in Matrix Operations (Lists of Lists)

### Matrix Creation
```python
# 2D matrix using nested lists
matrix_2x3 = [[1, 2, 3], [4, 5, 6]]
matrix_3x3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Identity matrix
def create_identity(n):
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

identity_3x3 = create_identity(3)               # [[1,0,0], [0,1,0], [0,0,1]]

# Zero matrix
def create_zeros(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]

zeros_2x3 = create_zeros(2, 3)                  # [[0,0,0], [0,0,0]]

# Matrix from range
def create_range_matrix(rows, cols, start=1):
    return [[start + i*cols + j for j in range(cols)] for i in range(rows)]

range_matrix = create_range_matrix(3, 3)        # [[1,2,3], [4,5,6], [7,8,9]]

# Random matrix
import random
def create_random_matrix(rows, cols, min_val=0, max_val=10):
    return [[random.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

random_matrix = create_random_matrix(2, 3)
```

### Basic Matrix Operations
```python
# Matrix dimensions
def matrix_dimensions(matrix):
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    return rows, cols

rows, cols = matrix_dimensions(matrix_2x3)      # (2, 3)

# Matrix element access
element = matrix_2x3[0][1]                      # Access row 0, column 1
matrix_2x3[1][2] = 10                           # Set element

# Matrix printing
def print_matrix(matrix, title="Matrix"):
    print(f"{title}:")
    for row in matrix:
        print(" ".join(f"{elem:4}" for elem in row))
    print()

print_matrix(matrix_3x3, "3x3 Matrix")

# Matrix transpose
def transpose(matrix):
    rows, cols = matrix_dimensions(matrix)
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

transposed = transpose(matrix_2x3)              # [[1,4], [2,5], [3,6]]

# Using zip for transpose
def transpose_zip(matrix):
    return list(map(list, zip(*matrix)))

transposed_zip = transpose_zip(matrix_2x3)
```

### Matrix Arithmetic (Built-in)
```python
# Matrix addition
def matrix_add(A, B):
    rows, cols = matrix_dimensions(A)
    if matrix_dimensions(B) != (rows, cols):
        raise ValueError("Matrices must have same dimensions")
    return [[A[i][j] + B[i][j] for j in range(cols)] for i in range(rows)]

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
C = matrix_add(A, B)                            # [[6, 8], [10, 12]]

# Matrix subtraction
def matrix_subtract(A, B):
    rows, cols = matrix_dimensions(A)
    if matrix_dimensions(B) != (rows, cols):
        raise ValueError("Matrices must have same dimensions")
    return [[A[i][j] - B[i][j] for j in range(cols)] for i in range(rows)]

D = matrix_subtract(A, B)                       # [[-4, -4], [-4, -4]]

# Scalar multiplication
def scalar_multiply(matrix, scalar):
    rows, cols = matrix_dimensions(matrix)
    return [[matrix[i][j] * scalar for j in range(cols)] for i in range(rows)]

scaled = scalar_multiply(A, 3)                  # [[3, 6], [9, 12]]

# Matrix multiplication
def matrix_multiply(A, B):
    rows_A, cols_A = matrix_dimensions(A)
    rows_B, cols_B = matrix_dimensions(B)
    
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must equal number of rows in B")
    
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

# Example: 2x3 * 3x2 = 2x2
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]
product = matrix_multiply(A, B)                 # [[58, 64], [139, 154]]
```

### Matrix Utilities (Built-in)
```python
# Check if matrix is square
def is_square(matrix):
    rows, cols = matrix_dimensions(matrix)
    return rows == cols

# Check if matrix is symmetric
def is_symmetric(matrix):
    if not is_square(matrix):
        return False
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix[j][i]:
                return False
    return True

# Matrix trace (sum of diagonal elements)
def trace(matrix):
    if not is_square(matrix):
        raise ValueError("Matrix must be square")
    return sum(matrix[i][i] for i in range(len(matrix)))

# Determinant (2x2 matrix)
def determinant_2x2(matrix):
    if matrix_dimensions(matrix) != (2, 2):
        raise ValueError("Matrix must be 2x2")
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

# Determinant (3x3 matrix using cofactor expansion)
def determinant_3x3(matrix):
    if matrix_dimensions(matrix) != (3, 3):
        raise ValueError("Matrix must be 3x3")
    
    a, b, c = matrix[0]
    det = a * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
    det -= b * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    det += c * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    
    return det

# Matrix flattening
def flatten_matrix(matrix):
    return [element for row in matrix for element in row]

# Matrix from flat list
def matrix_from_flat(flat_list, rows, cols):
    if len(flat_list) != rows * cols:
        raise ValueError("List length must equal rows * cols")
    return [flat_list[i*cols:(i+1)*cols] for i in range(rows)]
```

## NumPy Matrices and Arrays

### NumPy Array Creation
```python
import numpy as np

# Basic array creation
arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Matrix creation functions
zeros = np.zeros((3, 4))                        # 3x4 matrix of zeros
ones = np.ones((2, 3))                          # 2x3 matrix of ones
full = np.full((2, 2), 7)                       # 2x2 matrix filled with 7
eye = np.eye(4)                                 # 4x4 identity matrix
identity = np.identity(3)                       # 3x3 identity matrix

# Diagonal matrices
diag_from_array = np.diag([1, 2, 3, 4])        # Diagonal matrix from array
diag_from_matrix = np.diag(arr_2d)              # Extract diagonal from matrix

# Range-based creation
arange_2d = np.arange(12).reshape(3, 4)         # 3x4 matrix: 0-11
linspace_2d = np.linspace(0, 1, 12).reshape(3, 4)  # 3x4 matrix: 0 to 1

# Random matrices
random_uniform = np.random.random((3, 3))       # Uniform distribution [0, 1)
random_normal = np.random.randn(3, 3)           # Standard normal distribution
random_int = np.random.randint(1, 10, (3, 3))   # Random integers

# Special matrices
tri_upper = np.triu(np.ones((4, 4)))            # Upper triangular
tri_lower = np.tril(np.ones((4, 4)))            # Lower triangular
```

### NumPy Array Properties
```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Shape and dimensions
print(matrix.shape)                             # (2, 3)
print(matrix.ndim)                              # 2
print(matrix.size)                              # 6
print(matrix.dtype)                             # int64

# Memory layout
print(matrix.flags)                             # Array flags
print(matrix.strides)                           # Memory strides
print(matrix.nbytes)                            # Total bytes consumed

# Matrix properties
is_fortran = matrix.flags['F_CONTIGUOUS']       # Fortran-contiguous
is_c = matrix.flags['C_CONTIGUOUS']             # C-contiguous

# Reshaping
reshaped = matrix.reshape(3, 2)                 # Reshape to 3x2
flattened = matrix.flatten()                    # Flatten to 1D (copy)
raveled = matrix.ravel()                        # Flatten to 1D (view if possible)

# Transpose
transposed = matrix.T                           # Transpose
transposed_method = matrix.transpose()          # Transpose method
transposed_axes = matrix.transpose(1, 0)        # Transpose with axis specification
```

### NumPy Array Indexing and Slicing
```python
import numpy as np

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# Basic indexing
element = matrix[1, 2]                          # Element at row 1, col 2 (7)
row = matrix[1]                                 # Second row [5, 6, 7, 8]
col = matrix[:, 2]                              # Third column [3, 7, 11]

# Slicing
submatrix = matrix[0:2, 1:3]                    # Rows 0-1, cols 1-2
every_other = matrix[::2, ::2]                  # Every other row and column

# Advanced indexing
# Boolean indexing
mask = matrix > 6
filtered = matrix[mask]                         # Elements > 6: [7, 8, 9, 10, 11, 12]

# Fancy indexing
rows = [0, 2]
cols = [1, 3]
selected = matrix[np.ix_(rows, cols)]           # Select specific rows and columns

# Negative indexing
last_row = matrix[-1]                           # Last row
last_element = matrix[-1, -1]                   # Last element
```

### NumPy Matrix Operations
```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise operations
addition = A + B                                # [[6, 8], [10, 12]]
subtraction = A - B                             # [[-4, -4], [-4, -4]]
multiplication = A * B                          # Element-wise: [[5, 12], [21, 32]]
division = A / B                                # Element-wise division
power = A ** 2                                  # Element-wise power

# Matrix operations
dot_product = np.dot(A, B)                      # Matrix multiplication
matmul = A @ B                                  # Matrix multiplication (Python 3.5+)
matrix_power = np.linalg.matrix_power(A, 3)     # A^3

# Scalar operations
scalar_add = A + 5                              # Add 5 to all elements
scalar_mult = A * 3                             # Multiply all elements by 3

# Comparison operations
greater = A > 2                                 # Boolean matrix
equal = A == B                                  # Element-wise equality
```

### NumPy Linear Algebra
```python
import numpy as np
import numpy.linalg as la

A = np.array([[1, 2], [3, 4]])
B = np.array([[2, 0], [1, 3]])

# Matrix decompositions
# LU decomposition (requires scipy)
# from scipy.linalg import lu
# P, L, U = lu(A)

# QR decomposition
Q, R = la.qr(A)

# Singular Value Decomposition (SVD)
U, s, Vt = la.svd(A)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = la.eig(A)

# Cholesky decomposition (for positive definite matrices)
symmetric_pos_def = np.array([[4, 2], [2, 3]])
chol = la.cholesky(symmetric_pos_def)

# Matrix properties
det = la.det(A)                                 # Determinant
rank = la.matrix_rank(A)                        # Matrix rank
trace = np.trace(A)                             # Trace
norm = la.norm(A)                               # Frobenius norm
norm_2 = la.norm(A, 2)                          # 2-norm
cond = la.cond(A)                               # Condition number

# Matrix inverse
inv = la.inv(A)                                 # Matrix inverse
pinv = la.pinv(A)                               # Pseudo-inverse

# Solving linear systems
# Solve Ax = b
b = np.array([1, 2])
x = la.solve(A, b)                              # Solve for x

# Least squares solution
# For overdetermined systems
A_over = np.array([[1, 1], [1, 2], [1, 3]])
b_over = np.array([6, 8, 10])
x_lstsq = la.lstsq(A_over, b_over, rcond=None)[0]
```

### NumPy Matrix Functions
```python
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Aggregation functions
sum_all = np.sum(matrix)                        # Sum of all elements
sum_rows = np.sum(matrix, axis=1)               # Sum along rows
sum_cols = np.sum(matrix, axis=0)               # Sum along columns

mean_all = np.mean(matrix)                      # Mean of all elements
std_all = np.std(matrix)                        # Standard deviation
var_all = np.var(matrix)                        # Variance

min_all = np.min(matrix)                        # Minimum element
max_all = np.max(matrix)                        # Maximum element
argmin = np.argmin(matrix)                      # Index of minimum (flattened)
argmax = np.argmax(matrix)                      # Index of maximum (flattened)

# Matrix-specific functions
diag = np.diag(matrix)                          # Diagonal elements
triu = np.triu(matrix)                          # Upper triangular part
tril = np.tril(matrix)                          # Lower triangular part

# Flipping and rotating
fliplr = np.fliplr(matrix)                      # Flip left-right
flipud = np.flipud(matrix)                      # Flip up-down
rot90 = np.rot90(matrix)                        # Rotate 90 degrees

# Concatenation and stacking
hstack = np.hstack([matrix, matrix])            # Horizontal stack
vstack = np.vstack([matrix, matrix])            # Vertical stack
concatenate = np.concatenate([matrix, matrix], axis=0)  # General concatenation

# Splitting
hsplit = np.hsplit(matrix, 3)                   # Split horizontally
vsplit = np.vsplit(matrix, 3)                   # Split vertically
```

## SciPy Linear Algebra

### SciPy Advanced Linear Algebra
```python
import numpy as np
import scipy.linalg as la

A = np.array([[1, 2], [3, 4]], dtype=float)

# More decompositions
# Schur decomposition
T, Z = la.schur(A)

# Hessenberg decomposition
H, Q = la.hessenberg(A, calc_q=True)

# LU decomposition with pivoting
P, L, U = la.lu(A)

# LDL decomposition
L, D, perm = la.ldl(A + A.T)  # Make symmetric first

# Matrix functions
matrix_exp = la.expm(A)                         # Matrix exponential
matrix_log = la.logm(A)                         # Matrix logarithm
matrix_sqrt = la.sqrtm(A)                       # Matrix square root
matrix_sin = la.sinm(A)                         # Matrix sine
matrix_cos = la.cosm(A)                         # Matrix cosine

# Polar decomposition
U, P = la.polar(A)

# Procrustes analysis
A_target = np.array([[2, 1], [1, 2]], dtype=float)
R, s = la.orthogonal_procrustes(A, A_target)

# Sylvester equation: AX + XB = Q
B = np.array([[0, 1], [1, 0]], dtype=float)
Q = np.array([[1, 0], [0, 1]], dtype=float)
X = la.solve_sylvester(A, B, Q)

# Lyapunov equation: AX + XA^T + Q = 0
X_lyap = la.solve_lyapunov(A, -Q)

# Matrix sign
sign_A = la.signm(A)
```

### SciPy Sparse Matrices
```python
import numpy as np
import scipy.sparse as sp

# Create sparse matrices
# Compressed Sparse Row (CSR)
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 1, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
csr_matrix = sp.csr_matrix((data, (row, col)), shape=(3, 3))

# Compressed Sparse Column (CSC)
csc_matrix = sp.csc_matrix((data, (row, col)), shape=(3, 3))

# Coordinate format (COO)
coo_matrix = sp.coo_matrix((data, (row, col)), shape=(3, 3))

# Dictionary of Keys (DOK)
dok_matrix = sp.dok_matrix((3, 3))
dok_matrix[0, 0] = 1
dok_matrix[0, 2] = 2
dok_matrix[1, 1] = 3

# List of Lists (LIL)
lil_matrix = sp.lil_matrix((3, 3))
lil_matrix[0, :2] = [1, 2]
lil_matrix[1, 1] = 3

# Convert between formats
csr_from_dok = dok_matrix.tocsr()
dense_from_sparse = csr_matrix.toarray()
sparse_from_dense = sp.csr_matrix(np.eye(3))

# Sparse matrix operations
A_sparse = sp.csr_matrix([[1, 0, 2], [0, 0, 3], [4, 5, 6]])
B_sparse = sp.csr_matrix([[1, 2, 0], [0, 1, 1], [1, 0, 1]])

# Arithmetic operations
sum_sparse = A_sparse + B_sparse
product_sparse = A_sparse.dot(B_sparse)
scalar_mult = A_sparse * 2

# Sparse linear algebra
from scipy.sparse.linalg import spsolve, norm, eigs

# Solve sparse linear system
b = np.array([1, 2, 3])
x = spsolve(A_sparse, b)

# Eigenvalues of sparse matrix
eigenvals, eigenvecs = eigs(A_sparse, k=2)  # Find 2 eigenvalues

# Sparse matrix norms
norm_sparse = norm(A_sparse)
```

## Specialized Matrix Libraries

### SymPy for Symbolic Matrices
```python
import sympy as sp

# Symbolic variables
x, y, z = sp.symbols('x y z')

# Symbolic matrix
M = sp.Matrix([[1, 2], [3, x]])
N = sp.Matrix([[x, 0], [y, z]])

# Symbolic operations
sum_sym = M + N
product_sym = M * N
det_sym = M.det()                               # Determinant: x - 6
inv_sym = M.inv()                               # Symbolic inverse

# Eigenvalues and eigenvectors (symbolic)
eigenvals = M.eigenvals()
eigenvects = M.eigenvects()

# Matrix calculus
diff_M = M.diff(x)                              # Derivative with respect to x

# Substitution
M_substituted = M.subs(x, 5)                    # Substitute x = 5

# Solving matrix equations
A = sp.Matrix([[1, 2], [3, 4]])
b = sp.Matrix([x, y])
solution = sp.solve(A * sp.Matrix([x, y]) - sp.Matrix([1, 2]), [x, y])

# Special matrices
identity_sym = sp.eye(3)                        # 3x3 identity
zeros_sym = sp.zeros(2, 3)                      # 2x3 zero matrix
ones_sym = sp.ones(2, 2)                        # 2x2 ones matrix
```

### PyTorch Tensors (for Machine Learning)
```python
import torch

# Tensor creation
tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
zeros_tensor = torch.zeros(3, 4)
ones_tensor = torch.ones(2, 3)
random_tensor = torch.randn(3, 3)               # Random normal
eye_tensor = torch.eye(4)

# Tensor operations
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# Element-wise operations
addition = A + B
multiplication = A * B
division = A / B

# Matrix operations
matmul = torch.matmul(A, B)                     # Matrix multiplication
matmul_op = A @ B                               # Alternative syntax

# Linear algebra
det = torch.det(A)                              # Determinant
inv = torch.inverse(A)                          # Inverse
eigenvals, eigenvecs = torch.eig(A, eigenvectors=True)

# SVD
U, S, V = torch.svd(A)

# QR decomposition
Q, R = torch.qr(A)

# Solve linear system
b = torch.tensor([1.0, 2.0])
x = torch.solve(b.unsqueeze(1), A)[0]

# GPU acceleration (if CUDA available)
if torch.cuda.is_available():
    A_gpu = A.cuda()
    B_gpu = B.cuda()
    result_gpu = torch.matmul(A_gpu, B_gpu)
    result_cpu = result_gpu.cpu()
```

## Matrix Applications and Algorithms

### Image Processing with Matrices
```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple image (matrix)
def create_test_image(size=10):
    return np.random.randint(0, 256, (size, size), dtype=np.uint8)

image = create_test_image(8)

# Image transformations
def rotate_90(image):
    return np.rot90(image)

def flip_horizontal(image):
    return np.fliplr(image)

def flip_vertical(image):
    return np.flipud(image)

# Convolution (simplified)
def apply_kernel(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    result = np.zeros((h - kh + 1, w - kw + 1))
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    
    return result

# Edge detection kernel
edge_kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])

edges = apply_kernel(image.astype(float), edge_kernel)

# Blur kernel
blur_kernel = np.ones((3, 3)) / 9
blurred = apply_kernel(image.astype(float), blur_kernel)
```

### Graph Theory with Adjacency Matrices
```python
import numpy as np

# Adjacency matrix for a graph
# Graph: 0--1--2
#        |  |
#        3--4
adjacency = np.array([[0, 1, 0, 1, 0],
                      [1, 0, 1, 0, 1],
                      [0, 1, 0, 0, 0],
                      [1, 0, 0, 0, 1],
                      [0, 1, 0, 1, 0]])

# Graph properties
num_vertices = adjacency.shape[0]
num_edges = np.sum(adjacency) // 2              # Undirected graph

# Degree of each vertex
degrees = np.sum(adjacency, axis=1)

# Path counting (powers of adjacency matrix)
paths_length_2 = np.linalg.matrix_power(adjacency, 2)
paths_length_3 = np.linalg.matrix_power(adjacency, 3)

# Distance matrix (Floyd-Warshall algorithm)
def floyd_warshall(adj_matrix):
    n = adj_matrix.shape[0]
    dist = adj_matrix.copy().astype(float)
    
    # Initialize distances
    dist[dist == 0] = np.inf
    np.fill_diagonal(dist, 0)
    
    # Floyd-Warshall
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    
    return dist

distances = floyd_warshall(adjacency)

# Laplacian matrix
degree_matrix = np.diag(degrees)
laplacian = degree_matrix - adjacency

# Number of spanning trees (Matrix-Tree theorem)
laplacian_minor = laplacian[:-1, :-1]
num_spanning_trees = round(np.linalg.det(laplacian_minor))
```

### Markov Chains
```python
import numpy as np

# Transition matrix for a Markov chain
# States: Sunny, Cloudy, Rainy
transition_matrix = np.array([[0.7, 0.2, 0.1],    # From Sunny
                              [0.3, 0.5, 0.2],    # From Cloudy
                              [0.2, 0.3, 0.5]])   # From Rainy

# Initial state distribution
initial_state = np.array([1, 0, 0])             # Start sunny

# State after n steps
def markov_step(initial, transition, steps):
    state = initial.copy()
    for _ in range(steps):
        state = state @ transition
    return state

state_after_5 = markov_step(initial_state, transition_matrix, 5)

# Steady state (eigenvector of eigenvalue 1)
eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
steady_state_idx = np.argmin(np.abs(eigenvals - 1))
steady_state = np.real(eigenvecs[:, steady_state_idx])
steady_state = steady_state / np.sum(steady_state)

# Fundamental matrix (for absorbing chains)
def fundamental_matrix(Q):
    """Q is the submatrix of transient states"""
    I = np.eye(Q.shape[0])
    return np.linalg.inv(I - Q)
```

### Principal Component Analysis (PCA)
```python
import numpy as np

def pca(X, n_components=None):
    """
    Principal Component Analysis
    X: data matrix (n_samples, n_features)
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Covariance matrix
    cov_matrix = np.cov(X_centered.T)
    
    # Eigenvalue decomposition
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Select components
    if n_components is not None:
        eigenvecs = eigenvecs[:, :n_components]
        eigenvals = eigenvals[:n_components]
    
    # Transform data
    X_transformed = X_centered @ eigenvecs
    
    return X_transformed, eigenvecs, eigenvals

# Example usage
# Generate sample data
np.random.seed(42)
data = np.random.randn(100, 5)
data[:, 1] = data[:, 0] + np.random.randn(100) * 0.1  # Correlated feature

# Apply PCA
transformed, components, explained_variance = pca(data, n_components=3)

# Explained variance ratio
explained_variance_ratio = explained_variance / np.sum(explained_variance)
```

## Performance Optimization

### Vectorization vs Loops
```python
import numpy as np
import time

# Matrix multiplication comparison
def matrix_mult_loops(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    C = np.zeros((rows_A, cols_B))
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i, j] += A[i, k] * B[k, j]
    return C

def benchmark_matrix_operations():
    # Create test matrices
    size = 100
    A = np.random.randn(size, size)
    B = np.random.randn(size, size)
    
    # Time loop-based multiplication
    start = time.time()
    C_loops = matrix_mult_loops(A, B)
    time_loops = time.time() - start
    
    # Time NumPy multiplication
    start = time.time()
    C_numpy = A @ B
    time_numpy = time.time() - start
    
    print(f"Loop-based: {time_loops:.4f}s")
    print(f"NumPy: {time_numpy:.4f}s")
    print(f"Speedup: {time_loops/time_numpy:.1f}x")

# benchmark_matrix_operations()
```

### Memory-Efficient Operations
```python
import numpy as np

# In-place operations
def efficient_matrix_ops():
    A = np.random.randn(1000, 1000)
    B = np.random.randn(1000, 1000)
    
    # Memory-efficient: in-place operations
    A += B          # Instead of A = A + B
    A *= 2          # Instead of A = A * 2
    
    # Use views instead of copies when possible
    submatrix = A[100:200, 100:200]  # View, not copy
    
    # Pre-allocate arrays
    result = np.empty_like(A)
    np.add(A, B, out=result)  # Store result in pre-allocated array
    
    return result

# Block matrix operations for large matrices
def block_matrix_multiply(A, B, block_size=64):
    """Block matrix multiplication for better cache efficiency"""
    n, k = A.shape
    k, m = B.shape
    C = np.zeros((n, m))
    
    for i in range(0, n, block_size):
        for j in range(0, m, block_size):
            for l in range(0, k, block_size):
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, m)
                l_end = min(l + block_size, k)
                
                C[i:i_end, j:j_end] += A[i:i_end, l:l_end] @ B[l:l_end, j:j_end]
    
    return C
```

### Parallel Matrix Operations
```python
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def parallel_matrix_operation(matrices, operation):
    """Apply operation to multiple matrices in parallel"""
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(operation, matrices))
    return results

# Example: compute eigenvalues for multiple matrices
def compute_eigenvalues(matrix):
    return np.linalg.eigvals(matrix)

# Create test matrices
matrices = [np.random.randn(50, 50) for _ in range(10)]

# Sequential processing
start = time.time()
results_seq = [compute_eigenvalues(m) for m in matrices]
time_seq = time.time() - start

# Parallel processing
start = time.time()
results_par = parallel_matrix_operation(matrices, compute_eigenvalues)
time_par = time.time() - start

print(f"Sequential: {time_seq:.4f}s")
print(f"Parallel: {time_par:.4f}s")
print(f"Speedup: {time_seq/time_par:.1f}x")
```

## Matrix Best Practices

### Numerical Stability
```python
import numpy as np

# Condition number checking
def check_condition_number(matrix, threshold=1e12):
    """Check if matrix is well-conditioned"""
    cond_num = np.linalg.cond(matrix)
    if cond_num > threshold:
        print(f"Warning: Matrix is ill-conditioned (cond = {cond_num:.2e})")
    return cond_num

# Regularization for ill-conditioned matrices
def regularized_inverse(matrix, reg_param=1e-6):
    """Compute regularized inverse"""
    n = matrix.shape[0]
    regularized = matrix + reg_param * np.eye(n)
    return np.linalg.inv(regularized)

# Stable rank computation
def stable_rank(matrix):
    """Compute stable rank using SVD"""
    _, s, _ = np.linalg.svd(matrix)
    return np.sum(s) / np.max(s)

# Gram-Schmidt orthogonalization
def gram_schmidt(vectors):
    """Gram-Schmidt orthogonalization process"""
    orthogonal = []
    for v in vectors:
        # Subtract projections onto previous orthogonal vectors
        for u in orthogonal:
            v = v - np.dot(v, u) / np.dot(u, u) * u
        # Normalize
        v = v / np.linalg.norm(v)
        orthogonal.append(v)
    return np.array(orthogonal)
```

### Error Handling and Validation
```python
import numpy as np

def validate_matrix_operation(A, B, operation='multiply'):
    """Validate matrices for common operations"""
    if operation == 'multiply':
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Cannot multiply {A.shape} and {B.shape} matrices")
    elif operation == 'add':
        if A.shape != B.shape:
            raise ValueError(f"Cannot add matrices of shapes {A.shape} and {B.shape}")
    elif operation == 'solve':
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square for solving Ax=b")
        if A.shape[0] != B.shape[0]:
            raise ValueError("Incompatible dimensions for solving Ax=b")

def safe_matrix_inverse(matrix):
    """Safely compute matrix inverse with error checking"""
    try:
        # Check if matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square")
        
        # Check condition number
        cond_num = np.linalg.cond(matrix)
        if cond_num > 1e12:
            print(f"Warning: Matrix is ill-conditioned (cond = {cond_num:.2e})")
        
        # Compute inverse
        inv = np.linalg.inv(matrix)
        
        # Verify inverse
        identity_check = matrix @ inv
        if not np.allclose(identity_check, np.eye(matrix.shape[0]), atol=1e-10):
            print("Warning: Inverse verification failed")
        
        return inv
    
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error: {e}")
        return None
```

---

*This document covers comprehensive matrix operations in Python including built-in list operations, NumPy arrays, SciPy linear algebra, specialized libraries, applications, and performance optimization techniques. For the most up-to-date information, refer to the official documentation of the respective libraries.*