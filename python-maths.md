# Python Mathematics and Scientific Computing

This document provides a comprehensive guide to mathematical functions, operations, and scientific computing in Python with syntax and usage examples.

## Built-in Mathematical Functions

### Basic Arithmetic Operations
```python
# Basic arithmetic operators
a, b = 10, 3

addition = a + b                                # 13
subtraction = a - b                             # 7
multiplication = a * b                          # 30
division = a / b                                # 3.3333...
floor_division = a // b                         # 3
modulus = a % b                                 # 1
exponentiation = a ** b                         # 1000

# Augmented assignment operators
x = 10
x += 5                                          # x = 15
x -= 3                                          # x = 12
x *= 2                                          # x = 24
x /= 4                                          # x = 6.0
x //= 2                                         # x = 3.0
x %= 2                                          # x = 1.0
x **= 3                                         # x = 1.0

# Comparison operators
print(10 > 5)                                   # True
print(10 >= 10)                                 # True
print(5 < 10)                                   # True
print(5 <= 5)                                   # True
print(10 == 10)                                 # True
print(10 != 5)                                  # True
```

### Built-in Mathematical Functions
```python
# abs() - Absolute value
print(abs(-5))                                  # 5
print(abs(-3.14))                               # 3.14
print(abs(3+4j))                                # 5.0 (magnitude of complex number)

# round() - Rounding
print(round(3.14159))                           # 3
print(round(3.14159, 2))                        # 3.14
print(round(3.14159, 4))                        # 3.1416
print(round(1234.5678, -2))                     # 1200.0

# min() and max() - Minimum and maximum
print(min(1, 5, 3, 9, 2))                       # 1
print(max(1, 5, 3, 9, 2))                       # 9
print(min([1, 5, 3, 9, 2]))                     # 1
print(max([1, 5, 3, 9, 2]))                     # 9

# With key function
words = ['apple', 'pie', 'cherry']
print(min(words, key=len))                      # 'pie'
print(max(words, key=len))                      # 'cherry'

# sum() - Sum of iterable
print(sum([1, 2, 3, 4, 5]))                     # 15
print(sum([1, 2, 3, 4, 5], 10))                 # 25 (start value)
print(sum(range(1, 101)))                       # 5050

# pow() - Power function
print(pow(2, 3))                                # 8
print(pow(2, 3, 5))                             # 3 (2^3 % 5)
print(pow(2, -1))                               # 0.5

# divmod() - Division and modulus
quotient, remainder = divmod(17, 5)
print(quotient, remainder)                      # 3, 2

# Complex number operations
z1 = 3 + 4j
z2 = 1 + 2j

print(z1 + z2)                                  # (4+6j)
print(z1 * z2)                                  # (-5+10j)
print(abs(z1))                                  # 5.0
print(z1.real)                                  # 3.0
print(z1.imag)                                  # 4.0
print(z1.conjugate())                           # (3-4j)
```

### Number Base Conversions
```python
# Binary, octal, and hexadecimal
number = 42

binary = bin(number)                            # '0b101010'
octal = oct(number)                             # '0o52'
hexadecimal = hex(number)                       # '0x2a'

print(f"Decimal: {number}")
print(f"Binary: {binary}")
print(f"Octal: {octal}")
print(f"Hexadecimal: {hexadecimal}")

# Convert back to decimal
print(int('101010', 2))                         # 42
print(int('52', 8))                             # 42
print(int('2a', 16))                            # 42

# Using format() for different bases
print(format(42, 'b'))                          # '101010'
print(format(42, 'o'))                          # '52'
print(format(42, 'x'))                          # '2a'
print(format(42, 'X'))                          # '2A'

# Character and ASCII conversions
print(ord('A'))                                 # 65
print(chr(65))                                  # 'A'
print(ord('€'))                                 # 8364
print(chr(8364))                                # '€'
```

## Math Module

### Basic Mathematical Functions
```python
import math

# Constants
print(math.pi)                                  # 3.141592653589793
print(math.e)                                   # 2.718281828459045
print(math.tau)                                 # 6.283185307179586 (2*pi)
print(math.inf)                                 # inf
print(math.nan)                                 # nan

# Power and logarithmic functions
print(math.sqrt(16))                            # 4.0
print(math.pow(2, 3))                           # 8.0
print(math.exp(2))                              # 7.38905609893065 (e^2)

# Logarithms
print(math.log(math.e))                         # 1.0 (natural log)
print(math.log(100, 10))                        # 2.0 (log base 10)
print(math.log10(100))                          # 2.0
print(math.log2(8))                             # 3.0

# Trigonometric functions (angles in radians)
angle_rad = math.pi / 4                         # 45 degrees
print(math.sin(angle_rad))                      # 0.7071067811865476
print(math.cos(angle_rad))                      # 0.7071067811865476
print(math.tan(angle_rad))                      # 0.9999999999999999

# Convert between degrees and radians
angle_deg = 45
angle_rad = math.radians(angle_deg)             # 0.7853981633974483
angle_deg_back = math.degrees(angle_rad)        # 45.0

# Inverse trigonometric functions
print(math.asin(0.5))                           # 0.5235987755982989
print(math.acos(0.5))                           # 1.0471975511965979
print(math.atan(1))                             # 0.7853981633974483
print(math.atan2(1, 1))                         # 0.7853981633974483

# Hyperbolic functions
print(math.sinh(1))                             # 1.1752011936438014
print(math.cosh(1))                             # 1.5430806348152437
print(math.tanh(1))                             # 0.7615941559557649

# Floor, ceiling, and truncation
print(math.floor(3.7))                          # 3
print(math.ceil(3.2))                           # 4
print(math.trunc(3.7))                          # 3
print(math.trunc(-3.7))                         # -3

# Absolute value and sign
print(math.fabs(-3.5))                          # 3.5
print(math.copysign(5, -1))                     # -5.0

# Modulo operation
print(math.fmod(7.5, 2.5))                      # 2.5

# Greatest common divisor and least common multiple
print(math.gcd(48, 18))                         # 6
print(math.lcm(4, 6))                           # 12 (Python 3.9+)

# Factorial and combinations
print(math.factorial(5))                        # 120
print(math.comb(10, 3))                         # 120 (10 choose 3, Python 3.8+)
print(math.perm(10, 3))                         # 720 (10 permute 3, Python 3.8+)
```

### Special Mathematical Functions
```python
import math

# Error function and gamma function
print(math.erf(1))                              # 0.8427007929497149
print(math.erfc(1))                             # 0.15729920705028513
print(math.gamma(5))                            # 24.0 (gamma(n) = (n-1)!)

# Check for special values
print(math.isfinite(100))                       # True
print(math.isfinite(math.inf))                  # False
print(math.isinf(math.inf))                     # True
print(math.isnan(math.nan))                     # True
print(math.isclose(0.1 + 0.2, 0.3))            # True

# Distance and norm functions
point1 = (0, 0)
point2 = (3, 4)
distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
print(distance)                                 # 5.0

# Using math.dist (Python 3.8+)
print(math.dist(point1, point2))                # 5.0

# Remainder functions
print(math.remainder(23, 7))                    # 2.0 (IEEE remainder)
print(23 % 7)                                   # 2 (Python modulo)

# Next float functions (Python 3.9+)
print(math.nextafter(1.0, 2.0))                 # 1.0000000000000002
print(math.ulp(1.0))                            # 2.220446049250313e-16
```

## Random Module

### Random Number Generation
```python
import random

# Set seed for reproducibility
random.seed(42)

# Basic random functions
print(random.random())                          # Random float between 0 and 1
print(random.uniform(1, 10))                    # Random float between 1 and 10
print(random.randint(1, 6))                     # Random integer between 1 and 6 (inclusive)
print(random.randrange(0, 10, 2))               # Random even number between 0 and 8

# Random choice from sequence
colors = ['red', 'blue', 'green', 'yellow']
print(random.choice(colors))                    # Random color

# Multiple random choices
print(random.choices(colors, k=3))               # 3 random colors (with replacement)
print(random.choices(colors, weights=[1, 2, 3, 4], k=3))  # Weighted choices

# Sampling without replacement
print(random.sample(colors, 2))                 # 2 random colors without replacement

# Shuffle a list in place
numbers = list(range(1, 11))
random.shuffle(numbers)
print(numbers)                                  # Shuffled list

# Random bytes
print(random.randbytes(8))                      # 8 random bytes

# Triangular distribution
print(random.triangular(0, 10, 5))              # Triangular dist with mode at 5

# Beta distribution
print(random.betavariate(2, 5))                 # Beta distribution

# Exponential distribution
print(random.expovariate(1.5))                  # Exponential distribution

# Gamma distribution
print(random.gammavariate(2, 3))                # Gamma distribution

# Gaussian (normal) distribution
print(random.gauss(0, 1))                       # Mean=0, std=1
print(random.normalvariate(0, 1))               # Same as gauss

# Log-normal distribution
print(random.lognormvariate(0, 1))              # Log-normal distribution

# Pareto distribution
print(random.paretovariate(1))                  # Pareto distribution

# Von Mises distribution
print(random.vonmisesvariate(0, 1))             # Von Mises distribution

# Weibull distribution
print(random.weibullvariate(1, 2))              # Weibull distribution
```

### Random Utilities and Custom Distributions
```python
import random
import math

class RandomUtilities:
    @staticmethod
    def random_walk_1d(steps, start=0):
        """Generate 1D random walk"""
        position = start
        path = [position]
        
        for _ in range(steps):
            step = random.choice([-1, 1])
            position += step
            path.append(position)
        
        return path
    
    @staticmethod
    def random_walk_2d(steps, start=(0, 0)):
        """Generate 2D random walk"""
        x, y = start
        path = [(x, y)]
        
        for _ in range(steps):
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            x += dx
            y += dy
            path.append((x, y))
        
        return path
    
    @staticmethod
    def weighted_random_choice(items, weights):
        """Weighted random choice"""
        total = sum(weights)
        r = random.random() * total
        
        cumulative = 0
        for item, weight in zip(items, weights):
            cumulative += weight
            if r <= cumulative:
                return item
        
        return items[-1]
    
    @staticmethod
    def generate_password(length=12, chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'):
        """Generate random password"""
        return ''.join(random.choice(chars) for _ in range(length))
    
    @staticmethod
    def monte_carlo_pi(n_samples=1000000):
        """Estimate π using Monte Carlo method"""
        inside_circle = 0
        
        for _ in range(n_samples):
            x = random.random()
            y = random.random()
            if x*x + y*y <= 1:
                inside_circle += 1
        
        return 4 * inside_circle / n_samples

# Example usage
# walk_1d = RandomUtilities.random_walk_1d(100)
# walk_2d = RandomUtilities.random_walk_2d(100)
# pi_estimate = RandomUtilities.monte_carlo_pi(1000000)
# password = RandomUtilities.generate_password(16)
```

## Statistics Module

### Descriptive Statistics
```python
import statistics as stats

# Sample data
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grades = [85, 90, 78, 92, 88, 76, 95, 89, 84, 91]

# Measures of central tendency
print(stats.mean(data))                         # 5.5
print(stats.median(data))                       # 5.5
print(stats.mode([1, 2, 2, 3, 4]))             # 2

# Multiple modes
print(stats.multimode([1, 1, 2, 2, 3]))        # [1, 2]

# Geometric and harmonic means
print(stats.geometric_mean(data))               # 4.528728688116765
print(stats.harmonic_mean(data))                # 3.414171521474055

# Measures of spread
print(stats.pstdev(data))                       # 2.8722813232690143 (population std dev)
print(stats.stdev(data))                        # 3.0276503540974917 (sample std dev)
print(stats.pvariance(data))                    # 8.25 (population variance)
print(stats.variance(data))                     # 9.166666666666666 (sample variance)

# Quantiles
print(stats.quantiles(data, n=4))               # [3.25, 5.5, 7.75] (quartiles)
print(stats.quantiles(data, n=10))              # Deciles

# Advanced statistics (Python 3.8+)
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Covariance
print(stats.covariance(x, y))                   # 5.0

# Correlation coefficient
print(stats.correlation(x, y))                  # 1.0 (perfect positive correlation)

# Linear regression
slope, intercept = stats.linear_regression(x, y)
print(f"y = {slope}x + {intercept}")            # y = 2.0x + 0.0
```

### Statistical Analysis Functions
```python
import statistics as stats
import math

class StatisticalAnalysis:
    @staticmethod
    def z_score(value, mean, std_dev):
        """Calculate z-score"""
        return (value - mean) / std_dev
    
    @staticmethod
    def percentile_rank(data, value):
        """Calculate percentile rank of a value"""
        sorted_data = sorted(data)
        n = len(sorted_data)
        rank = sum(1 for x in sorted_data if x < value)
        return (rank / n) * 100
    
    @staticmethod
    def interquartile_range(data):
        """Calculate IQR"""
        q1, q3 = stats.quantiles(data, n=4)[0], stats.quantiles(data, n=4)[2]
        return q3 - q1
    
    @staticmethod
    def outliers_iqr(data):
        """Detect outliers using IQR method"""
        q1, q3 = stats.quantiles(data, n=4)[0], stats.quantiles(data, n=4)[2]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        return outliers
    
    @staticmethod
    def five_number_summary(data):
        """Calculate five-number summary"""
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        minimum = min(sorted_data)
        maximum = max(sorted_data)
        median = stats.median(sorted_data)
        
        if n >= 4:
            q1, q3 = stats.quantiles(sorted_data, n=4)[0], stats.quantiles(sorted_data, n=4)[2]
        else:
            q1 = q3 = median
        
        return {
            'min': minimum,
            'q1': q1,
            'median': median,
            'q3': q3,
            'max': maximum
        }
    
    @staticmethod
    def coefficient_of_variation(data):
        """Calculate coefficient of variation"""
        mean = stats.mean(data)
        std_dev = stats.stdev(data)
        return (std_dev / mean) * 100
    
    @staticmethod
    def skewness(data):
        """Calculate skewness (Pearson's method)"""
        mean = stats.mean(data)
        std_dev = stats.stdev(data)
        n = len(data)
        
        sum_cubed_deviations = sum((x - mean)**3 for x in data)
        skewness = (n / ((n-1) * (n-2))) * (sum_cubed_deviations / (std_dev**3))
        
        return skewness
    
    @staticmethod
    def kurtosis(data):
        """Calculate kurtosis"""
        mean = stats.mean(data)
        std_dev = stats.stdev(data)
        n = len(data)
        
        sum_fourth_deviations = sum((x - mean)**4 for x in data)
        kurt = (n * (n+1) / ((n-1) * (n-2) * (n-3))) * (sum_fourth_deviations / (std_dev**4)) - (3 * (n-1)**2 / ((n-2) * (n-3)))
        
        return kurt

# Example usage
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]  # Data with outlier

analysis = StatisticalAnalysis()
print("Five-number summary:", analysis.five_number_summary(data))
print("Outliers:", analysis.outliers_iqr(data))
print("Coefficient of variation:", analysis.coefficient_of_variation(data))
```

## NumPy for Mathematical Computing

### Basic NumPy Arrays and Operations
```python
import numpy as np

# Array creation
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# Array creation functions
zeros = np.zeros(5)                             # Array of zeros
ones = np.ones((2, 3))                          # 2x3 array of ones
full = np.full((3, 3), 7)                       # 3x3 array filled with 7
eye = np.eye(4)                                 # 4x4 identity matrix
arange = np.arange(0, 10, 2)                    # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)                 # [0.0, 0.25, 0.5, 0.75, 1.0]

# Random arrays
random_array = np.random.random(5)              # Random floats [0, 1)
random_int = np.random.randint(1, 10, 5)        # Random integers
normal = np.random.normal(0, 1, 5)              # Normal distribution

# Array properties
print(arr2.shape)                               # (2, 3)
print(arr2.dtype)                               # int64
print(arr2.size)                                # 6
print(arr2.ndim)                                # 2

# Basic operations
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(a + b)                                    # [6, 8, 10, 12]
print(a * b)                                    # [5, 12, 21, 32] (element-wise)
print(a ** 2)                                   # [1, 4, 9, 16]
print(np.sqrt(a))                               # [1.0, 1.414, 1.732, 2.0]

# Mathematical functions
print(np.sin(a))                                # Sine of each element
print(np.cos(a))                                # Cosine of each element
print(np.exp(a))                                # Exponential of each element
print(np.log(a))                                # Natural log of each element

# Aggregation functions
print(np.sum(a))                                # 10
print(np.mean(a))                               # 2.5
print(np.std(a))                                # 1.118
print(np.min(a))                                # 1
print(np.max(a))                                # 4
print(np.argmin(a))                             # 0 (index of minimum)
print(np.argmax(a))                             # 3 (index of maximum)
```

### Advanced NumPy Mathematical Operations
```python
import numpy as np

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = np.dot(A, B)                                # Matrix multiplication
C = A @ B                                       # Alternative syntax (Python 3.5+)
print(C)                                        # [[19, 22], [43, 50]]

# Element-wise operations
print(A * B)                                    # Element-wise multiplication
print(A / B)                                    # Element-wise division

# Linear algebra operations
eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

det = np.linalg.det(A)                          # Determinant
print("Determinant:", det)

inv = np.linalg.inv(A)                          # Inverse matrix
print("Inverse:", inv)

# Solving linear systems Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print("Solution x:", x)

# Singular Value Decomposition
U, s, Vt = np.linalg.svd(A)
print("SVD - U:", U)
print("SVD - s:", s)
print("SVD - Vt:", Vt)

# QR decomposition
Q, R = np.linalg.qr(A)
print("QR - Q:", Q)
print("QR - R:", R)

# Matrix norms
print("Frobenius norm:", np.linalg.norm(A))
print("2-norm:", np.linalg.norm(A, 2))
print("1-norm:", np.linalg.norm(A, 1))

# Boolean operations and logical functions
arr = np.array([1, 2, 3, 4, 5])
print(arr > 3)                                  # [False, False, False, True, True]
print(np.all(arr > 0))                          # True
print(np.any(arr > 4))                          # True
print(np.where(arr > 3, arr, 0))                # [0, 0, 0, 4, 5]

# Statistical functions
data = np.random.normal(100, 15, 1000)          # Normal distribution
print(f"Mean: {np.mean(data):.2f}")
print(f"Std: {np.std(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"25th percentile: {np.percentile(data, 25):.2f}")
print(f"75th percentile: {np.percentile(data, 75):.2f}")

# Correlation and covariance
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5
correlation_matrix = np.corrcoef(x, y)
covariance_matrix = np.cov(x, y)
print("Correlation matrix:", correlation_matrix)
print("Covariance matrix:", covariance_matrix)
```

### NumPy Polynomial Operations
```python
import numpy as np
import matplotlib.pyplot as plt

# Polynomial coefficients (highest degree first)
# p(x) = 2x^3 - 3x^2 + x - 5
coeffs = [2, -3, 1, -5]

# Evaluate polynomial at specific points
x_values = np.array([1, 2, 3, 4])
y_values = np.polyval(coeffs, x_values)
print("P(x) values:", y_values)

# Polynomial roots
roots = np.roots(coeffs)
print("Roots:", roots)

# Polynomial from roots
new_coeffs = np.poly(roots)
print("Reconstructed coefficients:", new_coeffs)

# Polynomial arithmetic
p1 = [1, 2, 3]    # x^2 + 2x + 3
p2 = [1, 1]       # x + 1

# Addition and subtraction
p_add = np.polyadd(p1, [0, 1, 1])               # Add x + 1
p_sub = np.polysub(p1, [0, 1, 1])               # Subtract x + 1

# Multiplication and division
p_mul = np.polymul(p1, p2)
p_div, remainder = np.polydiv(p_mul, p2)

print("Multiplication:", p_mul)
print("Division:", p_div)
print("Remainder:", remainder)

# Polynomial derivatives and integrals
p_deriv = np.polyder(coeffs)                    # Derivative
p_integr = np.polyint(coeffs)                   # Integral

print("Derivative coefficients:", p_deriv)
print("Integral coefficients:", p_integr)

# Polynomial fitting
# Generate noisy data
x_data = np.linspace(0, 5, 50)
y_true = 2 * x_data**2 + 3 * x_data + 1
y_data = y_true + np.random.normal(0, 2, 50)

# Fit polynomial of degree 2
fitted_coeffs = np.polyfit(x_data, y_data, 2)
print("Fitted coefficients:", fitted_coeffs)

# Evaluate fitted polynomial
y_fitted = np.polyval(fitted_coeffs, x_data)

# Calculate R-squared
ss_res = np.sum((y_data - y_fitted) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared: {r_squared:.4f}")
```

## SciPy Scientific Computing

### SciPy Optimization
```python
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# Function minimization
def objective_function(x):
    return x**2 + 10*np.sin(x)

# Find minimum
result = optimize.minimize_scalar(objective_function, bounds=(-10, 10), method='bounded')
print(f"Minimum at x = {result.x:.4f}, f(x) = {result.fun:.4f}")

# Multi-dimensional optimization
def rosenbrock(x):
    """Rosenbrock function"""
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# Starting point
x0 = [0, 0]

# Minimize using different methods
result_bfgs = optimize.minimize(rosenbrock, x0, method='BFGS')
result_nelder = optimize.minimize(rosenbrock, x0, method='Nelder-Mead')

print("BFGS result:", result_bfgs.x)
print("Nelder-Mead result:", result_nelder.x)

# Root finding
def equation(x):
    return x**3 - 2*x - 5

root = optimize.fsolve(equation, 2.0)[0]
print(f"Root: {root:.6f}")

# System of equations
def equations(vars):
    x, y = vars
    eq1 = x**2 + y**2 - 4
    eq2 = x - y - 1
    return [eq1, eq2]

solution = optimize.fsolve(equations, [1, 1])
print(f"System solution: x = {solution[0]:.4f}, y = {solution[1]:.4f}")

# Curve fitting
def model_function(x, a, b, c):
    return a * np.exp(-b * x) + c

# Generate sample data
x_data = np.linspace(0, 4, 50)
y_true = model_function(x_data, 2.5, 1.3, 0.5)
y_data = y_true + 0.2 * np.random.normal(size=len(x_data))

# Fit the curve
popt, pcov = optimize.curve_fit(model_function, x_data, y_data)
print(f"Fitted parameters: a = {popt[0]:.4f}, b = {popt[1]:.4f}, c = {popt[2]:.4f}")

# Parameter uncertainties
param_errors = np.sqrt(np.diag(pcov))
print(f"Parameter errors: {param_errors}")
```

### SciPy Integration
```python
import numpy as np
from scipy import integrate

# Numerical integration
def integrand(x):
    return np.exp(-x**2)

# Definite integral
result, error = integrate.quad(integrand, 0, np.inf)
print(f"∫₀^∞ e^(-x²) dx = {result:.6f} ± {error:.2e}")

# Multiple integration
def integrand_2d(y, x):
    return x * y**2

result_2d, error_2d = integrate.dblquad(integrand_2d, 0, 2, lambda x: 0, lambda x: 1)
print(f"Double integral result: {result_2d:.6f}")

# Triple integration
def integrand_3d(z, y, x):
    return x * y * z

result_3d, error_3d = integrate.tplquad(integrand_3d, 0, 1, lambda x: 0, lambda x: 1, lambda x, y: 0, lambda x, y: 1)
print(f"Triple integral result: {result_3d:.6f}")

# Ordinary Differential Equations (ODEs)
def harmonic_oscillator(t, y):
    """Simple harmonic oscillator: d²x/dt² = -ω²x"""
    x, v = y
    omega = 1.0
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

# Initial conditions: x(0) = 1, v(0) = 0
y0 = [1, 0]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 100)

# Solve ODE
sol = integrate.solve_ivp(harmonic_oscillator, t_span, y0, t_eval=t_eval)

# Extract solution
t = sol.t
x = sol.y[0]
v = sol.y[1]

print(f"ODE solved at {len(t)} time points")

# Lorenz system (chaotic)
def lorenz(t, xyz, sigma=10, rho=28, beta=8/3):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Initial conditions
xyz0 = [1, 1, 1]
t_span = (0, 25)
t_eval = np.linspace(0, 25, 10000)

# Solve Lorenz system
lorenz_sol = integrate.solve_ivp(lorenz, t_span, xyz0, t_eval=t_eval, method='RK45')

print(f"Lorenz system solved with {len(lorenz_sol.t)} points")
```

### SciPy Interpolation and Signal Processing
```python
import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt

# Interpolation
x = np.linspace(0, 10, 11)
y = np.sin(x)

# Linear interpolation
f_linear = interpolate.interp1d(x, y, kind='linear')

# Cubic spline interpolation
f_cubic = interpolate.interp1d(x, y, kind='cubic')

# B-spline interpolation
tck = interpolate.splrep(x, y, s=0)
f_bspline = lambda x_new: interpolate.splev(x_new, tck)

# Evaluate interpolations
x_new = np.linspace(0, 10, 101)
y_linear = f_linear(x_new)
y_cubic = f_cubic(x_new)
y_bspline = f_bspline(x_new)

# 2D interpolation
x_2d = np.linspace(0, 4, 5)
y_2d = np.linspace(0, 4, 5)
X_2d, Y_2d = np.meshgrid(x_2d, y_2d)
Z_2d = np.sin(X_2d) * np.cos(Y_2d)

# Create interpolator
f_2d = interpolate.interp2d(x_2d, y_2d, Z_2d, kind='cubic')

# Signal processing
# Generate sample signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs)
signal_clean = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
noise = 0.2 * np.random.randn(len(t))
signal_noisy = signal_clean + noise

# FFT
fft_result = np.fft.fft(signal_noisy)
freqs = np.fft.fftfreq(len(t), 1/fs)

# Power spectral density
f_psd, psd = signal.welch(signal_noisy, fs, nperseg=256)

# Filter design
# Low-pass filter
nyquist = fs / 2
low_cutoff = 80 / nyquist
b, a = signal.butter(4, low_cutoff, btype='low')

# Apply filter
signal_filtered = signal.filtfilt(b, a, signal_noisy)

# Peak finding
peaks, _ = signal.find_peaks(signal_clean, height=0.5)

# Convolution
kernel = np.array([1, 2, 1]) / 4  # Simple smoothing kernel
signal_convolved = signal.convolve(signal_noisy, kernel, mode='same')

print(f"Found {len(peaks)} peaks in the signal")
print(f"Filter order: {len(b) - 1}")
```

## SymPy Symbolic Mathematics

### Basic Symbolic Operations
```python
import sympy as sp
from sympy import symbols, Function, Eq, solve, diff, integrate, limit, series

# Define symbols
x, y, z = symbols('x y z')
t = symbols('t', real=True)
n = symbols('n', integer=True)

# Basic expressions
expr1 = x**2 + 2*x + 1
expr2 = (x + 1)**2
print(f"expr1 = {expr1}")
print(f"expr2 = {expr2}")
print(f"Are they equal? {sp.simplify(expr1 - expr2) == 0}")

# Simplification
complex_expr = (x**2 - 1)/(x - 1)
simplified = sp.simplify(complex_expr)
print(f"Simplified: {simplified}")

# Expansion and factoring
expanded = sp.expand((x + y)**3)
factored = sp.factor(x**2 - 4)
print(f"Expanded (x+y)³: {expanded}")
print(f"Factored x²-4: {factored}")

# Substitution
expr = x**2 + 2*x + 1
result = expr.subs(x, 3)
print(f"Substituting x=3: {result}")

# Multiple substitutions
result_multi = expr.subs([(x, y + 1)])
print(f"Substituting x=y+1: {result_multi}")

# Solving equations
equation = Eq(x**2 - 4, 0)
solutions = solve(equation, x)
print(f"Solutions to x²-4=0: {solutions}")

# System of equations
eq1 = Eq(x + y, 5)
eq2 = Eq(x - y, 1)
system_solution = solve([eq1, eq2], [x, y])
print(f"System solution: {system_solution}")

# Calculus - Derivatives
f = x**3 + 2*x**2 - x + 5
f_prime = diff(f, x)
f_double_prime = diff(f, x, 2)  # Second derivative

print(f"f(x) = {f}")
print(f"f'(x) = {f_prime}")
print(f"f''(x) = {f_double_prime}")

# Partial derivatives
g = x**2 * y + y**3
dg_dx = diff(g, x)
dg_dy = diff(g, y)
print(f"∂g/∂x = {dg_dx}")
print(f"∂g/∂y = {dg_dy}")
```

### Advanced SymPy Operations
```python
import sympy as sp
from sympy import *

x, y, z, t = symbols('x y z t')

# Integration
# Indefinite integrals
integral1 = integrate(x**2, x)
integral2 = integrate(sin(x), x)
integral3 = integrate(exp(-x**2), x)

print(f"∫x² dx = {integral1}")
print(f"∫sin(x) dx = {integral2}")
print(f"∫e^(-x²) dx = {integral3}")

# Definite integrals
definite1 = integrate(x**2, (x, 0, 1))
definite2 = integrate(sin(x), (x, 0, pi))
definite3 = integrate(exp(-x), (x, 0, oo))

print(f"∫₀¹ x² dx = {definite1}")
print(f"∫₀^π sin(x) dx = {definite2}")
print(f"∫₀^∞ e^(-x) dx = {definite3}")

# Multiple integrals
double_integral = integrate(integrate(x*y, x), y)
print(f"∬ xy dx dy = {double_integral}")

# Limits
limit1 = limit(sin(x)/x, x, 0)
limit2 = limit((1 + 1/x)**x, x, oo)
limit3 = limit(x**2/exp(x), x, oo)

print(f"lim(x→0) sin(x)/x = {limit1}")
print(f"lim(x→∞) (1+1/x)^x = {limit2}")
print(f"lim(x→∞) x²/e^x = {limit3}")

# Series expansion
series1 = series(sin(x), x, 0, n=6)
series2 = series(exp(x), x, 0, n=5)
series3 = series(1/(1-x), x, 0, n=5)

print(f"sin(x) series: {series1}")
print(f"e^x series: {series2}")
print(f"1/(1-x) series: {series3}")

# Differential equations
f = Function('f')
ode1 = Eq(f(x).diff(x), f(x))  # f'(x) = f(x)
solution1 = dsolve(ode1, f(x))
print(f"Solution to f'=f: {solution1}")

# Second-order ODE
ode2 = Eq(f(x).diff(x, 2) + f(x), 0)  # f'' + f = 0
solution2 = dsolve(ode2, f(x))
print(f"Solution to f''+f=0: {solution2}")

# Matrices
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print(f"Matrix A: {A}")
print(f"Determinant: {A.det()}")
print(f"Inverse: {A.inv()}")
print(f"Eigenvalues: {A.eigenvals()}")
print(f"Eigenvectors: {A.eigenvects()}")

# Matrix operations
print(f"A + B = {A + B}")
print(f"A * B = {A * B}")
print(f"A^T = {A.T}")

# Symbolic linear algebra
x1, x2 = symbols('x1 x2')
system_matrix = Matrix([[1, 2], [3, 4]])
rhs = Matrix([5, 6])
symbolic_solution = system_matrix.inv() * rhs
print(f"Symbolic solution: {symbolic_solution}")
```

### SymPy Applications
```python
import sympy as sp
from sympy import *
import matplotlib.pyplot as plt
import numpy as np

# Physics applications
# Kinematic equations
t, v0, a, s0 = symbols('t v_0 a s_0')

# Position as function of time
s = s0 + v0*t + sp.Rational(1,2)*a*t**2
v = diff(s, t)  # Velocity
acceleration = diff(v, t)  # Acceleration

print(f"Position: s(t) = {s}")
print(f"Velocity: v(t) = {v}")
print(f"Acceleration: a(t) = {acceleration}")

# Energy conservation
m, g, h, v = symbols('m g h v', positive=True)
KE = sp.Rational(1,2) * m * v**2  # Kinetic energy
PE = m * g * h  # Potential energy
total_energy = KE + PE

print(f"Total energy: E = {total_energy}")

# Fourier series
x = symbols('x', real=True)
n = symbols('n', integer=True, positive=True)

# Square wave Fourier series
def square_wave_fourier(x, n_terms=5):
    series_sum = 0
    for k in range(1, n_terms + 1, 2):  # Odd terms only
        term = (4/pi) * sin(k*x) / k
        series_sum += term
    return series_sum

# Symbolic representation
fourier_term = (4/pi) * sin((2*n-1)*x) / (2*n-1)
print(f"Fourier series term: {fourier_term}")

# Probability distributions
# Normal distribution PDF
mu, sigma = symbols('mu sigma', real=True, positive=True)
normal_pdf = exp(-(x - mu)**2 / (2*sigma**2)) / (sigma * sqrt(2*pi))
print(f"Normal PDF: {normal_pdf}")

# Moment generating function
mgf = integrate(exp(t*x) * normal_pdf, (x, -oo, oo))
print(f"MGF: {mgf}")

# Economic applications
# Compound interest
P, r, n_periods = symbols('P r n', positive=True)
compound_interest = P * (1 + r)**n_periods
print(f"Compound interest: A = {compound_interest}")

# Present value
discount_rate = symbols('discount_rate', positive=True)
future_value = symbols('FV', positive=True)
present_value = future_value / (1 + discount_rate)**n_periods
print(f"Present value: PV = {present_value}")

# Optimization problems
# Minimize cost function
cost_function = x**2 + 4*x + 7
critical_points = solve(diff(cost_function, x), x)
minimum_cost = cost_function.subs(x, critical_points[0])
print(f"Minimum cost at x = {critical_points[0]}: {minimum_cost}")

# Lagrange multipliers
# Optimize f(x,y) = x² + y² subject to g(x,y) = x + y - 1 = 0
lam = symbols('lambda')
f = x**2 + y**2
g = x + y - 1

# Lagrangian
L = f + lam * g

# Find critical points
critical_eqs = [diff(L, x), diff(L, y), diff(L, lam)]
lagrange_solution = solve(critical_eqs, [x, y, lam])
print(f"Lagrange solution: {lagrange_solution}")

# Number theory
# Greatest common divisor
gcd_result = gcd(48, 18)
print(f"gcd(48, 18) = {gcd_result}")

# Prime factorization
factors = factorint(60)
print(f"Prime factors of 60: {factors}")

# Continued fractions
cf = continued_fraction(pi)
print(f"π continued fraction: {cf}")

# Rational approximation
rational_approx = nsimplify(pi, rational=True)
print(f"Rational approximation of π: {rational_approx}")
```

## Specialized Mathematical Libraries

### Decimal for High-Precision Arithmetic
```python
from decimal import Decimal, getcontext, ROUND_HALF_UP
import math

# Set precision
getcontext().prec = 50

# High-precision calculations
a = Decimal('0.1')
b = Decimal('0.2')
c = a + b
print(f"High precision: 0.1 + 0.2 = {c}")

# Compare with float
float_result = 0.1 + 0.2
print(f"Float precision: 0.1 + 0.2 = {float_result}")

# Financial calculations
price = Decimal('19.99')
tax_rate = Decimal('0.085')
tax = price * tax_rate
total = price + tax

print(f"Price: ${price}")
print(f"Tax: ${tax:.2f}")
print(f"Total: ${total:.2f}")

# Rounding modes
value = Decimal('2.5')
rounded_up = value.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
print(f"Rounded value: {rounded_up}")

# Square root with high precision
sqrt_2 = Decimal(2).sqrt()
print(f"√2 with 50 decimal places: {sqrt_2}")

# Compound interest calculation
principal = Decimal('1000.00')
rate = Decimal('0.05')
periods = 10

compound_amount = principal * (1 + rate) ** periods
print(f"Compound amount: ${compound_amount:.2f}")
```

### Fractions for Exact Rational Arithmetic
```python
from fractions import Fraction

# Create fractions
f1 = Fraction(1, 3)
f2 = Fraction(2, 5)
f3 = Fraction('0.25')
f4 = Fraction(0.1).limit_denominator()

print(f"1/3 = {f1}")
print(f"2/5 = {f2}")
print(f"0.25 = {f3}")
print(f"0.1 ≈ {f4}")

# Arithmetic operations
addition = f1 + f2
subtraction = f1 - f2
multiplication = f1 * f2
division = f1 / f2

print(f"1/3 + 2/5 = {addition}")
print(f"1/3 - 2/5 = {subtraction}")
print(f"1/3 × 2/5 = {multiplication}")
print(f"1/3 ÷ 2/5 = {division}")

# Powers
power = f1 ** 2
print(f"(1/3)² = {power}")

# Converting to decimal
decimal_value = float(f1)
print(f"1/3 as decimal: {decimal_value}")

# Greatest common divisor
from math import gcd
fraction_gcd = Fraction(gcd(12, 18), 24)
print(f"GCD fraction: {fraction_gcd}")

# Continued fraction representation
def continued_fraction(x, max_terms=10):
    """Convert fraction to continued fraction representation"""
    cf = []
    for _ in range(max_terms):
        if x.denominator == 1:
            cf.append(x.numerator)
            break
        integer_part = x.numerator // x.denominator
        cf.append(integer_part)
        x = 1 / (x - integer_part)
    return cf

cf_repr = continued_fraction(Fraction(22, 7))
print(f"22/7 as continued fraction: {cf_repr}")
```

### Combinatorics and Probability
```python
import math
from itertools import permutations, combinations, combinations_with_replacement
from itertools import product
import numpy as np

# Combinatorics functions
def factorial(n):
    """Calculate factorial"""
    return math.factorial(n)

def permutation(n, r):
    """Calculate permutation P(n,r)"""
    return factorial(n) // factorial(n - r)

def combination(n, r):
    """Calculate combination C(n,r)"""
    return factorial(n) // (factorial(r) * factorial(n - r))

# Examples
print(f"5! = {factorial(5)}")
print(f"P(10, 3) = {permutation(10, 3)}")
print(f"C(10, 3) = {combination(10, 3)}")

# Using itertools for small sets
items = ['A', 'B', 'C', 'D']

# All permutations
all_perms = list(permutations(items, 2))
print(f"Permutations of 2 from {items}: {all_perms}")

# All combinations
all_combs = list(combinations(items, 2))
print(f"Combinations of 2 from {items}: {all_combs}")

# Combinations with replacement
combs_with_repl = list(combinations_with_replacement(items, 2))
print(f"Combinations with replacement: {combs_with_repl}")

# Cartesian product
cart_product = list(product(['A', 'B'], [1, 2, 3]))
print(f"Cartesian product: {cart_product}")

# Probability distributions
class ProbabilityDistributions:
    @staticmethod
    def binomial_probability(n, k, p):
        """Binomial probability P(X = k)"""
        return combination(n, k) * (p ** k) * ((1 - p) ** (n - k))
    
    @staticmethod
    def poisson_probability(k, lam):
        """Poisson probability P(X = k)"""
        return (lam ** k) * math.exp(-lam) / factorial(k)
    
    @staticmethod
    def normal_pdf(x, mu=0, sigma=1):
        """Normal distribution PDF"""
        return (1 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    @staticmethod
    def normal_cdf_approx(x, mu=0, sigma=1):
        """Approximate normal CDF using error function"""
        return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

# Examples
prob_dist = ProbabilityDistributions()

# Binomial: probability of 3 successes in 10 trials with p=0.3
binom_prob = prob_dist.binomial_probability(10, 3, 0.3)
print(f"Binomial P(X=3|n=10,p=0.3): {binom_prob:.4f}")

# Poisson: probability of 2 events with rate 1.5
poisson_prob = prob_dist.poisson_probability(2, 1.5)
print(f"Poisson P(X=2|λ=1.5): {poisson_prob:.4f}")

# Normal distribution
normal_prob = prob_dist.normal_pdf(1, 0, 1)
normal_cdf = prob_dist.normal_cdf_approx(1, 0, 1)
print(f"Normal PDF(1): {normal_prob:.4f}")
print(f"Normal CDF(1): {normal_cdf:.4f}")
```

### Complex Analysis
```python
import cmath
import numpy as np
import matplotlib.pyplot as plt

# Complex number operations
z1 = 3 + 4j
z2 = 1 + 2j

# Basic operations
addition = z1 + z2
multiplication = z1 * z2
division = z1 / z2

print(f"z1 = {z1}")
print(f"z2 = {z2}")
print(f"z1 + z2 = {addition}")
print(f"z1 × z2 = {multiplication}")
print(f"z1 ÷ z2 = {division}")

# Polar form
magnitude = abs(z1)
phase = cmath.phase(z1)
polar_form = cmath.polar(z1)

print(f"|z1| = {magnitude}")
print(f"arg(z1) = {phase} radians = {math.degrees(phase)} degrees")
print(f"Polar form: {polar_form}")

# Convert back from polar
rectangular = cmath.rect(magnitude, phase)
print(f"Back to rectangular: {rectangular}")

# Complex functions
print(f"e^z1 = {cmath.exp(z1)}")
print(f"ln(z1) = {cmath.log(z1)}")
print(f"z1^2 = {z1 ** 2}")
print(f"√z1 = {cmath.sqrt(z1)}")

# Trigonometric functions
print(f"sin(z1) = {cmath.sin(z1)}")
print(f"cos(z1) = {cmath.cos(z1)}")
print(f"tan(z1) = {cmath.tan(z1)}")

# Hyperbolic functions
print(f"sinh(z1) = {cmath.sinh(z1)}")
print(f"cosh(z1) = {cmath.cosh(z1)}")
print(f"tanh(z1) = {cmath.tanh(z1)}")

# Mandelbrot set example
def mandelbrot(c, max_iter=100):
    """Check if point c is in Mandelbrot set"""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

# Generate Mandelbrot set
def generate_mandelbrot(width=800, height=600, max_iter=100):
    xmin, xmax = -2.5, 1.5
    ymin, ymax = -1.5, 1.5
    
    mandelbrot_set = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            x = xmin + (xmax - xmin) * j / width
            y = ymin + (ymax - ymin) * i / height
            c = complex(x, y)
            mandelbrot_set[i, j] = mandelbrot(c, max_iter)
    
    return mandelbrot_set

# Small example
mandelbrot_small = generate_mandelbrot(100, 100, 50)
print(f"Generated {mandelbrot_small.shape} Mandelbrot set")

# Julia set
def julia_set(z, c=-0.7 + 0.27015j, max_iter=100):
    """Julia set iteration"""
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

# Complex polynomial roots
def find_polynomial_roots():
    """Find roots of complex polynomial"""
    # z^3 - 1 = 0 (cube roots of unity)
    roots = []
    for k in range(3):
        angle = 2 * math.pi * k / 3
        root = cmath.exp(1j * angle)
        roots.append(root)
    
    return roots

cube_roots = find_polynomial_roots()
print("Cube roots of unity:")
for i, root in enumerate(cube_roots):
    print(f"  ω{i} = {root:.6f}")
```

## Performance Optimization

### Vectorization and NumPy Optimization
```python
import numpy as np
import time
from numba import jit

# Compare loop vs vectorized operations
def compare_performance():
    # Large array
    n = 1000000
    a = np.random.randn(n)
    b = np.random.randn(n)
    
    # Python loop
    start_time = time.time()
    result_loop = []
    for i in range(n):
        result_loop.append(a[i] * b[i] + 1)
    result_loop = np.array(result_loop)
    loop_time = time.time() - start_time
    
    # NumPy vectorized
    start_time = time.time()
    result_vectorized = a * b + 1
    vectorized_time = time.time() - start_time
    
    print(f"Loop time: {loop_time:.4f} seconds")
    print(f"Vectorized time: {vectorized_time:.4f} seconds")
    print(f"Speedup: {loop_time / vectorized_time:.2f}x")
    
    # Verify results are the same
    print(f"Results match: {np.allclose(result_loop, result_vectorized)}")

# compare_performance()

# Memory-efficient operations
def memory_efficient_operations():
    # Large arrays
    a = np.random.randn(10000, 10000)
    b = np.random.randn(10000, 10000)
    
    # Memory-inefficient: creates temporary arrays
    # result = (a + b) * (a - b)  # Creates two temporary arrays
    
    # Memory-efficient: in-place operations
    result = np.empty_like(a)
    np.add(a, b, out=result)
    temp = np.subtract(a, b)
    np.multiply(result, temp, out=result)
    
    return result

# JIT compilation with Numba
@jit(nopython=True)
def monte_carlo_pi_jit(n):
    """Monte Carlo estimation of π with JIT compilation"""
    count = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1:
            count += 1
    return 4.0 * count / n

def compare_monte_carlo():
    n = 1000000
    
    # Regular Python
    start_time = time.time()
    pi_estimate_regular = monte_carlo_pi_regular(n)
    regular_time = time.time() - start_time
    
    # JIT compiled
    start_time = time.time()
    pi_estimate_jit = monte_carlo_pi_jit(n)
    jit_time = time.time() - start_time
    
    print(f"Regular time: {regular_time:.4f} seconds")
    print(f"JIT time: {jit_time:.4f} seconds")
    print(f"Speedup: {regular_time / jit_time:.2f}x")

def monte_carlo_pi_regular(n):
    """Regular Monte Carlo estimation"""
    count = 0
    for i in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1:
            count += 1
    return 4.0 * count / n

# Broadcasting for efficient operations
def broadcasting_examples():
    # 2D array operations
    matrix = np.random.randn(1000, 1000)
    
    # Subtract mean from each column (broadcasting)
    column_means = np.mean(matrix, axis=0)
    centered_matrix = matrix - column_means  # Broadcasting
    
    # Divide by standard deviation of each row
    row_stds = np.std(matrix, axis=1, keepdims=True)
    normalized_matrix = matrix / row_stds  # Broadcasting
    
    print(f"Original shape: {matrix.shape}")
    print(f"Column means shape: {column_means.shape}")
    print(f"Row stds shape: {row_stds.shape}")
    
    return centered_matrix, normalized_matrix

# Efficient linear algebra
def efficient_linear_algebra():
    # Use appropriate data types
    A_float32 = np.random.randn(1000, 1000).astype(np.float32)
    A_float64 = np.random.randn(1000, 1000).astype(np.float64)
    
    # Benchmark matrix multiplication
    start_time = time.time()
    result_32 = A_float32 @ A_float32
    time_32 = time.time() - start_time
    
    start_time = time.time()
    result_64 = A_float64 @ A_float64
    time_64 = time.time() - start_time
    
    print(f"Float32 time: {time_32:.4f} seconds")
    print(f"Float64 time: {time_64:.4f} seconds")
    
    # Use optimized BLAS/LAPACK when possible
    # NumPy automatically uses optimized libraries if available
    
    return result_32, result_64
```

---

*This document covers comprehensive mathematical computing in Python including built-in functions, standard library modules (math, random, statistics), scientific computing with NumPy and SciPy, symbolic mathematics with SymPy, specialized libraries for high-precision arithmetic, and performance optimization techniques. For the most up-to-date information, refer to the official documentation of the respective libraries.*