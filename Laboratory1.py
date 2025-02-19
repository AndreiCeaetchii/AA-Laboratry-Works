import time
import matplotlib.pyplot as plt
import numpy as np
import sys
from statistics import mean

# Increase recursion limit
sys.setrecursionlimit(3000)

MOD = 1000000007


# Function to calculate the N-th Fibonacci number using Fast Doubling method
def fib_fast_doubling(n):
    res = [0, 0]  # Initialize a list to hold F(n) and F(n+1)

    def FastDoubling(n, res):
        # Base Condition
        if n == 0:
            res[0] = 0
            res[1] = 1
            return

        FastDoubling(n // 2, res)  # Recursive call

        a = res[0]  # F(n)
        b = res[1]  # F(n+1)
        c = 2 * b - a

        if c < 0:
            c += MOD  # Modulo correction for negative numbers

        c = (a * c) % MOD  # F(2n)
        d = (a * a + b * b) % MOD  # F(2n+1)

        if n % 2 == 0:
            res[0] = c
            res[1] = d
        else:
            res[0] = d
            res[1] = (c + d) % MOD

    # Call the FastDoubling function with the input n
    FastDoubling(n, res)
    return res[0]  # Return the nth Fibonacci number


def fib_recursive(n):
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


def fib_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fib_matrix(n):
    def multiply(mat_a, mat_b):
        return [
            [mat_a[0][0] * mat_b[0][0] + mat_a[0][1] * mat_b[1][0],
             mat_a[0][0] * mat_b[0][1] + mat_a[0][1] * mat_b[1][1]],
            [mat_a[1][0] * mat_b[0][0] + mat_a[1][1] * mat_b[1][0],
             mat_a[1][0] * mat_b[0][1] + mat_a[1][1] * mat_b[1][1]]
        ]

    def matrix_power(mat, power):
        result = [[1, 0], [0, 1]]
        while power > 0:
            if power % 2 == 1:
                result = multiply(result, mat)
            mat = multiply(mat, mat)
            power = power // 2
        return result

    if n <= 1:
        return n
    mat = [[1, 1], [1, 0]]
    result = matrix_power(mat, n - 1)
    return result[0][0]


def fib_binet(n):
    if n > 71:  # Adjust this threshold based on your system's limitations
        raise ValueError(f"Binet's formula is not suitable for n > 71. Use other algorithms.")

    phi = (1 + np.sqrt(5)) / 2
    return int((phi ** n - (-1 / phi) ** n) / np.sqrt(5))


def fib_exponentiation_by_squaring(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1

    # Fibonacci exponentiation by squaring
    def exp_by_squaring(n):
        if n == 0:
            return [1, 0]  # F(0) = 0, F(1) = 1
        elif n == 1:
            return [1, 1]

        half = exp_by_squaring(n // 2)
        a = half[0]  # F(n//2)
        b = half[1]  # F(n//2 + 1)

        # Calculate next Fibonacci values
        if n % 2 == 0:
            return [a * a + b * b, b * (2 * a + b)]
        else:
            return [b * b + a * a, b * (2 * a + b)]

    result = exp_by_squaring(n)
    return result[0]


def measure_time(func, n, runs=3):
    """Measure execution time with multiple runs for more accurate results"""
    times = []
    for _ in range(runs):
        start_time = time.perf_counter()  # Using perf_counter for higher precision
        func(n)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return mean(times)  # Return average time


# Define different ranges for different algorithms
recursive_range = (5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 34)  # Small range for recursive
regular_range = (501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589,
                 15849, 19950, 25120, 31620, 39810, 50120, 63100, 79430, 100000, 125890, 158490, 199500,
                 251200, 316200, 398100, 501200, 631000, 794300, 1000000, 1258900, 1584900)

algorithms = {
    "Recursive": (fib_recursive, recursive_range),
    "Iterative": (fib_iterative, regular_range),
    "Matrix": (fib_matrix, regular_range),
    "Fast Doubling": (fib_fast_doubling, regular_range),  # Updated Fast Doubling
    "Binet": (fib_binet, recursive_range),
    "Exponentiation by Squaring": (fib_exponentiation_by_squaring, regular_range)
}

# Collect timing data
results = {}
for alg_name, (func, n_range) in algorithms.items():
    print(f"Testing {alg_name}...")
    times = []
    n_values = list(n_range)
    for n in n_values:
        time_taken = measure_time(func, n)
        times.append(time_taken)
    results[alg_name] = (n_values, times)

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

# Plot recursive algorithm separately (it's much slower)
recursive_n, recursive_times = results["Recursive"]
binet_n, binet_times = results["Binet"]
ax1.plot(recursive_n, recursive_times, label="Recursive", marker='o')
ax1.plot(binet_n, binet_times, label="Binet", marker='o')
ax1.set_title('Recursive Fibonacci Performance (Exponential Growth)')
ax1.set_xlabel('n')
ax1.set_ylabel('Execution Time (seconds)')
ax1.grid(True)
ax1.legend()
ax1.set_yscale('log')

# Plot other algorithms
for alg_name, (n_values, times) in results.items():
    if alg_name != "Recursive" and alg_name != "Binet":  # Skip recursive in second plot
        ax2.plot(n_values, times, label=alg_name, marker='o', markersize=3)

ax2.set_title('Other Fibonacci Algorithms Performance Comparison')
ax2.set_xlabel('n')
ax2.set_ylabel('Execution Time (seconds)')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# Print summary statistics and a table of n vs. execution time
print("\nPerformance Summary:")
for alg_name, (n_values, times) in results.items():
    avg_time = mean(times)
    max_time = max(times)
    print(f"\n{alg_name}:")
    print(f"Average time: {avg_time:.6f} seconds")
    print(f"Maximum time: {max_time:.6f} seconds")
    print(f"Tested up to n = {max(n_values)}")

    # Print table header
    print("{:>10} | {:>20}".format("n", "Execution Time (s)"))
    print("-" * 35)
    # Print each n value and its execution time in scientific notation
    for n, t in zip(n_values, times):
        print("{:>10} | {:>20.6e}".format(n, t))
