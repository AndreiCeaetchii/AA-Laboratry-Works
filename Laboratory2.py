import random
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd

sys.setrecursionlimit(10 ** 6)


def generate_random_integers(n, low=0, high=1000):
    return [random.randint(low, high) for _ in range(n)]


def generate_random_floats(n, low=0.0, high=1000.0):
    return [random.uniform(low, high) for _ in range(n)]


def generate_random_negative_integers(n, low=-1000, high=1000):
    return [random.randint(low, high) for _ in range(n)]


def generate_sorted_data(n):
    return sorted(generate_random_integers(n))


# Merge Sort with metrics
def merge_sort_with_metrics(arr, left, right):
    comparisons = 0
    swaps = 0
    recursion_depth = 0

    def merge(arr, left, mid, right):
        nonlocal comparisons, swaps
        n1 = mid - left + 1
        n2 = right - mid

        L = [0] * n1
        R = [0] * n2

        for i in range(n1):
            L[i] = arr[left + i]
        for j in range(n2):
            R[j] = arr[mid + 1 + j]

        i = j = 0
        k = left

        while i < n1 and j < n2:
            comparisons += 1
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            swaps += 1
            k += 1

        while i < n1:
            arr[k] = L[i]
            i += 1
            k += 1
            swaps += 1

        while j < n2:
            arr[k] = R[j]
            j += 1
            k += 1
            swaps += 1

    def sort(arr, left, right):
        nonlocal recursion_depth
        if left < right:
            recursion_depth += 1
            mid = (left + right) // 2
            sort(arr, left, mid)
            sort(arr, mid + 1, right)
            merge(arr, left, mid, right)

    sort(arr, left, right)
    return comparisons, swaps, recursion_depth


def heap_sort_with_metrics(arr):
    comparisons = 0
    swaps = 0

    def heapify(arr, n, i):
        nonlocal comparisons, swaps
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2

        if l < n:
            comparisons += 1
            if arr[l] > arr[largest]:
                largest = l

        if r < n:
            comparisons += 1
            if arr[r] > arr[largest]:
                largest = r

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            swaps += 1
            heapify(arr, n, largest)

    n = len(arr)

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        swaps += 1
        heapify(arr, i, 0)

    return comparisons, swaps


def quick_sort_with_metrics(arr, low, high):
    comparisons = 0
    swaps = 0
    recursion_depth = 0

    def partition(arr, low, high):
        nonlocal comparisons, swaps
        pivot = arr[high]
        i = low - 1

        for j in range(low, high):
            comparisons += 1
            if arr[j] < pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                swaps += 1

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        swaps += 1
        return i + 1

    def sort(arr, low, high):
        nonlocal recursion_depth
        if low < high:
            recursion_depth += 1
            pi = partition(arr, low, high)
            sort(arr, low, pi - 1)
            sort(arr, pi + 1, high)

    sort(arr, low, high)
    return comparisons, swaps, recursion_depth


def bubble_sort_with_metrics(arr):
    comparisons = 0
    swaps = 0

    n = len(arr)

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            comparisons += 1
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swaps += 1
                swapped = True
        if not swapped:
            break

    return comparisons, swaps


def measure_algorithm(algorithm, arr, *args):
    arr_copy = arr.copy()
    start_time = time.time()
    metrics = algorithm(arr_copy, *args)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time, metrics


input_sizes = [100, 1000, 10000]
input_types = {
    "Random Integers": generate_random_integers,
    "Random Floats": generate_random_floats,
    "Random Negative Integers": generate_random_negative_integers,
    "Sorted Data": generate_sorted_data,
}

algorithms = {
    "Merge Sort": merge_sort_with_metrics,
    "Heap Sort": heap_sort_with_metrics,
    "Quick Sort": quick_sort_with_metrics,
    "Bubble Sort": bubble_sort_with_metrics,
}

results = []

# Perform tests
for input_name, input_generator in input_types.items():
    for n in input_sizes:
        arr = input_generator(n)
        for algo_name, algo_func in algorithms.items():
            if algo_name == "Merge Sort" or algo_name == "Quick Sort":
                time_taken, metrics = measure_algorithm(algo_func, arr, 0, len(arr) - 1)
            else:
                time_taken, metrics = measure_algorithm(algo_func, arr)
            results.append({
                "Input Type": input_name,
                "Input Size": n,
                "Algorithm": algo_name,
                "Time (s)": time_taken,
                "Comparisons": metrics[0],
                "Swaps": metrics[1],
                "Recursion Depth": metrics[2] if len(metrics) > 2 else 0,
            })

results_df = pd.DataFrame(results)

input_types_list = ["Random Integers", "Random Floats", "Random Negative Integers", "Sorted Data"]

for input_type in input_types_list:
    print(f"\nResults for {input_type}:")
    input_type_df = results_df[results_df["Input Type"] == input_type]
    print(input_type_df.to_string(index=False))

for input_name in input_types.keys():
    plt.figure(figsize=(10, 6))
    for algo_name in algorithms.keys():
        subset = results_df[(results_df["Input Type"] == input_name) & (results_df["Algorithm"] == algo_name)]
        plt.plot(subset["Input Size"], subset["Time (s)"], label=algo_name)
    plt.title(f"Time vs Input Size for {input_name}")
    plt.xlabel("Input Size")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()
