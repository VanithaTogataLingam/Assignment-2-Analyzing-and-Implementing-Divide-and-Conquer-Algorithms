import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 1: Merge Sort Implementation

def merge_sort(arr):
    """
    This function implements the merge sort algorithm, a divide-and-conquer approach.
    It recursively divides the array in half, sorts each half, and merges them back together.
    """
    if len(arr) > 1:
        mid = len(arr) // 2  # Find the middle of the array
        left_half = arr[:mid]  # Left sub-array
        right_half = arr[mid:]  # Right sub-array

        # Recursively call merge_sort on both halves
        merge_sort(left_half)
        merge_sort(right_half)

        # Merge the sorted halves
        i = j = k = 0

        # Merge the left and right arrays into the original array
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        # Check if any elements were left in the left sub-array
        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        # Check if any elements were left in the right sub-array
        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1


def measure_time_merge_sort(arr):
    """
    Function to measure and return the execution time of merge_sort algorithm
    """
    start_time = time.time()
    merge_sort(arr)
    return time.time() - start_time


# Part 2: Quick Sort Implementation

def quick_sort(arr):
    """
    This function implements the quick sort algorithm, another divide-and-conquer method.
    It selects a pivot and partitions the array into smaller and larger elements, 
    then recursively sorts the two sub-arrays.
    """
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]  # Select the pivot element (middle element)
    left = [x for x in arr if x < pivot]  # Elements less than pivot
    middle = [x for x in arr if x == pivot]  # Elements equal to pivot
    right = [x for x in arr if x > pivot]  # Elements greater than pivot
    return quick_sort(left) + middle + quick_sort(right)


def measure_time_quick_sort(arr):
    """
    Function to measure and return the execution time of quick_sort algorithm
    """
    start_time = time.time()
    quick_sort(arr)
    return time.time() - start_time


# Part 3: Data Generation

def generate_data(size):
    """
    Generates three datasets: random, sorted, and reverse sorted arrays.
    """
    random_data = random.sample(range(size), size)
    sorted_data = sorted(random_data)
    reverse_sorted_data = sorted_data[::-1]
    return random_data, sorted_data, reverse_sorted_data


# Part 4: Performance Comparison

def compare_algorithms():
    """
    This function compares the performance of merge sort and quick sort on various data types 
    and array sizes. It returns the time taken by each algorithm.
    """
    sizes = [1000, 5000, 10000]  # Different array sizes for testing
    results = []

    for size in sizes:
        random_data, sorted_data, reverse_sorted_data = generate_data(size)

        # Measure Merge Sort times
        merge_time_random = measure_time_merge_sort(random_data.copy())
        merge_time_sorted = measure_time_merge_sort(sorted_data.copy())
        merge_time_reverse = measure_time_merge_sort(reverse_sorted_data.copy())

        # Measure Quick Sort times
        quick_time_random = measure_time_quick_sort(random_data.copy())
        quick_time_sorted = measure_time_quick_sort(sorted_data.copy())
        quick_time_reverse = measure_time_quick_sort(reverse_sorted_data.copy())

        results.append({
            "size": size,
            "merge_random": merge_time_random,
            "merge_sorted": merge_time_sorted,
            "merge_reverse": merge_time_reverse,
            "quick_random": quick_time_random,
            "quick_sorted": quick_time_sorted,
            "quick_reverse": quick_time_reverse
        })

    return results


# Part 5: Display Results

# Run the comparison of algorithms
results = compare_algorithms()

# Display the results in a user-friendly table format using pandas
df_results = pd.DataFrame(results)

# Display the dataframe in the notebook
print("Algorithm Performance Comparison Results:")
print(df_results)

# Plotting the results
plt.figure(figsize=(10, 6))
for size in df_results['size']:
    merge_random = df_results[df_results['size'] == size]['merge_random'].values[0]
    merge_sorted = df_results[df_results['size'] == size]['merge_sorted'].values[0]
    merge_reverse = df_results[df_results['size'] == size]['merge_reverse'].values[0]
    quick_random = df_results[df_results['size'] == size]['quick_random'].values[0]
    quick_sorted = df_results[df_results['size'] == size]['quick_sorted'].values[0]
    quick_reverse = df_results[df_results['size'] == size]['quick_reverse'].values[0]

    plt.plot([1, 2, 3, 4, 5, 6], [merge_random, merge_sorted, merge_reverse, quick_random, quick_sorted, quick_reverse],
             label=f"Size {size}", marker='o')

plt.xticks([1, 2, 3, 4, 5, 6], ['Merge (Random)', 'Merge (Sorted)', 'Merge (Reverse)', 'Quick (Random)', 'Quick (Sorted)', 'Quick (Reverse)'])
plt.xlabel('Algorithms and Data Types')
plt.ylabel('Execution Time (seconds)')
plt.title('Comparison of Merge Sort and Quick Sort Performance')
plt.legend()
plt.grid(True)
plt.show()
