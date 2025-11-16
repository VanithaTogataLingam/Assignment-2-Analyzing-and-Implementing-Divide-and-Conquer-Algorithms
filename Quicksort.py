import random
import time

# Function for deterministic quicksort with optimized pivot selection (median of three)
def quicksort(arr):
    """Implementation of the deterministic Quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    else:
        # Use median-of-three to choose the pivot
        pivot = median_of_three(arr)
        
        # Partitioning the array around the pivot
        left = [x for x in arr if x < pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + [pivot] + quicksort(right)

# Function to select pivot as median of the first, middle, and last element
def median_of_three(arr):
    """Choose pivot as the median of first, middle, and last element."""
    first = arr[0]
    middle = arr[len(arr) // 2]
    last = arr[-1]
    
    # Return the median of these three elements
    return sorted([first, middle, last])[1]

# Function for randomized quicksort
def quicksort_randomized(arr):
    """Implementation of the randomized Quicksort algorithm."""
    if len(arr) <= 1:
        return arr
    else:
        # Randomly choose a pivot index from the array
        pivot_index = random.randint(0, len(arr) - 1)
        
        # Swap the pivot with the last element to use the same partitioning logic
        arr[pivot_index], arr[-1] = arr[-1], arr[pivot_index]
        
        pivot = arr[-1]
        
        # Partitioning: divide the array into two subarrays
        left = [x for x in arr[:-1] if x <= pivot]
        right = [x for x in arr[:-1] if x > pivot]
        
        # Recursively apply randomized Quicksort to left and right subarrays, then combine
        return quicksort_randomized(left) + [pivot] + quicksort_randomized(right)

# Function to compare the performance of both Quicksort algorithms
def test_quicksort(arr):
    """Helper function to test and time both quicksort algorithms."""
    # Measure time for deterministic quicksort
    start = time.time()
    quicksort(arr)
    deterministic_time = time.time() - start
    
    # Measure time for randomized quicksort
    start = time.time()
    quicksort_randomized(arr)
    randomized_time = time.time() - start
    
    return deterministic_time, randomized_time

# Function to generate arrays for performance testing
def generate_arrays():
    """Generate arrays of different sizes and types for testing."""
    sizes = [100, 1000, 5000, 10000]
    arrays = {}
    for size in sizes:
        # Generate reverse sorted arrays to simulate worst-case scenario
        arrays[size] = list(range(size, 0, -1))
    return arrays

# Main execution
if __name__ == "__main__":
    # Generate test arrays
    arrays = generate_arrays()
    
    # Test each array size and measure performance
    for size, arr in arrays.items():
        print(f"Testing with array size: {size}")
        
        # Get the time taken for both deterministic and randomized quicksort
        deterministic_time, randomized_time = test_quicksort(arr)
        
        # Print the results for comparison
        print(f"Deterministic Quicksort Time: {deterministic_time:.6f} seconds")
        print(f"Randomized Quicksort Time: {randomized_time:.6f} seconds")
        print("-" * 50)
    
    # Example of running the algorithms on a sample array
    example_arr = [3, 6, 8, 10, 1, 2, 1]
    print("Example array:", example_arr)
    print("Deterministic Quicksort result:", quicksort(example_arr))
    print("Randomized Quicksort result:", quicksort_randomized(example_arr))
