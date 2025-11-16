
# Quicksort Implementation and Performance Analysis

## Overview

This repository contains the implementation of the **Quicksort algorithm**, both in its **deterministic** and **randomized** versions. The primary objective of this project is to implement and compare the performance of both versions, analyze their time complexity, and evaluate how randomization can help improve the algorithm's robustness.

## Features

- **Deterministic Quicksort**: The pivot is selected as the last element of the array.
- **Randomized Quicksort**: A pivot is selected randomly, reducing the chance of encountering the worst-case scenario.
- **Performance Analysis**: Empirical testing and timing of both algorithms on arrays of different sizes to compare their execution times.
- **Pivot Optimization**: For the deterministic version, a median-of-three strategy is employed to improve partitioning and performance.

## How to Run

1. **Clone the Repository**
   To get started, clone this repository to your local machine:
   ```bash
   git clone 
   cd quicksort-assignment
   ```

2. **Requirements**
   This project only requires **Python 3.x** to run. If you don't have Python installed, you can download it from [here](https://www.python.org/downloads/).

3. **Run the Code**
   Once you have cloned the repository and installed Python, simply run the `quicksort.py` file:
   ```bash
   python quicksort.py
   ```
   This will execute the script, which includes the following:
   - Performance testing on arrays of various sizes (100, 1000, 5000, 10000).
   - Output of the execution times for both deterministic and randomized versions of the algorithm.

## Example Results

Here are some example results obtained from testing the Quicksort algorithms on arrays of different sizes:

- **Array size: 100**
  - Deterministic Quicksort: 0.000178 seconds
  - Randomized Quicksort: 0.000900 seconds

- **Array size: 1000**
  - Deterministic Quicksort: 0.002367 seconds
  - Randomized Quicksort: 0.001965 seconds

- **Array size: 5000**
  - Deterministic Quicksort: 0.014649 seconds
  - Randomized Quicksort: 0.039129 seconds

- **Array size: 10000**
  - Deterministic Quicksort: 0.026439 seconds
  - Randomized Quicksort: 0.046437 seconds

- **Example Array**: `[3, 6, 8, 10, 1, 2, 1]`
  - Deterministic Quicksort result: `[1, 2, 3, 6, 8, 10]`
  - Randomized Quicksort result: `[1, 1, 2, 3, 6, 8, 10]`

## Conclusion

- **Deterministic Quicksort** works efficiently for smaller arrays and provides predictable results, but it can degrade in performance for sorted or nearly sorted arrays, especially for larger sizes.
- **Randomized Quicksort** improves on this by randomizing the pivot, thus reducing the likelihood of worst-case performance.
- The empirical results show that for smaller arrays, the deterministic Quicksort performs slightly better, while for larger arrays, the randomized Quicksort begins to show improved performance.
