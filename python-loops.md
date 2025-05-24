# Python Loops and Control Flow

This document provides a comprehensive guide to all Python loop constructs, control flow statements, and related functions, methods, packages, and built-ins with syntax and usage examples.

## For Loops

### Basic For Loop Syntax
```python
# Basic iteration over sequence
for item in [1, 2, 3, 4, 5]:
    print(item)

# Iterate over string
for char in "hello":
    print(char)

# Iterate over dictionary
person = {"name": "Alice", "age": 30, "city": "New York"}
for key in person:
    print(key, person[key])

# Iterate over dictionary items
for key, value in person.items():
    print(f"{key}: {value}")

# Iterate over dictionary keys explicitly
for key in person.keys():
    print(key)

# Iterate over dictionary values
for value in person.values():
    print(value)
```

### Range Function in For Loops
```python
# range(stop)
for i in range(5):
    print(i)                                    # 0, 1, 2, 3, 4

# range(start, stop)
for i in range(2, 8):
    print(i)                                    # 2, 3, 4, 5, 6, 7

# range(start, stop, step)
for i in range(0, 10, 2):
    print(i)                                    # 0, 2, 4, 6, 8

# Negative step
for i in range(10, 0, -1):
    print(i)                                    # 10, 9, 8, 7, 6, 5, 4, 3, 2, 1

# Negative range
for i in range(-5, 0):
    print(i)                                    # -5, -4, -3, -2, -1
```

### Enumerate in For Loops
```python
fruits = ["apple", "banana", "orange"]

# Basic enumerate
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# 0: apple
# 1: banana
# 2: orange

# Enumerate with custom start
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}: {fruit}")
# 1: apple
# 2: banana
# 3: orange

# Enumerate with step (using range)
for i in range(0, len(fruits), 2):
    print(f"{i}: {fruits[i]}")
```

### Zip in For Loops
```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
cities = ["New York", "London", "Tokyo"]

# Zip two lists
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Zip multiple lists
for name, age, city in zip(names, ages, cities):
    print(f"{name}, {age}, lives in {city}")

# Zip with enumerate
for index, (name, age) in enumerate(zip(names, ages)):
    print(f"{index}: {name} ({age})")

# Zip with different lengths (stops at shortest)
short_list = [1, 2]
long_list = [1, 2, 3, 4, 5]
for a, b in zip(short_list, long_list):
    print(a, b)                                 # (1, 1), (2, 2)
```

### Nested For Loops
```python
# Basic nested loops
for i in range(3):
    for j in range(3):
        print(f"({i}, {j})")

# Matrix iteration
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for row in matrix:
    for element in row:
        print(element, end=" ")
    print()  # New line after each row

# Nested loops with enumerate
for i, row in enumerate(matrix):
    for j, element in enumerate(row):
        print(f"matrix[{i}][{j}] = {element}")

# List comprehension equivalent
flattened = [element for row in matrix for element in row]
```

### For Loop with Else
```python
# Else clause executes if loop completes normally (no break)
for i in range(5):
    print(i)
    if i == 10:  # This condition is never true
        break
else:
    print("Loop completed normally")            # This will execute

# Else clause doesn't execute if break is encountered
for i in range(5):
    print(i)
    if i == 3:
        break
else:
    print("This won't print")                   # This won't execute

# Practical example: searching
numbers = [1, 2, 3, 4, 5]
target = 6

for num in numbers:
    if num == target:
        print(f"Found {target}")
        break
else:
    print(f"{target} not found")                # This will execute
```

## While Loops

### Basic While Loop Syntax
```python
# Basic while loop
count = 0
while count < 5:
    print(count)
    count += 1

# While with complex condition
x, y = 10, 1
while x > 0 and y < 100:
    print(f"x: {x}, y: {y}")
    x -= 1
    y *= 2

# Infinite loop (be careful!)
# while True:
#     print("This runs forever")
#     break  # Use break to exit
```

### While Loop with Input
```python
# Input validation loop
while True:
    user_input = input("Enter a number (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    try:
        number = int(user_input)
        print(f"You entered: {number}")
        break
    except ValueError:
        print("Invalid input. Please enter a number.")

# Password validation
attempts = 3
while attempts > 0:
    password = input("Enter password: ")
    if password == "secret123":
        print("Access granted!")
        break
    else:
        attempts -= 1
        print(f"Incorrect password. {attempts} attempts remaining.")
else:
    print("Access denied!")
```

### While Loop with Else
```python
# Else clause executes if loop completes normally
count = 0
while count < 3:
    print(count)
    count += 1
else:
    print("While loop completed normally")      # This will execute

# Else clause doesn't execute if break is encountered
count = 0
while count < 10:
    print(count)
    if count == 2:
        break
    count += 1
else:
    print("This won't print")                   # This won't execute
```

## Control Flow Statements

### Break Statement
```python
# Break in for loop
for i in range(10):
    if i == 5:
        break
    print(i)                                    # Prints 0, 1, 2, 3, 4

# Break in while loop
count = 0
while True:
    if count == 3:
        break
    print(count)
    count += 1

# Break in nested loops (only breaks inner loop)
for i in range(3):
    for j in range(3):
        if j == 1:
            break                               # Only breaks inner loop
        print(f"({i}, {j})")

# Breaking out of nested loops using flag
found = False
for i in range(3):
    for j in range(3):
        if i == 1 and j == 1:
            found = True
            break
    if found:
        break
    print(f"Outer loop: {i}")

# Breaking out of nested loops using function
def search_matrix():
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                return f"Found at ({i}, {j})"   # Returns from function
        print(f"Finished row {i}")
    return "Not found"
```

### Continue Statement
```python
# Continue in for loop
for i in range(5):
    if i == 2:
        continue                                # Skip iteration when i == 2
    print(i)                                    # Prints 0, 1, 3, 4

# Continue in while loop
count = 0
while count < 5:
    count += 1
    if count == 3:
        continue                                # Skip when count == 3
    print(count)                                # Prints 1, 2, 4, 5

# Continue with complex condition
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for num in numbers:
    if num % 2 == 0:                            # Skip even numbers
        continue
    if num > 7:                                 # Skip numbers > 7
        continue
    print(num)                                  # Prints 1, 3, 5, 7

# Processing files example
import os
files = ["file1.txt", "file2.py", "file3.txt", "file4.jpg"]
for filename in files:
    if not filename.endswith('.txt'):
        continue                                # Skip non-txt files
    print(f"Processing {filename}")
```

### Pass Statement
```python
# Pass as placeholder
for i in range(5):
    if i == 2:
        pass                                    # Do nothing, placeholder
    else:
        print(i)

# Pass in function definition
def todo_function():
    pass                                        # Function body to be implemented

# Pass in class definition
class TodoClass:
    pass                                        # Class body to be implemented

# Pass in exception handling
try:
    risky_operation()
except SpecificException:
    pass                                        # Ignore this exception

# Pass with comment for clarity
for item in items:
    if item.is_valid():
        process_item(item)
    else:
        pass  # Invalid items are ignored for now
```

## Iterator Protocol

### Creating Custom Iterators
```python
# Iterator class
class CountUp:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start >= self.end:
            raise StopIteration
        current = self.start
        self.start += 1
        return current

# Using custom iterator
counter = CountUp(1, 5)
for num in counter:
    print(num)                                  # 1, 2, 3, 4

# Iterator with list
iterator_list = [1, 2, 3, 4, 5]
iter_obj = iter(iterator_list)
print(next(iter_obj))                           # 1
print(next(iter_obj))                           # 2
```

### Generator Functions
```python
# Simple generator
def count_up_to(max_count):
    count = 1
    while count <= max_count:
        yield count
        count += 1

# Using generator
for num in count_up_to(5):
    print(num)                                  # 1, 2, 3, 4, 5

# Generator with return value
def fibonacci(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1
    return "Fibonacci sequence complete"

# Using generator
fib = fibonacci(5)
for num in fib:
    print(num)                                  # 0, 1, 1, 2, 3

try:
    next(fib)
except StopIteration as e:
    print(e.value)                              # "Fibonacci sequence complete"

# Generator expression
squares = (x**2 for x in range(5))
for square in squares:
    print(square)                               # 0, 1, 4, 9, 16
```

### Built-in Iterator Functions

#### `iter(object, sentinel=None)`
```python
# Create iterator from iterable
numbers = [1, 2, 3, 4, 5]
number_iter = iter(numbers)
print(next(number_iter))                        # 1

# Iterator with sentinel value
import random
random.seed(42)
random_iter = iter(lambda: random.randint(1, 6), 6)  # Roll until 6
for roll in random_iter:
    print(f"Rolled: {roll}")
print("Rolled a 6!")

# File iterator
with open("file.txt", "r") as f:
    line_iter = iter(f.readline, "")            # Read until empty line
    for line in line_iter:
        print(line.strip())
```

#### `next(iterator, default=None)`
```python
numbers = iter([1, 2, 3])

# Basic next
print(next(numbers))                            # 1
print(next(numbers))                            # 2
print(next(numbers))                            # 3

# Next with default (prevents StopIteration)
print(next(numbers, "No more items"))           # "No more items"

# Manual iteration
def manual_iteration(iterable):
    iterator = iter(iterable)
    while True:
        try:
            item = next(iterator)
            print(f"Processing: {item}")
        except StopIteration:
            print("Iteration complete")
            break
```

## Itertools Module

### Infinite Iterators
```python
import itertools

# count() - infinite arithmetic sequence
counter = itertools.count(10, 2)                # 10, 12, 14, 16, ...
for i, value in enumerate(counter):
    if i >= 5:
        break
    print(value)                                # 10, 12, 14, 16, 18

# cycle() - infinite repetition
colors = itertools.cycle(['red', 'green', 'blue'])
for i, color in enumerate(colors):
    if i >= 7:
        break
    print(color)                                # red, green, blue, red, green, blue, red

# repeat() - repeat value
repeater = itertools.repeat('hello', 3)
for word in repeater:
    print(word)                                 # hello, hello, hello

# Using infinite iterators in loops
for x, y in zip(range(5), itertools.count(10, 2)):
    print(f"{x}: {y}")                          # 0: 10, 1: 12, 2: 14, 3: 16, 4: 18
```

### Terminating Iterators
```python
import itertools
import operator

# accumulate() - cumulative operation
numbers = [1, 2, 3, 4, 5]
cumulative = itertools.accumulate(numbers)
for value in cumulative:
    print(value)                                # 1, 3, 6, 10, 15

# accumulate with custom function
products = itertools.accumulate(numbers, operator.mul)
for value in products:
    print(value)                                # 1, 2, 6, 24, 120

# chain() - flatten multiple iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list3 = [7, 8, 9]
chained = itertools.chain(list1, list2, list3)
for value in chained:
    print(value)                                # 1, 2, 3, 4, 5, 6, 7, 8, 9

# dropwhile() and takewhile()
numbers = [1, 3, 5, 8, 9, 10, 11, 13]
# Drop while condition is true
dropped = itertools.dropwhile(lambda x: x < 8, numbers)
for value in dropped:
    print(value)                                # 8, 9, 10, 11, 13

# Take while condition is true
taken = itertools.takewhile(lambda x: x < 8, numbers)
for value in taken:
    print(value)                                # 1, 3, 5

# groupby() - group consecutive equal elements
data = [1, 1, 2, 2, 2, 3, 1, 1]
for key, group in itertools.groupby(data):
    print(f"{key}: {list(group)}")              # 1: [1, 1], 2: [2, 2, 2], 3: [3], 1: [1, 1]
```

### Combinatorial Iterators
```python
import itertools

# product() - Cartesian product
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
for color, size in itertools.product(colors, sizes):
    print(f"{color} {size}")                    # red S, red M, red L, blue S, blue M, blue L

# permutations() - all permutations
letters = ['A', 'B', 'C']
for perm in itertools.permutations(letters, 2):
    print(perm)                                 # ('A', 'B'), ('A', 'C'), ('B', 'A'), etc.

# combinations() - combinations without repetition
for combo in itertools.combinations(letters, 2):
    print(combo)                                # ('A', 'B'), ('A', 'C'), ('B', 'C')

# combinations_with_replacement() - combinations with repetition
for combo in itertools.combinations_with_replacement(['A', 'B'], 2):
    print(combo)                                # ('A', 'A'), ('A', 'B'), ('B', 'B')
```

## Loop Patterns and Idioms

### Common Loop Patterns
```python
# Processing pairs of consecutive elements
numbers = [1, 2, 3, 4, 5]
for current, next_val in zip(numbers, numbers[1:]):
    print(f"{current} -> {next_val}")

# Processing with previous value
def process_with_previous(iterable):
    iterator = iter(iterable)
    previous = next(iterator)
    for current in iterator:
        yield previous, current
        previous = current

for prev, curr in process_with_previous([1, 2, 3, 4, 5]):
    print(f"{prev} -> {curr}")

# Chunking data
def chunk_data(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

data = list(range(20))
for chunk in chunk_data(data, 5):
    print(chunk)                                # [0,1,2,3,4], [5,6,7,8,9], etc.

# Parallel iteration with different step sizes
list1 = [1, 2, 3, 4, 5, 6, 7, 8]
list2 = ['a', 'b', 'c', 'd']
for i, (num, letter) in enumerate(zip(list1[::2], list2)):
    print(f"{i}: {num}, {letter}")
```

### Loop with Multiple Conditions
```python
# Multiple break conditions
for i in range(100):
    if i > 50 and i % 7 == 0:                  # Multiple conditions for break
        print(f"Breaking at {i}")
        break
    if i % 10 == 0:
        print(f"Milestone: {i}")

# Complex continue conditions
numbers = range(50)
for num in numbers:
    # Skip numbers divisible by 2 OR 3 OR greater than 30
    if num % 2 == 0 or num % 3 == 0 or num > 30:
        continue
    print(num)

# State-based loop control
state = "start"
for i in range(10):
    if state == "start" and i >= 3:
        state = "middle"
        print("Entered middle state")
    elif state == "middle" and i >= 7:
        state = "end"
        print("Entered end state")
    
    print(f"i={i}, state={state}")
```

### Exception Handling in Loops
```python
# Continue on exceptions
numbers = ["1", "2", "invalid", "4", "5"]
for num_str in numbers:
    try:
        num = int(num_str)
        print(f"Processed: {num}")
    except ValueError:
        print(f"Skipping invalid value: {num_str}")
        continue

# Accumulate errors
errors = []
results = []
for num_str in numbers:
    try:
        result = int(num_str) * 2
        results.append(result)
    except ValueError as e:
        errors.append(f"Error with {num_str}: {e}")

print(f"Results: {results}")
print(f"Errors: {errors}")

# Break on specific exceptions
for i in range(10):
    try:
        if i == 5:
            raise ValueError("Critical error")
        print(f"Processing {i}")
    except ValueError:
        print("Critical error encountered, stopping loop")
        break
```

## Loop Performance and Optimization

### Efficient Loop Patterns
```python
import time

# List comprehension vs loop
def time_comparison():
    # Traditional loop
    start = time.time()
    result1 = []
    for i in range(100000):
        if i % 2 == 0:
            result1.append(i**2)
    time1 = time.time() - start
    
    # List comprehension
    start = time.time()
    result2 = [i**2 for i in range(100000) if i % 2 == 0]
    time2 = time.time() - start
    
    print(f"Loop time: {time1:.4f}s")
    print(f"List comprehension time: {time2:.4f}s")

# Generator vs list for memory efficiency
def memory_efficient_processing():
    # Memory-heavy list
    def get_all_squares():
        return [x**2 for x in range(1000000)]
    
    # Memory-efficient generator
    def get_squares_generator():
        return (x**2 for x in range(1000000))
    
    # Process with generator
    squares_gen = get_squares_generator()
    for i, square in enumerate(squares_gen):
        if i >= 10:  # Process only first 10
            break
        print(square)
```

### Avoiding Common Pitfalls
```python
# DON'T modify list while iterating
numbers = [1, 2, 3, 4, 5]
# Bad - can cause issues
# for num in numbers:
#     if num % 2 == 0:
#         numbers.remove(num)  # Don't do this!

# Good - iterate over copy or use list comprehension
for num in numbers.copy():  # or numbers[:]
    if num % 2 == 0:
        numbers.remove(num)

# Better - use list comprehension
numbers = [num for num in numbers if num % 2 != 0]

# DON'T create functions inside loops unnecessarily
# Bad
results = []
for i in range(1000):
    def square(x):  # Function created 1000 times!
        return x**2
    results.append(square(i))

# Good
def square(x):
    return x**2

results = []
for i in range(1000):
    results.append(square(i))

# Best
results = [i**2 for i in range(1000)]
```

## Advanced Loop Constructs

### Context Managers in Loops
```python
# Multiple file processing
filenames = ["file1.txt", "file2.txt", "file3.txt"]

for filename in filenames:
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if 'error' in line.lower():
                    print(f"{filename}:{line_num}: {line.strip()}")
    except FileNotFoundError:
        print(f"File {filename} not found")

# Custom context manager in loop
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        print(f"Operation took {time.time() - self.start:.2f} seconds")

operations = [lambda: sum(range(1000)), lambda: [x**2 for x in range(1000)]]
for i, operation in enumerate(operations):
    with Timer():
        result = operation()
        print(f"Operation {i} completed")
```

### Parallel Loop Processing
```python
import concurrent.futures
import threading
import multiprocessing

# Thread-based parallel processing
def process_item(item):
    # Simulate some work
    import time
    time.sleep(0.1)
    return item ** 2

items = list(range(10))

# Sequential processing
start = time.time()
results_sequential = [process_item(item) for item in items]
print(f"Sequential: {time.time() - start:.2f}s")

# Parallel processing with threads
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results_parallel = list(executor.map(process_item, items))
print(f"Parallel (threads): {time.time() - start:.2f}s")

# Parallel processing with processes
start = time.time()
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    results_parallel = list(executor.map(process_item, items))
print(f"Parallel (processes): {time.time() - start:.2f}s")
```

### Async Loops (asyncio)
```python
import asyncio
import aiohttp

# Async iteration
async def async_range(start, stop):
    for i in range(start, stop):
        await asyncio.sleep(0.1)  # Simulate async work
        yield i

async def process_async_sequence():
    async for value in async_range(0, 5):
        print(f"Async value: {value}")

# Run async function
# asyncio.run(process_async_sequence())

# Async loop with HTTP requests
async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_multiple_urls():
    urls = [
        'http://httpbin.org/delay/1',
        'http://httpbin.org/delay/2',
        'http://httpbin.org/delay/1'
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

# asyncio.run(fetch_multiple_urls())
```

## Loop Debugging and Profiling

### Debugging Loops
```python
import logging

# Setup logging for loop debugging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

def debug_loop_example():
    numbers = [1, 2, 3, 4, 5]
    total = 0
    
    for i, num in enumerate(numbers):
        logging.debug(f"Iteration {i}: processing {num}")
        total += num
        logging.debug(f"Running total: {total}")
        
        if total > 10:
            logging.warning(f"Total exceeded 10 at iteration {i}")
            break
    
    return total

# Conditional debugging
DEBUG = True
for i in range(10):
    if DEBUG and i % 2 == 0:
        print(f"Debug: Processing even number {i}")
    # Regular processing
    result = i ** 2
```

### Loop Profiling
```python
import cProfile
import timeit

# Profile loop performance
def profile_loops():
    # Different loop implementations
    def method1():
        result = []
        for i in range(10000):
            result.append(i**2)
        return result
    
    def method2():
        return [i**2 for i in range(10000)]
    
    def method3():
        return list(map(lambda x: x**2, range(10000)))
    
    # Time each method
    time1 = timeit.timeit(method1, number=100)
    time2 = timeit.timeit(method2, number=100)
    time3 = timeit.timeit(method3, number=100)
    
    print(f"Method 1 (loop): {time1:.4f}s")
    print(f"Method 2 (comprehension): {time2:.4f}s")
    print(f"Method 3 (map): {time3:.4f}s")

# Memory profiling
import sys

def memory_comparison():
    # Generator (memory efficient)
    gen = (x**2 for x in range(100000))
    print(f"Generator size: {sys.getsizeof(gen)} bytes")
    
    # List (memory heavy)
    lst = [x**2 for x in range(100000)]
    print(f"List size: {sys.getsizeof(lst)} bytes")
```

## Specialized Loop Libraries

### more-itertools (Third-party)
```python
# Install: pip install more-itertools
# import more_itertools as mit

# # Chunking
# data = range(20)
# for chunk in mit.chunked(data, 5):
#     print(list(chunk))  # [0,1,2,3,4], [5,6,7,8,9], etc.

# # Windowed iteration
# for window in mit.windowed(range(10), 3):
#     print(window)  # (0,1,2), (1,2,3), (2,3,4), etc.

# # Flatten nested structures
# nested = [[1, 2], [3, 4], [5, 6]]
# flattened = list(mit.flatten(nested))  # [1, 2, 3, 4, 5, 6]

# # Partition data
# def is_even(x):
#     return x % 2 == 0

# evens, odds = mit.partition(is_even, range(10))
# print(list(evens))  # [1, 3, 5, 7, 9]
# print(list(odds))   # [0, 2, 4, 6, 8]
```

### NumPy Vectorized Operations
```python
import numpy as np

# Vectorized operations (avoid explicit loops)
def numpy_vs_loops():
    # Pure Python loop
    def python_loop(arr):
        result = []
        for x in arr:
            result.append(x**2 + 2*x + 1)
        return result
    
    # NumPy vectorized
    def numpy_vectorized(arr):
        return arr**2 + 2*arr + 1
    
    # Compare performance
    data = list(range(100000))
    np_data = np.array(data)
    
    import time
    
    # Python loop
    start = time.time()
    result1 = python_loop(data)
    time1 = time.time() - start
    
    # NumPy vectorized
    start = time.time()
    result2 = numpy_vectorized(np_data)
    time2 = time.time() - start
    
    print(f"Python loop: {time1:.4f}s")
    print(f"NumPy vectorized: {time2:.4f}s")
    print(f"Speedup: {time1/time2:.2f}x")

# numpy_vs_loops()
```

## Best Practices and Guidelines

### Loop Best Practices
```python
# 1. Use appropriate data structures
# Good: Use set for membership testing
valid_ids = {1, 2, 3, 4, 5}
for user_id in user_ids:
    if user_id in valid_ids:  # O(1) lookup
        process_user(user_id)

# Bad: Use list for membership testing
# valid_ids = [1, 2, 3, 4, 5]
# for user_id in user_ids:
#     if user_id in valid_ids:  # O(n) lookup
#         process_user(user_id)

# 2. Minimize work inside loops
# Good: Move invariant calculations outside
base_value = calculate_base()  # Do this once
results = []
for item in items:
    result = item * base_value  # Use pre-calculated value
    results.append(result)

# 3. Use enumerate instead of manual indexing
# Good
for index, item in enumerate(items):
    print(f"{index}: {item}")

# Less good
for i in range(len(items)):
    print(f"{i}: {items[i]}")

# 4. Use zip for parallel iteration
# Good
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Less good
for i in range(len(names)):
    print(f"{names[i]} is {ages[i]} years old")

# 5. Use list comprehensions for simple transformations
# Good
squares = [x**2 for x in numbers if x > 0]

# Less good
squares = []
for x in numbers:
    if x > 0:
        squares.append(x**2)
```

### When to Use Different Loop Types
```python
# Use for loops when:
# - You know the number of iterations
# - You're iterating over a sequence
# - You need the index

# Use while loops when:
# - The number of iterations is unknown
# - You're waiting for a condition
# - You're implementing algorithms with complex termination conditions

# Use list comprehensions when:
# - Creating new lists from existing iterables
# - Simple transformations and filtering
# - Readability is improved

# Use generator expressions when:
# - Working with large datasets
# - Memory efficiency is important
# - You don't need all results immediately

# Use map/filter when:
# - Applying functions to sequences
# - Working with multiple iterables
# - Functional programming style is preferred
```

---

*This document covers comprehensive loop constructs and control flow in Python including for/while loops, control statements, iterators, generators, itertools, advanced patterns, and performance considerations. For the most up-to-date information, refer to the official Python documentation.*