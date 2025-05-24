# Python Arrays, Lists, and Sequence Operations

This document provides a comprehensive guide to all Python array, list, enumerate, and related functions, methods, packages, and built-ins with syntax and usage examples.

## Lists (Built-in Sequence Type)

### List Creation
```python
# Empty list
empty_list = []
empty_list = list()

# List with initial values
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
nested = [[1, 2], [3, 4], [5, 6]]

# List from other iterables
from_string = list("hello")                     # ['h', 'e', 'l', 'l', 'o']
from_range = list(range(5))                     # [0, 1, 2, 3, 4]
from_tuple = list((1, 2, 3))                    # [1, 2, 3]

# List comprehensions
squares = [x**2 for x in range(5)]              # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0]   # [0, 2, 4, 6, 8]
nested_comp = [[x, x**2] for x in range(3)]    # [[0, 0], [1, 1], [2, 4]]
```

### List Methods

#### Adding Elements

##### `list.append(item)`
Adds an item to the end of the list.
```python
fruits = ["apple", "banana"]
fruits.append("orange")                         # ["apple", "banana", "orange"]
fruits.append(["grape", "mango"])               # ["apple", "banana", "orange", ["grape", "mango"]]
```

##### `list.insert(index, item)`
Inserts an item at the specified position.
```python
fruits = ["apple", "banana", "orange"]
fruits.insert(1, "grape")                       # ["apple", "grape", "banana", "orange"]
fruits.insert(0, "mango")                       # ["mango", "apple", "grape", "banana", "orange"]
fruits.insert(-1, "kiwi")                       # Insert before last element
```

##### `list.extend(iterable)`
Extends the list by appending all items from the iterable.
```python
fruits = ["apple", "banana"]
fruits.extend(["orange", "grape"])              # ["apple", "banana", "orange", "grape"]
fruits.extend("hi")                             # ["apple", "banana", "orange", "grape", "h", "i"]
fruits.extend(range(3))                         # Adds [0, 1, 2]
```

#### Removing Elements

##### `list.remove(item)`
Removes the first occurrence of the specified item.
```python
fruits = ["apple", "banana", "apple", "orange"]
fruits.remove("apple")                          # ["banana", "apple", "orange"]
# fruits.remove("grape")                        # ValueError: not in list
```

##### `list.pop(index=-1)`
Removes and returns the item at the specified position (last item by default).
```python
fruits = ["apple", "banana", "orange"]
last = fruits.pop()                             # Returns "orange", list: ["apple", "banana"]
first = fruits.pop(0)                           # Returns "apple", list: ["banana"]
# empty_list.pop()                              # IndexError: pop from empty list
```

##### `list.clear()`
Removes all items from the list.
```python
fruits = ["apple", "banana", "orange"]
fruits.clear()                                  # []
```

##### `del` statement
Removes items by index or slice.
```python
fruits = ["apple", "banana", "orange", "grape"]
del fruits[1]                                   # ["apple", "orange", "grape"]
del fruits[0:2]                                 # ["grape"]
del fruits[:]                                   # [] (clear all)
```

#### Finding and Counting

##### `list.index(item, start=0, end=len(list))`
Returns the index of the first occurrence of the item.
```python
fruits = ["apple", "banana", "apple", "orange"]
index = fruits.index("apple")                   # Returns 0
index = fruits.index("apple", 1)                # Returns 2 (start from index 1)
# fruits.index("grape")                         # ValueError: not in list
```

##### `list.count(item)`
Returns the number of occurrences of the item.
```python
fruits = ["apple", "banana", "apple", "orange"]
count = fruits.count("apple")                   # Returns 2
count = fruits.count("grape")                   # Returns 0
```

##### `in` and `not in` operators
Check if item exists in list.
```python
fruits = ["apple", "banana", "orange"]
"apple" in fruits                               # True
"grape" in fruits                               # False
"grape" not in fruits                           # True
```

#### Sorting and Reversing

##### `list.sort(key=None, reverse=False)`
Sorts the list in place.
```python
numbers = [3, 1, 4, 1, 5, 9]
numbers.sort()                                  # [1, 1, 3, 4, 5, 9]
numbers.sort(reverse=True)                      # [9, 5, 4, 3, 1, 1]

words = ["apple", "Banana", "cherry"]
words.sort()                                    # ["Banana", "apple", "cherry"]
words.sort(key=str.lower)                       # ["apple", "Banana", "cherry"]
words.sort(key=len)                             # ["apple", "cherry", "Banana"]

# Complex sorting
students = [("Alice", 85), ("Bob", 90), ("Charlie", 78)]
students.sort(key=lambda x: x[1])               # Sort by grade
```

##### `list.reverse()`
Reverses the list in place.
```python
fruits = ["apple", "banana", "orange"]
fruits.reverse()                                # ["orange", "banana", "apple"]
```

#### Copying

##### `list.copy()`
Returns a shallow copy of the list.
```python
original = [1, 2, [3, 4]]
copied = original.copy()                        # Shallow copy
copied[0] = 99                                  # original unchanged
copied[2][0] = 99                               # original[2] also changes!

# Deep copy
import copy
deep_copied = copy.deepcopy(original)
```

### List Operations

#### Concatenation and Repetition
```python
# Concatenation
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = list1 + list2                        # [1, 2, 3, 4, 5, 6]

# Augmented assignment
list1 += list2                                  # list1 becomes [1, 2, 3, 4, 5, 6]

# Repetition
repeated = [1, 2] * 3                           # [1, 2, 1, 2, 1, 2]
zeros = [0] * 5                                 # [0, 0, 0, 0, 0]

# Be careful with mutable objects
matrix = [[0] * 3] * 3                          # Wrong! All rows are the same object
matrix = [[0] * 3 for _ in range(3)]            # Correct
```

#### Indexing and Slicing
```python
fruits = ["apple", "banana", "orange", "grape", "mango"]

# Indexing
first = fruits[0]                               # "apple"
last = fruits[-1]                               # "mango"
second_last = fruits[-2]                        # "grape"

# Slicing
subset = fruits[1:4]                            # ["banana", "orange", "grape"]
first_three = fruits[:3]                        # ["apple", "banana", "orange"]
last_two = fruits[-2:]                          # ["grape", "mango"]
every_second = fruits[::2]                      # ["apple", "orange", "mango"]
reversed_list = fruits[::-1]                    # Reverse order

# Slice assignment
fruits[1:3] = ["kiwi", "peach"]                 # Replace elements
fruits[1:1] = ["inserted"]                      # Insert elements
fruits[1:3] = []                                # Delete elements
```

#### Length and Boolean Context
```python
fruits = ["apple", "banana", "orange"]

# Length
length = len(fruits)                            # 3

# Boolean context
if fruits:                                      # True if not empty
    print("List has items")

empty_list = []
if not empty_list:                              # True if empty
    print("List is empty")
```

## Built-in Functions for Sequences

### `enumerate(iterable, start=0)`
Returns an enumerate object with index-value pairs.
```python
fruits = ["apple", "banana", "orange"]

# Basic enumeration
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# 0: apple
# 1: banana
# 2: orange

# Custom start value
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}: {fruit}")
# 1: apple
# 2: banana
# 3: orange

# Convert to list
enumerated = list(enumerate(fruits))            # [(0, 'apple'), (1, 'banana'), (2, 'orange')]

# Enumerate with unpacking
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
for i, (name, age) in enumerate(data):
    print(f"Person {i}: {name} is {age} years old")
```

### `zip(*iterables)`
Returns an iterator of tuples from multiple iterables.
```python
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]
cities = ["New York", "London", "Tokyo"]

# Basic zip
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Multiple iterables
for name, age, city in zip(names, ages, cities):
    print(f"{name}, {age}, lives in {city}")

# Convert to list
pairs = list(zip(names, ages))                  # [('Alice', 25), ('Bob', 30), ('Charlie', 35)]

# Unzip
names, ages = zip(*pairs)                       # Unpack tuples back to separate lists

# Zip with different lengths (stops at shortest)
short_list = [1, 2]
long_list = [1, 2, 3, 4, 5]
result = list(zip(short_list, long_list))       # [(1, 1), (2, 2)]

# Zip longest (requires itertools)
import itertools
result = list(itertools.zip_longest(short_list, long_list, fillvalue=0))
# [(1, 1), (2, 2), (0, 3), (0, 4), (0, 5)]
```

### `map(function, iterable, ...)`
Applies a function to every item of one or more iterables.
```python
numbers = [1, 2, 3, 4, 5]

# Apply function to each element
squared = list(map(lambda x: x**2, numbers))    # [1, 4, 9, 16, 25]
strings = list(map(str, numbers))               # ['1', '2', '3', '4', '5']

# Multiple iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
sums = list(map(lambda x, y: x + y, list1, list2))  # [5, 7, 9]

# Using built-in functions
words = ["hello", "world", "python"]
lengths = list(map(len, words))                 # [5, 5, 6]
upper_words = list(map(str.upper, words))       # ['HELLO', 'WORLD', 'PYTHON']
```

### `filter(function, iterable)`
Filters elements based on a function that returns True/False.
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Filter even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4, 6, 8, 10]

# Filter positive numbers
mixed = [-2, -1, 0, 1, 2, 3]
positive = list(filter(lambda x: x > 0, mixed))      # [1, 2, 3]

# Filter None values (function=None removes falsy values)
mixed_data = [1, 0, "hello", "", None, False, "world"]
filtered = list(filter(None, mixed_data))            # [1, 'hello', 'world']

# Filter strings by length
words = ["a", "hello", "hi", "python", "code"]
long_words = list(filter(lambda x: len(x) > 3, words))  # ['hello', 'python', 'code']
```

### `sorted(iterable, key=None, reverse=False)`
Returns a new sorted list from the items in iterable.
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = sorted(numbers)                # [1, 1, 2, 3, 4, 5, 6, 9]
reverse_sorted = sorted(numbers, reverse=True)  # [9, 6, 5, 4, 3, 2, 1, 1]

# Sort strings
words = ["apple", "Banana", "cherry", "Date"]
sorted_words = sorted(words)                    # ['Banana', 'Date', 'apple', 'cherry']
case_insensitive = sorted(words, key=str.lower) # ['apple', 'Banana', 'cherry', 'Date']

# Sort by length
by_length = sorted(words, key=len)              # ['Date', 'apple', 'cherry', 'Banana']

# Sort complex objects
students = [("Alice", 85), ("Bob", 90), ("Charlie", 78)]
by_grade = sorted(students, key=lambda x: x[1]) # [('Charlie', 78), ('Alice', 85), ('Bob', 90)]
by_name = sorted(students, key=lambda x: x[0])  # [('Alice', 85), ('Bob', 90), ('Charlie', 78)]

# Multiple sort criteria
from operator import itemgetter
data = [("Alice", 25, 85), ("Bob", 30, 85), ("Charlie", 25, 90)]
sorted_data = sorted(data, key=itemgetter(2, 1)) # Sort by grade, then age
```

### `reversed(seq)`
Returns a reverse iterator.
```python
numbers = [1, 2, 3, 4, 5]
reversed_numbers = list(reversed(numbers))      # [5, 4, 3, 2, 1]

# Iterate in reverse
for num in reversed(numbers):
    print(num)  # 5, 4, 3, 2, 1

# Reverse string
text = "hello"
reversed_text = "".join(reversed(text))         # "olleh"
```

### `min()` and `max()`
Find minimum and maximum values.
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]

# Basic min/max
minimum = min(numbers)                          # 1
maximum = max(numbers)                          # 9

# Multiple arguments
minimum = min(3, 1, 4, 1, 5)                   # 1
maximum = max(3, 1, 4, 1, 5)                   # 5

# With key function
words = ["apple", "banana", "cherry"]
shortest = min(words, key=len)                  # "apple"
longest = max(words, key=len)                   # "banana"

# Complex objects
students = [("Alice", 85), ("Bob", 90), ("Charlie", 78)]
best_student = max(students, key=lambda x: x[1])  # ("Bob", 90)
worst_student = min(students, key=lambda x: x[1]) # ("Charlie", 78)

# Default value for empty iterables
empty_list = []
result = min(empty_list, default=0)             # Returns 0 instead of error
```

### `sum(iterable, start=0)`
Sums all items in an iterable.
```python
numbers = [1, 2, 3, 4, 5]
total = sum(numbers)                            # 15
total_with_start = sum(numbers, 10)             # 25

# Sum of specific attributes
prices = [("apple", 1.50), ("banana", 0.80), ("orange", 2.00)]
total_price = sum(price for name, price in prices)  # 4.30

# Flatten list of lists
nested = [[1, 2], [3, 4], [5, 6]]
flattened = sum(nested, [])                     # [1, 2, 3, 4, 5, 6]
```

### `all(iterable)` and `any(iterable)`
Test if all/any elements are true.
```python
# all() - returns True if all elements are true
numbers = [1, 2, 3, 4, 5]
all_positive = all(x > 0 for x in numbers)     # True
all_even = all(x % 2 == 0 for x in numbers)    # False

boolean_list = [True, True, True]
all_true = all(boolean_list)                    # True

empty_result = all([])                          # True (vacuously true)

# any() - returns True if any element is true
mixed = [0, 1, 2]
any_truthy = any(mixed)                         # True
any_positive = any(x > 0 for x in mixed)       # True

boolean_list = [False, False, True]
any_true = any(boolean_list)                    # True

empty_result = any([])                          # False
```

## Tuples

### Tuple Creation and Operations
```python
# Empty tuple
empty_tuple = ()
empty_tuple = tuple()

# Tuple with values
coordinates = (3, 4)
colors = ("red", "green", "blue")
mixed = (1, "hello", 3.14, True)

# Single element tuple (note the comma)
single = (42,)                                  # Tuple with one element
not_tuple = (42)                                # This is just an integer

# Tuple from other iterables
from_list = tuple([1, 2, 3])                   # (1, 2, 3)
from_string = tuple("hello")                    # ('h', 'e', 'l', 'l', 'o')

# Tuple unpacking
point = (3, 4)
x, y = point                                    # x=3, y=4

# Extended unpacking
numbers = (1, 2, 3, 4, 5)
first, *middle, last = numbers                  # first=1, middle=[2,3,4], last=5
first, second, *rest = numbers                  # first=1, second=2, rest=[3,4,5]
```

### Tuple Methods
```python
numbers = (1, 2, 3, 2, 4, 2, 5)

# count() - count occurrences
count_twos = numbers.count(2)                   # 3

# index() - find first occurrence
index_of_four = numbers.index(4)                # 4
# numbers.index(6)                              # ValueError: not in tuple
```

### Named Tuples
```python
from collections import namedtuple

# Define named tuple
Point = namedtuple('Point', ['x', 'y'])
Person = namedtuple('Person', 'name age city')   # Space-separated fields

# Create instances
p1 = Point(3, 4)
p2 = Point(x=1, y=2)

person = Person("Alice", 30, "New York")

# Access fields
print(p1.x, p1.y)                               # 3 4
print(person.name)                              # Alice

# Named tuple methods
print(p1._fields)                               # ('x', 'y')
point_dict = p1._asdict()                       # {'x': 3, 'y': 4}
new_point = p1._replace(x=10)                   # Point(x=10, y=4)

# Create from iterable
coordinates = [5, 6]
p3 = Point._make(coordinates)                   # Point(x=5, y=6)
```

## Arrays (`array` module)

### Array Creation and Types
```python
import array

# Type codes
# 'b': signed char (1 byte)
# 'B': unsigned char (1 byte)
# 'h': signed short (2 bytes)
# 'H': unsigned short (2 bytes)
# 'i': signed int (4 bytes)
# 'I': unsigned int (4 bytes)
# 'l': signed long (4 bytes)
# 'L': unsigned long (4 bytes)
# 'f': float (4 bytes)
# 'd': double (8 bytes)

# Create arrays
int_array = array.array('i', [1, 2, 3, 4, 5])
float_array = array.array('f', [1.1, 2.2, 3.3])
char_array = array.array('b', [65, 66, 67])     # ASCII values

# From other iterables
from_range = array.array('i', range(10))
from_list = array.array('d', [1.0, 2.0, 3.0])

# Empty array
empty_array = array.array('i')
```

### Array Methods
```python
import array

arr = array.array('i', [1, 2, 3, 4, 5])

# Adding elements
arr.append(6)                                   # [1, 2, 3, 4, 5, 6]
arr.insert(0, 0)                                # [0, 1, 2, 3, 4, 5, 6]
arr.extend([7, 8, 9])                           # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Removing elements
arr.remove(5)                                   # Remove first occurrence of 5
popped = arr.pop()                              # Remove and return last element
popped_index = arr.pop(0)                       # Remove and return element at index 0

# Finding elements
index = arr.index(3)                            # Find index of first occurrence
count = arr.count(2)                            # Count occurrences

# Reversing
arr.reverse()                                   # Reverse in place

# Converting
as_list = arr.tolist()                          # Convert to list
as_bytes = arr.tobytes()                        # Convert to bytes
as_string = arr.tostring()                      # Convert to string (deprecated)

# Creating from bytes
new_arr = array.array('i')
new_arr.frombytes(as_bytes)

# Buffer info
buffer_info = arr.buffer_info()                 # (address, length)
```

## Collections Module

### `collections.deque` (Double-ended queue)
```python
from collections import deque

# Create deque
dq = deque([1, 2, 3, 4, 5])
dq_with_maxlen = deque(maxlen=3)                # Fixed size deque

# Adding elements
dq.append(6)                                    # Add to right: [1, 2, 3, 4, 5, 6]
dq.appendleft(0)                                # Add to left: [0, 1, 2, 3, 4, 5, 6]
dq.extend([7, 8])                               # Extend right: [..., 6, 7, 8]
dq.extendleft([-2, -1])                         # Extend left: [-1, -2, 0, 1, ...]

# Removing elements
right = dq.pop()                                # Remove from right
left = dq.popleft()                             # Remove from left

# Rotating
dq.rotate(1)                                    # Rotate right by 1
dq.rotate(-2)                                   # Rotate left by 2

# Other operations
dq.reverse()                                    # Reverse in place
count = dq.count(2)                             # Count occurrences
dq.clear()                                      # Remove all elements

# Fixed size deque
fixed_dq = deque(maxlen=3)
fixed_dq.extend([1, 2, 3, 4, 5])               # Only keeps last 3: [3, 4, 5]
```

### `collections.Counter`
```python
from collections import Counter

# Create counter
counter = Counter([1, 2, 3, 2, 3, 3])          # Counter({3: 3, 2: 2, 1: 1})
counter = Counter("hello world")                # Count characters
counter = Counter(a=3, b=1)                     # From keyword arguments

# Counter operations
most_common = counter.most_common()             # List of (element, count) pairs
top_three = counter.most_common(3)              # Top 3 most common

# Update counter
counter.update([1, 2, 3])                      # Add counts
counter.subtract([1, 2])                       # Subtract counts

# Counter arithmetic
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)                                  # Counter({'a': 4, 'b': 3})
print(c1 - c2)                                  # Counter({'a': 2})
print(c1 & c2)                                  # Intersection: Counter({'a': 1, 'b': 1})
print(c1 | c2)                                  # Union: Counter({'a': 3, 'b': 2})

# Convert to other types
elements = list(counter.elements())             # List of all elements
keys = list(counter.keys())                     # List of unique elements
values = list(counter.values())                 # List of counts
```

### `collections.defaultdict`
```python
from collections import defaultdict

# Default dictionary with list
dd_list = defaultdict(list)
dd_list['key1'].append(1)                       # Automatically creates empty list
dd_list['key1'].append(2)                       # {'key1': [1, 2]}

# Default dictionary with int
dd_int = defaultdict(int)
dd_int['count'] += 1                            # Automatically starts at 0

# Default dictionary with set
dd_set = defaultdict(set)
dd_set['items'].add('apple')
dd_set['items'].add('banana')

# Custom default factory
def default_value():
    return "N/A"

dd_custom = defaultdict(default_value)
print(dd_custom['missing_key'])                 # "N/A"

# Group items
from collections import defaultdict
words = ["apple", "banana", "apricot", "blueberry", "cherry"]
grouped = defaultdict(list)
for word in words:
    grouped[word[0]].append(word)               # Group by first letter
```

### `collections.OrderedDict`
```python
from collections import OrderedDict

# Create ordered dictionary
od = OrderedDict()
od['first'] = 1
od['second'] = 2
od['third'] = 3

# Ordered dict from pairs
od = OrderedDict([('a', 1), ('b', 2), ('c', 3)])

# Move to end
od.move_to_end('a')                             # Move 'a' to end
od.move_to_end('b', last=False)                 # Move 'b' to beginning

# Pop items
last_item = od.popitem()                        # Remove and return last item
first_item = od.popitem(last=False)             # Remove and return first item
```

## NumPy Arrays (Third-party)

### NumPy Array Basics
```python
import numpy as np

# Create arrays
arr1d = np.array([1, 2, 3, 4, 5])
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# Array creation functions
zeros = np.zeros(5)                             # [0. 0. 0. 0. 0.]
ones = np.ones((2, 3))                          # 2x3 array of ones
full = np.full((2, 2), 7)                       # 2x2 array filled with 7
eye = np.eye(3)                                 # 3x3 identity matrix
arange = np.arange(0, 10, 2)                    # [0 2 4 6 8]
linspace = np.linspace(0, 1, 5)                 # [0.   0.25 0.5  0.75 1.  ]

# Random arrays
random_arr = np.random.random((2, 3))           # Random values 0-1
random_int = np.random.randint(1, 10, size=5)   # Random integers
normal = np.random.normal(0, 1, 5)              # Normal distribution
```

### NumPy Array Operations
```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

# Array properties
print(arr.shape)                                # (5,)
print(arr.dtype)                                # int64
print(arr.ndim)                                 # 1
print(arr.size)                                 # 5

# Reshaping
reshaped = arr.reshape(5, 1)                    # 5x1 array
flattened = arr2d.flatten()                     # 1D array
ravel = arr2d.ravel()                           # 1D view (if possible)

# Mathematical operations
arr2 = np.array([6, 7, 8, 9, 10])
sum_arr = arr + arr2                            # Element-wise addition
product = arr * arr2                            # Element-wise multiplication
power = arr ** 2                                # Element-wise power

# Aggregation functions
total = np.sum(arr)                             # Sum all elements
mean = np.mean(arr)                             # Average
std = np.std(arr)                               # Standard deviation
minimum = np.min(arr)                           # Minimum value
maximum = np.max(arr)                           # Maximum value

# Boolean indexing
mask = arr > 3                                  # [False False False True True]
filtered = arr[mask]                            # [4 5]
```

## Itertools Module

### Infinite Iterators
```python
import itertools

# count() - infinite arithmetic progression
counter = itertools.count(10, 2)                # 10, 12, 14, 16, ...
first_five = list(itertools.islice(counter, 5)) # [10, 12, 14, 16, 18]

# cycle() - infinite repetition
cycler = itertools.cycle(['A', 'B', 'C'])       # A, B, C, A, B, C, ...
first_ten = list(itertools.islice(cycler, 10))

# repeat() - repeat value
repeater = itertools.repeat('hello', 3)         # hello, hello, hello
infinite_repeat = itertools.repeat(42)          # 42, 42, 42, ...
```

### Finite Iterators
```python
import itertools

# accumulate() - cumulative operation
numbers = [1, 2, 3, 4, 5]
cumulative_sum = list(itertools.accumulate(numbers))        # [1, 3, 6, 10, 15]
cumulative_product = list(itertools.accumulate(numbers, operator.mul))  # [1, 2, 6, 24, 120]

# chain() - flatten iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
chained = list(itertools.chain(list1, list2))   # [1, 2, 3, 4, 5, 6]
from_iterable = list(itertools.chain.from_iterable([[1, 2], [3, 4], [5, 6]]))

# compress() - filter by selectors
data = ['A', 'B', 'C', 'D', 'E']
selectors = [1, 0, 1, 0, 1]
filtered = list(itertools.compress(data, selectors))  # ['A', 'C', 'E']

# dropwhile() and takewhile()
numbers = [1, 3, 5, 8, 9, 10, 11]
dropped = list(itertools.dropwhile(lambda x: x < 8, numbers))  # [8, 9, 10, 11]
taken = list(itertools.takewhile(lambda x: x < 8, numbers))    # [1, 3, 5]

# filterfalse() - opposite of filter
evens = list(itertools.filterfalse(lambda x: x % 2, range(10)))  # [0, 2, 4, 6, 8]

# groupby() - group consecutive elements
data = [1, 1, 2, 2, 2, 3, 1, 1]
grouped = [(k, list(g)) for k, g in itertools.groupby(data)]     # [(1, [1, 1]), (2, [2, 2, 2]), (3, [3]), (1, [1, 1])]

# islice() - slice iterator
numbers = range(100)
sliced = list(itertools.islice(numbers, 5, 15, 2))  # [5, 7, 9, 11, 13]
```

### Combinatorial Iterators
```python
import itertools

# product() - Cartesian product
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
combinations = list(itertools.product(colors, sizes))
# [('red', 'S'), ('red', 'M'), ('red', 'L'), ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]

# permutations() - all permutations
letters = ['A', 'B', 'C']
perms = list(itertools.permutations(letters, 2))     # [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# combinations() - combinations without repetition
combos = list(itertools.combinations(letters, 2))    # [('A', 'B'), ('A', 'C'), ('B', 'C')]

# combinations_with_replacement() - combinations with repetition
combos_rep = list(itertools.combinations_with_replacement(['A', 'B'], 2))  # [('A', 'A'), ('A', 'B'), ('B', 'B')]
```

## Advanced List Techniques

### List Comprehensions
```python
# Basic list comprehension
squares = [x**2 for x in range(10)]             # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
evens = [x for x in range(20) if x % 2 == 0]    # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# Multiple conditions
filtered = [x for x in range(20) if x % 2 == 0 if x > 10]  # [12, 14, 16, 18]

# Nested loops
pairs = [(x, y) for x in range(3) for y in range(3)]  # [(0,0), (0,1), (0,2), (1,0), ...]

# With function calls
words = ["hello", "world", "python"]
lengths = [len(word) for word in words]          # [5, 5, 6]

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
transposed = [[row[i] for row in matrix] for i in range(3)]  # [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

# Conditional expression
values = [1, -2, 3, -4, 5]
abs_values = [x if x >= 0 else -x for x in values]  # [1, 2, 3, 4, 5]
```

### Generator Expressions
```python
# Generator expression (memory efficient)
squares_gen = (x**2 for x in range(10))         # Generator object
print(next(squares_gen))                        # 0
print(next(squares_gen))                        # 1

# Convert to list when needed
squares_list = list(squares_gen)                # Remaining values

# Use in functions
sum_of_squares = sum(x**2 for x in range(10))   # More memory efficient
```

### Matrix Operations
```python
# Create matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Access elements
element = matrix[1][2]                          # 6 (row 1, column 2)

# Matrix transpose
transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Using zip for transpose
transposed = list(zip(*matrix))                 # [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
transposed = [list(row) for row in zip(*matrix)]  # Convert back to lists

# Flatten matrix
flattened = [item for row in matrix for item in row]  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Matrix addition
matrix1 = [[1, 2], [3, 4]]
matrix2 = [[5, 6], [7, 8]]
result = [[matrix1[i][j] + matrix2[i][j] for j in range(len(matrix1[0]))] 
          for i in range(len(matrix1))]
```

## Performance Considerations

### Memory Usage
```python
import sys

# List vs generator memory usage
list_comp = [x**2 for x in range(1000)]
gen_exp = (x**2 for x in range(1000))

print(sys.getsizeof(list_comp))                 # Much larger
print(sys.getsizeof(gen_exp))                   # Much smaller

# List vs array memory usage
import array
python_list = [1] * 1000
array_obj = array.array('i', [1] * 1000)

print(sys.getsizeof(python_list))               # Larger
print(sys.getsizeof(array_obj))                 # Smaller
```

### Performance Tips
```python
# Prefer list comprehensions over loops
# Slow
result = []
for i in range(1000):
    if i % 2 == 0:
        result.append(i**2)

# Fast
result = [i**2 for i in range(1000) if i % 2 == 0]

# Use appropriate data structures
# For frequent insertions/deletions at both ends
from collections import deque
dq = deque()

# For counting
from collections import Counter
counter = Counter()

# For lookups
lookup_set = set(large_list)                    # O(1) lookup vs O(n) for list

# Preallocate lists when size is known
result = [None] * 1000                          # Faster than appending
for i in range(1000):
    result[i] = process(i)
```

### Common Patterns
```python
# Chunking a list
def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

data = list(range(20))
chunks = chunk_list(data, 5)                    # [[0,1,2,3,4], [5,6,7,8,9], ...]

# Remove duplicates while preserving order
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Or using dict (Python 3.7+)
def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

# Find common elements
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
common = list(set(list1) & set(list2))          # [4, 5]

# Find differences
diff1 = list(set(list1) - set(list2))           # [1, 2, 3]
diff2 = list(set(list2) - set(list1))           # [6, 7, 8]
```

---

*This document covers comprehensive array, list, and sequence operations in Python including built-in types, standard library modules, and third-party libraries like NumPy. For the most up-to-date information, refer to the official Python documentation.*