# Python Dictionaries and Sets

This document provides a comprehensive guide to Python dictionaries, sets, and related operations with syntax and usage examples.

## Dictionaries

### Dictionary Creation

```python
# Empty dictionary
empty_dict = {}
empty_dict = dict()

# Dictionary with initial values
person = {"name": "Alice", "age": 30, "city": "New York"}
mixed_types = {1: "one", "two": 2, (3, 4): "tuple key"}

# Dictionary from keyword arguments
config = dict(host="localhost", port=8080, debug=True)

# Dictionary from list of tuples
pairs = [("a", 1), ("b", 2), ("c", 3)]
from_pairs = dict(pairs)                    # {'a': 1, 'b': 2, 'c': 3}

# Dictionary from zip
keys = ["name", "age", "city"]
values = ["Bob", 25, "London"]
person = dict(zip(keys, values))            # {'name': 'Bob', 'age': 25, 'city': 'London'}

# Dictionary comprehension
squares = {x: x**2 for x in range(6)}       # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
evens = {x: x**2 for x in range(10) if x % 2 == 0}  # {0: 0, 2: 4, 4: 16, 6: 36, 8: 64}

# Nested dictionaries
users = {
    "user1": {"name": "Alice", "role": "admin"},
    "user2": {"name": "Bob", "role": "user"}
}
```

### Accessing Dictionary Elements

```python
person = {"name": "Alice", "age": 30, "city": "New York"}

# Direct access
name = person["name"]                       # "Alice"
# age = person["country"]                   # KeyError: 'country'

# get() method - safe access
name = person.get("name")                   # "Alice"
country = person.get("country")             # None
country = person.get("country", "USA")      # "USA" (default value)

# Check if key exists
if "name" in person:
    print(person["name"])

has_age = "age" in person                   # True
no_country = "country" not in person        # True

# Access all keys, values, items
keys = person.keys()                        # dict_keys(['name', 'age', 'city'])
values = person.values()                    # dict_values(['Alice', 30, 'New York'])
items = person.items()                      # dict_items([('name', 'Alice'), ('age', 30), ('city', 'New York')])

# Convert to lists
keys_list = list(person.keys())             # ['name', 'age', 'city']
values_list = list(person.values())         # ['Alice', 30, 'New York']
items_list = list(person.items())           # [('name', 'Alice'), ('age', 30), ('city', 'New York')]
```

### Modifying Dictionaries

#### Adding and Updating

```python
person = {"name": "Alice", "age": 30}

# Add new key-value pair
person["city"] = "New York"                 # {'name': 'Alice', 'age': 30, 'city': 'New York'}

# Update existing value
person["age"] = 31                          # {'name': 'Alice', 'age': 31, 'city': 'New York'}

# update() method - merge dictionaries
person.update({"age": 32, "country": "USA"})  # Update existing and add new
person.update([("phone", "123-456")], email="alice@example.com")

# Merge operator (Python 3.9+)
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
merged = dict1 | dict2                      # {'a': 1, 'b': 3, 'c': 4} (dict2 wins)

# Augmented merge (Python 3.9+)
dict1 |= dict2                              # dict1 is now merged with dict2

# setdefault() - set value if key doesn't exist
person.setdefault("gender", "Female")       # Returns "Female", adds to dict
person.setdefault("name", "Bob")            # Returns "Alice", doesn't change dict
```

#### Removing Elements

```python
person = {"name": "Alice", "age": 30, "city": "New York", "country": "USA"}

# del statement
del person["country"]                       # Removes "country" key
# del person["phone"]                       # KeyError: 'phone'

# pop() - remove and return value
age = person.pop("age")                     # Returns 30, removes "age"
phone = person.pop("phone", "N/A")          # Returns "N/A" (default), no error

# popitem() - remove and return last inserted item (Python 3.7+)
item = person.popitem()                     # Returns ('city', 'New York')

# clear() - remove all items
person.clear()                              # {}
```

### Dictionary Methods

```python
person = {"name": "Alice", "age": 30, "city": "New York"}

# copy() - shallow copy
person_copy = person.copy()
person_copy["name"] = "Bob"                 # Original unchanged

# Deep copy
import copy
deep_copy = copy.deepcopy(person)

# fromkeys() - create dict with keys from sequence
keys = ["a", "b", "c"]
default_dict = dict.fromkeys(keys)          # {'a': None, 'b': None, 'c': None}
default_dict = dict.fromkeys(keys, 0)       # {'a': 0, 'b': 0, 'c': 0}

# values(), keys(), items() are dynamic views
person = {"name": "Alice"}
keys_view = person.keys()
print(keys_view)                            # dict_keys(['name'])
person["age"] = 30
print(keys_view)                            # dict_keys(['name', 'age']) - updated!
```

### Iterating Over Dictionaries

```python
person = {"name": "Alice", "age": 30, "city": "New York"}

# Iterate over keys (default)
for key in person:
    print(key)                              # name, age, city

for key in person.keys():
    print(key)                              # Same as above

# Iterate over values
for value in person.values():
    print(value)                            # Alice, 30, New York

# Iterate over items
for key, value in person.items():
    print(f"{key}: {value}")                # name: Alice, age: 30, city: New York

# Enumerate dictionary
for i, (key, value) in enumerate(person.items()):
    print(f"{i}: {key} = {value}")          # 0: name = Alice, 1: age = 30, 2: city = New York

# Iterate and modify (create new dict)
updated = {k: v for k, v in person.items() if isinstance(v, str)}
```

### Dictionary Comprehensions

```python
# Basic comprehension
squares = {x: x**2 for x in range(6)}       # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# With condition
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}

# Transform dictionary
person = {"name": "Alice", "age": 30, "city": "New York"}
upper_keys = {k.upper(): v for k, v in person.items()}  # {'NAME': 'Alice', 'AGE': 30, 'CITY': 'New York'}

# Filter dictionary
numbers = {"a": 1, "b": 2, "c": 3, "d": 4}
evens_only = {k: v for k, v in numbers.items() if v % 2 == 0}  # {'b': 2, 'd': 4}

# Invert dictionary (swap keys and values)
inverted = {v: k for k, v in numbers.items()}  # {1: 'a', 2: 'b', 3: 'c', 4: 'd'}

# Nested comprehension
matrix_dict = {f"row{i}": {f"col{j}": i*j for j in range(3)} for i in range(3)}
```

### Nested Dictionaries

```python
# Create nested dictionary
employees = {
    "emp1": {
        "name": "Alice",
        "dept": "Engineering",
        "salary": 80000
    },
    "emp2": {
        "name": "Bob",
        "dept": "Marketing",
        "salary": 70000
    }
}

# Access nested values
alice_salary = employees["emp1"]["salary"]  # 80000

# Safely access nested values
bob_bonus = employees.get("emp2", {}).get("bonus", 0)  # 0 (no KeyError)

# Update nested values
employees["emp1"]["salary"] = 85000

# Add to nested dictionary
employees["emp3"] = {"name": "Charlie", "dept": "Sales", "salary": 75000}

# Iterate nested dictionary
for emp_id, details in employees.items():
    print(f"{emp_id}: {details['name']} - ${details['salary']}")

# Flatten nested dictionary
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

flat = flatten_dict(employees)
# {'emp1_name': 'Alice', 'emp1_dept': 'Engineering', 'emp1_salary': 85000, ...}
```

### Advanced Dictionary Operations

```python
# Merge multiple dictionaries
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
dict3 = {"c": 5, "d": 6}

# Using unpacking
merged = {**dict1, **dict2, **dict3}        # {'a': 1, 'b': 3, 'c': 5, 'd': 6}

# Using ChainMap (doesn't merge, creates view)
from collections import ChainMap
chain = ChainMap(dict1, dict2, dict3)       # First dict has priority
print(chain["b"])                           # 2 (from dict1)

# Sorting dictionary by keys
unsorted = {"c": 3, "a": 1, "b": 2}
sorted_by_key = dict(sorted(unsorted.items()))  # {'a': 1, 'b': 2, 'c': 3}

# Sorting by values
sorted_by_value = dict(sorted(unsorted.items(), key=lambda x: x[1]))

# Reverse dictionary
reversed_dict = dict(reversed(list(unsorted.items())))

# Get key with max/min value
scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
best = max(scores, key=scores.get)          # "Bob"
worst = min(scores, key=scores.get)         # "Charlie"

# Dictionary from two lists
keys = ["a", "b", "c"]
values = [1, 2, 3]
combined = dict(zip(keys, values))          # {'a': 1, 'b': 2, 'c': 3}
```

## Sets

### Set Creation

```python
# Empty set (must use set(), {} creates empty dict)
empty_set = set()

# Set with initial values
numbers = {1, 2, 3, 4, 5}
mixed = {1, "hello", 3.14, True}

# Set from iterable
from_list = set([1, 2, 3, 2, 1])           # {1, 2, 3} - duplicates removed
from_string = set("hello")                  # {'h', 'e', 'l', 'o'}
from_range = set(range(5))                  # {0, 1, 2, 3, 4}

# Set comprehension
squares = {x**2 for x in range(6)}          # {0, 1, 4, 9, 16, 25}
evens = {x for x in range(10) if x % 2 == 0}  # {0, 2, 4, 6, 8}

# Frozen set (immutable)
frozen = frozenset([1, 2, 3, 4, 5])        # Immutable set
```

### Set Operations

#### Adding and Removing Elements

```python
colors = {"red", "green", "blue"}

# add() - add single element
colors.add("yellow")                        # {'red', 'green', 'blue', 'yellow'}
colors.add("red")                           # No change, already exists

# update() - add multiple elements
colors.update(["orange", "purple"])         # Add from list
colors.update({"pink", "brown"}, ["white"])  # Multiple iterables

# remove() - remove element (raises KeyError if not found)
colors.remove("red")
# colors.remove("black")                    # KeyError: 'black'

# discard() - remove element (no error if not found)
colors.discard("green")
colors.discard("black")                     # No error

# pop() - remove and return arbitrary element
color = colors.pop()                        # Returns and removes random element
# empty_set.pop()                           # KeyError: 'pop from an empty set'

# clear() - remove all elements
colors.clear()                              # set()
```

### Set Mathematical Operations

```python
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}
set3 = {1, 2, 3}

# Union - all elements from both sets
union = set1 | set2                         # {1, 2, 3, 4, 5, 6, 7, 8}
union = set1.union(set2)                    # Same as above
union = set1.union(set2, set3)              # Union of multiple sets

# Intersection - common elements
intersection = set1 & set2                  # {4, 5}
intersection = set1.intersection(set2)      # Same as above

# Difference - elements in first but not in second
difference = set1 - set2                    # {1, 2, 3}
difference = set1.difference(set2)          # Same as above

# Symmetric difference - elements in either but not both
sym_diff = set1 ^ set2                      # {1, 2, 3, 6, 7, 8}
sym_diff = set1.symmetric_difference(set2)  # Same as above

# Update operations (modify set in place)
set1 |= set2                                # Union update
set1 &= set2                                # Intersection update
set1 -= set2                                # Difference update
set1 ^= set2                                # Symmetric difference update

# Multiple sets
set_a = {1, 2, 3}
set_b = {2, 3, 4}
set_c = {3, 4, 5}
all_union = set_a | set_b | set_c          # {1, 2, 3, 4, 5}
all_intersection = set_a & set_b & set_c   # {3}
```

### Set Comparisons

```python
set1 = {1, 2, 3}
set2 = {1, 2, 3, 4, 5}
set3 = {1, 2, 3}
set4 = {4, 5, 6}

# Subset - all elements of set1 are in set2
is_subset = set1 <= set2                    # True
is_subset = set1.issubset(set2)             # True

# Proper subset - subset but not equal
is_proper_subset = set1 < set2              # True

# Superset - all elements of set2 are in set1
is_superset = set2 >= set1                  # True
is_superset = set2.issuperset(set1)         # True

# Proper superset
is_proper_superset = set2 > set1            # True

# Equality
is_equal = set1 == set3                     # True

# Disjoint - no common elements
is_disjoint = set1.isdisjoint(set4)         # True (no overlap)
is_disjoint = set1.isdisjoint(set2)         # False (have overlap)
```

### Set Methods

```python
numbers = {1, 2, 3, 4, 5}

# copy() - shallow copy
numbers_copy = numbers.copy()

# in operator - membership test
has_three = 3 in numbers                    # True
no_six = 6 not in numbers                   # True

# len() - number of elements
size = len(numbers)                         # 5

# min(), max(), sum() - work with numeric sets
minimum = min(numbers)                      # 1
maximum = max(numbers)                      # 5
total = sum(numbers)                        # 15

# sorted() - returns sorted list
sorted_list = sorted(numbers)               # [1, 2, 3, 4, 5]
sorted_desc = sorted(numbers, reverse=True) # [5, 4, 3, 2, 1]
```

### Iterating Over Sets

```python
colors = {"red", "green", "blue", "yellow"}

# Basic iteration (order not guaranteed)
for color in colors:
    print(color)

# With enumerate
for i, color in enumerate(colors):
    print(f"{i}: {color}")

# Iterate sorted
for color in sorted(colors):
    print(color)                            # Alphabetical order

# Set comprehension
upper_colors = {color.upper() for color in colors}
```

### Frozen Sets

```python
# Create frozen set
frozen = frozenset([1, 2, 3, 4, 5])

# Can't modify
# frozen.add(6)                             # AttributeError: 'frozenset' object has no attribute 'add'

# Can be used as dict key or set element
dict_with_frozen_key = {frozen: "value"}
set_of_sets = {frozenset([1, 2]), frozenset([3, 4])}

# All read-only operations work
union = frozen | {6, 7}                     # Returns new frozenset
intersection = frozen & {3, 4, 5, 6}        # Returns new frozenset
```

## Advanced Collections

### `defaultdict`

```python
from collections import defaultdict

# Default dictionary with list
word_index = defaultdict(list)
sentence = "the quick brown fox jumps over the lazy dog"
for i, word in enumerate(sentence.split()):
    word_index[word[0]].append(word)        # Automatically creates list

# Default dictionary with int (counting)
counter = defaultdict(int)
for char in "hello world":
    counter[char] += 1                      # Automatically starts at 0

# Default dictionary with set
grouped = defaultdict(set)
grouped['fruits'].add('apple')
grouped['fruits'].add('banana')

# Custom default factory
def default_value():
    return "N/A"

custom_dict = defaultdict(default_value)
print(custom_dict["missing"])               # "N/A"
```

### `Counter`

```python
from collections import Counter

# Count elements
letters = Counter("hello world")            # Counter({'l': 3, 'o': 2, 'h': 1, ...})
numbers = Counter([1, 2, 3, 2, 3, 3])      # Counter({3: 3, 2: 2, 1: 1})

# Most common elements
most_common = letters.most_common(3)        # [('l', 3), ('o', 2), ('h', 1)]

# Counter operations
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)                              # Counter({'a': 4, 'b': 3})
print(c1 - c2)                              # Counter({'a': 2}) - only positive
print(c1 & c2)                              # Counter({'a': 1, 'b': 1}) - min
print(c1 | c2)                              # Counter({'a': 3, 'b': 2}) - max

# Update counter
c1.update(c2)                               # Add counts
c1.subtract(c2)                             # Subtract counts

# Elements
c = Counter(a=2, b=3)
list(c.elements())                          # ['a', 'a', 'b', 'b', 'b']
```

### `OrderedDict`

```python
from collections import OrderedDict

# Maintain insertion order (Python 3.7+ dicts are ordered too)
od = OrderedDict()
od['first'] = 1
od['second'] = 2
od['third'] = 3

# Move to end
od.move_to_end('first')                     # 'first' now at end
od.move_to_end('second', last=False)        # 'second' now at beginning

# Pop items in order
last = od.popitem()                         # ('first', 1)
first = od.popitem(last=False)              # ('second', 2)
```

### `ChainMap`

```python
from collections import ChainMap

# Combine multiple dicts
defaults = {"color": "red", "user": "guest"}
custom = {"user": "admin"}
config = ChainMap(custom, defaults)         # custom takes priority

print(config["user"])                       # "admin" (from custom)
print(config["color"])                      # "red" (from defaults)

# Update only affects first mapping
config["theme"] = "dark"
print(custom)                               # {'user': 'admin', 'theme': 'dark'}
print(defaults)                             # {'color': 'red', 'user': 'guest'}

# Add new child
local = {"debug": True}
config = config.new_child(local)            # local now has highest priority
```

## Common Patterns and Best Practices

### Dictionary Patterns

```python
# Safe get with complex default
config = {}
value = config.get("key", {}).get("nested_key", "default")

# Counting occurrences
words = ["apple", "banana", "apple", "cherry", "banana"]
word_count = {}
for word in words:
    word_count[word] = word_count.get(word, 0) + 1

# Or use Counter
from collections import Counter
word_count = Counter(words)

# Group items by property
students = [
    {"name": "Alice", "grade": "A"},
    {"name": "Bob", "grade": "B"},
    {"name": "Charlie", "grade": "A"}
]
by_grade = {}
for student in students:
    grade = student["grade"]
    by_grade.setdefault(grade, []).append(student)

# Or use defaultdict
from collections import defaultdict
by_grade = defaultdict(list)
for student in students:
    by_grade[student["grade"]].append(student)

# Dictionary as switch statement
def operation(op, x, y):
    operations = {
        "add": lambda: x + y,
        "sub": lambda: x - y,
        "mul": lambda: x * y,
        "div": lambda: x / y
    }
    return operations.get(op, lambda: "Invalid")()
```

### Set Patterns

```python
# Remove duplicates from list (preserves order in Python 3.7+)
items = [1, 2, 3, 2, 1, 4, 5, 4]
unique = list(dict.fromkeys(items))         # [1, 2, 3, 4, 5]

# Or use set (doesn't preserve order)
unique = list(set(items))

# Fast membership testing
allowed_users = {"alice", "bob", "charlie"}  # O(1) lookup
if username in allowed_users:
    grant_access()

# Find unique elements across lists
list1 = [1, 2, 3, 4, 5]
list2 = [4, 5, 6, 7, 8]
only_in_list1 = list(set(list1) - set(list2))  # [1, 2, 3]
only_in_list2 = list(set(list2) - set(list1))  # [6, 7, 8]
in_both = list(set(list1) & set(list2))     # [4, 5]
in_either = list(set(list1) | set(list2))   # [1, 2, 3, 4, 5, 6, 7, 8]

# Filter duplicates while processing
seen = set()
for item in items:
    if item not in seen:
        process(item)
        seen.add(item)
```

---

*This document covers comprehensive dictionary and set operations in Python. For the most up-to-date information, refer to the official Python documentation.*
