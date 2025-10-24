# Python Decorators and Context Managers

This document provides a comprehensive guide to Python decorators, context managers, and related concepts with syntax and usage examples.

## Function Decorators

### Basic Decorators

```python
# Simple decorator
def my_decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Before function call
# Hello!
# After function call

# Decorator without @ syntax (equivalent)
def say_goodbye():
    print("Goodbye!")

say_goodbye = my_decorator(say_goodbye)

# Decorator with arguments
def decorator_with_args(func):
    def wrapper(*args, **kwargs):
        print(f"Arguments: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@decorator_with_args
def add(a, b):
    return a + b

add(5, 3)                                   # Arguments: (5, 3), {}  Result: 8

# Decorator that returns value
def uppercase_decorator(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result.upper()
    return wrapper

@uppercase_decorator
def get_greeting(name):
    return f"hello, {name}"

print(get_greeting("Alice"))                # HELLO, ALICE
```

### Preserving Function Metadata

```python
from functools import wraps

# Without wraps - metadata lost
def my_decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Greet someone by name"""
    return f"Hello, {name}"

print(greet.__name__)                       # wrapper (not ideal)
print(greet.__doc__)                        # None

# With wraps - metadata preserved
def my_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name):
    """Greet someone by name"""
    return f"Hello, {name}"

print(greet.__name__)                       # greet
print(greet.__doc__)                        # Greet someone by name
```

### Decorators with Arguments

```python
# Decorator factory - returns decorator
def repeat(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(times=3)
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Hello!
# Hello!
# Hello!

# Parameterized decorator with default
def log(prefix="LOG"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"{prefix}: Calling {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@log(prefix="DEBUG")
def process_data():
    print("Processing...")

process_data()                              # DEBUG: Calling process_data

@log()                                      # Default prefix
def save_data():
    print("Saving...")

save_data()                                 # LOG: Calling save_data
```

### Multiple Decorators

```python
def decorator1(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator 1")
        return func(*args, **kwargs)
    return wrapper

def decorator2(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print("Decorator 2")
        return func(*args, **kwargs)
    return wrapper

# Applied bottom to top
@decorator1
@decorator2
def my_function():
    print("Function")

my_function()
# Output:
# Decorator 1
# Decorator 2
# Function

# Equivalent to:
# my_function = decorator1(decorator2(my_function))
```

### Common Decorator Patterns

```python
# Timing decorator
import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

# Caching/Memoization decorator
def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))                       # Fast with memoization

# Authentication decorator
def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            raise PermissionError("Not authenticated")
        return func(*args, **kwargs)
    return wrapper

@require_auth
def delete_user(user_id):
    # Delete user
    pass

# Retry decorator
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def unstable_api_call():
    # Might fail randomly
    pass

# Validation decorator
def validate_args(*validators):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg, validator in zip(args, validators):
                if not validator(arg):
                    raise ValueError(f"Invalid argument: {arg}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_args(lambda x: x > 0, lambda x: isinstance(x, str))
def process(number, text):
    print(f"Number: {number}, Text: {text}")

# Rate limiting decorator
from collections import deque
from time import time

def rate_limit(max_calls, period):
    calls = deque()

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time()
            # Remove old calls
            while calls and calls[0] < now - period:
                calls.popleft()

            if len(calls) >= max_calls:
                raise Exception("Rate limit exceeded")

            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=5, period=60)
def api_call():
    print("API called")
```

## Class Decorators

### Decorating Classes

```python
# Class decorator
def singleton(cls):
    instances = {}
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        print("Creating database connection")

db1 = Database()                            # Creating database connection
db2 = Database()                            # No output, returns same instance
print(db1 is db2)                           # True

# Add attributes to class
def add_methods(cls):
    cls.new_method = lambda self: "New method"
    return cls

@add_methods
class MyClass:
    pass

obj = MyClass()
print(obj.new_method())                     # New method

# Dataclass-like decorator
def dataclass(cls):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls.__name__}({args})"

    cls.__init__ = __init__
    cls.__repr__ = __repr__
    return cls

@dataclass
class Point:
    pass

p = Point(x=10, y=20)
print(p)                                    # Point(x=10, y=20)
```

### Method Decorators

```python
# Property decorator
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @radius.deleter
    def radius(self):
        del self._radius

    @property
    def area(self):
        return 3.14159 * self._radius ** 2

c = Circle(5)
print(c.radius)                             # 5
c.radius = 10                               # Use setter
print(c.area)                               # 314.159

# Static method
class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

print(MathUtils.add(5, 3))                  # 8

# Class method
class Person:
    count = 0

    def __init__(self, name):
        self.name = name
        Person.count += 1

    @classmethod
    def get_count(cls):
        return cls.count

    @classmethod
    def from_string(cls, person_str):
        name = person_str.split("-")[0]
        return cls(name)

p1 = Person("Alice")
p2 = Person.from_string("Bob-25")
print(Person.get_count())                   # 2

# Abstract method
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass

class Dog(Animal):
    def make_sound(self):
        return "Woof!"

# Cached property (Python 3.8+)
from functools import cached_property

class DataProcessor:
    def __init__(self, data):
        self.data = data

    @cached_property
    def expensive_computation(self):
        print("Computing...")
        return sum(x ** 2 for x in self.data)

processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.expensive_computation)      # Computing... 55
print(processor.expensive_computation)      # 55 (cached, no computation)
```

## Context Managers

### Using Context Managers

```python
# File context manager
with open("file.txt") as f:
    content = f.read()
# File automatically closed

# Multiple context managers
with open("input.txt") as infile, open("output.txt", "w") as outfile:
    content = infile.read()
    outfile.write(content.upper())

# Nested context managers (older style)
with open("input.txt") as infile:
    with open("output.txt", "w") as outfile:
        content = infile.read()
        outfile.write(content)

# Lock context manager
from threading import Lock

lock = Lock()
with lock:
    # Critical section
    shared_resource += 1

# Database transaction
# Pseudo-code
with database.transaction():
    database.insert(record1)
    database.insert(record2)
    # Automatically commits or rolls back
```

### Creating Context Managers with Classes

```python
# Basic context manager class
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file:
            self.file.close()
        # Return False to propagate exceptions
        # Return True to suppress exceptions
        return False

with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")

# Context manager with exception handling
class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None

    def __enter__(self):
        self.connection = connect(self.host, self.port)
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        if self.connection:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
            self.connection.close()
        return False

with DatabaseConnection("localhost", 5432) as conn:
    conn.execute("INSERT INTO users VALUES (...)")
    # Automatically commits if no exception, rollbacks otherwise

# Context manager that suppresses exceptions
class IgnoreErrors:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Return True to suppress all exceptions
        return True

with IgnoreErrors():
    raise ValueError("This error is suppressed")
print("Execution continues")

# Timer context manager
import time

class Timer:
    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.duration = self.end - self.start
        print(f"{self.name} took {self.duration:.4f} seconds")
        return False

with Timer("Data processing"):
    time.sleep(1)
    # Process data
```

### Creating Context Managers with contextlib

```python
from contextlib import contextmanager

# Basic generator-based context manager
@contextmanager
def file_manager(filename, mode):
    f = open(filename, mode)
    try:
        yield f
    finally:
        f.close()

with file_manager("test.txt", "w") as f:
    f.write("Hello!")

# Context manager with setup and teardown
@contextmanager
def temporary_directory():
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

with temporary_directory() as temp_dir:
    # Use temp_dir
    pass
# Directory automatically deleted

# Context manager for changing directory
import os

@contextmanager
def change_directory(path):
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)

with change_directory("/tmp"):
    # Work in /tmp
    pass
# Back to original directory

# Context manager with value
@contextmanager
def timer(name="Operation"):
    start = time.time()
    yield lambda: time.time() - start
    duration = time.time() - start
    print(f"{name} took {duration:.4f} seconds")

with timer("Process") as get_elapsed:
    time.sleep(1)
    print(f"Elapsed so far: {get_elapsed():.4f}")
```

### contextlib Utilities

```python
from contextlib import (
    suppress,
    redirect_stdout,
    redirect_stderr,
    nullcontext,
    closing,
    ExitStack
)

# suppress - ignore specific exceptions
with suppress(FileNotFoundError, PermissionError):
    os.remove("file.txt")
# No error if file doesn't exist

# redirect_stdout - capture stdout
import io

with redirect_stdout(io.StringIO()) as output:
    print("This goes to StringIO")
    print("Not to console")

print(output.getvalue())                    # This goes to StringIO\nNot to console

# redirect_stderr - capture stderr
with redirect_stderr(io.StringIO()) as errors:
    sys.stderr.write("Error message")

# nullcontext - conditional context manager
def process(use_file=False):
    cm = open("file.txt") if use_file else nullcontext()
    with cm as f:
        # f is file or None
        pass

# closing - ensure close() is called
from contextlib import closing
import urllib.request

with closing(urllib.request.urlopen("http://example.com")) as page:
    content = page.read()

# ExitStack - dynamic context managers
with ExitStack() as stack:
    files = [stack.enter_context(open(f"file{i}.txt")) for i in range(5)]
    # All files automatically closed

# ExitStack with callbacks
with ExitStack() as stack:
    stack.callback(print, "Cleanup 1")
    stack.callback(print, "Cleanup 2")
    # Do work
# Prints: Cleanup 2, Cleanup 1 (LIFO order)

# Conditional context managers with ExitStack
def process_files(use_lock=False):
    with ExitStack() as stack:
        if use_lock:
            stack.enter_context(lock)

        files = [stack.enter_context(open(f)) for f in filenames]
        # Process files
```

## Advanced Decorator Patterns

### Decorator Classes

```python
# Decorator as a class
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Call {self.count} to {self.func.__name__}")
        return self.func(*args, **kwargs)

@CountCalls
def say_hello():
    print("Hello!")

say_hello()                                 # Call 1 to say_hello
say_hello()                                 # Call 2 to say_hello

# Parameterized decorator class
class Repeat:
    def __init__(self, times):
        self.times = times

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(self.times):
                result = func(*args, **kwargs)
            return result
        return wrapper

@Repeat(times=3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")                              # Hello, Alice! (3 times)

# Stateful decorator
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()

            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded")

            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

@RateLimiter(max_calls=3, period=60)
def api_call():
    print("API called")
```

### Decorator with Optional Arguments

```python
# Decorator that works with or without arguments
def optional_decorator(func=None, *, prefix="DEFAULT"):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            print(f"{prefix}: {f.__name__}")
            return f(*args, **kwargs)
        return wrapper

    if func is None:
        # Called with arguments: @optional_decorator(prefix="...")
        return decorator
    else:
        # Called without arguments: @optional_decorator
        return decorator(func)

@optional_decorator
def func1():
    pass

@optional_decorator(prefix="CUSTOM")
def func2():
    pass

func1()                                     # DEFAULT: func1
func2()                                     # CUSTOM: func2
```

### Chaining and Composing Decorators

```python
# Decorator composition
def compose(*decorators):
    def decorator(func):
        for dec in reversed(decorators):
            func = dec(func)
        return func
    return decorator

@compose(decorator1, decorator2, decorator3)
def my_function():
    pass

# Equivalent to:
# @decorator1
# @decorator2
# @decorator3
# def my_function():
#     pass

# Parameterized decorator composition
def apply_decorators(*dec_factories):
    def decorator(func):
        for dec_factory in reversed(dec_factories):
            func = dec_factory()(func)
        return func
    return decorator

@apply_decorators(
    lambda: timer,
    lambda: memoize,
    lambda: retry(max_attempts=3)
)
def expensive_function():
    pass
```

### Debugging Decorators

```python
# Preserve function for inspection
def debug_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result

    # Add reference to original function
    wrapper.__wrapped__ = func
    return wrapper

@debug_decorator
def add(a, b):
    return a + b

# Access original function
original_add = add.__wrapped__
result = original_add(2, 3)                 # No debug output

# Decorator that logs to file
import logging

def log_to_file(filename):
    logging.basicConfig(filename=filename, level=logging.INFO)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"Calling {func.__name__} with {args}, {kwargs}")
            result = func(*args, **kwargs)
            logging.info(f"Result: {result}")
            return result
        return wrapper
    return decorator

@log_to_file("function_calls.log")
def calculate(x, y):
    return x * y
```

## Best Practices

### When to Use Decorators

```python
# Good use cases:
# - Logging
# - Timing/profiling
# - Access control/authentication
# - Caching/memoization
# - Input validation
# - Retry logic
# - Rate limiting

# Bad use cases:
# - Complex business logic (hard to debug)
# - Modifying function behavior in non-obvious ways
# - Too many layers (hard to understand)

# Keep decorators simple and focused
# Bad - too much logic in decorator
def complex_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 50 lines of complex logic
        pass
    return wrapper

# Good - move complex logic to separate functions
def validate_input(args, kwargs):
    # Validation logic
    pass

def simple_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        validate_input(args, kwargs)
        return func(*args, **kwargs)
    return wrapper
```

### Performance Considerations

```python
# Use functools.lru_cache for memoization (more efficient)
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Decorator overhead
# Decorators add function call overhead
# For performance-critical code, consider alternatives

# Bad for tight loops
@debug_decorator
def process_item(item):
    return item * 2

for item in range(1000000):
    process_item(item)                      # Decorator called 1M times

# Good - apply decorator sparingly
def process_items(items):
    return [item * 2 for item in items]

@debug_decorator
def main():
    items = range(1000000)
    result = process_items(items)           # Decorator called once
```

---

*This document covers comprehensive decorator and context manager usage in Python. For the most up-to-date information, refer to the official Python documentation.*
