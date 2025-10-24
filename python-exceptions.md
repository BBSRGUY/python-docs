# Python Exception Handling

This document provides a comprehensive guide to Python exceptions, error handling, and related concepts with syntax and usage examples.

## Exception Basics

### Try-Except Blocks

```python
# Basic try-except
try:
    result = 10 / 0
except:
    print("An error occurred")

# Catch specific exception
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# Catch multiple exceptions
try:
    value = int("abc")
except (ValueError, TypeError):
    print("Invalid conversion")

# Separate handlers for different exceptions
try:
    file = open("missing.txt")
    content = file.read()
    number = int(content)
except FileNotFoundError:
    print("File not found")
except ValueError:
    print("Invalid number in file")
except Exception as e:
    print(f"Unexpected error: {e}")

# Access exception object
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")                    # Error: division by zero
    print(f"Type: {type(e)}")               # Type: <class 'ZeroDivisionError'>
```

### Try-Except-Else

```python
# Else clause - runs if no exception occurs
try:
    file = open("data.txt")
    content = file.read()
except FileNotFoundError:
    print("File not found")
else:
    print("File read successfully")
    print(f"Content: {content}")
    file.close()

# Example with user input
try:
    age = int(input("Enter your age: "))
except ValueError:
    print("That's not a valid number")
else:
    print(f"You are {age} years old")
    if age >= 18:
        print("You are an adult")
```

### Try-Except-Finally

```python
# Finally clause - always executes
try:
    file = open("data.txt")
    content = file.read()
except FileNotFoundError:
    print("File not found")
finally:
    print("Cleanup code always runs")

# Resource cleanup example
file = None
try:
    file = open("data.txt")
    content = file.read()
except FileNotFoundError:
    print("File not found")
finally:
    if file:
        file.close()
    print("File closed")

# Complete try-except-else-finally
try:
    result = risky_operation()
except SpecificError:
    handle_error()
else:
    print("Success!")
finally:
    cleanup()
```

### Raising Exceptions

```python
# Raise exception
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

# Raise with no argument (re-raise current exception)
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Logging error...")
    raise                                   # Re-raises the same exception

# Raise from another exception
try:
    result = int("abc")
except ValueError as e:
    raise TypeError("Invalid input") from e

# Raise custom message
def validate_age(age):
    if age < 0:
        raise ValueError(f"Age cannot be negative: {age}")
    if age > 150:
        raise ValueError(f"Age seems unrealistic: {age}")
    return age

# Raise without traceback (Python 3.3+)
try:
    result = 10 / 0
except ZeroDivisionError:
    raise ValueError("Bad input") from None  # Suppresses original exception
```

## Built-in Exceptions

### Common Exceptions

```python
# ValueError - invalid value
try:
    int("abc")
except ValueError:
    print("Cannot convert to integer")

# TypeError - wrong type
try:
    "hello" + 5
except TypeError:
    print("Cannot concatenate str and int")

# KeyError - missing dictionary key
try:
    data = {"a": 1}
    value = data["b"]
except KeyError:
    print("Key not found")

# IndexError - invalid index
try:
    lst = [1, 2, 3]
    value = lst[10]
except IndexError:
    print("Index out of range")

# AttributeError - invalid attribute
try:
    value = "hello".nonexistent
except AttributeError:
    print("Attribute doesn't exist")

# FileNotFoundError - file doesn't exist
try:
    with open("missing.txt") as f:
        content = f.read()
except FileNotFoundError:
    print("File not found")

# ZeroDivisionError - division by zero
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")

# ImportError - module not found
try:
    import nonexistent_module
except ImportError:
    print("Module not found")

# NameError - undefined variable
try:
    print(undefined_variable)
except NameError:
    print("Variable not defined")

# RuntimeError - general runtime error
try:
    raise RuntimeError("Something went wrong")
except RuntimeError as e:
    print(f"Runtime error: {e}")

# StopIteration - iterator exhausted
try:
    it = iter([1, 2, 3])
    next(it)
    next(it)
    next(it)
    next(it)                                # Raises StopIteration
except StopIteration:
    print("Iterator exhausted")

# AssertionError - assertion failed
try:
    assert 1 == 2, "One does not equal two"
except AssertionError as e:
    print(f"Assertion failed: {e}")
```

### Exception Hierarchy

```python
"""
BaseException
├── SystemExit
├── KeyboardInterrupt
├── GeneratorExit
└── Exception
    ├── StopIteration
    ├── ArithmeticError
    │   ├── FloatingPointError
    │   ├── OverflowError
    │   └── ZeroDivisionError
    ├── AssertionError
    ├── AttributeError
    ├── BufferError
    ├── EOFError
    ├── ImportError
    │   └── ModuleNotFoundError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── MemoryError
    ├── NameError
    │   └── UnboundLocalError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── FileExistsError
    │   ├── PermissionError
    │   └── TimeoutError
    ├── RuntimeError
    │   ├── NotImplementedError
    │   └── RecursionError
    ├── SyntaxError
    │   └── IndentationError
    ├── SystemError
    ├── TypeError
    ├── ValueError
    │   └── UnicodeError
    └── Warning
        ├── DeprecationWarning
        ├── UserWarning
        └── FutureWarning
"""

# Catching by hierarchy
try:
    # Some operation
    pass
except LookupError:                         # Catches both IndexError and KeyError
    print("Lookup failed")

try:
    # Some operation
    pass
except ArithmeticError:                     # Catches ZeroDivisionError, OverflowError, etc.
    print("Arithmetic error")
```

## Custom Exceptions

### Creating Custom Exceptions

```python
# Simple custom exception
class CustomError(Exception):
    pass

# Raise custom exception
raise CustomError("Something went wrong")

# Custom exception with attributes
class ValidationError(Exception):
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

# Use custom exception
try:
    raise ValidationError("email", "Invalid format")
except ValidationError as e:
    print(f"Field: {e.field}")              # Field: email
    print(f"Message: {e.message}")          # Message: Invalid format

# Custom exception with default message
class InsufficientFundsError(Exception):
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        message = f"Insufficient funds: balance={balance}, amount={amount}"
        super().__init__(message)

# More complex custom exception
class HTTPError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")

    def is_client_error(self):
        return 400 <= self.status_code < 500

    def is_server_error(self):
        return 500 <= self.status_code < 600

try:
    raise HTTPError(404, "Page not found")
except HTTPError as e:
    print(e)                                # HTTP 404: Page not found
    if e.is_client_error():
        print("Client error")
```

### Exception Hierarchies

```python
# Create exception hierarchy
class ApplicationError(Exception):
    """Base exception for application"""
    pass

class DatabaseError(ApplicationError):
    """Database-related errors"""
    pass

class ConnectionError(DatabaseError):
    """Database connection errors"""
    pass

class QueryError(DatabaseError):
    """Database query errors"""
    pass

class ValidationError(ApplicationError):
    """Validation errors"""
    pass

# Use exception hierarchy
try:
    # Database operation
    raise QueryError("Invalid SQL syntax")
except DatabaseError as e:                  # Catches all database errors
    print(f"Database error: {e}")
except ApplicationError as e:               # Catches all application errors
    print(f"Application error: {e}")

# More specific handling
try:
    # Some operation
    raise ConnectionError("Cannot connect to database")
except ConnectionError:
    print("Connection failed, retrying...")
except DatabaseError:
    print("Database error occurred")
except ApplicationError:
    print("Application error occurred")
```

## Context Managers

### Using Context Managers

```python
# File handling with context manager
with open("data.txt") as file:
    content = file.read()
# File automatically closed

# Multiple context managers
with open("input.txt") as infile, open("output.txt", "w") as outfile:
    content = infile.read()
    outfile.write(content.upper())

# Context manager with exception
try:
    with open("data.txt") as file:
        content = file.read()
        number = int(content)
except FileNotFoundError:
    print("File not found")
except ValueError:
    print("Invalid number")
# File still closed even if exception occurs

# Suppress exceptions
from contextlib import suppress

with suppress(FileNotFoundError):
    with open("missing.txt") as file:
        content = file.read()
# No exception raised if file not found
```

### Creating Context Managers

```python
# Using class
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
        # Return True to suppress exception, False to propagate
        return False

# Use custom context manager
with FileManager("data.txt", "r") as file:
    content = file.read()

# Using contextlib.contextmanager decorator
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    file = None
    try:
        file = open(filename, mode)
        yield file
    finally:
        if file:
            file.close()

# Use decorator-based context manager
with file_manager("data.txt", "r") as file:
    content = file.read()

# Context manager for timing
import time

@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f"{name} took {end - start:.2f} seconds")

with timer("Operation"):
    # Do something
    time.sleep(1)

# Context manager for database transaction
@contextmanager
def transaction(connection):
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise

# Context manager that suppresses exceptions
@contextmanager
def safe_operation():
    try:
        yield
    except Exception as e:
        print(f"Error occurred: {e}")
        # Exception suppressed
```

## Advanced Exception Handling

### Exception Chaining

```python
# Exception chaining with 'from'
try:
    try:
        result = int("abc")
    except ValueError as e:
        raise TypeError("Invalid input") from e
except TypeError as e:
    print(e)                                # Invalid input
    print(e.__cause__)                      # original ValueError

# Implicit exception chaining
try:
    try:
        result = 10 / 0
    except ZeroDivisionError:
        result = int("abc")                 # New exception in handler
except ValueError as e:
    print(e.__context__)                    # Original ZeroDivisionError

# Suppress context
try:
    result = int("abc")
except ValueError:
    raise TypeError("Bad input") from None  # No __cause__ or __context__
```

### Exception Groups (Python 3.11+)

```python
# ExceptionGroup for multiple exceptions
def process_items(items):
    errors = []
    for item in items:
        try:
            process(item)
        except ValueError as e:
            errors.append(e)

    if errors:
        raise ExceptionGroup("Processing failed", errors)

# Catch exception groups
try:
    process_items([1, "bad", 3, "invalid"])
except ExceptionGroup as eg:
    for e in eg.exceptions:
        print(f"Error: {e}")

# Selective handling with except*
try:
    raise ExceptionGroup("multiple errors", [
        ValueError("bad value"),
        TypeError("bad type"),
        KeyError("missing key")
    ])
except* ValueError as eg:
    print(f"Caught {len(eg.exceptions)} ValueError(s)")
except* TypeError as eg:
    print(f"Caught {len(eg.exceptions)} TypeError(s)")
```

### Assertions

```python
# Basic assertion
x = 5
assert x > 0                                # Passes
assert x > 10                               # AssertionError

# Assertion with message
age = -5
assert age >= 0, f"Age cannot be negative: {age}"

# Use in functions
def divide(a, b):
    assert b != 0, "Divisor cannot be zero"
    return a / b

# Assertions for debugging (can be disabled with -O flag)
def calculate_discount(price, discount):
    assert 0 <= discount <= 100, "Discount must be between 0 and 100"
    assert price > 0, "Price must be positive"
    return price * (1 - discount / 100)

# Disable assertions
# python -O script.py                      # Assertions are ignored
```

### Warnings

```python
import warnings

# Issue warning
warnings.warn("This is deprecated", DeprecationWarning)

# Different warning types
warnings.warn("This might cause issues", UserWarning)
warnings.warn("This will change in future", FutureWarning)
warnings.warn("This is experimental", RuntimeWarning)

# Filter warnings
warnings.filterwarnings("ignore")           # Ignore all warnings
warnings.filterwarnings("error")            # Turn warnings into exceptions
warnings.filterwarnings("always")           # Always show warnings
warnings.filterwarnings("default")          # Show warning once per location

# Specific warning filter
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Catch warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Warnings ignored in this block
    warnings.warn("This won't be shown")

# Custom warning
class CustomWarning(UserWarning):
    pass

warnings.warn("Custom warning message", CustomWarning)

# Show warning details
warnings.warn("Detailed warning", UserWarning, stacklevel=2)
```

### Traceback

```python
import traceback
import sys

# Print exception traceback
try:
    result = 10 / 0
except ZeroDivisionError:
    traceback.print_exc()

# Get traceback as string
try:
    result = 10 / 0
except ZeroDivisionError:
    tb_str = traceback.format_exc()
    print(tb_str)

# Get exception info
try:
    result = 10 / 0
except:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print(f"Type: {exc_type}")
    print(f"Value: {exc_value}")
    print(f"Traceback: {exc_traceback}")

# Extract traceback information
try:
    result = 10 / 0
except:
    tb_lines = traceback.format_tb(sys.exc_info()[2])
    for line in tb_lines:
        print(line)

# Print stack trace
def function_a():
    function_b()

def function_b():
    traceback.print_stack()

# Format exception
try:
    result = 10 / 0
except ZeroDivisionError:
    exc_info = sys.exc_info()
    formatted = traceback.format_exception(*exc_info)
    for line in formatted:
        print(line, end="")
```

## Best Practices

### Exception Handling Patterns

```python
# Be specific with exceptions
# Bad
try:
    process_data()
except:                                     # Catches everything!
    pass

# Good
try:
    process_data()
except ValueError:
    handle_value_error()
except FileNotFoundError:
    handle_file_not_found()

# Don't suppress exceptions silently
# Bad
try:
    risky_operation()
except Exception:
    pass                                    # Silent failure

# Good
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise

# Use finally for cleanup
# Bad
file = open("data.txt")
try:
    content = file.read()
except:
    file.close()
    raise
file.close()

# Good
file = open("data.txt")
try:
    content = file.read()
finally:
    file.close()

# Even better - use context manager
with open("data.txt") as file:
    content = file.read()

# Fail fast
def process_order(order):
    if not order:
        raise ValueError("Order cannot be None")
    if order.total < 0:
        raise ValueError("Order total cannot be negative")
    # Process valid order
    pass
```

### EAFP vs LBYL

```python
# LBYL (Look Before You Leap)
if key in dictionary:
    value = dictionary[key]
else:
    value = default_value

if os.path.exists(filename):
    with open(filename) as file:
        content = file.read()

# EAFP (Easier to Ask for Forgiveness than Permission) - Pythonic
try:
    value = dictionary[key]
except KeyError:
    value = default_value

try:
    with open(filename) as file:
        content = file.read()
except FileNotFoundError:
    handle_missing_file()

# EAFP is often better in Python
# - Avoids race conditions
# - Cleaner code
# - Better performance in success case
```

### Error Recovery

```python
# Retry logic
def retry_operation(func, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(1)

# Use retry
result = retry_operation(lambda: risky_api_call())

# Fallback values
def get_config_value(key):
    try:
        return config[key]
    except KeyError:
        return default_config[key]

# Graceful degradation
def get_user_data(user_id):
    try:
        return database.get_user(user_id)
    except DatabaseError:
        logger.warning("Database unavailable, using cache")
        try:
            return cache.get_user(user_id)
        except CacheError:
            logger.error("Cache also unavailable")
            return get_default_user()

# Circuit breaker pattern
class CircuitBreaker:
    def __init__(self, failure_threshold=5):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.is_open = False

    def call(self, func):
        if self.is_open:
            raise Exception("Circuit breaker is open")

        try:
            result = func()
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
            raise
```

### Logging Exceptions

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log exceptions
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    raise

# Log with traceback
try:
    result = risky_operation()
except Exception:
    logger.exception("Operation failed")    # Includes traceback

# Different log levels
try:
    result = risky_operation()
except ValueError:
    logger.warning("Invalid value")
except FileNotFoundError:
    logger.error("File not found")
except Exception:
    logger.critical("Critical error")
    raise
```

---

*This document covers comprehensive exception handling in Python. For the most up-to-date information, refer to the official Python documentation.*
