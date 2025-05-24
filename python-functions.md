# Python Built-in Functions

This document provides a comprehensive list of all Python built-in functions with their syntax and usage examples.

## Numeric Functions

### `abs(x)`
Returns the absolute value of a number.
```python
abs(-5)        # Returns 5
abs(3.14)      # Returns 3.14
abs(-2+3j)     # Returns 3.605551275463989
```

### `divmod(a, b)`
Returns a tuple containing the quotient and remainder when dividing a by b.
```python
divmod(10, 3)  # Returns (3, 1)
divmod(9, 4)   # Returns (2, 1)
```

### `max(*args, key=None)`
Returns the largest item in an iterable or the largest of two or more arguments.
```python
max(1, 2, 3)           # Returns 3
max([1, 2, 3])         # Returns 3
max("abc", key=len)    # Returns "abc"
```

### `min(*args, key=None)`
Returns the smallest item in an iterable or the smallest of two or more arguments.
```python
min(1, 2, 3)           # Returns 1
min([1, 2, 3])         # Returns 1
min("abc", "ab", key=len)  # Returns "ab"
```

### `pow(base, exp, mod=None)`
Returns base raised to the power exp; if mod is present, returns base**exp % mod.
```python
pow(2, 3)      # Returns 8
pow(2, 3, 5)   # Returns 3 (8 % 5)
```

### `round(number, ndigits=None)`
Returns a floating point number rounded to ndigits precision after the decimal point.
```python
round(3.14159, 2)  # Returns 3.14
round(1234.5, -1)  # Returns 1230.0
```

### `sum(iterable, start=0)`
Sums start and the items of an iterable from left to right and returns the total.
```python
sum([1, 2, 3])     # Returns 6
sum([1, 2, 3], 10) # Returns 16
```

## Type Conversion Functions

### `bool(x=False)`
Returns a Boolean value, i.e. one of True or False.
```python
bool(1)        # Returns True
bool(0)        # Returns False
bool("hello")  # Returns True
bool("")       # Returns False
```

### `bytearray(source=b'', encoding='utf-8', errors='strict')`
Returns a new array of bytes.
```python
bytearray(5)           # Returns bytearray(b'\x00\x00\x00\x00\x00')
bytearray("hello", "utf-8")  # Returns bytearray(b'hello')
```

### `bytes(source=b'', encoding='utf-8', errors='strict')`
Returns a new "bytes" object.
```python
bytes(5)               # Returns b'\x00\x00\x00\x00\x00'
bytes("hello", "utf-8") # Returns b'hello'
```

### `complex(real=0, imag=0)`
Returns a complex number with the value real + imag*1j.
```python
complex(1, 2)    # Returns (1+2j)
complex("3+4j")  # Returns (3+4j)
```

### `dict(**kwarg)` / `dict(mapping, **kwarg)` / `dict(iterable, **kwarg)`
Creates a new dictionary.
```python
dict()                    # Returns {}
dict(a=1, b=2)           # Returns {'a': 1, 'b': 2}
dict([('a', 1), ('b', 2)]) # Returns {'a': 1, 'b': 2}
```

### `float(x=0.0)`
Returns a floating point number constructed from a number or string.
```python
float(3)        # Returns 3.0
float("3.14")   # Returns 3.14
```

### `frozenset(iterable=set())`
Returns a new frozenset object.
```python
frozenset([1, 2, 3])     # Returns frozenset({1, 2, 3})
frozenset("hello")       # Returns frozenset({'h', 'e', 'l', 'o'})
```

### `int(x=0)` / `int(x, base=10)`
Returns an integer object constructed from a number or string.
```python
int(3.14)       # Returns 3
int("42")       # Returns 42
int("ff", 16)   # Returns 255
```

### `list(iterable=())`
Returns a list whose items are the same and in the same order as iterable's items.
```python
list("hello")           # Returns ['h', 'e', 'l', 'l', 'o']
list(range(3))          # Returns [0, 1, 2]
```

### `set(iterable=set())`
Returns a new set object.
```python
set([1, 2, 3, 2])       # Returns {1, 2, 3}
set("hello")            # Returns {'h', 'e', 'l', 'o'}
```

### `str(object='')` / `str(object=b'', encoding='utf-8', errors='strict')`
Returns a string version of object.
```python
str(123)        # Returns "123"
str([1, 2, 3])  # Returns "[1, 2, 3]"
```

### `tuple(iterable=())`
Returns a tuple whose items are the same and in the same order as iterable's items.
```python
tuple([1, 2, 3])        # Returns (1, 2, 3)
tuple("hello")          # Returns ('h', 'e', 'l', 'l', 'o')
```

## Sequence Functions

### `all(iterable)`
Returns True if all elements of the iterable are true (or if the iterable is empty).
```python
all([True, True, True])   # Returns True
all([True, False, True])  # Returns False
all([])                   # Returns True
```

### `any(iterable)`
Returns True if any element of the iterable is true.
```python
any([False, True, False]) # Returns True
any([False, False])       # Returns False
any([])                   # Returns False
```

### `enumerate(iterable, start=0)`
Returns an enumerate object.
```python
list(enumerate(['a', 'b', 'c']))      # Returns [(0, 'a'), (1, 'b'), (2, 'c')]
list(enumerate(['a', 'b'], start=1))  # Returns [(1, 'a'), (2, 'b')]
```

### `filter(function, iterable)`
Constructs an iterator from those elements of iterable for which function returns true.
```python
list(filter(lambda x: x > 0, [-1, 0, 1, 2]))  # Returns [1, 2]
list(filter(None, [0, 1, False, True]))        # Returns [1, True]
```

### `len(s)`
Returns the length (the number of items) of an object.
```python
len("hello")      # Returns 5
len([1, 2, 3])    # Returns 3
len({"a": 1})     # Returns 1
```

### `map(function, iterable, ...)`
Returns an iterator that applies function to every item of iterable.
```python
list(map(str, [1, 2, 3]))           # Returns ['1', '2', '3']
list(map(lambda x: x**2, [1, 2, 3])) # Returns [1, 4, 9]
```

### `range(stop)` / `range(start, stop, step=1)`
Returns an immutable sequence of numbers.
```python
list(range(5))        # Returns [0, 1, 2, 3, 4]
list(range(1, 5))     # Returns [1, 2, 3, 4]
list(range(0, 10, 2)) # Returns [0, 2, 4, 6, 8]
```

### `reversed(seq)`
Returns a reverse iterator.
```python
list(reversed([1, 2, 3]))     # Returns [3, 2, 1]
list(reversed("hello"))       # Returns ['o', 'l', 'l', 'e', 'h']
```

### `sorted(iterable, key=None, reverse=False)`
Returns a new sorted list from the items in iterable.
```python
sorted([3, 1, 2])             # Returns [1, 2, 3]
sorted([3, 1, 2], reverse=True) # Returns [3, 2, 1]
sorted(["apple", "pie"], key=len) # Returns ["pie", "apple"]
```

### `zip(*iterables)`
Returns an iterator of tuples.
```python
list(zip([1, 2, 3], ['a', 'b', 'c']))  # Returns [(1, 'a'), (2, 'b'), (3, 'c')]
list(zip([1, 2], ['a', 'b'], [10, 20])) # Returns [(1, 'a', 10), (2, 'b', 20)]
```

## Input/Output Functions

### `input(prompt='')`
Reads a line from input, converts it to a string (stripping a trailing newline), and returns that.
```python
name = input("Enter your name: ")  # Waits for user input
```

### `print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)`
Prints objects to the text stream file, separated by sep and followed by end.
```python
print("Hello", "World")           # Prints: Hello World
print("Hello", "World", sep="-")  # Prints: Hello-World
print("Hello", end="")            # Prints: Hello (no newline)
```

### `open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)`
Opens file and returns a corresponding file object.
```python
f = open("file.txt", "r")         # Opens file for reading
f = open("file.txt", "w")         # Opens file for writing
f = open("file.txt", "a")         # Opens file for appending
```

## Object Inspection Functions

### `callable(object)`
Returns True if the object argument appears callable, False if not.
```python
callable(print)       # Returns True
callable(42)          # Returns False
callable(lambda: 1)   # Returns True
```

### `dir(object=None)`
Returns a list of valid attributes for that object.
```python
dir(str)              # Returns list of string methods
dir()                 # Returns names in current scope
```

### `getattr(object, name, default=None)`
Returns the value of the named attribute of object.
```python
getattr(list, "append")        # Returns <method 'append' of 'list' objects>
getattr(list, "nonexistent", "default")  # Returns "default"
```

### `globals()`
Returns a dictionary representing the current global symbol table.
```python
globals()             # Returns dict of global variables
```

### `hasattr(object, name)`
Returns True if the string is the name of one of the object's attributes.
```python
hasattr(list, "append")        # Returns True
hasattr(list, "nonexistent")   # Returns False
```

### `id(object)`
Returns the "identity" of an object.
```python
id(42)                # Returns unique integer for object
```

### `isinstance(object, classinfo)`
Returns True if the object argument is an instance of the classinfo argument.
```python
isinstance(42, int)           # Returns True
isinstance("hello", str)      # Returns True
isinstance([1, 2], (list, tuple)) # Returns True
```

### `issubclass(class, classinfo)`
Returns True if class is a subclass of classinfo.
```python
issubclass(bool, int)         # Returns True
issubclass(str, int)          # Returns False
```

### `locals()`
Updates and returns a dictionary representing the current local symbol table.
```python
def func():
    x = 1
    return locals()           # Returns {'x': 1}
```

### `type(object)` / `type(name, bases, dict)`
Returns the type of an object or creates a new type object.
```python
type(42)              # Returns <class 'int'>
type("hello")         # Returns <class 'str'>
```

### `vars(object=None)`
Returns the __dict__ attribute for a module, class, instance, or any other object.
```python
class Example:
    def __init__(self):
        self.x = 1
obj = Example()
vars(obj)             # Returns {'x': 1}
```

## Iterator Functions

### `iter(object, sentinel=None)`
Returns an iterator object.
```python
iter([1, 2, 3])       # Returns list iterator
iter("hello")         # Returns string iterator
```

### `next(iterator, default=None)`
Retrieves the next item from the iterator.
```python
it = iter([1, 2, 3])
next(it)              # Returns 1
next(it)              # Returns 2
next(it, "end")       # Returns 3
next(it, "end")       # Returns "end"
```

## Attribute Functions

### `delattr(object, name)`
Deletes the named attribute from object.
```python
class Example:
    x = 1
obj = Example()
delattr(obj, 'x')     # Removes attribute x
```

### `setattr(object, name, value)`
Sets the named attribute on the given object to the specified value.
```python
class Example:
    pass
obj = Example()
setattr(obj, 'x', 42) # Sets obj.x = 42
```

## Code Execution Functions

### `compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1)`
Compiles the source into a code or AST object.
```python
code = compile("print('hello')", "<string>", "exec")
exec(code)            # Prints: hello
```

### `eval(expression, globals=None, locals=None)`
Evaluates the given expression and returns the result.
```python
eval("2 + 3")         # Returns 5
eval("len('hello')")  # Returns 5
```

### `exec(object, globals=None, locals=None)`
Executes the given object (string, bytes, or code object).
```python
exec("print('hello')")  # Prints: hello
exec("x = 42")          # Creates variable x
```

## Formatting Functions

### `ascii(object)`
Returns a string containing a printable representation of an object with non-ASCII characters escaped.
```python
ascii("hello")        # Returns "'hello'"
ascii("café")         # Returns "'caf\\xe9'"
```

### `bin(x)`
Converts an integer number to a binary string prefixed with "0b".
```python
bin(10)               # Returns '0b1010'
bin(-10)              # Returns '-0b1010'
```

### `format(value, format_spec='')`
Converts a value to a "formatted" representation.
```python
format(42, 'b')       # Returns '101010' (binary)
format(3.14159, '.2f') # Returns '3.14'
```

### `hex(x)`
Converts an integer number to a lowercase hexadecimal string prefixed with "0x".
```python
hex(255)              # Returns '0xff'
hex(16)               # Returns '0x10'
```

### `oct(x)`
Converts an integer number to an octal string prefixed with "0o".
```python
oct(8)                # Returns '0o10'
oct(64)               # Returns '0o100'
```

### `ord(c)`
Returns an integer representing the Unicode character.
```python
ord('A')              # Returns 65
ord('€')              # Returns 8364
```

### `chr(i)`
Returns the string representing a character whose Unicode code point is the integer i.
```python
chr(65)               # Returns 'A'
chr(8364)             # Returns '€'
```

### `repr(object)`
Returns a string containing a printable representation of an object.
```python
repr("hello")         # Returns "'hello'"
repr([1, 2, 3])       # Returns '[1, 2, 3]'
```

## Memory Management Functions

### `hash(object)`
Returns the hash value of the object (if it has one).
```python
hash("hello")         # Returns hash value
hash((1, 2, 3))       # Returns hash value
```

### `memoryview(object)`
Returns a memory view object created from the given argument.
```python
data = bytearray(b"hello")
mv = memoryview(data) # Returns memory view
```

## Special Functions

### `slice(stop)` / `slice(start, stop, step=None)`
Returns a slice object representing the set of indices.
```python
s = slice(2, 8, 2)
[1, 2, 3, 4, 5, 6, 7, 8, 9][s]  # Returns [3, 5, 7]
```

### `object()`
Returns a new featureless object.
```python
obj = object()        # Creates new object
```

### `property(fget=None, fset=None, fdel=None, doc=None)`
Returns a property attribute.
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
```

### `classmethod(function)`
Returns a class method for the given function.
```python
class MyClass:
    @classmethod
    def my_method(cls):
        return "class method"
```

### `staticmethod(function)`
Returns a static method for the given function.
```python
class MyClass:
    @staticmethod
    def my_method():
        return "static method"
```

### `super(type=None, object_or_type=None)`
Returns a proxy object that delegates method calls to a parent or sibling class.
```python
class Parent:
    def method(self):
        return "parent"

class Child(Parent):
    def method(self):
        return super().method() + " child"
```

### `__import__(name, globals=None, locals=None, fromlist=(), level=0)`
Imports a module (advanced use, prefer `import` statement).
```python
math = __import__('math')
math.sqrt(16)         # Returns 4.0
```

---

*This document covers all Python built-in functions as of Python 3.11. For the most up-to-date information, refer to the official Python documentation.*