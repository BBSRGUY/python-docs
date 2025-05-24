# Python String Functions and Methods

This document provides a comprehensive guide to all Python string-related functions, methods, packages, and built-ins with syntax and usage examples.

## String Literals and Creation

### String Literals
```python
# Single quotes
s1 = 'Hello World'

# Double quotes
s2 = "Hello World"

# Triple quotes (multiline)
s3 = '''This is a
multiline string'''

s4 = """Another
multiline string"""

# Raw strings (escape sequences not interpreted)
s5 = r'C:\Users\name\file.txt'

# f-strings (formatted string literals)
name = "Alice"
age = 30
s6 = f"My name is {name} and I'm {age} years old"

# Bytes strings
s7 = b'Hello'

# Unicode strings
s8 = u'Hello Unicode: \u03A9'
```

## String Methods

### Case Conversion Methods

#### `str.upper()`
Returns a copy of the string with all characters converted to uppercase.
```python
"hello world".upper()          # Returns "HELLO WORLD"
"Hello World".upper()          # Returns "HELLO WORLD"
```

#### `str.lower()`
Returns a copy of the string with all characters converted to lowercase.
```python
"HELLO WORLD".lower()          # Returns "hello world"
"Hello World".lower()          # Returns "hello world"
```

#### `str.capitalize()`
Returns a copy of the string with its first character capitalized and the rest lowercased.
```python
"hello world".capitalize()     # Returns "Hello world"
"HELLO WORLD".capitalize()     # Returns "Hello world"
```

#### `str.title()`
Returns a titlecased version of the string where words start with an uppercase character.
```python
"hello world".title()          # Returns "Hello World"
"hello-world".title()          # Returns "Hello-World"
```

#### `str.swapcase()`
Returns a copy of the string with uppercase characters converted to lowercase and vice versa.
```python
"Hello World".swapcase()       # Returns "hELLO wORLD"
"PyThOn".swapcase()            # Returns "pYtHoN"
```

#### `str.casefold()`
Returns a casefolded copy of the string (more aggressive than lower()).
```python
"HELLO".casefold()             # Returns "hello"
"ß".casefold()                 # Returns "ss"
```

### Searching and Testing Methods

#### `str.find(sub, start=0, end=len(string))`
Returns the lowest index where substring is found, or -1 if not found.
```python
"hello world".find("world")    # Returns 6
"hello world".find("foo")      # Returns -1
"hello world".find("l", 3)     # Returns 3
```

#### `str.rfind(sub, start=0, end=len(string))`
Returns the highest index where substring is found, or -1 if not found.
```python
"hello world".rfind("l")       # Returns 9
"hello world".rfind("foo")     # Returns -1
```

#### `str.index(sub, start=0, end=len(string))`
Like find(), but raises ValueError when the substring is not found.
```python
"hello world".index("world")   # Returns 6
# "hello world".index("foo")   # Raises ValueError
```

#### `str.rindex(sub, start=0, end=len(string))`
Like rfind(), but raises ValueError when the substring is not found.
```python
"hello world".rindex("l")      # Returns 9
# "hello world".rindex("foo")  # Raises ValueError
```

#### `str.count(sub, start=0, end=len(string))`
Returns the number of non-overlapping occurrences of substring.
```python
"hello world".count("l")       # Returns 3
"banana".count("ana")          # Returns 1
```

#### `str.startswith(prefix, start=0, end=len(string))`
Returns True if string starts with the specified prefix.
```python
"hello world".startswith("hello")    # Returns True
"hello world".startswith(("hi", "hello"))  # Returns True
```

#### `str.endswith(suffix, start=0, end=len(string))`
Returns True if string ends with the specified suffix.
```python
"hello world".endswith("world")      # Returns True
"hello world".endswith((".txt", ".py"))  # Returns False
```

#### `str.in` operator
Tests for substring membership.
```python
"world" in "hello world"        # Returns True
"foo" in "hello world"          # Returns False
```

### Character Classification Methods

#### `str.isalpha()`
Returns True if all characters are alphabetic and there is at least one character.
```python
"hello".isalpha()              # Returns True
"hello123".isalpha()           # Returns False
"".isalpha()                   # Returns False
```

#### `str.isdigit()`
Returns True if all characters are digits and there is at least one character.
```python
"123".isdigit()                # Returns True
"12.3".isdigit()               # Returns False
"".isdigit()                   # Returns False
```

#### `str.isalnum()`
Returns True if all characters are alphanumeric and there is at least one character.
```python
"hello123".isalnum()           # Returns True
"hello 123".isalnum()          # Returns False
```

#### `str.isspace()`
Returns True if there are only whitespace characters and there is at least one character.
```python
"   ".isspace()                # Returns True
" \t\n".isspace()              # Returns True
"hello ".isspace()             # Returns False
```

#### `str.islower()`
Returns True if all cased characters are lowercase and there is at least one cased character.
```python
"hello".islower()              # Returns True
"Hello".islower()              # Returns False
"hello123".islower()           # Returns True
```

#### `str.isupper()`
Returns True if all cased characters are uppercase and there is at least one cased character.
```python
"HELLO".isupper()              # Returns True
"Hello".isupper()              # Returns False
"HELLO123".isupper()           # Returns True
```

#### `str.istitle()`
Returns True if string is titlecased.
```python
"Hello World".istitle()        # Returns True
"Hello world".istitle()        # Returns False
```

#### `str.isdecimal()`
Returns True if all characters are decimal characters.
```python
"123".isdecimal()              # Returns True
"12.3".isdecimal()             # Returns False
```

#### `str.isnumeric()`
Returns True if all characters are numeric characters.
```python
"123".isnumeric()              # Returns True
"½".isnumeric()                # Returns True
"12.3".isnumeric()             # Returns False
```

#### `str.isascii()`
Returns True if all characters are ASCII characters or the string is empty.
```python
"hello".isascii()              # Returns True
"café".isascii()               # Returns False
"".isascii()                   # Returns True
```

#### `str.isprintable()`
Returns True if all characters are printable or the string is empty.
```python
"hello world".isprintable()    # Returns True
"hello\nworld".isprintable()   # Returns False
```

#### `str.isidentifier()`
Returns True if the string is a valid identifier according to Python syntax.
```python
"hello_world".isidentifier()   # Returns True
"123hello".isidentifier()      # Returns False
"hello world".isidentifier()   # Returns False
```

### Splitting and Joining Methods

#### `str.split(sep=None, maxsplit=-1)`
Returns a list of words in the string, using sep as the delimiter.
```python
"hello world python".split()         # Returns ['hello', 'world', 'python']
"hello,world,python".split(",")      # Returns ['hello', 'world', 'python']
"hello,world,python".split(",", 1)   # Returns ['hello', 'world,python']
```

#### `str.rsplit(sep=None, maxsplit=-1)`
Returns a list of words in the string, using sep as the delimiter, starting from the right.
```python
"hello,world,python".rsplit(",", 1)  # Returns ['hello,world', 'python']
```

#### `str.splitlines(keepends=False)`
Returns a list of lines in the string, breaking at line boundaries.
```python
"hello\nworld\npython".splitlines()     # Returns ['hello', 'world', 'python']
"hello\nworld\n".splitlines(True)       # Returns ['hello\n', 'world\n']
```

#### `str.partition(sep)`
Splits the string at the first occurrence of sep and returns a 3-tuple.
```python
"hello@world.com".partition("@")     # Returns ('hello', '@', 'world.com')
"hello world".partition("@")         # Returns ('hello world', '', '')
```

#### `str.rpartition(sep)`
Splits the string at the last occurrence of sep and returns a 3-tuple.
```python
"hello@world@com".rpartition("@")    # Returns ('hello@world', '@', 'com')
```

#### `str.join(iterable)`
Returns a string which is the concatenation of the strings in iterable.
```python
",".join(["hello", "world", "python"])    # Returns "hello,world,python"
" ".join(["hello", "world"])              # Returns "hello world"
"".join(["a", "b", "c"])                  # Returns "abc"
```

### Padding and Alignment Methods

#### `str.ljust(width, fillchar=' ')`
Returns the string left-justified in a string of length width.
```python
"hello".ljust(10)              # Returns "hello     "
"hello".ljust(10, "*")         # Returns "hello*****"
```

#### `str.rjust(width, fillchar=' ')`
Returns the string right-justified in a string of length width.
```python
"hello".rjust(10)              # Returns "     hello"
"hello".rjust(10, "*")         # Returns "*****hello"
```

#### `str.center(width, fillchar=' ')`
Returns the string centered in a string of length width.
```python
"hello".center(10)             # Returns "  hello   "
"hello".center(10, "*")        # Returns "**hello***"
```

#### `str.zfill(width)`
Returns a copy of the string left filled with ASCII '0' digits.
```python
"42".zfill(5)                  # Returns "00042"
"-42".zfill(5)                 # Returns "-0042"
```

### Trimming Methods

#### `str.strip(chars=None)`
Returns a copy of the string with leading and trailing characters removed.
```python
"  hello world  ".strip()      # Returns "hello world"
"xxxhello worldxxx".strip("x") # Returns "hello world"
```

#### `str.lstrip(chars=None)`
Returns a copy of the string with leading characters removed.
```python
"  hello world  ".lstrip()     # Returns "hello world  "
"xxxhello world".lstrip("x")   # Returns "hello world"
```

#### `str.rstrip(chars=None)`
Returns a copy of the string with trailing characters removed.
```python
"  hello world  ".rstrip()     # Returns "  hello world"
"hello worldxxx".rstrip("x")   # Returns "hello world"
```

#### `str.removeprefix(prefix)`
Returns a string with the given prefix string removed if present.
```python
"HelloWorld".removeprefix("Hello")  # Returns "World"
"HelloWorld".removeprefix("Hi")     # Returns "HelloWorld"
```

#### `str.removesuffix(suffix)`
Returns a string with the given suffix string removed if present.
```python
"HelloWorld".removesuffix("World")  # Returns "Hello"
"HelloWorld".removesuffix("Earth")  # Returns "HelloWorld"
```

### Replacement Methods

#### `str.replace(old, new, count=-1)`
Returns a copy of the string with all occurrences of substring old replaced by new.
```python
"hello world".replace("world", "python")    # Returns "hello python"
"hello hello".replace("hello", "hi", 1)     # Returns "hi hello"
```

#### `str.translate(table)`
Returns a copy of the string mapped through the given translation table.
```python
# Create translation table
table = str.maketrans("aeiou", "12345")
"hello world".translate(table)         # Returns "h2ll4 w4rld"

# Remove characters
table = str.maketrans("", "", "aeiou")
"hello world".translate(table)         # Returns "hll wrld"
```

#### `str.maketrans(x, y=None, z=None)`
Static method to create a translation table.
```python
# Character replacement
table = str.maketrans("abc", "123")

# Character removal
table = str.maketrans("", "", "aeiou")

# Dictionary mapping
table = str.maketrans({"a": "1", "b": "2", "c": "3"})
```

### Encoding and Decoding Methods

#### `str.encode(encoding='utf-8', errors='strict')`
Returns an encoded version of the string as a bytes object.
```python
"hello".encode()               # Returns b'hello'
"café".encode("utf-8")         # Returns b'caf\xc3\xa9'
"café".encode("ascii", "ignore")  # Returns b'caf'
```

#### `bytes.decode(encoding='utf-8', errors='strict')`
Returns a string decoded from the given bytes.
```python
b'hello'.decode()              # Returns "hello"
b'caf\xc3\xa9'.decode("utf-8") # Returns "café"
```

### Formatting Methods

#### `str.format(*args, **kwargs)`
Performs string formatting operation.
```python
"Hello {0}".format("World")                    # Returns "Hello World"
"Hello {name}".format(name="Alice")            # Returns "Hello Alice"
"{0:.2f}".format(3.14159)                      # Returns "3.14"
"{:>10}".format("hello")                       # Returns "     hello"
```

#### `str.format_map(mapping)`
Similar to format(**mapping), but mapping is used directly.
```python
"Hello {name}".format_map({"name": "Alice"})   # Returns "Hello Alice"
```

#### f-strings (Formatted String Literals)
```python
name = "Alice"
age = 30
f"Hello {name}, you are {age} years old"       # Returns "Hello Alice, you are 30 years old"
f"{3.14159:.2f}"                               # Returns "3.14"
f"{'hello':>10}"                               # Returns "     hello"
```

#### % formatting (old style)
```python
"Hello %s" % "World"                           # Returns "Hello World"
"Hello %s, you are %d years old" % ("Alice", 30)  # Returns "Hello Alice, you are 30 years old"
"Pi is %.2f" % 3.14159                         # Returns "Pi is 3.14"
```

## String Constants (from string module)

```python
import string

# ASCII letters
string.ascii_letters      # 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
string.ascii_lowercase    # 'abcdefghijklmnopqrstuvwxyz'
string.ascii_uppercase    # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# Digits
string.digits             # '0123456789'
string.hexdigits          # '0123456789abcdefABCDEF'
string.octdigits          # '01234567'

# Special characters
string.punctuation        # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
string.printable          # All printable ASCII characters
string.whitespace         # ' \t\n\r\x0b\x0c'
```

## String Module Functions

### `string.capwords(s, sep=None)`
Split the argument into words using split(), capitalize each word, then join using join().
```python
import string
string.capwords("hello world python")         # Returns "Hello World Python"
string.capwords("hello-world-python", "-")    # Returns "Hello-World-Python"
```

### Template Strings

#### `string.Template(template)`
A string class for supporting $-substitutions.
```python
import string

# Basic substitution
template = string.Template("Hello $name")
template.substitute(name="Alice")              # Returns "Hello Alice"

# Safe substitution (doesn't raise error for missing keys)
template = string.Template("Hello $name, welcome to $place")
template.safe_substitute(name="Alice")         # Returns "Hello Alice, welcome to $place"

# Dictionary substitution
template = string.Template("$greeting $name")
template.substitute({"greeting": "Hello", "name": "Alice"})  # Returns "Hello Alice"
```

### `string.Formatter`
Custom string formatting class.
```python
import string

formatter = string.Formatter()
formatter.format("Hello {0}", "World")         # Returns "Hello World"
```

## Regular Expressions (re module)

### Basic Pattern Matching

```python
import re

# Search for pattern
re.search(r'world', 'hello world')            # Returns match object
re.match(r'hello', 'hello world')             # Returns match object
re.fullmatch(r'hello world', 'hello world')   # Returns match object

# Find all occurrences
re.findall(r'\d+', 'abc 123 def 456')         # Returns ['123', '456']
re.finditer(r'\d+', 'abc 123 def 456')        # Returns iterator of match objects

# Split string
re.split(r'\s+', 'hello   world  python')     # Returns ['hello', 'world', 'python']

# Replace pattern
re.sub(r'\d+', 'X', 'abc 123 def 456')        # Returns 'abc X def X'
re.subn(r'\d+', 'X', 'abc 123 def 456')       # Returns ('abc X def X', 2)
```

### Common Regular Expression Patterns

```python
import re

# Email validation
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
re.match(email_pattern, 'user@example.com')

# Phone number
phone_pattern = r'^\d{3}-\d{3}-\d{4}$'
re.match(phone_pattern, '123-456-7890')

# URL validation
url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
re.match(url_pattern, 'https://www.example.com')

# Extract numbers
numbers = re.findall(r'-?\d+\.?\d*', 'Price: $19.99, Discount: -5.5%')

# Word boundaries
re.findall(r'\bword\b', 'word, sword, words')  # Returns ['word']
```

### Compiled Patterns

```python
import re

# Compile pattern for reuse
pattern = re.compile(r'\d+')
pattern.findall('abc 123 def 456')            # Returns ['123', '456']
pattern.search('abc 123 def 456')             # Returns match object

# Case-insensitive matching
pattern = re.compile(r'hello', re.IGNORECASE)
pattern.search('HELLO WORLD')                 # Returns match object
```

## Text Processing Libraries

### `textwrap` module

```python
import textwrap

# Wrap text
text = "This is a very long line that needs to be wrapped to fit within a certain width."
wrapped = textwrap.wrap(text, width=20)
# Returns ['This is a very long', 'line that needs to be', 'wrapped to fit within', 'a certain width.']

# Fill text
filled = textwrap.fill(text, width=20)
# Returns string with newlines

# Dedent text
dedented = textwrap.dedent("""
    This text has
    leading whitespace
    """)

# Indent text
indented = textwrap.indent("line1\nline2", "    ")  # Returns "    line1\n    line2"

# Shorten text
shortened = textwrap.shorten("This is a long text", width=10)  # Returns "This [...]"
```

### `difflib` module

```python
import difflib

# Compare sequences
a = "hello world"
b = "hello python"

# Get similarity ratio
ratio = difflib.SequenceMatcher(None, a, b).ratio()  # Returns similarity ratio

# Get differences
diff = list(difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm=''))

# Get close matches
close_matches = difflib.get_close_matches("hello", ["helo", "help", "hello", "world"])
```

### `unicodedata` module

```python
import unicodedata

# Normalize unicode
text = "café"
normalized = unicodedata.normalize('NFKD', text)

# Get character info
unicodedata.name('A')                          # Returns 'LATIN CAPITAL LETTER A'
unicodedata.category('A')                      # Returns 'Lu' (Letter, uppercase)
unicodedata.numeric('5')                       # Returns 5.0

# Remove accents
import unicodedata
def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if unicodedata.category(c) != 'Mn')

remove_accents("café")                         # Returns "cafe"
```

### `locale` module

```python
import locale

# Set locale
locale.setlocale(locale.LC_ALL, '')

# Format numbers according to locale
locale.format_string("%.2f", 1234.56, grouping=True)

# String collation
locale.strcoll("café", "cafe")                 # Locale-aware string comparison
```

## String Validation and Parsing

### Email Validation
```python
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

validate_email("user@example.com")             # Returns True
```

### URL Parsing
```python
from urllib.parse import urlparse, parse_qs

url = "https://example.com/path?param1=value1&param2=value2"
parsed = urlparse(url)
# ParseResult(scheme='https', netloc='example.com', path='/path', 
#            params='', query='param1=value1&param2=value2', fragment='')

# Parse query parameters
query_params = parse_qs(parsed.query)
# {'param1': ['value1'], 'param2': ['value2']}
```

### CSV Parsing
```python
import csv
from io import StringIO

# Parse CSV string
csv_string = "name,age,city\nAlice,30,New York\nBob,25,Los Angeles"
reader = csv.reader(StringIO(csv_string))
for row in reader:
    print(row)  # ['name', 'age', 'city'], ['Alice', '30', 'New York'], etc.

# DictReader
reader = csv.DictReader(StringIO(csv_string))
for row in reader:
    print(row)  # {'name': 'Alice', 'age': '30', 'city': 'New York'}, etc.
```

### JSON String Handling
```python
import json

# JSON string to Python object
json_string = '{"name": "Alice", "age": 30}'
data = json.loads(json_string)                 # Returns dict

# Python object to JSON string
data = {"name": "Alice", "age": 30}
json_string = json.dumps(data)                 # Returns JSON string
json_pretty = json.dumps(data, indent=2)       # Returns formatted JSON
```

## String Performance Tips

### String Concatenation
```python
# Inefficient (creates new string objects)
result = ""
for i in range(1000):
    result += str(i)

# Efficient (using join)
result = "".join(str(i) for i in range(1000))

# Efficient (using list)
parts = []
for i in range(1000):
    parts.append(str(i))
result = "".join(parts)
```

### String Formatting Performance
```python
name = "Alice"
age = 30

# f-strings (fastest)
result = f"Hello {name}, you are {age} years old"

# format() method
result = "Hello {}, you are {} years old".format(name, age)

# % formatting (slowest)
result = "Hello %s, you are %d years old" % (name, age)
```

## Common String Operations Examples

### Password Validation
```python
import re

def validate_password(password):
    """Validate password: 8+ chars, uppercase, lowercase, digit, special char"""
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    return True
```

### Extract Information from Text
```python
import re

def extract_info(text):
    """Extract emails, phone numbers, and URLs from text"""
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    
    return {
        'emails': emails,
        'phones': phones,
        'urls': urls
    }
```

### Clean and Normalize Text
```python
import re
import unicodedata

def clean_text(text):
    """Clean and normalize text"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove accents
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    return text
```

### Slug Generation
```python
import re
import unicodedata

def slugify(text):
    """Convert text to URL-friendly slug"""
    # Normalize unicode and remove accents
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Convert to lowercase and replace spaces/special chars with hyphens
    text = re.sub(r'[^\w\s-]', '', text.lower())
    text = re.sub(r'[-\s]+', '-', text)
    
    return text.strip('-')

slugify("Hello World! This is a Test.")        # Returns "hello-world-this-is-a-test"
```

---

*This document covers comprehensive string handling in Python including built-in methods, standard library modules, and common patterns. For the most up-to-date information, refer to the official Python documentation.*