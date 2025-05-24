# Python Input/Output Functions and Methods

This document provides a comprehensive guide to all Python input/output related functions, methods, packages, and built-ins with syntax and usage examples.

## Built-in I/O Functions

### `input(prompt='')`
Reads a line from standard input, converts it to a string, and returns it.
```python
name = input("Enter your name: ")              # Waits for user input
age = input()                                  # Reads input without prompt
number = int(input("Enter a number: "))        # Convert input to integer
```

### `print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)`
Prints objects to the text stream file.
```python
print("Hello World")                           # Basic print
print("Hello", "World", sep="-")               # Custom separator: Hello-World
print("Hello", end="")                         # No newline at end
print("Hello", file=sys.stderr)                # Print to stderr
print("Hello", flush=True)                     # Force flush output buffer

# Multiple values
print("Name:", "Alice", "Age:", 30)            # Name: Alice Age: 30
```

## File Operations

### `open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None)`
Opens a file and returns a file object.

#### File Modes
```python
# Reading modes
f = open("file.txt", "r")          # Read text (default)
f = open("file.txt", "rb")         # Read binary
f = open("file.txt", "rt")         # Read text (explicit)

# Writing modes
f = open("file.txt", "w")          # Write text (overwrites)
f = open("file.txt", "wb")         # Write binary
f = open("file.txt", "wt")         # Write text (explicit)

# Appending modes
f = open("file.txt", "a")          # Append text
f = open("file.txt", "ab")         # Append binary

# Read/Write modes
f = open("file.txt", "r+")         # Read and write
f = open("file.txt", "w+")         # Write and read (overwrites)
f = open("file.txt", "a+")         # Append and read

# Exclusive creation
f = open("file.txt", "x")          # Create file, fail if exists
```

#### File Opening with Parameters
```python
# Specify encoding
f = open("file.txt", "r", encoding="utf-8")

# Buffer size
f = open("file.txt", "r", buffering=1024)

# Error handling
f = open("file.txt", "r", errors="ignore")     # Ignore decode errors
f = open("file.txt", "r", errors="replace")    # Replace with placeholder

# Newline handling
f = open("file.txt", "r", newline='')          # Don't translate newlines
f = open("file.txt", "r", newline='\n')        # Use specific newline
```

### File Context Managers
```python
# Recommended: Using with statement
with open("file.txt", "r") as f:
    content = f.read()
# File automatically closed

# Multiple files
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    data = infile.read()
    outfile.write(data.upper())
```

## File Object Methods

### Reading Methods

#### `file.read(size=-1)`
Reads and returns at most size characters from the file.
```python
with open("file.txt", "r") as f:
    content = f.read()                         # Read entire file
    content = f.read(10)                       # Read 10 characters
```

#### `file.readline(size=-1)`
Reads and returns one line from the file.
```python
with open("file.txt", "r") as f:
    line = f.readline()                        # Read one line
    line = f.readline(10)                      # Read at most 10 chars of line
```

#### `file.readlines(hint=-1)`
Reads and returns a list of lines from the file.
```python
with open("file.txt", "r") as f:
    lines = f.readlines()                      # Read all lines into list
    lines = f.readlines(100)                   # Read lines up to ~100 chars
```

### Writing Methods

#### `file.write(string)`
Writes the string to the file and returns the number of characters written.
```python
with open("file.txt", "w") as f:
    f.write("Hello World")                     # Write string
    chars_written = f.write("Hello")           # Returns 5
```

#### `file.writelines(lines)`
Writes a list of lines to the file.
```python
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
with open("file.txt", "w") as f:
    f.writelines(lines)                        # Write all lines
```

### File Position Methods

#### `file.tell()`
Returns the current file position.
```python
with open("file.txt", "r") as f:
    pos = f.tell()                             # Get current position
    f.read(10)
    new_pos = f.tell()                         # Position after reading
```

#### `file.seek(offset, whence=0)`
Changes the file position to the given byte offset.
```python
with open("file.txt", "r") as f:
    f.seek(0)                                  # Go to beginning
    f.seek(10)                                 # Go to position 10
    f.seek(0, 2)                               # Go to end (whence=2)
    f.seek(-10, 2)                             # Go 10 bytes before end
    f.seek(5, 1)                               # Move 5 bytes forward from current
```

### File Status Methods

#### `file.readable()`
Returns True if the file can be read.
```python
with open("file.txt", "r") as f:
    if f.readable():
        content = f.read()
```

#### `file.writable()`
Returns True if the file can be written to.
```python
with open("file.txt", "w") as f:
    if f.writable():
        f.write("Hello")
```

#### `file.seekable()`
Returns True if the file supports seek() and tell().
```python
with open("file.txt", "r") as f:
    if f.seekable():
        f.seek(10)
```

#### `file.closed`
Returns True if the file is closed.
```python
f = open("file.txt", "r")
print(f.closed)                                # False
f.close()
print(f.closed)                                # True
```

### File Utility Methods

#### `file.flush()`
Flushes the write buffers of the file.
```python
with open("file.txt", "w") as f:
    f.write("Hello")
    f.flush()                                   # Force write to disk
```

#### `file.close()`
Closes the file.
```python
f = open("file.txt", "r")
f.close()                                       # Manual close (not recommended)
```

#### `file.truncate(size=None)`
Truncates the file to at most size bytes.
```python
with open("file.txt", "r+") as f:
    f.truncate(100)                             # Truncate to 100 bytes
    f.truncate()                                # Truncate at current position
```

## File Iteration

### Reading Files Line by Line
```python
# Method 1: Direct iteration (recommended)
with open("file.txt", "r") as f:
    for line in f:
        print(line.strip())                     # Process each line

# Method 2: Using readlines()
with open("file.txt", "r") as f:
    for line in f.readlines():
        print(line.strip())

# Method 3: Using readline() in loop
with open("file.txt", "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        print(line.strip())
```

## Binary File Operations

### Reading Binary Files
```python
# Read binary file
with open("image.jpg", "rb") as f:
    data = f.read()                             # Read all bytes
    chunk = f.read(1024)                        # Read 1024 bytes

# Read binary in chunks
with open("large_file.bin", "rb") as f:
    while True:
        chunk = f.read(8192)                    # 8KB chunks
        if not chunk:
            break
        # Process chunk
```

### Writing Binary Files
```python
# Write binary data
data = b'\x89PNG\r\n\x1a\n'                    # Binary data
with open("output.bin", "wb") as f:
    f.write(data)

# Copy binary file
with open("source.bin", "rb") as src, open("dest.bin", "wb") as dst:
    dst.write(src.read())
```

## Standard Streams

### `sys.stdin`, `sys.stdout`, `sys.stderr`
```python
import sys

# Read from stdin
line = sys.stdin.readline()
for line in sys.stdin:
    print(f"Input: {line.strip()}")

# Write to stdout
sys.stdout.write("Hello World\n")

# Write to stderr
sys.stderr.write("Error message\n")

# Redirect output
original_stdout = sys.stdout
with open("output.txt", "w") as f:
    sys.stdout = f
    print("This goes to file")                  # Written to file
    sys.stdout = original_stdout
```

## The `io` Module

### String I/O

#### `io.StringIO(initial_value='', newline='\n')`
In-memory string buffer.
```python
import io

# Create string buffer
sio = io.StringIO()
sio.write("Hello ")
sio.write("World")
content = sio.getvalue()                        # Returns "Hello World"

# Initialize with content
sio = io.StringIO("Initial content")
content = sio.read()                            # Returns "Initial content"

# Use as file-like object
sio = io.StringIO()
print("Hello World", file=sio)
result = sio.getvalue()                         # Returns "Hello World\n"
```

### Bytes I/O

#### `io.BytesIO(initial_bytes=b'')`
In-memory bytes buffer.
```python
import io

# Create bytes buffer
bio = io.BytesIO()
bio.write(b"Hello ")
bio.write(b"World")
content = bio.getvalue()                        # Returns b"Hello World"

# Initialize with content
bio = io.BytesIO(b"Initial content")
content = bio.read()                            # Returns b"Initial content"

# Read/write operations
bio = io.BytesIO(b"Hello World")
bio.seek(6)
bio.write(b"Python")
result = bio.getvalue()                         # Returns b"Hello Python"
```

### Text I/O Wrapper

#### `io.TextIOWrapper(buffer, encoding=None, errors=None, newline=None)`
Text wrapper around binary buffer.
```python
import io

# Wrap binary buffer
bio = io.BytesIO(b"Hello World")
text_io = io.TextIOWrapper(bio, encoding="utf-8")
content = text_io.read()                        # Returns "Hello World"
```

### Base I/O Classes

#### `io.IOBase`
Base class for all I/O classes.
```python
import io

# Check if object is I/O object
f = open("file.txt", "r")
isinstance(f, io.IOBase)                        # Returns True
```

#### `io.RawIOBase`
Base class for raw binary I/O.

#### `io.BufferedIOBase`
Base class for buffered I/O.

#### `io.TextIOBase`
Base class for text I/O.

## File System Operations (`os` module)

### Path Operations
```python
import os

# Current working directory
cwd = os.getcwd()                               # Get current directory
os.chdir("/path/to/directory")                  # Change directory

# Path manipulation
path = os.path.join("folder", "subfolder", "file.txt")  # Cross-platform path
dirname = os.path.dirname("/path/to/file.txt")  # Returns "/path/to"
basename = os.path.basename("/path/to/file.txt") # Returns "file.txt"
name, ext = os.path.splitext("file.txt")        # Returns ("file", ".txt")

# Path information
exists = os.path.exists("file.txt")             # Check if path exists
is_file = os.path.isfile("file.txt")            # Check if it's a file
is_dir = os.path.isdir("directory")             # Check if it's a directory
size = os.path.getsize("file.txt")              # Get file size in bytes
```

### Directory Operations
```python
import os

# List directory contents
files = os.listdir(".")                         # List current directory
files = os.listdir("/path/to/directory")        # List specific directory

# Create directories
os.mkdir("new_directory")                       # Create single directory
os.makedirs("path/to/nested/directory")         # Create nested directories
os.makedirs("directory", exist_ok=True)         # Don't fail if exists

# Remove directories
os.rmdir("empty_directory")                     # Remove empty directory
import shutil
shutil.rmtree("directory_with_contents")        # Remove directory and contents
```

### File Operations
```python
import os
import shutil

# File operations
os.rename("old_name.txt", "new_name.txt")       # Rename file
shutil.copy("source.txt", "destination.txt")    # Copy file
shutil.copy2("source.txt", "dest.txt")          # Copy with metadata
shutil.move("source.txt", "destination.txt")    # Move file
os.remove("file.txt")                           # Delete file
os.unlink("file.txt")                           # Delete file (alias)

# File permissions
os.chmod("file.txt", 0o644)                     # Change file permissions
```

### Walking Directory Trees
```python
import os

# Walk directory tree
for root, dirs, files in os.walk("/path/to/directory"):
    print(f"Directory: {root}")
    for file in files:
        file_path = os.path.join(root, file)
        print(f"  File: {file_path}")
```

## Path Operations (`pathlib` module)

### `pathlib.Path`
Object-oriented path manipulation.
```python
from pathlib import Path

# Create Path objects
p = Path("folder/file.txt")
p = Path("/absolute/path/file.txt")
p = Path.home()                                 # User home directory
p = Path.cwd()                                  # Current working directory

# Path properties
print(p.name)                                   # File name
print(p.stem)                                   # File name without extension
print(p.suffix)                                 # File extension
print(p.parent)                                 # Parent directory
print(p.parents)                                # All parents
print(p.parts)                                  # Path components tuple

# Path operations
new_path = p / "subdirectory" / "file.txt"      # Join paths
absolute = p.resolve()                          # Get absolute path
relative = p.relative_to("/base/path")          # Get relative path
```

### Path Information
```python
from pathlib import Path

p = Path("file.txt")

# Check path properties
p.exists()                                      # Path exists
p.is_file()                                     # Is a file
p.is_dir()                                      # Is a directory
p.is_symlink()                                  # Is a symbolic link
p.is_absolute()                                 # Is absolute path

# File information
p.stat()                                        # File statistics
p.stat().st_size                                # File size
p.stat().st_mtime                               # Modification time
```

### Path File Operations
```python
from pathlib import Path

p = Path("file.txt")

# Read/write operations
content = p.read_text()                         # Read text file
content = p.read_text(encoding="utf-8")         # Read with encoding
data = p.read_bytes()                           # Read binary file

p.write_text("Hello World")                     # Write text file
p.write_text("Hello", encoding="utf-8")         # Write with encoding
p.write_bytes(b"Hello World")                   # Write binary file

# File operations
p.rename("new_name.txt")                        # Rename file
p.replace("new_name.txt")                       # Replace file
p.unlink()                                      # Delete file
p.unlink(missing_ok=True)                       # Don't fail if missing
```

### Directory Operations with pathlib
```python
from pathlib import Path

p = Path("directory")

# Directory operations
p.mkdir()                                       # Create directory
p.mkdir(parents=True)                           # Create with parents
p.mkdir(exist_ok=True)                          # Don't fail if exists
p.rmdir()                                       # Remove empty directory

# List directory contents
files = list(p.iterdir())                       # All items
files = list(p.glob("*.txt"))                   # Files matching pattern
files = list(p.rglob("*.txt"))                  # Recursive glob
files = [f for f in p.iterdir() if f.is_file()] # Only files
```

## Temporary Files (`tempfile` module)

### Temporary Files
```python
import tempfile

# Temporary file
with tempfile.NamedTemporaryFile(mode='w+t', delete=True) as temp:
    temp.write("Temporary content")
    temp.seek(0)
    content = temp.read()
    print(f"Temp file: {temp.name}")
# File automatically deleted

# Temporary file that persists
temp = tempfile.NamedTemporaryFile(delete=False)
temp.write(b"Temporary content")
temp_name = temp.name
temp.close()
# File still exists, must be manually deleted
os.unlink(temp_name)

# Temporary directory
with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir) / "temp_file.txt"
    temp_path.write_text("Hello")
# Directory automatically deleted
```

### Temporary File Creation
```python
import tempfile
import os

# Get temporary file descriptor
fd, path = tempfile.mkstemp()
try:
    with os.fdopen(fd, 'w') as temp_file:
        temp_file.write("Temporary content")
finally:
    os.unlink(path)

# Get temporary directory
temp_dir = tempfile.mkdtemp()
try:
    # Use temporary directory
    temp_file = os.path.join(temp_dir, "temp.txt")
    with open(temp_file, 'w') as f:
        f.write("Content")
finally:
    import shutil
    shutil.rmtree(temp_dir)

# Get temporary directory path
temp_dir = tempfile.gettempdir()                # System temp directory
```

## CSV File Operations (`csv` module)

### Reading CSV Files
```python
import csv

# Basic CSV reading
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)                              # List of strings

# CSV with custom delimiter
with open("data.csv", "r") as file:
    reader = csv.reader(file, delimiter=";")
    for row in reader:
        print(row)

# DictReader (first row as headers)
with open("data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)                              # Dictionary with column names
        print(row["column_name"])               # Access by column name
```

### Writing CSV Files
```python
import csv

# Basic CSV writing
with open("output.csv", "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age", "City"])    # Write header
    writer.writerow(["Alice", 30, "New York"])  # Write data row
    writer.writerows([                          # Write multiple rows
        ["Bob", 25, "Los Angeles"],
        ["Charlie", 35, "Chicago"]
    ])

# DictWriter
with open("output.csv", "w", newline='') as file:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()                        # Write header
    writer.writerow({"name": "Alice", "age": 30, "city": "New York"})
    writer.writerows([                          # Write multiple rows
        {"name": "Bob", "age": 25, "city": "Los Angeles"},
        {"name": "Charlie", "age": 35, "city": "Chicago"}
    ])
```

### CSV Dialects and Formatting
```python
import csv

# Register custom dialect
csv.register_dialect('custom', delimiter='|', quoting=csv.QUOTE_ALL)

# Use custom dialect
with open("data.csv", "r") as file:
    reader = csv.reader(file, dialect='custom')
    for row in reader:
        print(row)

# CSV formatting options
with open("output.csv", "w", newline='') as file:
    writer = csv.writer(file, 
                       delimiter=',',
                       quotechar='"',
                       quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["Value with, comma", "Normal value"])
```

## JSON File Operations (`json` module)

### Reading JSON Files
```python
import json

# Read JSON file
with open("data.json", "r") as file:
    data = json.load(file)                      # Parse JSON to Python object

# Read JSON from string
json_string = '{"name": "Alice", "age": 30}'
data = json.loads(json_string)                  # Parse JSON string
```

### Writing JSON Files
```python
import json

data = {"name": "Alice", "age": 30, "hobbies": ["reading", "swimming"]}

# Write JSON file
with open("output.json", "w") as file:
    json.dump(data, file)                       # Write object to file
    json.dump(data, file, indent=2)             # Pretty formatted
    json.dump(data, file, indent=2, ensure_ascii=False)  # Allow unicode

# Convert to JSON string
json_string = json.dumps(data)                  # Object to JSON string
json_pretty = json.dumps(data, indent=2)        # Pretty formatted string
```

### JSON Options
```python
import json

data = {"name": "Alice", "age": 30, "score": 95.5}

# JSON serialization options
json_str = json.dumps(data,
                     indent=2,                  # Pretty print
                     sort_keys=True,            # Sort dictionary keys
                     ensure_ascii=False,        # Allow unicode characters
                     separators=(',', ': '))    # Custom separators

# Custom JSON encoder
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

data_with_set = {"values": {1, 2, 3}}
json_str = json.dumps(data_with_set, cls=CustomEncoder)
```

## Pickle Serialization (`pickle` module)

### Pickle Operations
```python
import pickle

# Pickle (serialize) objects
data = {"name": "Alice", "numbers": [1, 2, 3], "nested": {"key": "value"}}

# Write pickle file
with open("data.pickle", "wb") as file:
    pickle.dump(data, file)                     # Serialize to file

# Serialize to bytes
pickled_data = pickle.dumps(data)               # Serialize to bytes

# Read pickle file
with open("data.pickle", "rb") as file:
    loaded_data = pickle.load(file)             # Deserialize from file

# Deserialize from bytes
loaded_data = pickle.loads(pickled_data)        # Deserialize from bytes
```

### Custom Pickle Behavior
```python
import pickle

class CustomClass:
    def __init__(self, value):
        self.value = value
    
    def __getstate__(self):
        # Custom serialization
        return {"custom_value": self.value}
    
    def __setstate__(self, state):
        # Custom deserialization
        self.value = state["custom_value"]

# Pickle custom objects
obj = CustomClass(42)
with open("custom.pickle", "wb") as file:
    pickle.dump(obj, file)
```

## Configuration Files (`configparser` module)

### Reading Configuration Files
```python
import configparser

# Create config parser
config = configparser.ConfigParser()

# Read config file
config.read("config.ini")

# Access values
value = config["section"]["key"]                # Direct access
value = config.get("section", "key")            # Method access
value = config.get("section", "key", fallback="default")  # With fallback

# Check sections and keys
sections = config.sections()                    # List all sections
has_section = config.has_section("section")     # Check if section exists
has_option = config.has_option("section", "key") # Check if key exists

# Get all items in section
items = config.items("section")                 # List of (key, value) tuples
```

### Writing Configuration Files
```python
import configparser

# Create config
config = configparser.ConfigParser()

# Add sections and values
config["DEFAULT"] = {
    "debug": "true",
    "log_level": "info"
}

config["database"] = {
    "host": "localhost",
    "port": "5432",
    "name": "mydb"
}

# Write to file
with open("config.ini", "w") as file:
    config.write(file)

# Add values programmatically
config.add_section("new_section")
config.set("new_section", "key", "value")
```

## YAML Files (`yaml` module - third-party)

```python
# Install: pip install PyYAML
import yaml

# Read YAML file
with open("config.yaml", "r") as file:
    data = yaml.safe_load(file)                 # Parse YAML to Python object

# Write YAML file
data = {"name": "Alice", "hobbies": ["reading", "swimming"]}
with open("output.yaml", "w") as file:
    yaml.dump(data, file, default_flow_style=False)  # Write object to YAML
```

## XML Files (`xml` module)

### XML Parsing with ElementTree
```python
import xml.etree.ElementTree as ET

# Parse XML file
tree = ET.parse("data.xml")
root = tree.getroot()

# Parse XML string
xml_string = "<root><item>value</item></root>"
root = ET.fromstring(xml_string)

# Access elements
for child in root:
    print(child.tag, child.text)

# Find elements
items = root.findall("item")                    # Find all item elements
item = root.find("item")                        # Find first item element
specific = root.find(".//specific")             # Find anywhere in tree

# Access attributes
for elem in root.iter():
    if "attribute" in elem.attrib:
        print(elem.attrib["attribute"])
```

### Creating XML
```python
import xml.etree.ElementTree as ET

# Create XML structure
root = ET.Element("root")
child = ET.SubElement(root, "child")
child.text = "Child content"
child.set("attribute", "value")

# Create tree and write to file
tree = ET.ElementTree(root)
tree.write("output.xml", encoding="utf-8", xml_declaration=True)

# Convert to string
xml_string = ET.tostring(root, encoding="unicode")
```

## SQLite Database I/O (`sqlite3` module)

### Basic SQLite Operations
```python
import sqlite3

# Connect to database
conn = sqlite3.connect("database.db")          # File database
conn = sqlite3.connect(":memory:")              # In-memory database

# Create cursor
cursor = conn.cursor()

# Execute SQL
cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ("Alice", 30))

# Execute many
users = [("Bob", 25), ("Charlie", 35)]
cursor.executemany("INSERT INTO users (name, age) VALUES (?, ?)", users)

# Fetch results
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()                        # All rows
row = cursor.fetchone()                         # One row
rows = cursor.fetchmany(5)                      # Limited rows

# Commit and close
conn.commit()
conn.close()
```

### SQLite with Context Manager
```python
import sqlite3

with sqlite3.connect("database.db") as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE age > ?", (25,))
    for row in cursor:
        print(row)
# Connection automatically closed
```

## Logging (`logging` module)

### Basic Logging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log'
)

# Log messages
logging.debug("Debug message")
logging.info("Info message")
logging.warning("Warning message")
logging.error("Error message")
logging.critical("Critical message")
```

### Advanced Logging
```python
import logging

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create file handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Use logger
logger.info("Application started")
```

## Advanced I/O Patterns

### Reading Large Files Efficiently
```python
def read_large_file(file_path, chunk_size=8192):
    """Read large file in chunks"""
    with open(file_path, "r") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Process large file
for chunk in read_large_file("large_file.txt"):
    # Process chunk
    pass
```

### File Backup and Rotation
```python
import os
import shutil
from datetime import datetime

def backup_file(source_path, backup_dir):
    """Create timestamped backup of file"""
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(source_path)
    name, ext = os.path.splitext(filename)
    backup_name = f"{name}_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_name)
    
    shutil.copy2(source_path, backup_path)
    return backup_path
```

### Atomic File Writing
```python
import os
import tempfile

def atomic_write(file_path, content):
    """Write file atomically to prevent corruption"""
    dir_name = os.path.dirname(file_path)
    with tempfile.NamedTemporaryFile(mode='w', dir=dir_name, delete=False) as temp_file:
        temp_file.write(content)
        temp_name = temp_file.name
    
    # Atomic rename
    os.rename(temp_name, file_path)
```

### File Monitoring
```python
import os
import time

def monitor_file_changes(file_path, callback):
    """Monitor file for changes"""
    last_modified = os.path.getmtime(file_path)
    
    while True:
        current_modified = os.path.getmtime(file_path)
        if current_modified != last_modified:
            callback(file_path)
            last_modified = current_modified
        time.sleep(1)

def file_changed(path):
    print(f"File {path} was modified")

# Usage
# monitor_file_changes("config.txt", file_changed)
```

## Performance Tips

### Buffered I/O
```python
# Use appropriate buffer sizes
with open("large_file.txt", "r", buffering=8192) as file:
    content = file.read()

# For binary files
with open("large_file.bin", "rb", buffering=8192) as file:
    data = file.read()
```

### Memory-Efficient File Processing
```python
# Generator for memory-efficient line processing
def process_lines(file_path):
    with open(file_path, "r") as file:
        for line in file:
            yield line.strip().upper()

# Process without loading entire file
for processed_line in process_lines("large_file.txt"):
    # Process line
    pass
```

### Concurrent I/O
```python
import concurrent.futures
import os

def process_file(file_path):
    """Process a single file"""
    with open(file_path, "r") as file:
        return len(file.readlines())

# Process multiple files concurrently
file_paths = ["file1.txt", "file2.txt", "file3.txt"]

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = executor.map(process_file, file_paths)
    for file_path, line_count in zip(file_paths, results):
        print(f"{file_path}: {line_count} lines")
```

---

*This document covers comprehensive input/output operations in Python including file handling, standard streams, various file formats, and advanced I/O patterns. For the most up-to-date information, refer to the official Python documentation.*