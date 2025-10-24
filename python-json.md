# Python JSON and Data Serialization

This document provides a comprehensive guide to Python JSON handling, pickling, and data serialization with syntax and usage examples.

## JSON Module

### Basic JSON Operations

```python
import json

# Python to JSON (serialization)
data = {
    "name": "Alice",
    "age": 30,
    "city": "New York",
    "is_student": False,
    "courses": ["Python", "JavaScript", "SQL"]
}

# dumps() - convert to JSON string
json_string = json.dumps(data)
print(json_string)
# {"name": "Alice", "age": 30, "city": "New York", "is_student": false, "courses": ["Python", "JavaScript", "SQL"]}

# loads() - parse JSON string
parsed_data = json.loads(json_string)
print(parsed_data)
# {'name': 'Alice', 'age': 30, 'city': 'New York', 'is_student': False, 'courses': ['Python', 'JavaScript', 'SQL']}

# dump() - write to file
with open("data.json", "w") as f:
    json.dump(data, f)

# load() - read from file
with open("data.json", "r") as f:
    loaded_data = json.load(f)
    print(loaded_data)
```

### JSON Formatting

```python
import json

data = {
    "name": "Alice",
    "age": 30,
    "address": {
        "street": "123 Main St",
        "city": "New York"
    }
}

# Pretty print with indentation
json_string = json.dumps(data, indent=4)
print(json_string)
# {
#     "name": "Alice",
#     "age": 30,
#     "address": {
#         "street": "123 Main St",
#         "city": "New York"
#     }
# }

# Custom indentation
json_string = json.dumps(data, indent=2)

# No indentation (compact)
json_string = json.dumps(data)

# Sort keys
json_string = json.dumps(data, sort_keys=True, indent=4)
print(json_string)
# {
#     "address": {
#         "city": "New York",
#         "street": "123 Main St"
#     },
#     "age": 30,
#     "name": "Alice"
# }

# Custom separators (default is (', ', ': '))
json_string = json.dumps(data, separators=(',', ':'))
print(json_string)                          # {"name":"Alice","age":30,...}

# More readable separators
json_string = json.dumps(data, separators=(', ', ' = '))
```

### Type Conversion

```python
import json

# Python to JSON type mapping
data = {
    "string": "hello",                      # str -> string
    "integer": 42,                          # int -> number
    "float": 3.14,                          # float -> number
    "boolean": True,                        # bool -> true/false
    "none": None,                           # None -> null
    "list": [1, 2, 3],                     # list -> array
    "tuple": (4, 5, 6),                    # tuple -> array
    "dict": {"key": "value"}                # dict -> object
}

json_string = json.dumps(data, indent=2)
print(json_string)

# JSON to Python type mapping
json_string = '{"name": "Alice", "age": 30, "active": true, "score": null}'
parsed = json.loads(json_string)
print(type(parsed["name"]))                 # <class 'str'>
print(type(parsed["age"]))                  # <class 'int'>
print(type(parsed["active"]))               # <class 'bool'>
print(type(parsed["score"]))                # <class 'NoneType'>
```

### Custom Encoding

```python
import json
from datetime import datetime, date
from decimal import Decimal

# Custom encoder class
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

# Use custom encoder
data = {
    "timestamp": datetime.now(),
    "date": date.today(),
    "amount": Decimal("99.99"),
    "tags": {"python", "json", "tutorial"}
}

json_string = json.dumps(data, cls=CustomEncoder, indent=2)
print(json_string)

# Alternative: use default parameter
def custom_encoder(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

json_string = json.dumps(data, default=custom_encoder, indent=2)
print(json_string)

# Simple lambda for common cases
json_string = json.dumps(data, default=str, indent=2)
```

### Custom Decoding

```python
import json
from datetime import datetime

# Custom decoder function
def custom_decoder(dct):
    # Convert ISO format strings to datetime
    for key, value in dct.items():
        if isinstance(value, str):
            try:
                dct[key] = datetime.fromisoformat(value)
            except (ValueError, AttributeError):
                pass
    return dct

# Use custom decoder with object_hook
json_string = '{"name": "Alice", "created": "2024-01-15T14:30:45"}'
data = json.loads(json_string, object_hook=custom_decoder)
print(type(data["created"]))                # <class 'datetime.datetime'>

# Custom decoder class
class CustomDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key, value in dct.items():
            if key.endswith('_date') and isinstance(value, str):
                try:
                    dct[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass
        return dct

json_string = '{"name": "Alice", "birth_date": "1990-05-15T00:00:00"}'
data = json.loads(json_string, cls=CustomDecoder)
print(type(data["birth_date"]))             # <class 'datetime.datetime'>
```

### Working with Complex Objects

```python
import json

# Serialize class instances
class Person:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email

    def to_dict(self):
        return {
            "name": self.name,
            "age": self.age,
            "email": self.email
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], data["age"], data["email"])

# Serialize
person = Person("Alice", 30, "alice@example.com")
json_string = json.dumps(person.to_dict(), indent=2)
print(json_string)

# Deserialize
data = json.loads(json_string)
person = Person.from_dict(data)
print(f"{person.name}, {person.age}, {person.email}")

# Using __dict__ for simple classes
class SimpleClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b

obj = SimpleClass(1, 2)
json_string = json.dumps(obj.__dict__)
print(json_string)                          # {"a": 1, "b": 2}

# Generic object encoder
def object_to_dict(obj):
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

json_string = json.dumps(obj, default=object_to_dict)
```

### Error Handling

```python
import json

# Handle JSON decode errors
invalid_json = '{"name": "Alice", "age": }'

try:
    data = json.loads(invalid_json)
except json.JSONDecodeError as e:
    print(f"JSON decode error: {e}")
    print(f"Line: {e.lineno}, Column: {e.colno}")
    print(f"Error message: {e.msg}")

# Handle encoding errors
class NonSerializable:
    pass

try:
    json_string = json.dumps({"obj": NonSerializable()})
except TypeError as e:
    print(f"Encoding error: {e}")

# Validate JSON
def is_valid_json(json_string):
    try:
        json.loads(json_string)
        return True
    except json.JSONDecodeError:
        return False

print(is_valid_json('{"name": "Alice"}'))   # True
print(is_valid_json('{"name": Alice}'))     # False
```

## Pickle Module

### Basic Pickling

```python
import pickle

# Serialize Python object
data = {
    "name": "Alice",
    "age": 30,
    "scores": [85, 92, 78],
    "metadata": {"last_update": "2024-01-15"}
}

# dumps() - serialize to bytes
pickled_data = pickle.dumps(data)
print(pickled_data)                         # b'\x80\x04\x95...'

# loads() - deserialize from bytes
unpickled_data = pickle.loads(pickled_data)
print(unpickled_data)

# dump() - write to file
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# load() - read from file
with open("data.pkl", "rb") as f:
    loaded_data = pickle.load(f)
    print(loaded_data)

# Pickle protocol versions (0-5)
# Higher protocols are more efficient but not backward compatible
pickled_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
```

### Pickling Complex Objects

```python
import pickle
from datetime import datetime

# Pickle supports many Python types
data = {
    "datetime": datetime.now(),
    "set": {1, 2, 3},
    "frozenset": frozenset([4, 5, 6]),
    "bytes": b"hello",
    "complex": 3 + 4j,
    "range": range(10),
    "function": lambda x: x * 2
}

pickled = pickle.dumps(data)
unpickled = pickle.loads(pickled)

# Pickle custom classes
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person('{self.name}', {self.age})"

person = Person("Alice", 30)
pickled_person = pickle.dumps(person)
unpickled_person = pickle.loads(pickled_person)
print(unpickled_person)                     # Person('Alice', 30)

# Pickle multiple objects
with open("objects.pkl", "wb") as f:
    pickle.dump(person, f)
    pickle.dump([1, 2, 3], f)
    pickle.dump({"key": "value"}, f)

# Unpickle multiple objects
with open("objects.pkl", "rb") as f:
    obj1 = pickle.load(f)
    obj2 = pickle.load(f)
    obj3 = pickle.load(f)
    print(obj1, obj2, obj3)
```

### Custom Pickle Behavior

```python
import pickle

# Customize pickling with __getstate__ and __setstate__
class CustomPickle:
    def __init__(self, data, password):
        self.data = data
        self.password = password
        self.cached_result = None

    def __getstate__(self):
        # Return state for pickling (exclude cached_result)
        state = self.__dict__.copy()
        del state['cached_result']
        return state

    def __setstate__(self, state):
        # Restore state from pickle
        self.__dict__.update(state)
        self.cached_result = None  # Re-initialize

obj = CustomPickle("important data", "secret123")
obj.cached_result = "expensive computation"

pickled = pickle.dumps(obj)
unpickled = pickle.loads(pickled)
print(unpickled.data)                       # important data
print(unpickled.cached_result)              # None (not pickled)

# Using __reduce__ for custom serialization
class ReduceExample:
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        # Return (callable, args) to reconstruct object
        return (self.__class__, (self.value,))

obj = ReduceExample(42)
pickled = pickle.dumps(obj)
unpickled = pickle.loads(pickled)
print(unpickled.value)                      # 42
```

### Pickle Security

```python
import pickle
import pickletools

# WARNING: Never unpickle untrusted data!
# Pickle can execute arbitrary code during unpickling

# Analyze pickle data
data = {"key": "value"}
pickled = pickle.dumps(data)
pickletools.dis(pickled)                    # Shows pickle opcodes

# Safer alternatives for untrusted data:
# - Use JSON instead
# - Validate data structure after unpickling
# - Use restricted unpickler

# Example: Restricted unpickler
class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Only allow safe classes
        if module == "__main__" and name == "SafeClass":
            return SafeClass
        raise pickle.UnpicklingError(f"Class {module}.{name} is forbidden")

class SafeClass:
    pass

# Use restricted unpickler
with open("data.pkl", "rb") as f:
    unpickler = RestrictedUnpickler(f)
    data = unpickler.load()
```

## Other Serialization Formats

### YAML

```python
# Requires: pip install pyyaml
import yaml

# Python to YAML
data = {
    "name": "Alice",
    "age": 30,
    "courses": ["Python", "JavaScript"],
    "address": {
        "city": "New York",
        "zip": "10001"
    }
}

# Serialize to YAML
yaml_string = yaml.dump(data, default_flow_style=False)
print(yaml_string)
# name: Alice
# age: 30
# courses:
# - Python
# - JavaScript
# address:
#   city: New York
#   zip: '10001'

# Parse YAML
parsed_data = yaml.safe_load(yaml_string)
print(parsed_data)

# Write to file
with open("data.yaml", "w") as f:
    yaml.dump(data, f)

# Read from file
with open("data.yaml", "r") as f:
    loaded_data = yaml.safe_load(f)

# Multiple documents in one file
docs = [{"doc": 1}, {"doc": 2}]
yaml_string = yaml.dump_all(docs)

# Load multiple documents
for doc in yaml.safe_load_all(yaml_string):
    print(doc)
```

### MessagePack

```python
# Requires: pip install msgpack
import msgpack

# Serialize
data = {"name": "Alice", "age": 30, "scores": [85, 92, 78]}
packed = msgpack.packb(data)
print(packed)                               # b'\x83\xa4name...'

# Deserialize
unpacked = msgpack.unpackb(packed)
print(unpacked)

# With file
with open("data.msgpack", "wb") as f:
    msgpack.pack(data, f)

with open("data.msgpack", "rb") as f:
    loaded_data = msgpack.unpack(f)

# Streaming
packer = msgpack.Packer()
for item in [1, 2, 3]:
    packed = packer.pack(item)
    print(packed)

unpacker = msgpack.Unpacker()
for packed in [b'\x01', b'\x02', b'\x03']:
    unpacker.feed(packed)
    for unpacked in unpacker:
        print(unpacked)
```

### CSV

```python
import csv

# Write CSV
data = [
    ["Name", "Age", "City"],
    ["Alice", 30, "New York"],
    ["Bob", 25, "Los Angeles"],
    ["Charlie", 35, "Chicago"]
]

with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Read CSV
with open("data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# CSV with dictionaries
data = [
    {"name": "Alice", "age": 30, "city": "New York"},
    {"name": "Bob", "age": 25, "city": "Los Angeles"}
]

with open("data.csv", "w", newline="") as f:
    fieldnames = ["name", "age", "city"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

with open("data.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row)                          # OrderedDict

# Custom delimiter
with open("data.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(data)
```

### XML

```python
import xml.etree.ElementTree as ET

# Create XML
root = ET.Element("data")
person = ET.SubElement(root, "person")
person.set("id", "1")

name = ET.SubElement(person, "name")
name.text = "Alice"

age = ET.SubElement(person, "age")
age.text = "30"

# Convert to string
xml_string = ET.tostring(root, encoding="unicode")
print(xml_string)
# <data><person id="1"><name>Alice</name><age>30</age></person></data>

# Pretty print
from xml.dom import minidom

xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
print(xml_str)

# Write to file
tree = ET.ElementTree(root)
tree.write("data.xml", encoding="utf-8", xml_declaration=True)

# Parse XML
tree = ET.parse("data.xml")
root = tree.getroot()

for person in root.findall("person"):
    name = person.find("name").text
    age = person.find("age").text
    person_id = person.get("id")
    print(f"Person {person_id}: {name}, {age}")

# XPath queries
for name in root.findall(".//name"):
    print(name.text)
```

## Data Validation

### JSON Schema Validation

```python
# Requires: pip install jsonschema
from jsonschema import validate, ValidationError

# Define schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number", "minimum": 0},
        "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
}

# Valid data
data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
try:
    validate(instance=data, schema=schema)
    print("Valid!")
except ValidationError as e:
    print(f"Validation error: {e.message}")

# Invalid data
invalid_data = {"name": "Bob", "age": -5}
try:
    validate(instance=invalid_data, schema=schema)
except ValidationError as e:
    print(f"Validation error: {e.message}")
```

### Pydantic

```python
# Requires: pip install pydantic
from pydantic import BaseModel, EmailStr, validator
from typing import List

class Person(BaseModel):
    name: str
    age: int
    email: EmailStr
    tags: List[str] = []

    @validator('age')
    def age_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v

# Valid data
person = Person(name="Alice", age=30, email="alice@example.com")
print(person.json())                        # JSON string
print(person.dict())                        # Dictionary

# Automatic type conversion
person = Person(name="Bob", age="25", email="bob@example.com")
print(type(person.age))                     # <class 'int'>

# Validation error
try:
    person = Person(name="Charlie", age=-5, email="invalid")
except ValidationError as e:
    print(e.json())

# Parse from JSON
json_string = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
person = Person.parse_raw(json_string)
```

## Performance Considerations

```python
import json
import pickle
import timeit

data = {"key": f"value{i}" for i in range(1000)}

# JSON performance
json_time = timeit.timeit(lambda: json.dumps(data), number=1000)
print(f"JSON serialization: {json_time:.4f}s")

# Pickle performance
pickle_time = timeit.timeit(lambda: pickle.dumps(data), number=1000)
print(f"Pickle serialization: {pickle_time:.4f}s")

# File I/O optimization
# Bad - multiple writes
with open("data.json", "w") as f:
    for item in large_list:
        f.write(json.dumps(item) + "\n")

# Good - single write
with open("data.json", "w") as f:
    json.dump(large_list, f)

# Memory-efficient JSON streaming
import ijson  # Requires: pip install ijson

# Stream parse large JSON files
with open("large.json", "rb") as f:
    objects = ijson.items(f, "item")
    for obj in objects:
        process(obj)  # Process one at a time
```

## Best Practices

```python
import json
import pickle

# Use JSON for data interchange
# - Human-readable
# - Language-independent
# - Safe for untrusted sources

# Use Pickle for Python-specific data
# - Preserves Python types exactly
# - Faster than JSON
# - Only for trusted sources

# Always specify encoding
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)

# Handle errors gracefully
def safe_load_json(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in {filename}")
        return {}

# Version your data formats
data = {
    "version": "1.0",
    "schema": "person",
    "data": {
        "name": "Alice",
        "age": 30
    }
}

# Validate before serializing
def serialize_person(person):
    if not isinstance(person.get("name"), str):
        raise ValueError("Name must be a string")
    if not isinstance(person.get("age"), int):
        raise ValueError("Age must be an integer")
    return json.dumps(person)

# Use context managers
with open("data.json", "w") as f:
    json.dump(data, f)  # File automatically closed

# Compress large files
import gzip

with gzip.open("data.json.gz", "wt", encoding="utf-8") as f:
    json.dump(large_data, f)

with gzip.open("data.json.gz", "rt", encoding="utf-8") as f:
    data = json.load(f)
```

---

*This document covers comprehensive JSON and data serialization in Python. For the most up-to-date information, refer to the official Python documentation.*
