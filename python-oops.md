# Python Object-Oriented Programming (OOP)

This document provides a comprehensive guide to object-oriented programming in Python, including classes, objects, inheritance, encapsulation, polymorphism, and advanced OOP concepts with syntax and usage examples.

## Basic Class Definition and Objects

### Simple Class Creation
```python
# Basic class definition
class Person:
    """A simple Person class"""
    
    # Class variable (shared by all instances)
    species = "Homo sapiens"
    
    # Constructor method
    def __init__(self, name, age):
        # Instance variables (unique to each instance)
        self.name = name
        self.age = age
    
    # Instance method
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."
    
    # Instance method with parameters
    def have_birthday(self):
        self.age += 1
        return f"Happy birthday! {self.name} is now {self.age} years old."

# Creating objects (instances)
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

# Accessing attributes
print(person1.name)                             # Alice
print(person1.age)                              # 25
print(person1.species)                          # Homo sapiens

# Calling methods
print(person1.introduce())                      # Hi, I'm Alice and I'm 25 years old.
print(person1.have_birthday())                  # Happy birthday! Alice is now 26 years old.

# Class variables are shared
print(Person.species)                           # Homo sapiens
print(person1.species)                          # Homo sapiens
print(person2.species)                          # Homo sapiens

# Modifying class variable affects all instances
Person.species = "Human"
print(person1.species)                          # Human
print(person2.species)                          # Human
```

### Instance vs Class Attributes
```python
class Counter:
    # Class variable
    total_count = 0
    
    def __init__(self, name):
        # Instance variable
        self.name = name
        self.count = 0
        
        # Modify class variable
        Counter.total_count += 1
    
    def increment(self):
        self.count += 1
    
    def get_info(self):
        return f"{self.name}: {self.count}, Total counters: {Counter.total_count}"

# Create instances
counter1 = Counter("Counter1")
counter2 = Counter("Counter2")

print(counter1.get_info())                      # Counter1: 0, Total counters: 2
print(counter2.get_info())                      # Counter2: 0, Total counters: 2

# Increment individual counters
counter1.increment()
counter1.increment()
counter2.increment()

print(counter1.get_info())                      # Counter1: 2, Total counters: 2
print(counter2.get_info())                      # Counter2: 1, Total counters: 2

# Access class variable directly
print(f"Total counters created: {Counter.total_count}")  # Total counters created: 2
```

## Method Types

### Instance, Class, and Static Methods
```python
class MathUtils:
    pi = 3.14159
    
    def __init__(self, value):
        self.value = value
    
    # Instance method - has access to self
    def square(self):
        """Instance method - operates on instance data"""
        return self.value ** 2
    
    # Class method - has access to cls (class itself)
    @classmethod
    def from_string(cls, value_str):
        """Class method - alternative constructor"""
        value = float(value_str)
        return cls(value)
    
    @classmethod
    def get_pi(cls):
        """Class method - access class variables"""
        return cls.pi
    
    # Static method - no access to self or cls
    @staticmethod
    def add(a, b):
        """Static method - utility function"""
        return a + b
    
    @staticmethod
    def is_even(number):
        """Static method - doesn't need class or instance data"""
        return number % 2 == 0

# Using instance methods
math_obj = MathUtils(5)
print(math_obj.square())                        # 25

# Using class methods
math_obj2 = MathUtils.from_string("7.5")        # Alternative constructor
print(math_obj2.value)                          # 7.5
print(MathUtils.get_pi())                       # 3.14159

# Using static methods
print(MathUtils.add(10, 20))                    # 30
print(MathUtils.is_even(42))                    # True

# Static methods can be called from instances too
print(math_obj.add(5, 10))                      # 15
```

### Property Decorators
```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Getter for celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter for celsius with validation"""
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        """Deleter for celsius"""
        print("Deleting celsius value")
        self._celsius = 0
    
    @property
    def fahrenheit(self):
        """Read-only property calculated from celsius"""
        return (self._celsius * 9/5) + 32
    
    @property
    def kelvin(self):
        """Read-only property calculated from celsius"""
        return self._celsius + 273.15

# Using properties
temp = Temperature(25)

# Get values
print(f"Celsius: {temp.celsius}")               # Celsius: 25
print(f"Fahrenheit: {temp.fahrenheit}")         # Fahrenheit: 77.0
print(f"Kelvin: {temp.kelvin}")                 # Kelvin: 298.15

# Set celsius (uses setter with validation)
temp.celsius = 30
print(f"New Celsius: {temp.celsius}")           # New Celsius: 30

# Try invalid value
try:
    temp.celsius = -300  # Below absolute zero
except ValueError as e:
    print(f"Error: {e}")

# Delete property
del temp.celsius                                # Deleting celsius value
print(f"After deletion: {temp.celsius}")       # After deletion: 0
```

## Access Control and Encapsulation

### Private and Protected Attributes
```python
class BankAccount:
    def __init__(self, account_number, initial_balance=0):
        # Public attribute
        self.account_number = account_number
        
        # Protected attribute (convention: single underscore)
        self._owner = None
        
        # Private attribute (name mangling: double underscore)
        self.__balance = initial_balance
        
        # Private method
        self.__validate_amount(initial_balance)
    
    # Private method
    def __validate_amount(self, amount):
        if amount < 0:
            raise ValueError("Amount cannot be negative")
    
    # Protected method (convention)
    def _log_transaction(self, transaction_type, amount):
        print(f"Transaction: {transaction_type} ${amount}")
    
    # Public methods to access private data
    def deposit(self, amount):
        self.__validate_amount(amount)
        self.__balance += amount
        self._log_transaction("DEPOSIT", amount)
    
    def withdraw(self, amount):
        self.__validate_amount(amount)
        if amount > self.__balance:
            raise ValueError("Insufficient funds")
        self.__balance -= amount
        self._log_transaction("WITHDRAW", amount)
    
    def get_balance(self):
        return self.__balance
    
    # Property for protected attribute
    @property
    def owner(self):
        return self._owner
    
    @owner.setter
    def owner(self, name):
        self._owner = name

# Using the class
account = BankAccount("123456789", 1000)

# Public attribute access
print(f"Account Number: {account.account_number}")  # Account Number: 123456789

# Protected attribute access (allowed but not recommended)
account._owner = "John Doe"
print(f"Owner: {account.owner}")                # Owner: John Doe

# Public method access
account.deposit(500)                            # Transaction: DEPOSIT $500
print(f"Balance: ${account.get_balance()}")     # Balance: $1500

# Try to access private attribute directly (won't work as expected)
# print(account.__balance)                      # AttributeError

# Private attributes are name-mangled
print(f"Private balance: ${account._BankAccount__balance}")  # Private balance: $1500

# Protected method can be called (but shouldn't be)
account._log_transaction("TEST", 100)          # Transaction: TEST $100

# Attempting to access private method
try:
    account.__validate_amount(100)
except AttributeError as e:
    print(f"Can't access private method: {e}")
```

### Access Control Patterns
```python
class AccessControlDemo:
    def __init__(self):
        # Public
        self.public_var = "Anyone can access this"
        
        # Protected (single underscore - convention)
        self._protected_var = "Subclasses and internal use"
        
        # Private (double underscore - name mangling)
        self.__private_var = "Only this class can access"
    
    def public_method(self):
        """Public method - anyone can call"""
        return "This is a public method"
    
    def _protected_method(self):
        """Protected method - internal use and subclasses"""
        return "This is a protected method"
    
    def __private_method(self):
        """Private method - only this class"""
        return "This is a private method"
    
    def access_all_methods(self):
        """Demonstrate access to all method types from within class"""
        print(f"Public: {self.public_method()}")
        print(f"Protected: {self._protected_method()}")
        print(f"Private: {self.__private_method()}")
    
    def show_all_attributes(self):
        """Show all attributes from within the class"""
        print(f"Public: {self.public_var}")
        print(f"Protected: {self._protected_var}")
        print(f"Private: {self.__private_var}")

# Demonstrate access
demo = AccessControlDemo()

# Public access works
print(demo.public_var)                          # Anyone can access this
print(demo.public_method())                     # This is a public method

# Protected access works but is discouraged
print(demo._protected_var)                      # Subclasses and internal use
print(demo._protected_method())                 # This is a protected method

# Private access doesn't work directly
try:
    print(demo.__private_var)
except AttributeError as e:
    print(f"Private access failed: {e}")

# But private attributes are accessible via name mangling
print(demo._AccessControlDemo__private_var)     # Only this class can access

# Access all from within the class
demo.access_all_methods()
demo.show_all_attributes()
```

## Inheritance

### Single Inheritance
```python
# Base class (Parent class)
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
        self.is_alive = True
    
    def make_sound(self):
        return "Some generic animal sound"
    
    def eat(self, food):
        return f"{self.name} is eating {food}"
    
    def sleep(self):
        return f"{self.name} is sleeping"
    
    def __str__(self):
        return f"{self.name} the {self.species}"

# Derived class (Child class)
class Dog(Animal):
    def __init__(self, name, breed):
        # Call parent constructor
        super().__init__(name, "Dog")
        self.breed = breed
        self.is_trained = False
    
    # Override parent method
    def make_sound(self):
        return "Woof! Woof!"
    
    # New method specific to Dog
    def fetch(self, item):
        return f"{self.name} is fetching the {item}"
    
    def train(self):
        self.is_trained = True
        return f"{self.name} has been trained!"

# Another derived class
class Cat(Animal):
    def __init__(self, name, indoor=True):
        super().__init__(name, "Cat")
        self.indoor = indoor
        self.lives_remaining = 9
    
    # Override parent method
    def make_sound(self):
        return "Meow!"
    
    # New method specific to Cat
    def climb(self):
        return f"{self.name} is climbing"
    
    def use_litter_box(self):
        if self.indoor:
            return f"{self.name} used the litter box"
        return f"{self.name} went outside"

# Using inheritance
# Create instances
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", indoor=True)

# Use inherited methods
print(dog.eat("dog food"))                      # Buddy is eating dog food
print(cat.sleep())                              # Whiskers is sleeping

# Use overridden methods
print(dog.make_sound())                         # Woof! Woof!
print(cat.make_sound())                         # Meow!

# Use child-specific methods
print(dog.fetch("ball"))                        # Buddy is fetching the ball
print(cat.climb())                              # Whiskers is climbing

# Check inheritance
print(isinstance(dog, Dog))                     # True
print(isinstance(dog, Animal))                  # True
print(isinstance(cat, Dog))                     # False

# Check class hierarchy
print(Dog.__mro__)                              # Method Resolution Order
```

### Multiple Inheritance
```python
# Multiple base classes
class Flyable:
    def __init__(self):
        self.can_fly = True
        self.altitude = 0
    
    def take_off(self):
        self.altitude = 100
        return "Taking off!"
    
    def land(self):
        self.altitude = 0
        return "Landing!"

class Swimmable:
    def __init__(self):
        self.can_swim = True
        self.depth = 0
    
    def dive(self):
        self.depth = 10
        return "Diving!"
    
    def surface(self):
        self.depth = 0
        return "Surfacing!"

class Bird(Animal, Flyable):
    def __init__(self, name, species, wing_span):
        # Call both parent constructors
        Animal.__init__(self, name, species)
        Flyable.__init__(self)
        self.wing_span = wing_span
    
    def make_sound(self):
        return "Tweet tweet!"

class Duck(Bird, Swimmable):
    def __init__(self, name):
        # Call all parent constructors
        Bird.__init__(self, name, "Duck", "Medium")
        Swimmable.__init__(self)
    
    def make_sound(self):
        return "Quack quack!"
    
    def migrate(self):
        return f"{self.name} is migrating south"

# Using multiple inheritance
duck = Duck("Donald")

# Methods from Animal
print(duck.eat("bread"))                        # Donald is eating bread

# Methods from Flyable
print(duck.take_off())                          # Taking off!

# Methods from Swimmable
print(duck.dive())                              # Diving!

# Duck-specific method
print(duck.migrate())                           # Donald is migrating south

# Check inheritance
print(isinstance(duck, Duck))                   # True
print(isinstance(duck, Bird))                   # True
print(isinstance(duck, Animal))                 # True
print(isinstance(duck, Flyable))                # True
print(isinstance(duck, Swimmable))              # True

# Method Resolution Order (MRO)
print(Duck.__mro__)
```

### Method Resolution Order (MRO) and Super()
```python
class A:
    def method(self):
        print("Method from A")
    
    def common_method(self):
        print("A's common method")

class B(A):
    def method(self):
        print("Method from B")
        super().method()  # Call parent method
    
    def common_method(self):
        print("B's common method")
        super().common_method()

class C(A):
    def method(self):
        print("Method from C")
        super().method()
    
    def common_method(self):
        print("C's common method")
        super().common_method()

class D(B, C):
    def method(self):
        print("Method from D")
        super().method()
    
    def common_method(self):
        print("D's common method")
        super().common_method()

# Demonstrate MRO
d = D()
print("MRO:", D.__mro__)                        # Shows method resolution order

print("\nCalling method():")
d.method()
# Output:
# Method from D
# Method from B
# Method from C
# Method from A

print("\nCalling common_method():")
d.common_method()
# Output:
# D's common method
# B's common method
# C's common method
# A's common method

# Cooperative inheritance example
class Shape:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'Unknown')
        super().__init__(**kwargs)  # Forward remaining kwargs
    
    def info(self):
        return f"Shape: {self.name}"

class Colorable:
    def __init__(self, **kwargs):
        self.color = kwargs.get('color', 'transparent')
        super().__init__(**kwargs)
    
    def info(self):
        base_info = super().info() if hasattr(super(), 'info') else ""
        return f"{base_info}, Color: {self.color}"

class Rectangle(Shape, Colorable):
    def __init__(self, width, height, **kwargs):
        self.width = width
        self.height = height
        super().__init__(**kwargs)
    
    def area(self):
        return self.width * self.height
    
    def info(self):
        base_info = super().info()
        return f"{base_info}, Area: {self.area()}"

# Cooperative inheritance in action
rect = Rectangle(10, 5, name="Rectangle", color="blue")
print(rect.info())                              # Shape: Rectangle, Color: blue, Area: 50
```

## Polymorphism

### Method Overriding and Dynamic Dispatch
```python
# Base class defining interface
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def start_engine(self):
        return "Engine started"
    
    def stop_engine(self):
        return "Engine stopped"
    
    def accelerate(self):
        raise NotImplementedError("Subclass must implement accelerate()")
    
    def get_info(self):
        return f"{self.brand} {self.model}"

# Different implementations
class Car(Vehicle):
    def __init__(self, brand, model, doors):
        super().__init__(brand, model)
        self.doors = doors
    
    def accelerate(self):
        return "Car accelerating smoothly"
    
    def honk(self):
        return "Beep beep!"

class Motorcycle(Vehicle):
    def __init__(self, brand, model, engine_size):
        super().__init__(brand, model)
        self.engine_size = engine_size
    
    def accelerate(self):
        return "Motorcycle accelerating quickly"
    
    def wheelie(self):
        return "Doing a wheelie!"

class Truck(Vehicle):
    def __init__(self, brand, model, cargo_capacity):
        super().__init__(brand, model)
        self.cargo_capacity = cargo_capacity
    
    def accelerate(self):
        return "Truck accelerating slowly but powerfully"
    
    def load_cargo(self, weight):
        return f"Loading {weight} tons of cargo"

# Polymorphism in action
def test_vehicle(vehicle):
    """Function that works with any Vehicle type"""
    print(f"Testing: {vehicle.get_info()}")
    print(f"Start: {vehicle.start_engine()}")
    print(f"Accelerate: {vehicle.accelerate()}")  # Different for each type
    print(f"Stop: {vehicle.stop_engine()}")
    print("-" * 40)

# Create different vehicle instances
vehicles = [
    Car("Toyota", "Camry", 4),
    Motorcycle("Harley", "Davidson", 1200),
    Truck("Ford", "F-150", 2.5)
]

# Polymorphic behavior
for vehicle in vehicles:
    test_vehicle(vehicle)  # Same function, different behavior

# Another example of polymorphism
def garage_service(vehicles):
    """Service all vehicles regardless of type"""
    for vehicle in vehicles:
        print(f"Servicing {vehicle.get_info()}")
        if hasattr(vehicle, 'honk'):
            print(f"  Testing horn: {vehicle.honk()}")
        if hasattr(vehicle, 'wheelie'):
            print(f"  Testing balance: {vehicle.wheelie()}")
        if hasattr(vehicle, 'load_cargo'):
            print(f"  Testing cargo: {vehicle.load_cargo(1.0)}")

garage_service(vehicles)
```

### Duck Typing
```python
# Duck typing: "If it walks like a duck and quacks like a duck, it's a duck"

class Duck:
    def quack(self):
        return "Quack!"
    
    def walk(self):
        return "Waddle waddle"

class Person:
    def quack(self):
        return "I'm imitating a duck: Quack!"
    
    def walk(self):
        return "Walking on two legs"

class Robot:
    def quack(self):
        return "BEEP: Quack sound simulation"
    
    def walk(self):
        return "WHIRR: Mechanical walking motion"

# Function that uses duck typing
def make_it_quack_and_walk(thing):
    """This function works with anything that can quack and walk"""
    print(f"Quacking: {thing.quack()}")
    print(f"Walking: {thing.walk()}")

# Duck typing in action
duck = Duck()
person = Person()
robot = Robot()

# All work with the same function
make_it_quack_and_walk(duck)    # Works with Duck
make_it_quack_and_walk(person)  # Works with Person
make_it_quack_and_walk(robot)   # Works with Robot

# Protocol-based example
class File:
    def read(self):
        return "Reading from file"
    
    def write(self, data):
        return f"Writing '{data}' to file"

class Network:
    def read(self):
        return "Reading from network"
    
    def write(self, data):
        return f"Sending '{data}' over network"

class StringIO:
    def __init__(self):
        self.data = ""
    
    def read(self):
        return f"Reading: {self.data}"
    
    def write(self, data):
        self.data += data
        return f"Appended '{data}' to string buffer"

# Function that works with any "file-like" object
def process_data(stream, data):
    """Process data using any object with read/write methods"""
    print(f"Before: {stream.read()}")
    print(f"Operation: {stream.write(data)}")
    print(f"After: {stream.read()}")

# All these work due to duck typing
file_obj = File()
network_obj = Network()
string_obj = StringIO()

process_data(file_obj, "test data")
process_data(network_obj, "network data")
process_data(string_obj, "buffer data")
```

## Abstract Base Classes and Interfaces

### Abstract Base Classes with ABC Module
```python
from abc import ABC, abstractmethod, abstractproperty
import math

# Abstract base class
class Shape(ABC):
    """Abstract base class for shapes"""
    
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def area(self):
        """Calculate the area of the shape"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate the perimeter of the shape"""
        pass
    
    @abstractproperty
    def description(self):
        """Description of the shape"""
        pass
    
    # Concrete method (can be used by subclasses)
    def display_info(self):
        return f"{self.description}: Area={self.area():.2f}, Perimeter={self.perimeter():.2f}"

# Concrete implementations
class Circle(Shape):
    def __init__(self, radius):
        super().__init__("Circle")
        self.radius = radius
    
    def area(self):
        return math.pi * self.radius ** 2
    
    def perimeter(self):
        return 2 * math.pi * self.radius
    
    @property
    def description(self):
        return f"Circle with radius {self.radius}"

class Rectangle(Shape):
    def __init__(self, width, height):
        super().__init__("Rectangle")
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)
    
    @property
    def description(self):
        return f"Rectangle {self.width}x{self.height}"

class Triangle(Shape):
    def __init__(self, a, b, c):
        super().__init__("Triangle")
        self.a, self.b, self.c = a, b, c
        
        # Validate triangle inequality
        if not (a + b > c and b + c > a and a + c > b):
            raise ValueError("Invalid triangle sides")
    
    def area(self):
        # Heron's formula
        s = self.perimeter() / 2
        return math.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))
    
    def perimeter(self):
        return self.a + self.b + self.c
    
    @property
    def description(self):
        return f"Triangle with sides {self.a}, {self.b}, {self.c}"

# Using abstract base classes
shapes = [
    Circle(5),
    Rectangle(4, 6),
    Triangle(3, 4, 5)
]

for shape in shapes:
    print(shape.display_info())

# Cannot instantiate abstract class
try:
    abstract_shape = Shape("test")  # This will raise TypeError
except TypeError as e:
    print(f"Cannot instantiate abstract class: {e}")

# Check if class is abstract
print(f"Is Shape abstract? {Shape.__abstractmethods__}")
print(f"Is Circle abstract? {Circle.__abstractmethods__}")
```

### Protocol and Interface Patterns
```python
from typing import Protocol, runtime_checkable

# Protocol definition (Python 3.8+)
@runtime_checkable
class Drawable(Protocol):
    """Protocol for drawable objects"""
    
    def draw(self) -> str:
        """Draw the object"""
        ...
    
    def get_bounds(self) -> tuple:
        """Get bounding box (x, y, width, height)"""
        ...

@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable objects"""
    
    def serialize(self) -> dict:
        """Serialize object to dictionary"""
        ...
    
    def deserialize(self, data: dict) -> None:
        """Deserialize from dictionary"""
        ...

# Classes implementing protocols
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def draw(self):
        return f"Drawing point at ({self.x}, {self.y})"
    
    def get_bounds(self):
        return (self.x, self.y, 1, 1)
    
    def serialize(self):
        return {"x": self.x, "y": self.y}
    
    def deserialize(self, data):
        self.x = data["x"]
        self.y = data["y"]

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
    
    def draw(self):
        return f"Drawing line from ({self.x1}, {self.y1}) to ({self.x2}, {self.y2})"
    
    def get_bounds(self):
        x = min(self.x1, self.x2)
        y = min(self.y1, self.y2)
        w = abs(self.x2 - self.x1)
        h = abs(self.y2 - self.y1)
        return (x, y, w, h)
    
    def serialize(self):
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}
    
    def deserialize(self, data):
        self.x1, self.y1 = data["x1"], data["y1"]
        self.x2, self.y2 = data["x2"], data["y2"]

# Functions using protocols
def render_scene(drawables: list[Drawable]):
    """Render all drawable objects"""
    for drawable in drawables:
        print(drawable.draw())
        bounds = drawable.get_bounds()
        print(f"  Bounds: {bounds}")

def save_objects(objects: list[Serializable]):
    """Save all serializable objects"""
    data = []
    for obj in objects:
        data.append(obj.serialize())
    return data

# Using protocols
point = Point(10, 20)
line = Line(0, 0, 100, 100)

# Check protocol compliance at runtime
print(f"Point implements Drawable: {isinstance(point, Drawable)}")
print(f"Line implements Serializable: {isinstance(line, Serializable)}")

# Use protocol-based functions
render_scene([point, line])
saved_data = save_objects([point, line])
print(f"Saved data: {saved_data}")
```

## Special Methods (Magic Methods)

### Common Magic Methods
```python
class Vector:
    """A 2D vector class demonstrating magic methods"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # String representation
    def __str__(self):
        """Human-readable string representation"""
        return f"Vector({self.x}, {self.y})"
    
    def __repr__(self):
        """Developer-friendly string representation"""
        return f"Vector(x={self.x}, y={self.y})"
    
    # Arithmetic operations
    def __add__(self, other):
        """Addition: v1 + v2"""
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        return NotImplemented
    
    def __sub__(self, other):
        """Subtraction: v1 - v2"""
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        return NotImplemented
    
    def __mul__(self, scalar):
        """Scalar multiplication: v * scalar"""
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented
    
    def __rmul__(self, scalar):
        """Reverse scalar multiplication: scalar * v"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        """Division: v / scalar"""
        if isinstance(scalar, (int, float)) and scalar != 0:
            return Vector(self.x / scalar, self.y / scalar)
        return NotImplemented
    
    # Comparison operations
    def __eq__(self, other):
        """Equality: v1 == v2"""
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        return False
    
    def __ne__(self, other):
        """Inequality: v1 != v2"""
        return not self.__eq__(other)
    
    def __lt__(self, other):
        """Less than: v1 < v2 (comparing magnitudes)"""
        if isinstance(other, Vector):
            return self.magnitude() < other.magnitude()
        return NotImplemented
    
    def __le__(self, other):
        """Less than or equal: v1 <= v2"""
        return self.__lt__(other) or self.__eq__(other)
    
    def __gt__(self, other):
        """Greater than: v1 > v2"""
        if isinstance(other, Vector):
            return self.magnitude() > other.magnitude()
        return NotImplemented
    
    def __ge__(self, other):
        """Greater than or equal: v1 >= v2"""
        return self.__gt__(other) or self.__eq__(other)
    
    # Unary operations
    def __neg__(self):
        """Negation: -v"""
        return Vector(-self.x, -self.y)
    
    def __pos__(self):
        """Positive: +v"""
        return Vector(self.x, self.y)
    
    def __abs__(self):
        """Absolute value: abs(v) - returns magnitude"""
        return self.magnitude()
    
    # Container-like behavior
    def __len__(self):
        """Length: len(v) - number of components"""
        return 2
    
    def __getitem__(self, index):
        """Indexing: v[0] for x, v[1] for y"""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Vector index out of range")
    
    def __setitem__(self, index, value):
        """Item assignment: v[0] = value"""
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            raise IndexError("Vector index out of range")
    
    def __iter__(self):
        """Iteration: for component in v"""
        yield self.x
        yield self.y
    
    def __contains__(self, value):
        """Membership: value in v"""
        return value == self.x or value == self.y
    
    # Hash and boolean
    def __hash__(self):
        """Hash: hash(v) - allows use in sets and as dict keys"""
        return hash((self.x, self.y))
    
    def __bool__(self):
        """Boolean: bool(v) - True if not zero vector"""
        return self.x != 0 or self.y != 0
    
    # Utility methods
    def magnitude(self):
        """Calculate magnitude of vector"""
        return (self.x ** 2 + self.y ** 2) ** 0.5
    
    def normalize(self):
        """Return normalized vector"""
        mag = self.magnitude()
        if mag == 0:
            return Vector(0, 0)
        return Vector(self.x / mag, self.y / mag)

# Demonstrate magic methods
v1 = Vector(3, 4)
v2 = Vector(1, 2)

# String representation
print(f"str(v1): {str(v1)}")                   # str(v1): Vector(3, 4)
print(f"repr(v1): {repr(v1)}")                 # repr(v1): Vector(x=3, y=4)

# Arithmetic
v3 = v1 + v2                                    # Vector(4, 6)
v4 = v1 - v2                                    # Vector(2, 2)
v5 = v1 * 2                                     # Vector(6, 8)
v6 = 3 * v1                                     # Vector(9, 12)
v7 = v1 / 2                                     # Vector(1.5, 2.0)

print(f"v1 + v2 = {v3}")
print(f"v1 * 2 = {v5}")

# Comparison
print(f"v1 == v2: {v1 == v2}")                 # False
print(f"v1 > v2: {v1 > v2}")                   # True (magnitude comparison)

# Unary operations
print(f"-v1 = {-v1}")                          # Vector(-3, -4)
print(f"abs(v1) = {abs(v1)}")                  # 5.0

# Container operations
print(f"len(v1) = {len(v1)}")                  # 2
print(f"v1[0] = {v1[0]}")                      # 3
print(f"list(v1) = {list(v1)}")                # [3, 4]
print(f"3 in v1: {3 in v1}")                   # True

# Hash and boolean
print(f"hash(v1) = {hash(v1)}")
print(f"bool(v1) = {bool(v1)}")                # True
print(f"bool(Vector(0, 0)) = {bool(Vector(0, 0))}")  # False
```

### Context Managers
```python
class FileManager:
    """Custom context manager for file operations"""
    
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        """Enter the context"""
        print(f"Opening file: {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context"""
        if self.file:
            print(f"Closing file: {self.filename}")
            self.file.close()
        
        # Handle exceptions
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
            return False  # Don't suppress the exception
        
        return True

# Using custom context manager
with FileManager('test.txt', 'w') as f:
    f.write('Hello, World!')

# Database connection context manager
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
        self.transaction_active = False
    
    def __enter__(self):
        print(f"Connecting to database: {self.connection_string}")
        # Simulate database connection
        self.connection = f"Connected to {self.connection_string}"
        self.transaction_active = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.transaction_active:
            if exc_type is None:
                print("Committing transaction")
            else:
                print("Rolling back transaction due to exception")
        
        if self.connection:
            print("Closing database connection")
            self.connection = None
        
        return False  # Don't suppress exceptions
    
    def execute(self, query):
        if not self.connection:
            raise RuntimeError("Not connected to database")
        return f"Executing: {query}"

# Using database context manager
try:
    with DatabaseConnection("postgresql://localhost/mydb") as db:
        result = db.execute("SELECT * FROM users")
        print(result)
        # Simulate an error
        # raise ValueError("Something went wrong")
except Exception as e:
    print(f"Caught exception: {e}")

# Timer context manager
import time

class Timer:
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        print(f"Starting {self.name}...")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        end_time = time.time()
        duration = end_time - self.start_time
        print(f"{self.name} completed in {duration:.4f} seconds")
        return False

# Using timer context manager
with Timer("Data processing"):
    time.sleep(1)  # Simulate work
    data = [i**2 for i in range(1000000)]
```

## Advanced OOP Concepts

### Descriptors
```python
class Validator:
    """Base descriptor for validation"""
    
    def __init__(self, name=None):
        self.name = name
    
    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to a class attribute"""
        self.name = name
    
    def __get__(self, obj, objtype=None):
        """Get attribute value"""
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        """Set attribute value with validation"""
        self.validate(value)
        obj.__dict__[self.name] = value
    
    def validate(self, value):
        """Override in subclasses"""
        pass

class PositiveNumber(Validator):
    """Descriptor that validates positive numbers"""
    
    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be a number")
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")

class NonEmptyString(Validator):
    """Descriptor that validates non-empty strings"""
    
    def validate(self, value):
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be a string")
        if not value.strip():
            raise ValueError(f"{self.name} cannot be empty")

class RangeValidator(Validator):
    """Descriptor that validates values within a range"""
    
    def __init__(self, min_val, max_val, name=None):
        super().__init__(name)
        self.min_val = min_val
        self.max_val = max_val
    
    def validate(self, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be a number")
        if not (self.min_val <= value <= self.max_val):
            raise ValueError(f"{self.name} must be between {self.min_val} and {self.max_val}")

# Using descriptors
class Product:
    name = NonEmptyString()
    price = PositiveNumber()
    rating = RangeValidator(1, 5)
    
    def __init__(self, name, price, rating):
        self.name = name
        self.price = price
        self.rating = rating
    
    def __str__(self):
        return f"{self.name}: ${self.price} (Rating: {self.rating}/5)"

# Test descriptor validation
try:
    product = Product("Laptop", 999.99, 4)
    print(product)  # Works fine
    
    # Test validations
    product.price = -100  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")

try:
    product.name = ""  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")

try:
    product.rating = 10  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")
```

### Metaclasses
```python
class SingletonMeta(type):
    """Metaclass that creates singleton instances"""
    
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseConnection(metaclass=SingletonMeta):
    """Singleton database connection"""
    
    def __init__(self, host="localhost"):
        if hasattr(self, 'initialized'):
            return
        
        self.host = host
        self.connected = False
        self.initialized = True
        print(f"Creating database connection to {host}")
    
    def connect(self):
        self.connected = True
        print(f"Connected to {self.host}")
    
    def disconnect(self):
        self.connected = False
        print(f"Disconnected from {self.host}")

# Test singleton behavior
db1 = DatabaseConnection("server1")
db2 = DatabaseConnection("server2")  # Same instance as db1

print(f"db1 is db2: {db1 is db2}")              # True
print(f"db1.host: {db1.host}")                  # server1 (first initialization)
print(f"db2.host: {db2.host}")                  # server1 (same instance)

# Validation metaclass
class ValidatedMeta(type):
    """Metaclass that adds validation to class creation"""
    
    def __new__(mcs, name, bases, namespace):
        # Ensure all methods have docstrings
        for key, value in namespace.items():
            if callable(value) and not key.startswith('_'):
                if not getattr(value, '__doc__', None):
                    raise ValueError(f"Method {key} in class {name} must have a docstring")
        
        # Add automatic string representation
        if '__str__' not in namespace:
            def auto_str(self):
                attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items())
                return f"{name}({attrs})"
            namespace['__str__'] = auto_str
        
        return super().__new__(mcs, name, bases, namespace)

class Person(metaclass=ValidatedMeta):
    """A person class with validated methods"""
    
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        """Greet someone"""
        return f"Hello, I'm {self.name}"
    
    def celebrate_birthday(self):
        """Celebrate birthday"""
        self.age += 1
        return f"Happy birthday! Now {self.age} years old"

# Test validated class
person = Person("Alice", 25)
print(person)  # Uses auto-generated __str__
print(person.greet())

# Registry metaclass
class RegistryMeta(type):
    """Metaclass that maintains a registry of all classes"""
    
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        mcs.registry[name] = cls
        return cls
    
    @classmethod
    def get_class(mcs, name):
        return mcs.registry.get(name)
    
    @classmethod
    def list_classes(mcs):
        return list(mcs.registry.keys())

class Animal(metaclass=RegistryMeta):
    pass

class Dog(Animal):
    pass

class Cat(Animal):
    pass

# Test registry
print("Registered classes:", RegistryMeta.list_classes())
DogClass = RegistryMeta.get_class('Dog')
dog = DogClass()
print(f"Created instance: {type(dog)}")
```

### Dataclasses and Advanced Class Features
```python
from dataclasses import dataclass, field, InitVar
from typing import List, ClassVar
import uuid
from datetime import datetime

# Basic dataclass
@dataclass
class Point:
    x: float
    y: float
    
    def distance_from_origin(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

# Advanced dataclass features
@dataclass(frozen=True, order=True)  # Immutable and comparable
class ImmutablePoint:
    x: float
    y: float

# Dataclass with default values and factories
@dataclass
class Person:
    name: str
    age: int = 0
    hobbies: List[str] = field(default_factory=list)  # Mutable default
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    # Class variable
    species: ClassVar[str] = "Homo sapiens"
    
    # Field not included in init
    full_name: str = field(init=False)
    
    def __post_init__(self):
        """Called after __init__"""
        self.full_name = f"{self.name} (ID: {self.id[:8]})"

# Dataclass with InitVar
@dataclass
class Rectangle:
    width: float
    height: float
    unit: InitVar[str] = "cm"  # Not stored as instance variable
    
    area: float = field(init=False)
    perimeter: float = field(init=False)
    description: str = field(init=False)
    
    def __post_init__(self, unit):
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)
        self.description = f"{self.width}x{self.height} {unit}"

# Custom field with validation
def validated_field(validator_func, **kwargs):
    """Create a field with validation"""
    def validate_and_set(instance, value):
        if not validator_func(value):
            raise ValueError(f"Invalid value: {value}")
        return value
    
    return field(**kwargs)

@dataclass
class BankAccount:
    account_number: str
    balance: float = field(default=0.0)
    
    def __post_init__(self):
        if self.balance < 0:
            raise ValueError("Balance cannot be negative")
    
    def deposit(self, amount: float):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount
    
    def withdraw(self, amount: float):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount

# Using dataclasses
point = Point(3.0, 4.0)
print(f"Point: {point}")
print(f"Distance from origin: {point.distance_from_origin()}")

# Frozen dataclass
immutable_point = ImmutablePoint(1.0, 2.0)
# immutable_point.x = 5.0  # This would raise an error

# Dataclass comparison (with order=True)
p1 = ImmutablePoint(1, 2)
p2 = ImmutablePoint(3, 4)
print(f"p1 < p2: {p1 < p2}")  # Compares as tuple

# Complex dataclass
person = Person("Alice", 30, ["reading", "swimming"])
print(f"Person: {person}")
print(f"Full name: {person.full_name}")

# Dataclass with InitVar
rect = Rectangle(10, 5, "inches")
print(f"Rectangle: {rect.description}")
print(f"Area: {rect.area}, Perimeter: {rect.perimeter}")

# Bank account with validation
account = BankAccount("12345", 1000.0)
account.deposit(500.0)
print(f"Account balance: ${account.balance}")

try:
    account.withdraw(2000.0)  # Should raise error
except ValueError as e:
    print(f"Error: {e}")
```

---

*This document covers comprehensive object-oriented programming in Python including basic classes, inheritance, polymorphism, encapsulation, abstract base classes, special methods, descriptors, metaclasses, and modern features like dataclasses. For the most up-to-date information, refer to the official Python documentation.*