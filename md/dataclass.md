A **dataclass** in Python is a clean, built-in way to define classes that are mainly used to **store data**—without writing a bunch of boilerplate code.

Think of it as:

> “I just want a class with attributes, not a whole ceremony.” 😄

---

## Why dataclasses exist

Before dataclasses, you had to write this kind of stuff manually:

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
```

That’s… a lot, for something simple.

---

## The dataclass version ✨

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
```

That’s it.

Python automatically generates:

* `__init__`
* `__repr__`
* `__eq__`
* and more

---

## What you get for free

```python
p1 = Point(1, 2)
p2 = Point(1, 2)

print(p1)        # Point(x=1, y=2)
print(p1 == p2) # True
```

No extra code. Super readable.

---

## Type hints are part of the deal

Dataclasses **use type annotations**:

```python
@dataclass
class User:
    name: str
    age: int
    active: bool = True   # default value
```

* Required fields come first
* Defaults come after

---

## Common options you’ll actually use

### 1. Make it immutable (like a tuple)

```python
@dataclass(frozen=True)
class Config:
    host: str
    port: int
```

Now:

```python
c = Config("localhost", 8080)
c.port = 9000   # ❌ error
```

Great for configs and safety.

---

### 2. Control what gets compared or printed

```python
from dataclasses import field

@dataclass
class Account:
    user: str
    password: str = field(repr=False, compare=False)
```

* `repr=False` → hides it in print
* `compare=False` → ignored in equality checks

---

### 3. Computed fields (`__post_init__`)

```python
@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height
```

`__post_init__` runs **after** `__init__`.

---

## When should you use a dataclass?

Use a dataclass when:

* The class is mostly **data**
* You want clean, readable code
* You don’t need complex inheritance or behavior-heavy logic

Typical uses:

* Configuration objects
* API response models
* ML / data science features
* DTOs (data transfer objects)

---

## When *not* to use it

Skip dataclasses if:

* The class is behavior-heavy
* Identity matters more than values
* You need full control over initialization logic

---

## Mental model (easy to remember)

* `tuple` → ordered, immutable
* `dict` → flexible, unstructured
* **`@dataclass` → structured, readable, Pythonic**

If you want, I can:

* compare `dataclass` vs `namedtuple` vs `pydantic`
* show how dataclasses help in **LeetCode / ML / backend**
* explain `slots=True` (performance boost)

Just say the word 👌
---
The `@dataclass` decorator automatically generates boilerplate special methods such as `__init__()` and `__repr__()` for classes primarily used to store data.
---
from dataclasses import dataclass

@dataclass
class User:
    id: int
    name: str
    email: str

# Automatically generates __init__(id, name, email) and __repr__()
user = User(1, "Alice", "alice@example.com")
print(user)  # Output: User(id=1, name='Alice', email='alice@example.com')