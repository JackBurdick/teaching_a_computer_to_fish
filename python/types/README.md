# Datatypes and Structures

## Some (useful+common) Primitive Datatypes

[python docs](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex)


### Numeric
[docs](https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex) [notebook](./examples/numeric.ipynb)
- `int` - [docs](https://docs.python.org/3/library/functions.html#int) - "unlimited precision"
- `float` - [docs](https://docs.python.org/3/library/functions.html#float) "usually implemented using `double` in C"

#### Boolean
[docs](https://docs.python.org/3/library/functions.html#bool) [notebook](./examples/boolean.ipynb)
- `bool()` - `True` or `False`

### Sequence Types
[notebook](./examples/sequence.ipynb)
- `list` - [docs](https://docs.python.org/3/library/stdtypes.html#list)
- `tuple` - [docs](https://docs.python.org/3/library/stdtypes.html#tuple)
- `range` - [docs](https://docs.python.org/3/library/stdtypes.html#range) advantage over list/tuple: small memory consumption - only start, stop, and step

### Text Sequence Type
[notebook](./examples/text.ipynb)
- `str` - [docs](https://docs.python.org/3/library/stdtypes.html#str) string
    - can be initialized with:
        - single quotes: `'hello'`
        - double quotes: `"hello"`
        - tripple quoted: `'''hello'''` or `"""hello"""`

### Set Types
[notebook](./examples/set.ipynb)
- `set` - [docs](https://docs.python.org/3/library/stdtypes.html#set) mutable
- `frozenset` - [docs](https://docs.python.org/3/library/stdtypes.html#frozenset) immutable and hashable (can be used as a dict key or an element of another set)

### Mapping Types
[notebook](./examples/mapping.ipynb)
- `dict` - [docs](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict)

### Note on Arrays

Python does include a built-in Array module (docs [Array](https://docs.python.org/3/library/array.html)) but it is far more common to sue the Numpy library when working with arrays. For more information on Numpy, please see the official [documentation](http://www.numpy.org/) or my notebooks (TODO: link my notebooks)

## Troubleshooting
The built-in `type()` function can be used to display the type of a variable

Example
```python
i = 1
print(f"i's type is {type(i)}")

i = 1.3
print(f"i's type is {type(i)}")

i = 10/3
print(f"i's type is {type(i)}")

i = 'c'
print(f"i's type is {type(i)}")

i = 'hello'
print(f"i's type is {type(i)}")

i = True
print(f"i's type is {type(i)}")
```

```
i's type is <class 'int'>
i's type is <class 'float'>
i's type is <class 'float'>
i's type is <class 'str'>
i's type is <class 'str'>
i's type is <class 'bool'>

```