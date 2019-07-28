# Type Hinting


[Documentation](http://mypy.readthedocs.io/en/latest/index.html)


## Installation
[Installation](http://mypy.readthedocs.io/en/latest/getting_started.html#getting-started)

I installed by:
```
1) Activate environment
     - source activate dl_edge
1) Install mypy
    - pip install -U mypy
1) Use:
    - mypy ex_01.py
```

## Example

Running mypy
```bash
mypy ex_01.py

```
Output:
```bash
ex_01.py:11: error: Argument 1 to "is_int_even_or_odd" has incompatible type "float"; expected "int"

```

TODO: more examples, better explanations, resources

TODO: jupyter integration