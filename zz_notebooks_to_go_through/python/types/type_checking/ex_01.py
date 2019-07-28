import sys
print("python version: {}".format((sys.version).split("|")[0]))

def is_int_even_or_odd(some_int: int) -> str:
    return 'even' if some_int % 2 == 0 else 'odd'

int_var: int = 3 
print(is_int_even_or_odd(int_var))

# will still run, but mypy will show error (see docs)
f_var: float = 3.3
print(is_int_even_or_odd(f_var))

