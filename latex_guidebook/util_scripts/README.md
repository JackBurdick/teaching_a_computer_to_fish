# sync code from ipynb to latex

## Example

### Latex Template Format

```latex
% {{{00000}}}
```

### .ipynb Format Example

```python
# {{{00000
for i in ["a","b","c", 'd']:
    print(i)
# END}}}
```

### Output example

```latex
% {{{00000}}}
\begin{lstlisting}[language=Python]
for i in ["a","b","c", 'd']:
    print(i)
\end{lstlisting}
```

## Table Format

```txt
< sync code > || < path to ipynb > || < path to latex > || < options >
```

### Options

**NOT IMPLEMENTED YET!**

```txt
o : include cell output in .ipynb
```