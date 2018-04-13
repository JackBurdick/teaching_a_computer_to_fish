# sync code from ipynb to latex

Utility script (`sync_code_from_table.py`) used (in coordination with the sync_table.txt) file to include python snippets from an .ipynb cell into a latex document.

## Example

### Latex Example

```latex
Blah blah blah blah .....

% {{{00000}}}


More blah blah blah...
```

### .ipynb Example

```python
# {{{00000
for i in ["a","b","c", 'd']:
    print(i)
# END}}}
```

### Output example

```latex
Blah blah blah blah .....

% {{{00000}}}
\begin{lstlisting}[language=Python]
for i in ["a","b","c", 'd']:
    print(i)
\end{lstlisting}

More blah blah blah...
```

## Table Format

```txt
< sync code > || < path to ipynb > || < path to latex > || < options >
```

### include options

*include cell output*:

```txt
o : include cell output in .ipynb
```

### TODO

- Include comment from cell above/below
```
ca : include comment from above
```