# Jupyter Notebook

[kernel_install_example]: ./misc/kernel_install.png

## TODO:
- example
- Common commands/examples
 - restart
 - timeit
 - displaying images
- styling
- exporting

## Helpful Commands

#### List all available kernels
 - `conda info --envs`


#### Accessing kernel from jupyter notebook
> if modules installed in the current environment are not appearing in jupyter notebook, you may consider trying the following;
1. `source activate <env>`
2. `conda install notebook ipykernel`
3. `python -m ipykernel install --user --name <NAME> --display-name "<NAME>"`

Where `--name` specifies the internal name used by jupyter to reference the kernel and `--display-name` specifies the name you'll see in the drop down (shown below, post running the above command)

![kernel selection menu][kernel_install_example]