# Anaconda

From the [homepage](https://www.anaconda.com/):

>"Includes 250+ popular data science packages and the conda package and virtual environment manager for Windows, Linux, and MacOS. Conda makes it quick and easy to install, run, and upgrade complex data science and machine learning environments like Scikit-learn, TensorFlow, and SciPy"

### Documentation and Installation

[Documentation](https://docs.anaconda.com/anaconda/)

[Download](https://www.anaconda.com/download/)

## Basics

### Environments

Create environment
> `conda create --name <env_name>`

Create environment with specific python version
> `conda create --n <env_name> python=<version_num>`

Create environment from `.yml` file
> `conda env create -f <yaml_file_name.yml>`

List all environments
> `conda info --envs`

Create environment yaml file

First, activate the environment you wish to export
> `conda env export > <yaml_file_name.yml>`

Removing an environment
> `conda remove --name <env_name> --all`

#### Activate Environment

Linux/MacOS
> `$ source activate <env_name>`

Windows
> `$ activate <env_name>`

#### Deactivate Environment

Linux/MacOS
> `$ source deactivate`

Windows
> `$ deactivate`

### Packages

Packages can be searched on the [cloud website](https://anaconda.org/)

Install package
> `conda install <package_name>`

Update package
> `conda update <package_name>`

Install specific package
> `conda install <package_name>=version`

Install package from a specific channel
> `conda install <package_name> -c <channel_name>`

Add channel
> `conda config --add channels <channel_name>`