
# Eager

## Installation
> [as of the time of this writing 20Dec17] you'll likely need to update to nightly build to use Eager. To do this, I;
1. Activated my environment
    - `source activate cpu_tf`
1. Found the .whl for the nightly build from [here](https://github.com/tensorflow/tensorflow)
1. Copied that above address (be sure to check py version, os, gpu/cpu)
1. upgraded tf through pip
    - `pip install --upgrade https://ci.tensorflow.org/view/tf-nightly/job/tf-nightly-linux/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON3.5,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tf_nightly-1.head-cp35-cp35m-linux_x86_64.whl`

## Potential hangups and fixes
```
ValueError: tfe.enable_eager_execution has to be called at program startup.
```
> this means you'll want to move the `tfe.enable_eager_execution()` up "to the top"/before you make any tensorflow calls.  Simply add `import tensorflow.contrib.eager as tfe` right after you import tensorflow, then enable _i.e_
```
import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
``` 