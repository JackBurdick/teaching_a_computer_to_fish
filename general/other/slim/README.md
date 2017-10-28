1. create directory to hold nets
    > `jack@notskynet:~/code/python/tf_models`
2. download
    1. move to new directory
    2. `git clone https://github.com/tensorflow/models`
3. ensure nets are present
    > jack@notskynet:~/code/python/tf_models/models/research/slim/nets
4. add slim nets to path
```python
import sys
sys.path.append("/home/jack/code/python/tf_models/models/research/slim")
```
5. test import
```python
from nets import inception_utils
```