# Scipy

## Installation

I use this [command](https://anaconda.org/anaconda/scipy)
> `conda install -c anaconda scipy`

## note
the module used for image io is being depreciated.  Instead we'll use [imageio](https://pypi.python.org/pypi/imageio)
> `conda install -c menpo imageio` [conda install](https://anaconda.org/menpo/imageio)

## Common Errors

 ```
 ImportError: cannot import name 'imread'
 ```
Potential option:
- Install imageio (above)
- Install [pillow](https://anaconda.org/anaconda/pil)
> `conda install -c anaconda pillow`
