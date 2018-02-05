# Running with Tensors

[//]: # (Image References)
[adversarial_example]: ./misc/adversarial_example.jpg
[experimental]: ./misc/experimental.png
[fcn_segmentation]: ./misc/fcn_segmentation.png
[synthetic_gradients]: ./misc/synthetic_gradients.png
[convolution_kernel]: ./misc/convolution_example.png

Much like running with scissors, running with tensors is _dangerous_.  I am not an expert, nor do I claim to be. This is an incomplete and non-exhaustive guide to using tensorflow + common (related) ml libraries in python.  This is largely for personal use, but I do hope that I can help alleviate some of the pain of DL/TF growing pains for someone else.

## Sample projects from included notebooks

### FCN Segmentation

[notebook](./segmentation/Segmentation_dev.ipynb)
![Example output from FCN segmentation][fcn_segmentation]

### Adversarial Example

[notebook](./TODO/adversarial_example/adversarial_example_lesion.ipynb)
![Example output from adversarial example][adversarial_example]

### Experimental

[notebook](./experimental/semi_auto_cnn/progressive_net.ipynb)
![Example experimental architecture][experimental]

### Synthetic Gradients

[notebook](./TODO/synthetic_gradients/compare_synthetic_gradient_to_bp.ipynb)
![Example output from synthetic gradients][synthetic_gradients]

## Examples

### Convolution Kernel Examples

[notebook](./convolution/conv_kernel.ipynb)
![Example of emboss kernel convolution output][convolution_kernel]

#### please note

> The content and structure will change frequently. Additionally, I have many _placeholder_ (see what I did there?) directories that are currently empty -- this is for reminding me of content I want to add.

#### plan

> This project serves two main purposes; 1) personal use 2) I wanted to create "the guide I wish I would've found" when I started. Eventually, I intend to turn these into a blog/(maybe youtube) series.