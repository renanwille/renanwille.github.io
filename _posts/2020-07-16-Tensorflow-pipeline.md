---
layout: post
title: Keras pipeline update - from sequence generator to tf.data
---

I was updating from Tensorflow 1.15 and Keras, when I got the message below,
after a little research. I also found that the function fit_generator was deprecated
since version 2.1 of Tensorflow and we are supposed to find other solutions to input data into the training pipeline.
After researching a little bit I found it difficult to find the information needed to port from the generator that
I was using to the new version. Besides that, I also discovered some problems
related to the generators APIs that would freeze my training process. So I
decided to investigate a little more how to move my python code into the tf.data
pipeline.

```
WARNING:tensorflow:multiprocessing can interact badly with TensorFlow
```

> Note: Community commenting multiprocess problems the same used in Sequence
> generators [Tensorflow issue #39523](https://github.com/tensorflow/tensorflow/issues/39523)

Well for starting I found the following article [Introduction to Keras for Engineers](https://keras.io/getting_started/intro_to_keras_for_engineers/). In here there are some tips into what inputs are available
in the new API:

* Numpy arrays;
* Tensorflow dataset objects;
* Python generators.

Keras still support Python generators by using keras.utils.Sequence class. But
as this implementation uses the underlining multithread/multiprocess and this can
interact badly with Tensorflow runtime. I decided to look into
using the Tensorflow dataset API. The dataset objects have some big advantages that
are:

* Processing data in CPU while the GPU is busy;
* Prefetching data on GPU memory so it can process the next batch right away.

To get up and running with this generator the first part is to list images and this
can be done by:

```python
dataset = tf.data.Dataset.list_files("path/to/images/*.jpg")
```

In my data pre-process normally I would use OpenCV or Albumentations code. Because
of this, I was hoping to still use this type of implementation to add data augmentations
to my code. It is possible to do that with tf.py_function in Tensorflow 2. Here is how to
insert this into the pipeline:

```python
def my_image_augmenter(args):
    img = cv2.imread(args.numpy().decode("utf-8"), cv2.IMREAD_COLOR)
    # insert your DA here ...
    return img

def main():
    dataset = tf.data.Dataset.list_files("path/to/images/*.jpg")

    # tf.py_function(<function>, <args list>, <return type list>)
    dataset = dataset.map(lambda args: tf.py_function(my_image_augmenter, [args], [tf.float32]))

    # To visualize the output for debugging
    for v in dataset.take(1):
        print(v[0].numpy())
```

This works really well but as Tensorflow must fallback to python, this isn't easy to
parallelize, because if you define more threads for this step by using `num_parallel_calls` arg
from `dataset.map` function, it will be blocked by python's Global Interpreter Lock(GIL). So to
get more efficiency at this point is better to use `tf.image` API and get real concurrency in
your data augmentations.

Besides that, after implementing these steps we can use some nice tricks of the tf.data.Dataset
to help us to create batches for working with data:

* `dataset = dataset.prefetch(<num>)` - with this function we can prefetch data to input into the GPU and
gain some time between moving memory blocks.

* `dataset = dataset.batch(<num>)` - This can be used to join single data inputs into batches of size <num>

* `dataset = dataset.shuffle(3, reshuffle_each_iteration=True)` - This can be used to shuffle data before creating batches

The nice thing about using the Tensorflow data pipeline is that it helps with pre-made functions. One can see the
complete list of the API here [Tensorflow Dataset API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).

One more note, as we want to use the `py_function` returns into the Keras pipeline sometimes you may need to set the
output shape so Keras can handle training correctly. This can be done with the function `set_shape()` of the output tensors.

Well by using this API I got rid of the locking problems when using Tensorflow, the only thing that remains is to parallelize the processing when needed. To do this we need to move more into using the Tensorflow `tf.image` API and get some speed. There is
nice support material in this link: [Kaggle - Rotation Augmentation Gpu Tpu](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96).

Hope these tips help you out!


