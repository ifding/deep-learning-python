# Behavioral Cloning

The goal of this project was to train a end-to-end deep learning model that would let a car drive itself around the track in a driving simulator. [Read full article here](http://navoshta.com/end-to-end-deep-learning/).


To make a better sense of it, let's consider an example of a **single recorded sample** that we turn into **16 training samples** by using frames from all three cameras and applying aforementioned augmentation pipeline.

<p align="center">
  <img src="images/frames_original.png" alt="Original"/>
</p>
<p align="center">
  <img src="images/frames_augmented.png" alt="Augmented and preprocessed"/>
</p>

Augmentation pipeline is applied using a Keras generator, which lets us do it in real-time on CPU while GPU is busy backpropagating!

## Model 

I started with the model described in [Nvidia paper](https://arxiv.org/abs/1604.07316) and kept simplifying and optimising it while making sure it performs well on both tracks. It was clear we wouldn't need that complicated model, as the data we are working with is way simpler and much more constrained than the one Nvidia team had to deal with when running their model. Eventually I settled on a fairly simple architecture with **3 convolutional layers and 3 fully connected layers**.

<p align="center">
  <img src="images/model.png" alt="Architecture"/>
</p>

This model can be very briefly encoded with Keras.

```python
from keras import models
from keras.layers import core, convolutional, pooling

model = models.Sequential()
model.add(convolutional.Convolution2D(16, 3, 3, input_shape=(32, 128, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(32, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(convolutional.Convolution2D(64, 3, 3, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2, 2)))
model.add(core.Flatten())
model.add(core.Dense(500, activation='relu'))
model.add(core.Dense(100, activation='relu'))
model.add(core.Dense(20, activation='relu'))
model.add(core.Dense(1))
``` 

I added dropout on 2 out of 3 dense layers to prevent overfitting, and the model proved to generalise quite well. The model was trained using **Adam optimiser with a learning rate = `1e-04` and mean squared error as a loss function**. I used 20% of the training data for validation (which means that we only used **6158 out of 7698 examples** for training), and the model seems to perform quite well after training for **~20 epochs**.

## Results

The car manages to drive just fine on both tracks pretty much endlessly. It rarely goes off the middle of the road, this is what driving looks like on track 2 (previously unseen).

You can check out a longer [video compilation](https://www.youtube.com/watch?v=J72Q9A0GeEo) of the car driving itself on both tracks.

Clearly this is a very basic example of end-to-end learning for self-driving cars, nevertheless it should give a rough idea of what these models are capable of, even considering all limitations of training and validating solely on a virtual driving simulator.



