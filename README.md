# Effect of resolution on performace on MNIST handwritten digits recognition

## Introduction

The main idea of this project is carry out recognition of handwritten digits from the MNIST dataset using coarse resolution 7x7 and use higher resolutions in case of degree of belief is below some threshold value. This approach may be useful for time-critical applications like autonomous driving. 

The original black-and-white MNIST pictures come with resolution 28x28 pixels. 

The code creates copies of the dataset with resolution 14x14 and 7x7 pixels. 

First, the code trains a simple Neural Network with one hidden layer for all three available resolution. The prediction procedure starts from the lowest resolution 7x7. If the softmax value is at least 0.95, the prediction is accepted. Otherwise, the prediction will use the picture with the finer resolution 14x14 pixels. If softmax value for 14x14 falls below 0.95, the finest resolution 28x28 pixels comes to play. 

The algorithm naively consider softmax value as a probability of truth, Bayesian degree of belief. The threshold value of 0.95 corresponds to confidence interval of two-sigma for Gaussian distribution. 

## Results

The results are quite remarkable, see pictures/screen\_output.txt
