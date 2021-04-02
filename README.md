MITx 6.86x
Machine Learning with Python-From Linear Models to Deep Learning

Project 2

Part 1
The goal of this project was to use linear and logistic regression, non-linear features, regularization, and kernel tricks to solve the famous digit recognition problem using the MNIST (Mixed National Institute of Standards and Technology) database.

I implemented the closed form solution to solve the linear regression which turned out to be inadequate for this task. I used scikit-learn's SVM for binary classification and multiclass classification. Then, I implemented the softmax regression using gradient descent. Finally, I experimented with different hyperparameters, different labels and different features, including kernelized features. 

Files description:

part1/linear_regression.py where I implemented linear regression
part1/svm.py where I implemented support vector machine
part1/softmax.py where I implemented multinomial regression
part1/features.py where I implemented principal component analysis (PCA) dimensionality reduction
part1/kernel.py where I implemented polynomial and Gaussian RBF kernels
part1/main.py where I tested the code

Part 2
In this section of the project I implemented a simple neural net from scratch in neural_nets.py. Then I use deep neural networks to perform the same classification task as with the simple neural network. This was done using PyTorch. First, I employed the most basic form of a deep neural network. Next, I applied convolutional neural networks to the same task. Finaly, I trained a few neural networks to solve the problem of hand-written digit recognition using a multi-digit version of MNIST.

Files description:
part2-nn/neural_nets.py in which I implemented a simple neural net from scratch
part2-mnist/nnet_fc.py where I started using PyTorch to classify MNIST digits
part2-mnist/nnet_conv.py where I used convolutional layers to boost performance
part2-twodigit/mlp.py and part2-twodigit/conv.py which are for a new, more difficult version of the MNIST dataset