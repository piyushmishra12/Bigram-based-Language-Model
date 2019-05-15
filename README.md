# Bigram-based-Language-Model
Bigram based language model in three separate methodologies without using any libraries for deep learning
## Markov Assumption with Add - one smoothing:
Basically finding the probability of the current word given the last word by simple counting and employing the chain rule of probability i.e. the Bayes' Theorem. Taking into consideration the Markov assumption which states that probability of current word given all the previous words is equivalent to the probability of the current word given the previous one word. To ensure that probabilities add up to one, add-one smoothing is used.

## Perceptron Model:
Employing logistic regression in one neuron. Probability of current word given previous word is the softmax function of the dot product of the weight matrix and the input vector. The optimum weight matrix is calculated using the gradient descent technique. The perceptron model is made completely using numpy and no other library for deep learning.

## Neural Network Model:
Employing the perceptronn model with slight variation. Hidden vector is tanh function of the dot product of the first weight matrix and input vector. Probability of current word given previous word is the softmax function of the dot product of the second weight matrix and the hidden vector. The optimum weights are calculated using the gradiend descent technique. The enntire architecture and model of the neural network is designed only using numpy and no other library is used for deep learning.
