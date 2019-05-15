from perceptron import encode_sents
from markov import get_bigram_prob

import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

if __name__ == "__main__":
    sentences, word2idx = encode_sents(2000)
    V = len(word2idx)
    print("Vocabulary size: ", V)
    
    start_idx = word2idx['START']
    end_idx = word2idx['END']
    
    bigram_prob = get_bigram_prob(V, sentences, start_idx, end_idx, smooth = 0.1)
    
    # train a neural network
    D = 100
    W1 = np.random.randn(V, D) / np.sqrt(V)
    W2 = np.random.randn(D, V) / np.sqrt(D)
    
    losses = []
    epochs = 1
    lr = 0.01
    
    def softmax(a):
        a = a - a.max()
        exp_a = np.exp(a)
        sigma = exp_a / exp_a.sum(axis = 1, keepdims = True)
        return sigma
    
    W_bigram = np.log(bigram_prob)
    bigram_losses = []
    
    t0 = datetime.now()
    for epoch in range(epochs):
        random.shuffle(sentences)
        
        j = 0
        for sentence in sentences:
            # one hot encoding
            sentence = [start_idx] + sentence + [end_idx]
            n = len(sentence)
            inputs = np.zeros((n - 1, V))
            targets = np.zeros((n - 1, V))
            inputs[np.arange(n - 1), sentence[:n - 1]] = 1
            targets[np.arange(n - 1), sentence[1:]] = 1
            
            # predictions
            hidden = np.tanh(inputs.dot(W1))
            predictions = softmax(hidden.dot(W2))
            
            # gradient descent
            W2 = W2 - (lr * hidden.T.dot(predictions - targets))
            term = (predictions - targets).dot(W2.T) * (1 - hidden * hidden)
            W1 = W1 - lr * inputs.T.dot(term)
            
            # tracking loss
            loss = -np.sum(targets * np.log(predictions)) / (n - 1)
            losses.append(loss)
            
            if epochs == 0:
                bigram_predictions = softmax(inputs.dot(W_bigram))
                bigram_loss = -np.sum(targets * np.log(bigram_predictions)) / (n - 1)
                bigram_losses.append(bigram_loss)
            
            if j % 10 == 0:
                print("epoch: ", epoch, " sentence: %s/%s" % (j, len(sentences)), " loss: ", loss)
            j = j + 1
    
    print("Elapsed time: ", datetime.now() - t0)
    plt.plot(losses)
    
    avg_bigram_loss = np.mean(bigram_losses)
    print("avg_bigram_loss:", avg_bigram_loss)
    plt.axhline(y=avg_bigram_loss, color='r', linestyle='-')
    
    def smoothed_loss(x, decay=0.99):
        y = np.zeros(len(x))
        last = 0
        for t in range(len(x)):
            z = decay * last + (1 - decay) * x[t]
            y[t] = z / (1 - decay ** (t + 1))
            last = z
        return y

    plt.plot(smoothed_loss(losses))
    plt.show()
    
    plt.subplot(1,2,1)
    plt.title("Neural Network Model")
    plt.imshow(np.tanh(W1).dot(W2))
    plt.subplot(1,2,2)
    plt.title("Bigram Probs")
    plt.imshow(W_bigram)
    plt.show()
# Elapsed time:  0:03:22.401611