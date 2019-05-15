import operator
import nltk
nltk.download('brown')
from nltk.corpus import brown

def encode_sents(n_vocab = 2000):
    sentences = brown.sents()
    indexed_sentences = []
    
    i = 2
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    
    word_idx_count = {
            0: float('inf'),
            1: float('inf')
            }
    
    for sentence in sentences:
        indexed_sentence = []
        for token in sentence:
            token = token.lower()
            if token not in word2idx:
                word2idx[token] = i
                idx2word.append(token)
                i = i + 1
            
            idx = word2idx[token]
            word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
            
            indexed_sentence.append(idx)
        indexed_sentences.append(indexed_sentence)
        
    # sorting words ased on descending order
    sorted_word_idx_map = sorted(word_idx_count.items(), key = operator.itemgetter(1), reverse = True)
    word2idx_small = {}
    new_idx_map = {} # mapping from old index to new
    new_idx = 0
    for idx, count in sorted_word_idx_map[:n_vocab]:
        word = idx2word[idx]
        print(word, count)
        word2idx_small[word] = new_idx
        new_idx_map[idx] = new_idx
        new_idx = new_idx + 1
    
    word2idx_small['UNKNOWN'] = new_idx
    unknown = new_idx
    
    sentences_small = []
    for sentence in indexed_sentences:
        if len(sentences) > 1:
            new_sentence = [new_idx_map[idx] if idx in new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)
            
    return sentences_small, word2idx_small

import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

import os
import sys
sys.path.append(os.path.abspath('..'))
from markov import get_bigram_prob

if __name__ == "__main__":
    sentences, word2idx = encode_sents(2000)
    
    V = len(word2idx)
    print("Vocaulary size: ", V)
    
    start_idx = word2idx['START']
    end_idx = word2idx['END']
    
    bigram_prob = get_bigram_prob(V, sentences, start_idx, end_idx, smooth = 0.1)
    
    # train a logistic model
    W = np.random.randn(V, V) / np.sqrt(V)
    
    losses = []
    epochs = 1
    lr = 0.1
    
    def softmax(a):
        a = a - a.max()
        exp_a = np.exp(a)
        sigma = exp_a / exp_a.sum(axis = 1, keepdims = True)
        return sigma
    
    # compare log bigram probability loss and perceptron loss
    W_bigram = np.log(bigram_prob)
    bigram_losses = []
    
    t0 = datetime.now()
    for epoch in range(epochs):
        #shuffle sentences
        random.shuffle(sentences)
        j = 0
        for sentence in sentences:
            # one hot encode the inputs and outputs
            sentence = [start_idx] + sentence + [end_idx]
            n = len(sentence)
            inputs = np.zeros((n - 1, V))
            targets = np.zeros((n - 1, V))
            inputs[np.arange(n - 1), sentence[:n - 1]] = 1
            targets[np.arange(n - 1), sentence[1:]] = 1
            
            # predictions
            predictions = softmax(inputs.dot(W))
            
            # gradient descent
            W = W - (lr * inputs.T.dot(predictions - targets))
            
            # cost function
            loss = -np.sum(targets * np.log(predictions)) / (n - 1)
            losses.append(loss)
            
            if epoch == 0:
                bigram_predictions = softmax(inputs.dot(W_bigram))
                bigram_loss = -np.sum(targets * np.log(predictions)) / (n - 1)
                bigram_losses.append(bigram_loss)
            
            if j % 10 == 0:
                    print("epoch: ", epoch, " sentence: %s/%s" % (j, len(sentences)), " loss:", loss)
            j = j + 1
        
    print("Elapsed time: ", datetime.now() - t0)
    plt.plot(losses)
    
    avg_bigram_loss = np.mean(bigram_losses)
    print("Average Bigram Loss: ", avg_bigram_loss)
    plt.axhline(y = avg_bigram_loss, color = 'r', linestyle = '-')
    
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
    plt.title("Logistic Model")
    plt.imshow(W)
    plt.subplot(1,2,2)
    plt.title("Bigram Probability")
    plt.imshow(W_bigram)
    plt.show()
            
# Elapsed time:  0:31:48.166314
# Average Bigram Loss:  4.715087473707792        