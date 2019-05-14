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
                idx2word.append(token)
                word2idx[token] = i
                i = i + 1
                
            idx = word2idx[token]
            word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
            
            indexed_sentence.append(idx)
        indexed_sentences.append(indexed_sentence)
    
    # sorting the words based on frequency in descending order
    sorted_word_idx_map = sorted(word_idx_count.items(), key = operator.itemgetter(1), reverse = True)
    word2idx_small = {}
    new_idx_map = {} # mapping from old index to new index
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
        if len(sentence) > 1:
            new_sentence = [new_idx_map[idx] if idx in new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)
    
    return sentences_small, word2idx_small


import numpy as np
import os
import sys
sys.path.append(os.path.abspath('..'))

# making a function to get the bigram model using Markov assumption and add-one smoothing
# on top of Bayes' probability theory
def get_bigram_prob(V, sentences, start_idx, end_idx, smooth = 1):
    # motive is to make a matrix that has p(wt|wt-1) values
    # i.e. matrix[previous word, current word] = p(wt|wt-1)
    prob_matrix = np.ones((V, V)) * smooth
    for sentence in sentences:
        for i in range(len(sentence)):
            if i == 0:
                prob_matrix[start_idx, sentence[i]] += 1
            else:
                prob_matrix[sentence[i-1], sentence[i]] += 1
            if i == len(sentence) - 1:
                prob_matrix[sentence[i], end_idx] += 1
    # matrix now contains count of all bigrams
    # normalising the counts to get probabilities
    prob_matrix = prob_matrix / prob_matrix.sum(axis = 1, keepdims = True)
    return prob_matrix


sentences, word2idx = encode_sents(10000)
V = len(word2idx)

start_idx = word2idx['START']
end_idx = word2idx['END']

bigram_prob = get_bigram_prob(V, sentences, start_idx, end_idx, smooth = 0.1)

# function to calculate the nnormalised log probability of a sentence
def get_score(sentence):
    score = 0
    for i in range(len(sentence)):
        if i == 0:
            score += np.log(bigram_prob[start_idx, sentence[i]])
        else:
            score += np.log(bigram_prob[sentence[i - 1], sentence[i]])
    score += np.log(bigram_prob[sentence[i], end_idx])
    
    # normalise the score
    score = score / (len(sentence) + 1)
    return score

from future.utils import iteritems
# function to map words ack from the index
idx2word = dict((i, w) for w, i in iteritems(word2idx))
def get_words(sentence):
    return ' '.join(idx2word[i] for i in sentence)

sample_prob = np.ones(V)
sample_prob[start_idx] = 0
sample_prob[end_idx] = 0
sample_prob /= sample_prob.sum()

real_idx = np.random.choice(len(sentences))
real = sentences[real_idx]
    
fake = np.random.choice(V, size = len(real), p = sample_prob)
    
print("Real: ", get_words(real), "Score: ", get_score(real))
# Real: for the convenience of guests bundle centers have been established
# throughout the city and suburbs where the UNKNOWN may be deposited
# between now and the date of the big event . Score:  -5.300286274942182
print("Fake: ", get_words(fake), "Score: ", get_score(fake))
# Fake:  declined enjoyment commute cracking halt everyone australia eugene
# killed academic marina everlasting marched sick partnership random remember
# end imperial effectively popularity jacket missing meats bull process ninety
# christians beans eternal rivers Score:  -9.379412803037678

