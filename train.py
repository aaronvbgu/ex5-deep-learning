import csv
import itertools
import numpy as np
import sys
import nltk
from enum import Enum
from imported_theano.rnn_theano import RNNTheano


class Token(Enum):

    START = "SENT_START"
    END = "SENT_END"
    UNKNOWN = "UNKNOWN"

print "/========== initiating ==========/"

train_size = 5000
data_path = 'will_play_text.csv'
# nltk.download('punkt')

# read data and tokenize
with open(data_path, 'rb') as data_file:
    print "/===== reading and tokenizing =====/"
    reader = csv.reader(data_file, skipinitialspace=True)
    reader.next()
    # reader = data_file.readlines();
    sentences = itertools.chain(*[nltk.sent_tokenize(sent[0].decode('utf-8').lower()) for sent in reader])
    sentences = ["%s %s %s" % (Token.START, sent, Token.END) for sent in sentences]
tokenized = [nltk.word_tokenize(sent) for sent in sentences]

# get frequent words, map words to vectors and tokenize infrequent words
freq = nltk.FreqDist(itertools.chain(*tokenized))
train_data = freq.most_common(train_size-1)
map_idx_to_word = [w[0] for w in train_data]
map_idx_to_word.append(Token.UNKNOWN)
map_word_to_idx = dict([(w, i) for i, w in enumerate(map_idx_to_word)])
for i, sent in enumerate(tokenized):
    tokenized[i] = [w if w in map_word_to_idx else Token.UNKNOWN for w in sent]

# build data
vector_x = np.asarray([[map_word_to_idx[w] for w in sent[:-1]] for sent in tokenized])
vector_y = np.asarray([[map_word_to_idx[w] for w in sent[1:]] for sent in tokenized])

# predict
# np.random.seed(10)
# model = RNN(train_size)
model = RNNTheano(train_size)
# o, s = model.get_words_probabilities(vector_x[10])
# print "/===== probabilities =====/"
# print o.shape
# print o
#
# predictions = model.get_next_word(vector_x[10])
# print "/===== prediction =====/"
# print predictions.shape
# print predictions


def train_with_sgd(model, vec_x, vec_y, learning_rate=0.005, num_of_iter=100):
    n = 0
    for epoch in range(num_of_iter):
        for i in range(len(vec_y)):
            model.sgd_step(vec_x[i], vec_y[i], learning_rate)
            n += 1


train_with_sgd(model, vector_x, vector_y)
print "/==== generate sentences ====/"


def print_sentence(s, iw):
    sentence = [iw[x] for x in s[1:-1]]
    print(" ".join(sentence))
    sys.stdout.flush()


def generate_sentence(model, iw, wi, min=5):
    sentence = [wi[Token.START]]
    print "sentence: ", sentence
    while not sentence[-1] == wi[Token.END]:
        next_word_probabilities = model.forward_propagation(sentence)[-1]
        print "next_word_probabilities:", next_word_probabilities
        samples = np.random.multinomial(1, next_word_probabilities)
        sample = np.argmax(samples)
        sentence.append(sample)
        if len(sentence) > 100 or sample == wi[Token.UNKNOWN]:
            return None
    if len(sentence) < min:
        return None
    return sentence


def generate(model, n, iw, wi):
    for i in range(n):
        sent = None
        while not sent:
            print "---------- no sent ----------"
            sent = generate_sentence(model, iw, wi)
        print_sentence(sent, iw)


generate(model, 10, map_idx_to_word, map_word_to_idx)
print "/========== finishing ==========/"
