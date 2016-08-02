import numpy as np
import nltk
import csv
import itertools
from enum import Enum
from rnn import RNN


class Token(Enum):

    START = "SENT_START"
    END = "SENT_END"
    UNKNOWN = "UNKNOWN"

print "/========== initiating ==========/"

train_size = 5000
data_path = 'my_path'
# nltk.download('punkt')

# read data and tokenize
with open(data_path, 'rb') as data_file:
    print "/===== reading and tokenizing =====/"
    reader = csv.reader(data_file, skipinitialspace=True)
    reader.next()
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
np.random.seed(10)
model = RNN(train_size)
o, s = model.forward_propagation(vector_x[10])
print "/===== forward propagation =====/"
print o.shape
print o

predictions = model.predict(vector_x[10])
print "/===== prediction =====/"
print predictions.shape
print predictions

print "/========== finishing ==========/"
