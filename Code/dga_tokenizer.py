from __future__ import division, print_function

from utils.utils import get_valid_characters
from keras.preprocessing import sequence
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# import old model
from keras.preprocessing.text import Tokenizer
from utils.utils import text_filter

def mapping():
    chars = get_valid_characters()
    charmap = {}
    for i, c in enumerate(chars):
        charmap[c] = i
    return charmap


def reverse_mapping():
    chars = get_valid_characters()
    charmap = {}
    for i, c in enumerate(chars):
        charmap[i] = c
    return charmap


def tokenize(data, maxlen=45):
    charmap = mapping()
    x_data = [[charmap[c] for c in list(x)] for x in data]
    return sequence.pad_sequences(x_data, maxlen=maxlen, padding='post', truncating='post')


def onehot_encoder(categories):
    enc = OneHotEncoder(sparse=False)
    enc.fit(categories)
    return enc


def encode(data, encoder, maxlen=45):
    x_data = [[ord(c) for c in list(x)] for x in data]
    encoder.transform([x_data[0]])
    l = [encoder.transform([d]) for d in x_data]
    x_data = sequence.pad_sequences(l, maxlen=maxlen)
    return x_data


def to_onehot(data, shape):
    train = np.zeros(shape, dtype=np.int8)
    for i, domain_characters in enumerate(data):
        for j, character in enumerate(domain_characters):
            train[i][j][character] = 1
