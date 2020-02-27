import string

from keras.preprocessing import sequence


def get_valid_characters():
    """
        Returns a string with all the valid characters
    :return:
    """
    return '$' + string.ascii_lowercase + string.digits + '-_.'

def mapping():
    chars = get_valid_characters()
    charmap = {}
    for i, c in enumerate(chars):
        charmap[c] = i
    return charmap


def tokenize(data, maxlen=45):
    charmap = mapping()
    x_data = [[charmap[c] for c in list(x.lower())] for x in data]
    return sequence.pad_sequences(x_data, maxlen=maxlen, padding='post', truncating='post')
