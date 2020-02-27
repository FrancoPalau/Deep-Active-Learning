import os
from keras.models import load_model

from nn_model.utils import tokenize

model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "multiclass_model.h5")
model = load_model(model_path)
model._make_predict_function()

def predict_multiclass(sample, threshold=0.9):
    encoded = tokenize([sample])
    prediction = model.predict(encoded)
    probabilities = (prediction[0][0], prediction[0][1], prediction[0][2])
    classes = prediction > threshold
    return classes.argmax(axis=1), probabilities

def predict(sample, threshold=0.9):
    encoded = tokenize([sample])
    prediction = model.predict(encoded)
    return int(prediction[0][0] >= threshold), float(prediction[0][0])

if __name__ == '__main__':
    print(predict_multiclass("tito.com"))
