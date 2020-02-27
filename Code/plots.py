from __future__ import division, print_function
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import entropy
import numpy as np


def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_scatter(legit, dga):
    legit_len, legit_entropy, dga_len, dga_entropy = [], [], [], []
    for x in legit:
        legit_len.append(len(x))
        legit_entropy.append(entropy.shannon_entropy(x))
    for x in dga:
        dga_len.append(len(x))
        dga_entropy.append(entropy.shannon_entropy(x))
    plt.scatter(legit_len, legit_entropy, s=140, c='#aaaaff', label='Legit', alpha=.2)
    plt.scatter(dga_len, dga_entropy, s=40, c='r', label='DGA', alpha=.3)
    plt.legend()
    plt.xlabel('Domain Length')
    plt.ylabel('Domain Entropy')
    plt.show()


def plot_probability_distribution(data):
    with plt.style.context('fivethirtyeight'):
        val = 0.  # this is the value where you want the data to appear on the y-axis.
        plt.plot(data, np.zeros_like(data) + val, 'x')
        plt.savefig("test.svg", format="svg")
        plt.show()


def boxplot(data):
    with plt.style.context('fivethirtyeight'):
        plt.boxplot(data)
        plt.savefig("boxplot_fp.svg", format="svg")
        plt.show()


def plot_training_curves(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
