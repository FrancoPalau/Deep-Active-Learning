from __future__ import division, print_function

import pickle
from keras import backend as K 
from keras.utils import to_categorical
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import argparse
import sys
import pandas as pd 
from dga_model import test_binary_model, train_model,build_model_graph, predict_class,binarize,tunnel_model, parameter_tunning
from dga_tokenizer import reverse_mapping
from layers.Attention import Attention
from utils.utils import get_valid_characters
import tensorflow as tf


#Set GPU memory fraction
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#K.set_session(session)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='DGA CNN Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model-file", help="File to load model from")
    parser.add_argument("--test-only", action="store_true",
                        help="Bypass training and test with previous weights.")
    parser.add_argument("--save-model", help="Save model to this file")
    parser.add_argument("--csv-preds", action="store_true", 
    					help="Genetate csv file with predictions")
    parser.add_argument("--tunning", action="store_true", 
    					help="Perform parameter sweep")
    args = parser.parse_args()
    threshold = .50

    if args.test_only:
        model = load_model(args.model_file, custom_objects={'Attention': Attention})
        print(model.summary())
        test_x_data, test_y_data = pickle.load(open('test_paper_tunnel.pkl', 'rb'))
 
        predictions = test_model(model, test_x_data, test_y_data, threshold)

        if args.csv_preds:
        	# To generate csv file with value predictions
        	df = pd.DataFrame(predictions)
        	#print(df.head())
        	df.to_csv("preds_test.csv",index=False)

        test_x_data = [test_x_data[i] for i, x in enumerate(predictions)
                       if (predict_class(x, thresh=threshold)) and
                       test_y_data[i] == 1]

        domains = []
        keymap = reverse_mapping()
        for i, sample in enumerate(test_x_data):
            domains.append("".join([keymap[j] for j in sample if j > 0]))

        for d in domains:
            #print(d)
            pass

        sys.exit(0)

    train_x_data, train_y_data = pickle.load(open('train_data.pkl', 'rb'))
    valid_x_data, valid_y_data = pickle.load(open('valid_data.pkl', 'rb'))
    maxlen = 1000
    charmap_size = len(get_valid_characters())
    print(train_x_data.shape, train_y_data.shape)
    print(valid_x_data.shape, valid_y_data.shape)

    if args.tunning:
    	model = KerasClassifier(build_fn=tunnel_model, input_shape=(50,maxlen), clear_session=True)
    	parameter_tunning(model, train_x_data, train_y_data)
    	sys.exit(0)

    model = build_model_graph(input_shape=(maxlen,maxlen), model='lstm_model_woodbridge')
    
    #checkpointer = ModelCheckpoint(filepath='/tmp/weights.model',
    #                               verbose=1, monitor='val_acc',
    #                               save_best_only=True)

    train_model(model, train_x_data, train_y_data, validation_data=(valid_x_data, valid_y_data),
                 batch_size=256, epochs=20, with_weights=False)#, checkpointer=checkpointer)

    if args.save_model:
        model.save(args.save_model)


    test_x_data, test_y_data = pickle.load(open('test_data.pkl', 'rb'))
    # test_x_data, test_y_data = pickle.load(open('train_data.pkl', 'rb'))
    test_binary_model(model, test_x_data, test_y_data, threshold)
