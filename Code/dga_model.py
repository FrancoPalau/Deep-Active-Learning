from __future__ import division, print_function

import numpy
from keras import backend as K
import tensorflow as tf
from keras import regularizers
from keras.metrics import sparse_categorical_accuracy
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, GRU, \
    GlobalMaxPool1D, MaxPooling1D, Lambda, LSTM
from keras.layers.merge import concatenate
from sklearn import metrics
from keras.models import Model
from keras.layers.core import Dense
from keras.layers import Convolution1D, AveragePooling1D, \
    Flatten, Input, Embedding, Dropout, K, BatchNormalization
import numpy as np
from sklearn.metrics import classification_report
from layers.Attention import Attention
from plots import plot_training_curves
from sklearn.model_selection import GridSearchCV

# Imports to run old models
from keras.models import Sequential


# Set GPU memory fraction
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#K.set_session(session)

def sum_1d(x):
    from keras import backend
    return backend.sum(x, axis=1)


def getconvmodel(model, filter_length, nb_filter, input_shape):
    conv = Convolution1D(nb_filter=nb_filter,
                         filter_length=filter_length,
                         border_mode='same',
                         activation='relu',
                         subsample_length=filter_length)(model)
    return AveragePooling1D()(conv)


def binarize(x, sz=45):
    from keras import backend
    return backend.tf.to_float(backend.tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))


def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 45


def max_1d(x):
    return K.max(x, axis=1)


def build_single_CNN_model(input_shape):
    nb_filter = 32
    input_length = input_shape[0]
    input_node = Input(shape=input_shape, dtype=numpy.float32)
    m = Embedding(input_dim=input_length, output_dim=32,
                           input_length=input_length)(input_node)
    # m = Lambda(binarize, output_shape=binarize_outshape)(input_node)
    m = Conv1D(nb_filter, 4, activation='selu', padding='valid', strides=1,
               input_shape=input_shape)(m)
    m = BatchNormalization()(m)
    # m = Dropout(0.5)(m)
    # m = Flatten()(m)
    m = Lambda(sum_1d)(m)
    m = Dense(64, activation='selu')(m)
    # m = Dropout(0.5)(m)
    m = Dense(1, activation='sigmoid', name='output')(m)
    model = Model(inputs=input_node, outputs=m)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'], weighted_metrics=['accuracy'])
    return model


def build_old_model(input_shape):
    model = Sequential()
    model.add(Embedding(input_dim=input_shape[0], output_dim=128, input_length=input_shape[1]))

    model.add(Convolution1D(nb_filter=64,
                            filter_length=3,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=model.output_shape[1]))
    model.add(Flatten())
    model.add(Dense(128))

    # model.add(GRU(128, return_sequences=False))
    # Add dropout if overfitting
    # model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    # model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])  # metrics=['accuracy']
    return model


def cacic_model(input_shape):
    model = Sequential()
    model.add(Embedding(input_dim=input_shape[0], output_dim=100, input_length=input_shape[1]))

    model.add(Convolution1D(nb_filter=256,
                            kernel_size=4,
                            strides=1,
                            activation='relu'))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

#def tunnel_model(size_dense_1, filters, kernel_sizes, input_shape=(45,45), clear_session=True):
def tunnel_model(input_shape):
    # if clear_session:
    #    K.clear_session()

    model = Sequential()
    model.add(Embedding(input_dim=input_shape[0], output_dim=100, input_length=input_shape[1]))

    model.add(Convolution1D(nb_filter=256,#256 #1024
                            kernel_size=4,#4
                            strides=1,
                            activation='relu'))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))#512
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

def lstm_model_woodbridge(input_shape):

    model=Sequential()
    #model.add(Embedding(max_features, 128, input_length=75))
    model.add(Embedding(input_dim=input_shape[0], output_dim=128,
                        input_length=input_shape[1], mask_zero=True))
    #model.add(Embedding(input_shape[0], 128))    
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

    return model

#def multiclass_model(size_dense_1, size_dense_2, size_dense_3, filters, kernel_sizes, input_shape=(45,45), clear_session=True):
def multiclass_model(input_shape=(45,45)):
    #if clear_session:
    #    K.clear_session()

    # Set GPU memory fraction
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config=config)
    # K.set_session(session)    

    model = Sequential()
    model.add(Embedding(input_dim=input_shape[0], output_dim=100, input_length=input_shape[1]))

    model.add(Convolution1D(nb_filter=512,
                            kernel_size=4,
                            strides=1,
                            activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])
    return model


def build_GRU_with_attention_model(input_shape):
    input_length = input_shape[0]
    input_node = Input(shape=input_shape, dtype=numpy.float32)
    m = Embedding(input_dim=input_length, output_dim=32,
                           input_length=input_length)(input_node)
    m = GRU(64, return_sequences=True, unroll=True)(m)
    m = Attention()(m)
    m = Dense(1, activation='sigmoid', name='output')(m)
    model = Model(inputs=input_node, outputs=m)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'], weighted_metrics=['accuracy'])
    return model


def build_concatenated_CNN_model(input_shape):
    embedding_dims = 32
    nb_filter = 24
    kernel_sizes = [2, 4, 8]
    maxlen = input_shape[0]
    input_node = Input(shape=(input_shape[0],))
    # char indices to one hot matrix, 1D sequence to 2D
    # embedded = Lambda(binarize, output_shape=binarize_outshape)(input_node)
    m = Embedding(input_dim=input_shape[1], output_dim=embedding_dims,
                  input_length=maxlen)(input_node)
    convs = []
    for kernel_size in kernel_sizes:
        conv = Conv1D(nb_filter, kernel_size, activation='relu')(m)
        # conv = AveragePooling1D(pool_size=2)(conv)
        convs.append(conv)
    if len(convs) > 1:
        m = concatenate(convs, axis=1)
    else:
        m = convs[0]

    m = Flatten()(m)
    m = Dense(1024, activation='relu')(m)
    # m = Dropout(0.5)(m)
    m = Dense(1, activation='sigmoid', name='output')(m)
    model = Model(inputs=input_node, outputs=m)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def build_model_graph(input_shape, model):
    models = {'GRU_with_attention': build_GRU_with_attention_model,
              'concatenated_CNN': build_concatenated_CNN_model,
              'single_CNN': build_single_CNN_model,
              'double_GRU': build_double_GRU_model,
              'old_model': build_old_model,
              'cacic_model': cacic_model,
              'tunnel_model': tunnel_model,
              'multiclass_model': multiclass_model,
              'lstm_model_woodbridge':lstm_model_woodbridge}
    print("Model",model,"selected")
    return models[model](input_shape)


def train_model(model, x_train, y_train, validation_data, batch_size, epochs=20, with_weights=False):#, checkpointer):
    print("Training model...")
    print(y_train.shape)
    weights = with_weights and {0: 2, 1: 1} or None
    # FIT THE MODEL
    history = model.fit(x_train, y_train,
                        validation_data=validation_data,
                        batch_size=batch_size,
                        epochs=epochs,
                        class_weight=weights,
                        verbose=1,
                        shuffle=False)
                        #callbacks=[checkpointer, EarlyStopping(patience=5)]
    #plot_training_curves(history)


def predict_class(probability, thresh=0.5):
    if probability < thresh:
        return 0
    return 1


def test_binary_model(model, x_test, y_test, threshold=0.5):
    test_preds = model.predict(x_test, batch_size=4096)

    # fpr, tpr, thresholds = metrics.roc_curve(y_test, test_preds, pos_label=1, drop_intermediate=True)
    # roc_auc = metrics.auc(fpr, tpr)
    # plot_roc_curve(fpr, tpr, roc_auc)
    #preds = [predict_class(x, thresh=threshold) for x in test_preds]

    preds = [predict_class(x, thresh=threshold) for x in test_preds]
    print(metrics.confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["normal","botnet"], digits=4))
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, preds).ravel()
    print("TN: ", tn)
    print("FP: ", fp)
    print("FN: ", fn)
    print("TP: ", tp)
    print("False Alarm Rate: ", fp / float(fp + tn))
    print("Precision Rate: ", tp / float(tp + fp))
    print("Recall Rate: ", tp / float(tp + fn))

    return test_preds


def test_model(model, x_test, y_test, threshold=0.5):
    test_preds = model.predict(x_test, batch_size=4096)
    
    preds = test_preds# > threshold
    print(metrics.confusion_matrix(y_test, preds.argmax(axis=1)))
    print(classification_report(y_test, preds.argmax(axis=1),
                                target_names=["normal","dga","tunnel"]))

    cnf_matrix = metrics.confusion_matrix(y_test, preds.argmax(axis=1))

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy

    print("False Alarm Rate: ", FPR)
    print("True Positive Rate: ", TPR)

    return test_preds


def false_alarm_rate(y_true, y_pred):
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return fp / (fp + fn)


def get_cnn_layer(model, index):
    return Model(inputs=model.input, outputs=model.get_layer(index=index).output)


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def char_block(in_layer, nb_filter=(64, 100), filter_length=(3, 3), subsample=(2, 1), pool_length=(2, 2)):
    block = in_layer
    for i in range(len(nb_filter)):

        block = Conv1D(filters=nb_filter[i],
                       kernel_size=filter_length[i],
                       padding='valid',
                       activation='tanh',
                       strides=subsample[i])(block)

        # block = BatchNormalization()(block)
        # block = Dropout(0.1)(block)
        if pool_length[i]:
            block = MaxPooling1D(pool_size=pool_length[i])(block)

    # block = Lambda(max_1d, output_shape=(nb_filter[-1],))(block)
    block = GlobalMaxPool1D()(block)
    block = Dense(128, activation='relu')(block)
    return block


def build_double_GRU_model(input_shape):
    input_node = Input(shape=input_shape)
    lstm_h = 92
    lstm_layer = GRU(lstm_h, input_shape=input_shape,
                     return_sequences=True, dropout=0.1, recurrent_dropout=0.1,
                     implementation=0)(input_node)
    lstm_layer2 = GRU(lstm_h, return_sequences=False, dropout=0.1,
                      recurrent_dropout=0.1, implementation=0)(lstm_layer)
    m = Dense(1, activation='sigmoid', name='output')(lstm_layer2)
    model = Model(inputs=input_node, outputs=m)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'], weighted_metrics=['accuracy'])
    return model


def parameter_tunning(model, train_x_data, train_y_data):
    # GridSearch
    # Use scikit-learn to grid search

    #dropouts = [0.3, 0.4, 0.5, 0.6, 0.7]
    #optimizers = [ 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    filters = [64, 256, 512, 1024]
    kernel_sizes = [4, 6, 8]
    size_dense_1 = [256, 512, 1024]
    #size_dense_2 = [256, 512, 1024]
    #size_dense_3 = [256, 512, 1024]
    epochs = [10]
    batch_size = [1024]
    param_grid = dict(epochs=epochs, batch_size=batch_size,
                      filters=filters, size_dense_1=size_dense_1,
                        kernel_sizes=kernel_sizes)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5,
                        scoring='f1',
                        verbose=10)
    grid_result = grid.fit(train_x_data, train_y_data, verbose=0)
    # # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # # Escritura archivo
    file = open('barrido_output.txt', 'w')

    file.write("-----F1----" + '\n')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        file.write("%f (%f) with: %r" % (mean, stdev, param) + '\n')
    
    # file.write("-----Macro----" + '\n')
    # means = grid_result.cv_results_['mean_test_f1_macro']
    # stds = grid_result.cv_results_['std_test_f1_macro']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     file.write("%f (%f) with: %r" % (mean, stdev, param) + '\n')
    
    # file.write("-----Weighted----" + '\n')
    # means = grid_result.cv_results_['mean_test_f1_weighted']
    # stds = grid_result.cv_results_['std_test_f1_weighted']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     file.write("%f (%f) with: %r" % (mean, stdev, param) + '\n')    
    
    file.close()

    return