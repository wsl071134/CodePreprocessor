# encoding=utf-8
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from keras.utils import np_utils
from keras import backend as K
import numpy as np


def build_LSTM_model(input_shape_1, input_shape_2, return_sequences):
    model = Sequential()
    if return_sequences:
        model.add(LSTM(input_shape=(input_shape_1, input_shape_2), units=60, return_sequences=True))
    else:
        model.add(LSTM(input_shape=(input_shape_1, input_shape_2), units=60, return_sequences=False))
    model.add(Dense(1,activation='sigmoid'))
    #model.add(Activation('relu'))
    #model.add(Activation('softmax'))
    #model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, train_x, train_y, test_x, test_y):
    scores1 = 0
    scores2 = 0
    predict = []
    try:
        model.fit(train_x, train_y, batch_size=512, epochs=500)
        predict = model.predict(test_x)
        scores1 = model.evaluate(train_x, train_y, verbose=0)
        scores2 = model.evaluate(test_x, test_y)
    except KeyboardInterrupt:
        print('训练准确率: %.2f%%' % (scores1[1] * 100))
        print('测试准确率: %.2f%%' % (scores2[1] * 100))
    print('训练准确率: %.2f%%' % (scores1[1] * 100))
    print('测试准确率: %.2f%%' % (scores2[1] * 100))
    return model, predict, test_y
