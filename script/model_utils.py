from keras.models import Sequential
from keras.layers import Dense,LSTM,Activation
from keras.utils import np_utils
from keras import backend as K
import numpy as np

import numpy as np
def build_model(input_shape1,input_shape2):
	model = Sequential()
	model.add(LSTM(input_shape=(input_shape1, input_shape2), units=60, return_sequences=True))
#	model.add(LSTM(input_shape=(input_shape1, input_shape2), units=60, return_sequences=False))
	model.add(Dense(output_dim=1))
#	model.add(Activation('tanh'))
	model.add(Activation('relu'))
	model.compile(loss = 'mse', optimizer = 'adam')
	return model

#训练模型
def train_model(train_x,train_y,test_x,test_y,model):
	try:
		model.fit(train_x, train_y, batch_size=512, nb_epoch=500, validation_split=0.1)
		predict = model.predict(test_x)
#		print(predict)

	except KeyboardInterrupt:
		print(predict)
		print(test_y)
#	print(predict)
#	print(test_y)
	return model,predict, test_y
