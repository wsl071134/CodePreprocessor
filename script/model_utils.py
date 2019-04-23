from keras.models import Sequential
from keras.layers import Dense,LSTM,Activation
from keras.utils import np_utils

import numpy as np
def build_model(input_shape1):
	model = Sequential()
	model.add(LSTM(input_shape=(input_shape1, 1), units=50, return_sequences=True))
	print(model.layers)
	model.add(LSTM(100, return_sequences=False))
	model.add(Dense(output_dim=2))
	model.add(Activation('linear'))
	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])
	return model

#训练模型
def train_model(train_x,train_y,test_x,test_y,model):
	try:
		model.fit(train_x, train_y, batch_size=512, nb_epoch=100, validation_split=0.1)
		predict = model.predict(test_x)
		predict = np.reshape(predict, (predict.size, ))

	except KeyboardInterrupt:
		print(predict)
		print(test_y)
	print(predict)
	print(test_y)
	return predict, test_y
