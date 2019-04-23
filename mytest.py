					#自己编写的预测程序，同时存在ABC和BCD为学习目标
					#加载数据Demo

import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,LSTM,Activation
from keras.utils import np_utils

keyword_list='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#对照转换表的构建
code_to_int=dict((c,i) for i,c in enumerate(keyword_list))
int_to_code=dict((i,c) for i,c in enumerate(keyword_list))
#从文件加载数据
def load_train_data(split=0.8):
	df=pd.read_csv('train_data/input_1.txt',sep='\n')
	data_X=init_input(df)
	df=pd.read_csv('train_data/output_1.txt',sep='\n')
	data_Y=init_output(df)

	split_boundary=int(data_X.shape[0]*split)
	train_x=data_X[:split_boundary]
	test_x=data_X[split_boundary:]

	split_boundary=int(data_Y.shape[0]*split)
	train_y=data_Y[:split_boundary]
	test_y=data_Y[split_boundary:]

	return train_x,train_y,test_x,test_y
#处理加载的输入数据，返回归一化后的数值矩阵
def init_input(df):
	temp = np.array(df).astype(str)
	data_temp=[]
	for x in temp:
		data_temp.append(x[0])
	#此时data_temp为字符串数组
	data_in=[]
	for x in data_temp:
		data_in.append([code_to_int[char] for char in x])
	#此时data_in为数字矩阵
	data_in=np.array(data_in).astype(int)
	#接下来对输入数据reshape并且归一化
	X=np.reshape(data_in,(len(data_in),data_in.shape[1],1))
	X=X/len(keyword_list)
	return X

#处理加载的输出数据
def init_output(df):
	Y = np.array(df).astype(int)
	return Y

#构建模型
def build_model():
	model = Sequential()
	model.add(LSTM(input_shape=(29, 1), units=50, return_sequences=True))
	print(model.layers)
	model.add(LSTM(100, return_sequences=False))
	model.add(Dense(output_dim=1))
	model.add(Activation('linear'))
	model.compile(loss='mse', optimizer='rmsprop')
	return model

#训练模型
def train_model(train_x,train_y,test_x,test_y):
	model = build_model()
	try:
		model.fit(train_x, train_y, batch_size=512, nb_epoch=372, validation_split=0.1)
		predict = model.predict(test_x)
		predict = np.reshape(predict, (predict.size, ))

	except KeyboardInterrupt:
		print(predict)
		print(test_y)
	print(predict)
	print(test_y)
	return predict, test_y


if __name__ == '__main__':
	train_x,train_y,test_x,test_y=load_train_data()
	train_model(train_x,train_y,test_x,test_y)
