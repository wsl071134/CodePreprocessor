#indeed,the sequence of letters are time steps of one feature
#rather than one time steps of seperate features.
#使用窗口的方式，添加更多数据的上下文
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

numpy.random.seed(7)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

char_to_int = dict((c,i) for i, c in enumerate(alphabet))
int_to_char = dict((i,c) for i, c in enumerate(alphabet))

seq_length = 3
dataX=[]
dataY=[]

for i in range(0,len(alphabet) - seq_length, 1):
    seq_in = alphabet[i:i + seq_length]
    seq_out = alphabet[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)
print(dataX)
print('******************************************************')
#shape表示的是张量的状态
#我觉得reshape之后的状态是len(dataX)是一个包含所有输入的大矩阵最外层的括号，第二层括号表示 ？
#最内层括号里有seq_length=3个元素
X = numpy.reshape(dataX, (len(dataX), seq_length, 1 ))#与1three2one.py的区别仅仅在于seq_length和1的次序相反
print(X)
X = X / float(len(alphabet))

print('.......................')
print(X)
print('.....................')

y=np_utils.to_categorical(dataY)
print(y)

model = Sequential()
model.add(LSTM(32, input_shape = (X.shape[1], X.shape[2])))
model.add(Dense(Y.shape[1], activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])
model.fit(X, y, epochs = 500, batch_size = 1, verbose =0)

scores = model.evaluate(X, y, verbose = 0)
print("MOdel Accuracy: %.2f%%" % (scores[1]*100))

for pattern in dataX:
    x = numpy.reshape(pattern, (1, len(pattern), 1))#对应上面不同的输入
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose = 0)
    index  = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, '->', result)
