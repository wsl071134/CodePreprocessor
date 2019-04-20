				#ABC预测D的程序。
				#照别人代码编写
				#其中模型构建、对照转换表等值得参考
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

keyword_list='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#对照转换表的构建
code_to_int=dict((c,i) for i,c in enumerate(keyword_list))
int_to_code=dict((i,c) for i,c in enumerate(keyword_list))
#print(char_to_int)
#测试表明，当存在关键字表时，可以完成关键字到编号的转化
#keyword=['int','float']
#code_to_int=dict((c,i) for i,c in enumerate(keyword))
#int_to_code=dict((i,c) for i,c in enumerate(keyword))
#print(code_to_int)
#此处指的是每一句话的长度，即单词个数
code_line_len=3
#定义转化为数字后的输入输出数据
dataX=[]
dataY=[]
#构造输入输出数据
for i in range(len(keyword_list)-code_line_len-1):
	seq_in=keyword_list[i:i+code_line_len]
    #包括下标为i的元素，但不包括下表为i+code_line_len的元素
	seq_out=keyword_list[i+code_line_len]
	dataX.append([code_to_int[char] for char in seq_in])
	dataY.append(code_to_int[seq_out])
	#print('输入输出数据如下：',seq_in,'->',seq_out)
print(dataX)
#dataX为22行，3列的数据
#将dataX升维，22*3*1
X=np.reshape(dataX,(len(dataX),code_line_len,1))
#print(X)
#将X归一化
X=X/float(len(keyword_list))
#print(X)
#将dataY向量化
Y=np_utils.to_categorical(dataY)
#print(Y)
model = Sequential()
model.add(LSTM(32, input_shape = (X.shape[1], X.shape[2])))
model.add(Dense(Y.shape[1], activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',metrics=['accuracy'])
model.fit(X,Y, epochs = 100, batch_size = 1, verbose =0)
scores = model.evaluate(X, Y, verbose = 0)
print("MOdel Accuracy: %.2f%%" % (scores[1]*100))

for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1))#对应上面不同的输入
    x = x / float(len(keyword_list))
    prediction = model.predict(x, verbose = 0)
    index  = np.argmax(prediction)
    result = int_to_code[index]
    seq_in = [int_to_code[value] for value in pattern]
    print(seq_in, '->', result)








