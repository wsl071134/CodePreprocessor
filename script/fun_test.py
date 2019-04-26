import data_utils as du
import model_utils as mu
import pandas as pd
from keras.utils import np_utils
import numpy as np

#测试读取输入数据，通过
in_data = du.load_str_list('../train_data/input.txt')
kw_list = du.load_str_list('../wordlist.txt')
#print(in_data)
#print(kw_list)
#测试读取输出数据，通过
out_data = du.load_output('../train_data/output.txt')
#6 v 6 seq2seq
Y=du.init_output_ByGroup(out_data,6)
#6行预测第一行
#Y=du.init_output(out_data)
#print(Y)
#测试读取正则，通过
pattern_list = du.load_pattern('../pattern.txt')
#print(pattern_list)
#测试提取变量及方法名，通过
variables_list = du.get_variables(pattern_list,in_data)
#print(variables_list)
#测试生成字典，通过
code_to_int,int_to_code = du.load_keyword_dict(kw_list,variables_list)
#print(variables_list)
#print(code_to_int)
#测试矩阵转换，基本通过，有些特殊情况尚待解决。如还原后的代码可读性不好等等
X = du.init_input_ByGroup(in_data,code_to_int,int_to_code,6)
#print(X)
train_x,train_y,test_x,test_y = du.load_train_data(X,Y,0.8)
#print(len(X))
#print(len(Y))
#print(train_y)
model=mu.build_model(6,200)
model,predict, test_y=mu.train_model(train_x,train_y,test_x,test_y,model)
#for i in range(len(test_y)):
#	print(predict[i],'->',test_y[i])
in_data = du.load_str_list('../train_data/code.txt')
X = du.init_input_ByGroup(in_data,code_to_int,int_to_code,6)
predict=model.predict(X)
result=[]
for x in predict:
	if(x[0]==0):
		result.append(0)
	else:
		result.append(1)
result=pd.DataFrame(result)
result.to_csv('code.csv',sep='\n',index=False,header=False)
print(predict)

