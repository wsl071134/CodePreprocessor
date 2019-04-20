#encodeing=utf-8

import re
import os
import numpy as np
import pandas as pd


#Author:	WangShiling
#Date:		2019-04-20 17:50
#Describe:	数据加载工具类，提供各种数据处理函数

'''
=================================================================================================
#					变量名参考表						#
=================================================================================================
#  str_list	——	字符串列表		in_data		——	输入数据的字符串表示	#
#  out_data	—— 	输出数据的原始表示	kw_list		——	关键词列表 		#
#  pattern_list	——	正则列表		variables_list	——	程序变量及方法名列表	#
#  Y		——	输出数据的矩阵表示	X		——	输入数据的数字矩阵表示	#
#  code_to_int	——	关键字对照字典		int_to_code	——	关键字对照字典		#
#  train_x	——	训练集输入数据		train_y		——	训练集输出数据		#
#  test_x	——	测试集输入数据		test_y		——	测试集输出数据		#
==================================================================================================
'''

#从文件加载字符串列表
def load_str_list(filename):
	'''
	以下功能均可由本方法实现：
	#加载输入文件		def load_input(filename)
	#加载关键字对照表	def load_kw_list(filename)
	'''
	df = pd.read_csv(filename,sep='\n',header = None)
	temp = np.array(df).astype(str)
	str_list = []
	for x in temp:
		str_list.append(x[0].strip())
	return str_list

def load_pattern(filename):
	temp = load_str_list(filename)
	pattern_list=[]
	for x in temp:
		pattern = re.compile(r''+x+'')
		pattern_list.append(pattern)
	return pattern_list	

#加载输出文件
def load_output(filename):
	df = pd.read_csv(filename,sep='\n')
	out_data = np.array(df).astype(int)
	return out_data

#初始化输入矩阵
def init_input(in_data,code_to_int,int_to_code):
	temp = []
	pattern = re.compile(r'([\W]+)')
	for x in in_data:
		data = re.split(pattern,x)
		temp0=[]
		for y in data:
			if len(y) == 0:
				continue
			try:
				temp0.append(code_to_int[y])
			except KeyError:
				temp0.append(-1)
		if len(temp0)!=200:
			for i in range(200-len(temp0)-1):
				temp0.append(-1)
		temp.append(temp0)
	temp=np.array(temp).astype(int)
	'''
	print(temp)
	for x in temp:
		code_str=[]
		for value in x:
			if value!=-1:
				code_str.append(int_to_code[value])
		print(code_str)
	'''
	X=np.reshape(temp,(len(temp),temp.shape[1],1))
	X=X/len(code_to_int)
	return X

#初始化输出矩阵
def init_output(out_data):
	Y=out_data
	return Y

#提取变量及方法名列表
def get_variables(pattern_list,in_data):
	variables_list = []
	temp = []
	for x in in_data:
		for y in pattern_list:
			result = y.findall(str(x))
			if result:
				for z in result:
					temp.append(z)
	variables_list = list(set(temp))
	variables_list.sort(key = temp.index)
	return variables_list

#生成转换字典
def load_keyword_dict(kw_list,variables_list):
	for x in variables_list:
		kw_list.append(x)
	kw_list = list(set(kw_list))
	code_to_int = dict((c,i) for i,c in enumerate(kw_list))
	int_to_code = dict((i,c) for i,c in enumerate(kw_list))
	return code_to_int,int_to_code

#加载训练数据
def load_train_data(X,Y,split):
	split_boundary = int(X.shape[0]*split)
	train_x = X[:split_boundary]
	test_x = X[split_boundary:]
	split_boundary = int(Y.shape[0]*split)
	train_y = Y[:split_boundary]
	test_y = Y[split_boundary:]
	return train_x,train_y,test_x,test_y



















