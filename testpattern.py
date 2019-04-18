import pandas as pd
import numpy as np
import re

pattern_obj_list = pd.read_csv('pattern.txt', sep='\n',header=None)
pattern_str_list = np.array(pattern_obj_list).astype(str)
pattern_list=[]
for x in pattern_str_list:
	pattern=re.compile(r''+x[0]+'')
	pattern_list.append(pattern)
df = pd.read_csv('test.txt', sep='\n',header=None)
data_all = np.array(df).astype(str)
data=[]
for x in np.nditer(data_all):
	for y in pattern_list:
		result=y.findall(str(x))
		if result:
			for z in result:
				data.append(z)
result = list(set(data))
result.sort(key=data.index)
print(result)

#re.split(pattern,string,maxsplit=0)分割字符串很有用的。
