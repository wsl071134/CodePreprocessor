					#通过正则拆分代码
					#构建转换对照表并进行文本到矩阵的转化Demo
import re

pattern=re.compile(r'([\W]+)')
data=re.split(pattern,'if(array[row][col]<0 || array[row][col]>1000)       ')
print(data)
#此处可从文件读
keyword_list=['bool','float','double','int','main']
#对照转换表的构建
code_to_int=dict((c,i) for i,c in enumerate(keyword_list))
int_to_code=dict((i,c) for i,c in enumerate(keyword_list))
#转换
data_in=[]
for x in data:
	if len(x)==0:
		continue
	try:
		data_in.append([code_to_int[x]])
	except KeyError:	
		data_in.append([-1])
print(data_in)
