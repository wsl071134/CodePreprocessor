import data_utils as du

#测试读取输入数据，通过
in_data = du.load_str_list('../train_data/code.txt')
kw_list = du.load_str_list('../wordlist.txt')
#print(in_data)
#print(kw_list)
#测试读取输出数据，通过
out_data = du.load_output('../train_data/output_1.txt')
Y=du.init_output(out_data)
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
X = du.init_input(in_data,code_to_int,int_to_code)
print(X)


