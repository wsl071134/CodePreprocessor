# encoding=utf-8
import re
import numpy as np
import pandas as pd
import collections as col


#  @Author:             WangShiling
#  @Create_date:        2019-04-27 11:54
#  @Description:        数据处理工具包，提供各种数据处理函数
#  @Update_date:        2019-04-29 10:21


def get_file_lines(filename):
    """从文件按行加载内容，返回字符串列表str_list=[],保留空行,但首行不允许有空行"""
    str_list = []
    df = pd.read_csv(filename, sep='\n', header=None, skip_blank_lines=False)
    temp = np.array(df).astype(str)
    for x in temp:
        if x[0] == 'nan':
            str_list.append('')
        else:
            str_list.append(x[0].strip())
    return str_list


def get_file_skip_blank(filename):
    """从文件按行加载内容，返回字符串列表str_list=['..',...''],不保留空行"""
    str_list = []
    df = pd.read_csv(filename, sep='\n', header=None, skip_blank_lines=True)
    temp = np.array(df).astype(str)
    for x in temp:
        str_list.append(x[0].strip())
    return str_list


def get_variables(str_list):
    """提取用户自定义变量以及代码中调用的方法名,返回字符串列表variable_list[]"""
    pattern_list = load_pattern_list()
    temp = []
    for data in str_list:
        if len(data) == 0:
            continue
        for pattern in pattern_list:
            result = pattern.findall(str(data))
            if result:
                for rst in result:
                    temp.append(rst)
    variable_list = list(set(temp))
    variable_list.sort(key=temp.index)
    return variable_list


def get_keyword_dict(variable_list):
    """生成词袋，即对照转换列表，返回有序词典类型code_to_int{},int_to_code{}"""
    kw_list = get_file_skip_blank('../static/kwlist.txt')
    for x in variable_list:
        kw_list.append(x)
    code_to_int = col.OrderedDict((c, i + 1) for i, c in enumerate(kw_list))
    int_to_code = col.OrderedDict((i + 1, c) for i, c in enumerate(kw_list))
    return code_to_int, int_to_code


def get_in_matrix(str_list, code_to_int):
    """将字符串列表根据词典转换为数字矩阵,空格、填充及其他无关紧要字符为0，返回替换后的数字表示in_matrix[[]...[]],每行长度为200"""
    in_matrix = []
    for x in str_list:
        line = split_code_line(x)
        if len(line) == 1 and line[0] == '':
            temp = []
            for i in range(200):
                temp.append(0)
            in_matrix.append(temp)
            continue
        temp = []
        for y in line:
            if len(y) == 0:
                continue
            try:
                temp.append(code_to_int[y])
            except KeyError:
                temp_code = split_other(y)
                for z in temp_code:
                    if len(z) == 0:
                        continue
                    try:
                        temp.append(code_to_int[z])
                    except KeyError:
                        temp.append(0)
        if len(temp) != 200:
            for i in range(200 - len(temp)):
                temp.append(0)
        in_matrix.append(temp)
    return in_matrix


def get_in_matrix_bin(in_matrix):
    """将数字表示转换为二进制表示，每词长度为10，每行长度为2000"""
    return in_matrix


def get_input_set(str_list, code_to_int, group_size):
    """以group_size为最小单位构造输入数据,返回reshape并且归一化后的input_x"""
    in_matrix = get_in_matrix(str_list, code_to_int)
    input_x = []
    if group_size == 1:
        input_x = in_matrix
        input_x = np.array(input_x).astype(int)
        input_x = np.reshape(input_x, (len(input_x), input_x.shape[1], 1))
    else:
        for i in range(group_size - 1):
            temp0 = []
            for j in range(200):
                temp0.append(0)
            in_matrix.append(temp0)
        for i in range(0, len(in_matrix) - group_size + 1):
            input_x.append(in_matrix[i:i + group_size])
        input_x = np.array(input_x).astype(int)
        input_x = np.reshape(input_x, (len(input_x), input_x.shape[1], input_x.shape[2]))
    input_x = input_x / len(code_to_int)
    return input_x


def get_out_matrix(filename):
    """读取输出数据"""
    temp = get_file_skip_blank(filename)
    out_matrix = list(np.array(temp).astype(int))
    return out_matrix


def get_output_set(filename, group_size):
    """以group_size为最小单位构造输出数据,返回reshape并且归一化后的output_y"""
    out_matrix = get_out_matrix(filename)
    output_y = []
    if group_size == 1:
        output_y = out_matrix
        output_y = np.array(output_y).astype(int)
        output_y = np.reshape(output_y, (len(output_y), 1))
    else:
        for i in range(group_size - 1):
            out_matrix.append(-1)
        for i in range(0, len(out_matrix) - group_size + 1):
            output_y.append(out_matrix[i:i + group_size])
        output_y = np.array(output_y).astype(int)
        output_y = np.reshape(output_y, (len(output_y), output_y.shape[1], 1))
    return output_y


def get_train_data(input_x, output_y, split):
    """加载模型训练所需数据，将训练集中输入输出按比例拆分"""
    split_boundary_x = int(input_x.shape[0] * split)
    split_boundary_y = int(output_y.shape[0] * split)
    train_x = input_x[:split_boundary_x]
    test_x = input_x[split_boundary_x:]
    train_y = output_y[:split_boundary_y]
    test_y = output_y[split_boundary_y:]
    return train_x, train_y, test_x, test_y


def get_train_data_one_step(in_file, out_file, in_group_size, out_group_size, split):
    """更加方便的调用，一步得到训练所需数据"""
    str_list = get_file_lines(in_file)
    variable_list = get_variables(str_list)
    code_to_int, int_to_code = get_keyword_dict(variable_list)
    input_x = get_input_set(str_list, code_to_int, in_group_size)
    output_y = get_output_set(out_file, out_group_size)
    return get_train_data(input_x, output_y, split)


def code_recover(in_matrix, int_to_code):
    """由数字表示还原代码"""
    code = []
    for x in in_matrix:
        code_line = []
        for value in x:
            if value != 0:
                code_line.append(int_to_code[value])
        code.append(' '.join(code_line))
    return code


def load_pattern_list():
    """加载正则匹配模式列表，返回正则匹配模式对象列表pattern_list[],一般不对外提供，无特殊需求，在外部程序无需调用"""
    temp = get_file_skip_blank('../static/pattern.txt')
    pattern_list = []
    for x in temp:
        pattern = re.compile(r'' + x + '')
        pattern_list.append(pattern)
    return pattern_list


def split_code_line(code_line):
    """拆分每一行代码"""
    pattern = re.compile(r'([^a-zA-Z0-9]{1,2})')
    line = []
    temp = code_line.split(' ')
    for x in temp:
        result = re.split(pattern, x)
        for r in result:
            line.append(r)
    return line


def split_other(other):
    """拆分错误分割，即将多个符号分割在一起的情况，如][,])等"""
    pattern = re.compile(r'([^0-9a-zA-Z\s])')
    temp = pattern.findall(other)
    return temp
