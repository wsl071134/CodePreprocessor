import data_utils as du
import model_utils as mu

str_list=du.get_file_lines('../data/test/test.txt')
v_list = du.get_variables(str_list)
code_to_int,int_to_code = du.get_keyword_dict(v_list)
train_x, train_y, test_x, test_y=du.get_train_data_one_step('../data/train/input.txt','../data/train/output.txt',6,1,0.9)
model=mu.build_LSTM_model(train_x.shape[1],train_x.shape[2],False)
model,predict,test_y=mu.train_model(model,train_x, train_y, test_x, test_y)
ind=du.get_input_set(str_list,code_to_int,6)
predict=model.predict(ind)
print(predict)