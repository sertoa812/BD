
import re
import gc
import sys
import linecache
import random
import numpy as np
import tensorflow as tf
import datetime
from sklearn.decomposition import PCA
rules = []
status_collections = []
use_data_train = []
use_data_check = []
use_data_raw_list = []

def read_lines(num = 0 ,file_name = 'kdd_data/kdd99_10',random_option = True):
    file_object = linecache.getlines(file_name)
    file_len = len(file_object)
    try:
        if random_option :
            random_line_chosen = []
            for index in range(0,147002):
                if num != 0:
                    if index == num:
                        break
                #将数据分为训练集和测试集
                list_chose = 0
                #选出147000个随机数放入测试集
                if index <=147000:
                    # 记录已选则的行数
                    random_line = random.randrange(0, file_len)
                    while random_line in random_line_chosen:
                        random_line = random.randrange(0, file_len)
                    random_line_chosen.append(random_line)
                    list_chose = random_line
                    do_with_line(file_object[list_chose], file_chose=2)
                #将剩下的数放入训练集
                else:
                    for i in range(0,file_len):
                        if i not in random_line_chosen:
                            list_chose = i
                            do_with_line(file_object[list_chose], file_chose=1)
                        else:
                            pass
        else:
            for index,item in enumerate(file_object):
                do_with_line(line = item,file_chose = 1)

    finally:
        file_object.clear()
        del file_object

def do_with_line(line = "" , file_chose = 1 ):
    """
    :param line: 要处理的行
    :param list_chose: 处理完成的行存储在哪一个数据集中
    :return:
    """
    if line != " ":
        features = re.split(r',' , line)
        features[41] = features[41].strip('\n')
        use_data_single = []
        '''提取3的流量特征'''
        if features[41] not in rules[41][3]:
            return
        '''提取3的特征流量结束'''
        for index,value in enumerate(rules):
            try:
                if index in [20]:
                    continue
                if index == 41:
                    for index_of_re,value_re in enumerate(value):
                        if features[index] in value_re:
                            use_data_single.append(str(index_of_re))

                            break
                    continue
                #无需处理的变量
                if value[0] == 0:
                    use_data_single.append(str(features[index]))
                #连续变量归一化
                if value[0] == 1:
                    use_data_single.append(str(float(features[index]) / float(value[1])*255))
                #离散变量连续化然后归一化
                if value[0] == 2:
                    use_data_single.append(str(value[1][features[index]]))

            except:
                print(features[index])
                print(value[1])
                sys.exit(1)
                #print(index)

        '''use_data_raw_list.append(use_data_single)'''
        #转为字符串
        '''if len(use_data_single)!=41:
            sys.exit(1)'''
        use_data_single_str = ','.join(use_data_single)
        if file_chose == 1:
            use_data_train.append(use_data_single_str)
        else:
            use_data_check.append(use_data_single_str)
def write_lines(file_name = "" , lines = []):
    lines_len =len(lines)
    for index in range(0,lines_len):
        if lines[index] != '':
            lines[index]+='\n'
    file_object = open(file_name, 'w')
    try:
        file_object.writelines(lines)
    finally:
        file_object.close()
        del file_object

def read_rules(file_name = 'kdd_data/data_rules'):
    file_object = open(file_name,'rU')
    try:
        for index,line in enumerate(file_object):
            str = line.strip('\n')
            if str != '':
                tmp = re.split(r':',str)
                rule_name = tmp[0]
                rule = tmp[1]
                #rules第一个变量为0时不需要处理，为1时为连续变量需要缩放，为2时为离散变量需要映射
                #连续性变量
                if is_number(rule):
                    rule_judge = []
                    if rule == '0':
                        rule_judge.append(0)
                        rule_judge.append(float(rule))
                    else :
                        rule_judge.append(1)
                        rule_judge.append(float(rule))

                    rules.append(rule_judge)
                # 离散值变量
                else :
                    discrete_object = re.split(r',' , rule)
                    # 结果分为四大类，按列表存储
                    if index == 41:
                        rule_judge = []
                        rule_normal = ['normal.'];rule_judge.append(rule_normal)
                        rule_dos = ['smurf.','neptune.','back.','teardrop.','land.','pod.'];rule_judge.append(rule_dos)
                        rule_r2l = ['warezmaster.','imap.','spy.','warezclient.','ftp_write.','phf.','guess_passwd.','multihop.'];rule_judge.append(rule_r2l)
                        rule_u2r = ['perl.','rootkit.','buffer_overflow.','loadmodule.'];rule_judge.append(rule_u2r)
                        rule_probe = ['portsweep.','satan.','ipsweep.','nmap.'];rule_judge.append(rule_probe)
                        rules.append(rule_judge)
                        continue

                    else:
                        discrete_map = {}
                        for single_index,single_object in enumerate(discrete_object):
                            if single_object[0] == ' ':
                                discrete_map[single_object[1:]] = single_index
                            else:
                                discrete_map[single_object] = single_index
                        rule_judge = []
                        rule_judge.append(2)
                        rule_judge.append(discrete_map)
                        rules.append(rule_judge)

    finally:
        file_object.close()
def is_number(data):
    try:
        float(data)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(data)
        return True
    except (ValueError,TypeError):
        pass

    return False
def __main__():
    read_rules()
    read_lines(debug = False)
    write_lines(file_name="KDD99_train_data" , lines=use_data_train)
    write_lines(file_name="KDD99_check_data" , lines=use_data_check)
    gc.collect()
    #从文档中读取数据
def get_lines_str_data(filename="" , num = 0 , offset = 0 , random_option = False):
    '''

    :param filename: 文件名
    :param num: 所取数量
    :param offset: 起始偏置
    :param random: bp使用随机模式读取，svm训练不使用
    :return:
    '''
    file_object = linecache.getlines(filename)
    file_object_count = len(file_object)
    raw_data = []
    # 读文件是否随机，此次随机为二次随机，svm课忽略

    if file_object_count < num :
        for index,item in enumerate(file_object):
            raw_data.append(file_object[index])

    else:
        if random_option == True:
            for index in range(0,num):
                #测试时用
                try:
                    raw_data.append(file_object[random.randrange(0,file_object_count)])
                except Exception:
                    print(str(Exception))
                    print("error random")
                    sys.exit(1)
        else :
            for index in range(0,num):
                raw_data.append(file_object[index])
    # file_object.clear()
    # del file_object
    return raw_data
def get_data_ann(filename="" , num = 0 , random_option = True):
    raw_data = get_lines_str_data(filename,num)
    x_list = []
    y_list = []
    #将前41行特征和第42行结果分开
    for index_raw_data , raw_data_line in enumerate(raw_data):
        x_value = []
        y_value = []
        raw_data_line_list = re.split(',',raw_data_line.strip('\n'))
        for index_line , value_line in enumerate(raw_data_line_list):
            if index_line <=40:
                if value_line.find('.') != -1:
                    x_value.append(float(value_line))
                else:
                    x_value.append(int(value_line))
            else:
                for i in range(0,23):#左闭右开区间，一个23个值
                    if int(value_line) == i:
                        y_value.append(1)
                    else:
                        y_value.append(0)
        x_list.append(x_value)
        y_list.append(y_value)

    x_np_array = np.array(x_list)
    y_np_array = np.array(y_list)
    return x_np_array,y_np_array
def get_data_svm(filename="" , num = 0 , random_option = False):
    #得到的rawdata是行的集合
    raw_data = get_lines_str_data(filename=filename , num = num , random_option = random_option)
    x_list = []
    y_list = []
    # 将前40行特征和第41行结果分开
    len_value_test = []
    for index_raw_data, raw_data_line in enumerate(raw_data):
        # rawdataline是字符串
        x_value = []
        y_value = []

        #rawdatalinelist是属性的集合
        raw_data_line_list = re.split(',', raw_data_line.strip('\n'))

        if len(raw_data_line_list) not in len_value_test:
            len_value_test.append(len(len_value_test))
        if len(raw_data_line_list) != 41:
            sys.exit(1)

        for index_line, value_line in enumerate(raw_data_line_list):
            if index_line < len(raw_data_line_list)-1:
                x_value.append(float(value_line))
            else:
                y_value.append(str(value_line))
        x_list.append(x_value)
        y_list.append(y_value)
    return x_list,y_list
def get_data_svm_collect(num = 0 , random_option = False):
    file_list = ['kdd_data/normal_data','kdd_data/probe_data','kdd_data/r2l_data','kdd_data/dos_data','kdd_data/u2r_data']
    data_collect_x = []
    data_collect_y = []
    for file in file_list:
        x,y = get_data_svm(filename = file , num = num , random_option = random_option)
        for object in x:
            data_collect_x.append(object)
        for object in y:
            data_collect_y.append(object)
        del x,y
    gc.collect()
    return data_collect_x,data_collect_y
def get_train_data_split(raw_data = []):

    normal_list = []
    dos_list = []
    r2l_list = []
    u2r_list = []
    probe_list = []
    for index_raw_data, raw_data_line in enumerate(raw_data):

        #rawdatalinelist是属性的集合
        raw_data_line_list = re.split(',', raw_data_line.strip('\n'))
        line_len = len(raw_data_line_list) - 1
        if raw_data_line_list[line_len] == '0':
            normal_list.append(','.join(raw_data_line_list))
            continue
        if raw_data_line_list[line_len] == '1':
            dos_list.append(','.join(raw_data_line_list))
            continue
        if raw_data_line_list[line_len] == '2':
            r2l_list.append(','.join(raw_data_line_list))
            continue
        if raw_data_line_list[line_len] == '3':
            u2r_list.append(','.join(raw_data_line_list))
            continue
        if raw_data_line_list[line_len] == '4':
            probe_list.append(','.join(raw_data_line_list))
            continue

    return normal_list,dos_list,r2l_list,u2r_list,probe_list
def make_tfrecords():
    writer = tf.python_io.TFRecordWriter('train_tfrecords')
    file_object = linecache.getlines('train_data')
    for line in file_object:
        example = tf.train.Example(features = tf.train.Features(feature={
            "label":tf.train.Feature(int64_list = tf.train._INT64LIST(value=[line[41]])),
            "characters":tf.train.Feature(float_list = tf.train._FLOATLIST(value=[line]))
        }))
def test():
    #get_data(filename='train_data')
    #get_data_svm(filename='train_data',num = 1)

    '''file_object_1 = linecache.getlines("KDD99_check_data")
    print(len(file_object_1))
    file_object_2 = linecache.getlines("KDD99_train_data")
    print(len(file_object_2))
    file_object_1.clear()
    file_object_2.clear()
    del file_object_1
    del file_object_2
    gc.collect()'''

    #实验pca
    '''read_rules()
    read_lines(debug = False,random_option=False)
    print(len(use_data_raw_list))
    pca = PCA(copy = True,n_components=7)
    pca.fit(use_data_raw_list)
    total = 0.0
    print(type(pca.explained_variance_ratio_))
    var_list = pca.explained_variance_ratio_.tolist()
    for index,item in enumerate(var_list):
        total += item
        print(str(index)+':'+str(total))
    #print(pca.explained_variance_ratio_)
    #write_lines(file_name='train_data',lines=use_data_check)'''

    #对原始百分之十数据处理并分割
    '''start_time = datetime.datetime.now()
    read_rules()
    read_lines(random_option=True)
    print('read no problem')
    write_lines(file_name='train_data' , lines = use_data_train)
    write_lines(file_name='check_data' , lines = use_data_check)
    end_time = datetime.datetime.now()
    print ((end_time-start_time).seconds)
    1782s
    '''

    #对原始数据处理并分割
    '''read_rules()
    #read_lines(random_option=False)

    read_lines(file_name='train_data',random_option=False)
    write_lines(file_name='u2r_data', lines=use_data_train)
    '''

    # 对训练数据分割处理
    '''normal_list,dos_list,r2l_list,u2r_list,probe_list = get_train_data_split(get_lines_str_data(filename='train_data',random_option=False))
    write_lines(file_name='normal_data',lines = normal_list)
    write_lines(file_name='dos_data',lines = dos_list)
    write_lines(file_name='r2l_data',lines = r2l_list)
    write_lines(file_name='u2r_data',lines = u2r_list)
    write_lines(file_name='probe_data',lines = probe_list)'''


    #对原始数据处理后的pca分析
    '''x,y= get_data_svm(filename="train_data",random_option=False)
    pca = PCA(copy=False)
    pca.fit(x)
    total = 0.0
    print(type(pca.explained_variance_ratio_))
    var_list = pca.explained_variance_ratio_.tolist()
    for index, item in enumerate(var_list):
        total += item
        print(str(index) + ':' + str(total))'''

test()
