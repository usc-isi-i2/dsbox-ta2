import csv
import random
import os
import copy
def get_random_float(min_num=0,max_num=10):
    return random.uniform(min_num,max_num)

def get_random_int(min_num=0,max_num=10):
    return random.randint(min_num,max_num)
def get_random_col_indexes(col_num,selected_num):
    s=set()
    while(len(list(s))<selected_num):
        s.add(random.randint(0,col_num-1))
    return list(s)
def create_dir():
    dirs=['clean_data','requirement_data','performance_data']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
def generate_float_clean_data(data_point_num=100,attribute_num=50):
    float_clean_data=[]
    for i in range(data_point_num):
        float_clean_data.append([get_random_float() for j in range(attribute_num+1)])
    for i in range(data_point_num):##replace final class column by int
        float_clean_data[i][-1]=get_random_int()
    return float_clean_data
def generate_int_clean_data(data_point_num=100,attribute_num=50):
    int_clean_data=[]
    for i in range(data_point_num):
        int_clean_data.append([get_random_int() for j in range(attribute_num+1)])
    return int_clean_data
def generate_clean_data(data_point_num=100,attribute_num=50):
    ##save two clean versions into csv,notice that the final column is the output class with int attribute.
    source_dir='clean_data'
    float_clean_data=generate_float_clean_data(data_point_num,attribute_num)
    int_clean_data=generate_int_clean_data(data_point_num,attribute_num)
    ##save to csv
    with open(source_dir+"/float_clean_data.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(float_clean_data)
    with open(source_dir+"/int_clean_data.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(int_clean_data)

def generate_requirement_data():
    source_dir='requirement_data'
    data_point_num=100
    attribute_num=50
    float_clean_data=generate_float_clean_data(data_point_num,attribute_num)
    int_clean_data=generate_int_clean_data(data_point_num,attribute_num)

    float_negative_version=copy.deepcopy(float_clean_data)
    int_negative_version=copy.deepcopy(int_clean_data)
    all_string_version=copy.deepcopy(float_clean_data)
    some_string_version=copy.deepcopy(float_clean_data)
    unique_value_version=copy.deepcopy(float_clean_data)
    one_missing_version=copy.deepcopy(float_clean_data)
    some_missing_version=copy.deepcopy(float_clean_data)
    ###
    for i in get_random_col_indexes(attribute_num,get_random_int(max_num=attribute_num)):
        for j in range(len(float_clean_data)):
            float_negative_version[j][i]=get_random_float(min_num=-10,max_num=0)
            int_negative_version[j][i]=get_random_int(min_num=-10,max_num=0)

    string_list=['A','B','C','D']
    for i in range(attribute_num):
        for j in range(data_point_num):
            all_string_version[j][i]=string_list[get_random_int(max_num=len(string_list)-1)]
    for i in get_random_col_indexes(attribute_num,get_random_int(max_num=attribute_num-1)):
        for j in range(data_point_num):
            some_string_version[j][i]=string_list[get_random_int(max_num=len(string_list)-1)]


    for i in range(data_point_num):
        unique_value_version[i][0]=1
        if random.uniform(0,1)<0.5:
            one_missing_version[i][0]=None


    for i in get_random_col_indexes(attribute_num,get_random_int(min_num=2,max_num=attribute_num-1)):
        for j in range(data_point_num):
            if random.uniform(0,1)<0.5:
                some_missing_version[j][i]=None
    ###
    with open(source_dir+"/float_negative_version.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(float_negative_version)
    with open(source_dir+"/int_negative_version.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(int_negative_version)
    with open(source_dir+"/all_string_version.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(all_string_version)
    with open(source_dir+"/some_string_version.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(some_string_version)
    with open(source_dir+"/unique_value_version.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(unique_value_version)
    with open(source_dir+"/one_missing_version.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(one_missing_version)
    with open(source_dir+"/some_missing_version.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(some_missing_version)


def generate_performance_data(min_data_point_num,max_data_point_num,min_attribute_num,max_attribute_num,step):
    source_dir='performance_data'
    #only generate float version
    for d in range(min_data_point_num,max_data_point_num,step):
        for a in range(min_attribute_num,max_attribute_num,step):
            print 'data point num:',d,'attr num:',a
            float_clean_data=generate_float_clean_data(data_point_num=d,attribute_num=a)
        with open(source_dir+"/float_{0}_{1}.csv".format(d,a), "wb") as f:
            writer = csv.writer(f)
            writer.writerows(float_clean_data)

##
create_dir()

generate_clean_data(data_point_num=100,attribute_num=50)

generate_requirement_data()

generate_performance_data(min_data_point_num=100,max_data_point_num=2500,min_attribute_num=50,max_attribute_num=500,step=100)
