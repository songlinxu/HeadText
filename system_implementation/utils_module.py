import matplotlib as mp 
mp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal,ndimage
from tslearn.preprocessing import TimeSeriesScalerMeanVariance,TimeSeriesScalerMinMax
from tslearn.neighbors import KNeighborsTimeSeriesClassifier, KNeighborsTimeSeries
from tslearn.metrics import dtw
from sklearn.metrics import accuracy_score
import joblib
from sklearn import preprocessing
import string, codecs, re, json, time, sys, itertools, math, os
##############################################   useful tools   ###################################################
def generate_id(class_num, sample_per_class):
    '''input: class number and sample per class'''
    order_array = np.zeros(class_num*sample_per_class)
    for m in range(class_num):
        for n in range(sample_per_class):
            order_array[n+sample_per_class*m] = m
    np.random.shuffle(order_array)
    return order_array

def imu_energy(data):
    energy = np.zeros(len(data))
    for i in range(len(data)):
        ax = data[i][0]
        ay = data[i][1]
        az = data[i][2]
        energy_point = ax*ax + ay*ay + az*az
        energy[i] = energy_point
    energy = np.sqrt(energy)
    return energy

##############################################   recognize data   ###################################################
def data_extend(target_data, max_length = 200):
    '''input is 3-d array'''
    target_data_length = len(target_data)
    extended_data = np.zeros((max_length,3))
    if target_data_length > max_length:
        print('too long------------------------------------------------------length: %d',target_data_length)
        extended_data = target_data[int((target_data_length-max_length)/2):int((target_data_length-max_length)/2+max_length)]
    else:
        extended_data[int((max_length-target_data_length)/2):int((max_length-target_data_length)/2+target_data_length)] = target_data
    return extended_data

def process_file_train(train_dir,extend_data_length):
    train_data = np.empty([0,3,extend_data_length])
    train_label = np.empty([0,1])
    for file_name in os.listdir(train_dir):
        filename = train_dir + '/' + file_name
        data_raw = np.loadtxt(filename)
        data_line = data_extend(data_raw,max_length=extend_data_length)
        data_line = data_line.reshape((1, data_line.shape[1], data_line.shape[0]))
        train_data = np.append(train_data, data_line, axis=0)
        train_label = np.append(train_label, int(file_name.split('_')[-1].split('.')[0]))
              
    np.save('train_data.npy', train_data)
    np.save('train_label.npy', train_label)
    
def process_file_test(test_dir,extend_data_length):
    test_data = np.empty([0,3,extend_data_length])
    test_label = np.empty([0,1])
    for file_name in os.listdir(test_dir):
        filename = test_dir + '/' + file_name
        data_raw = np.loadtxt(filename)
        data_line = data_extend(data_raw,max_length=extend_data_length)
        data_line = data_line.reshape((1, data_line.shape[1], data_line.shape[0]))
        test_data = np.append(test_data, data_line, axis=0)
        test_label = np.append(test_label, int(file_name.split('_')[-1].split('.')[0]))
              
    np.save('test_data.npy', test_data)
    np.save('test_label.npy', test_label)
                            

def train_model(train_data_file, train_label_file, user_name):
    train_X = np.load(train_data_file)
    train_y = np.load(train_label_file)
    knn_clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
    knn_clf.fit(train_X, train_y)
    print('finished training')
    joblib.dump(knn_clf, 'gesture_classify_'+user_name+'.pkl') 


def process_file_noise_svm_train_test(head_dir, noise_dir, extend_data_length):
    train_test_data = np.empty([0,extend_data_length])
    train_test_label = np.empty([0,1])
    for file_name in os.listdir(head_dir):
        if file_name[-4:] == '.txt':
            filename = head_dir + '/' + file_name
            data_raw = np.loadtxt(filename)
            data_line = data_extend(data_raw,max_length=extend_data_length)
            data_line_energy = imu_energy(data_line)
            train_test_data = np.append(train_test_data, data_line_energy.reshape((1, len(data_line_energy))), axis=0)
            train_test_label = np.append(train_test_label, int(file_name.split('_')[-1].split('.')[0]))
    
    for file_name in os.listdir(noise_dir):
        if file_name[-4:] == '.txt':
            filename = noise_dir + '/' + file_name
            data_raw = np.loadtxt(filename)
            data_line = data_extend(data_raw,max_length=extend_data_length)
            data_line_energy = imu_energy(data_line)
            train_test_data = np.append(train_test_data, data_line_energy.reshape((1, len(data_line_energy))), axis=0)
            train_test_label = np.append(train_test_label, int(file_name.split('_')[-1].split('.')[0]))
              
    np.save('noise_train_test_data.npy', train_test_data)
    np.save('noise_train_test_label.npy', train_test_label)


def calculate_prob(input_num_list, word_prob_dict, emission_matrix_dict, alpha = 0.7):
    predict_word_set = {}
    n = len(input_num_list)
    for potential_word in word_prob_dict.keys():
        if len(potential_word) >= n:
            multi_prob = 1
            m = len(potential_word)
            for k in range(n):
                emission_prob_key = input_num_list[k] + '_' + potential_word[k]
                multi_prob = multi_prob * emission_matrix_dict[emission_prob_key] * math.pow(alpha, (m-n))
            final_prob = multi_prob * word_prob_dict[potential_word]
            predict_word_set[potential_word] = final_prob
    predict_word_set_sorted = sorted(predict_word_set.items(), key=lambda d:d[1], reverse = True)
    result_top_3 = [candidate_item[0] for candidate_item in predict_word_set_sorted[0:3]]
   
    return result_top_3




