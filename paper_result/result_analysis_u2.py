import numpy as np 
import os,math 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import json

metrics_list = ['Speed', 'UER', 'CER', 'TER', 'Auto-Complete-Rate']
session_list = ['s1', 's2', 's3', 's4', 's5', 's6']
session_list_2 = ['p1s1', 'p1s2', 'p1s3', 'p1s4', 'p1s5', 'p1s6', 
                'p2s1', 'p2s2', 'p2s3', 'p2s4', 'p2s5', 'p2s6',
                'p3s1', 'p3s2', 'p3s3', 'p3s4', 'p3s5', 'p3s6',
                'p4s1', 'p4s2', 'p4s3', 'p4s4', 'p4s5', 'p4s6',
                'p5s1', 'p5s2', 'p5s3', 'p5s4', 'p5s5', 'p5s6',
                'p6s1', 'p6s2', 'p6s3', 'p6s4', 'p6s5', 'p6s6',
                'p7s1', 'p7s2', 'p7s3', 'p7s4', 'p7s5', 'p7s6',
                'p8s1', 'p8s2', 'p8s3', 'p8s4', 'p8s5', 'p8s6',
                'p9s1', 'p9s2', 'p9s3', 'p9s4', 'p9s5', 'p9s6',
                'p10s1', 'p10s2', 'p10s3', 'p10s4', 'p10s5', 'p10s6']

def func_calculate_prob(input_num_list, word_prob_dict, emission_matrix_dict, alpha = 0.7):
    # t1 = time.time()
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

def u2_text_entry_result(root_dir):
# Text Entry Speed ---------------------------------------------------------------------------------
    time_data = np.loadtxt(root_dir+'user_entry_word_time.txt')
    average_time_per_word = np.mean(time_data)
    text_entry_speed = 60/average_time_per_word
    # print('WPM: ',60/average_time_per_word)

# Text Entry Accuracy ------------------------------------------------------------------------------
# Uncorrected Error Rate
    truth_text = pd.read_csv(root_dir+'truth_text.txt',header=None)
    truth_text = np.array(truth_text)
    user_text = pd.read_csv(root_dir+'user_entry_word.txt',header=None)
    user_text = np.array(user_text)
    right_count = 0
    for k in range(min(len(truth_text),len(user_text))):
        if truth_text[k] == user_text[k]:
            right_count += 1
    uncorrected_error_rate = 1-right_count/min(len(truth_text),len(user_text))
    # print('Uncorrected Error Rate(UER): ', uncorrected_error_rate)

# Corrected Error Rate
    auto_correct_count = 0
    with open('word_prob_15000.json', 'r') as f1:
        word_prob_dict = json.load(fp=f1)
    with open('spatial_model_4_group_dict.json', 'r') as f2:
        emission_matrix_dict_auto_correct = json.load(fp=f2)
    with open('spatial_model_4_group_dict_raw.json', 'r') as f3:
        emission_matrix_dict_raw = json.load(fp=f3)
    user_text = pd.read_csv(root_dir+'user_entry_word.txt',header=None)
    user_text = np.array(user_text)
    user_input_gesture_set = pd.read_csv(root_dir+'user_input_gesture_set.txt',header=None)
    user_input_gesture_set = np.array(user_input_gesture_set)
    for i in range(len(user_text)):
        predict_top_3_raw = func_calculate_prob(list(str(user_input_gesture_set[i][0])), word_prob_dict, emission_matrix_dict_raw)
        if user_text[i] not in predict_top_3_raw:
            auto_correct_count += 1
    corrected_error_rate = auto_correct_count/len(user_text)
    # print('Corrected Error Rate(CER): ', corrected_error_rate)

# Total Error Rate
    total_error_rate = uncorrected_error_rate + corrected_error_rate
    # print('Total Error Rate(TER): ', total_error_rate)

# Auto Complete Rate --------------------------------------------------------------------------------
    user_text = pd.read_csv(root_dir+'user_entry_word.txt',header=None)
    user_text = np.array(user_text)
    user_self_input_length = pd.read_csv(root_dir+'user_self_input_word_length.txt',header=None)
    user_self_input_length = np.array(user_self_input_length)
    auto_complete_rate = np.zeros(len(user_text))
    for i in range(len(user_text)):
        each_word_len = len(user_text[i][0])
        auto_complete_rate[i] = 1 - user_self_input_length[i][0] / each_word_len
    mean_auto_complete_rate = np.mean(auto_complete_rate)
    # print('Average Auto-Complete Rate: ',np.mean(auto_complete_rate))

    return text_entry_speed, uncorrected_error_rate, corrected_error_rate, total_error_rate, mean_auto_complete_rate

all_root_folder = 'u2_data'

folders = os.listdir(all_root_folder)
folders.sort()
u2_text_entry_result_array = np.zeros((6,10,5))
u2_text_entry_result_array_2 = np.zeros((60,5))
i = 0
for user_each_root in folders:
    if user_each_root[0] == 'p':
        print('Processing...', user_each_root)
        user_each_root_path = all_root_folder + '/' + user_each_root + '/'
        user_id = int(user_each_root.split('_')[0][1:]) - 1
        session_id = int(user_each_root.split('_')[1][1:]) - 1
        result_item = u2_text_entry_result(user_each_root_path)
        u2_text_entry_result_array[session_id][user_id] = result_item
        u2_text_entry_result_array_2[i] = result_item
        i += 1
        print(result_item)
        print('\n')

result = np.mean(u2_text_entry_result_array, axis=1)
df = pd.DataFrame(np.mean(u2_text_entry_result_array, axis=1), index=session_list, columns=metrics_list)
df2 = pd.DataFrame(u2_text_entry_result_array_2, index=session_list_2, columns=metrics_list)
writer = pd.ExcelWriter('u2_result.xlsx')
writer_2 = pd.ExcelWriter('u2_result_details.xlsx')
df.to_excel(writer, "Sheet1")
df2.to_excel(writer_2, "Sheet1")
writer.save()
writer_2.save()
print(result)
print(u2_text_entry_result_array_2)


