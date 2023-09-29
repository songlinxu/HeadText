import numpy as np 
import os
import pandas as pd


all_root_folder = 'u1_data/Test_Session'

gesture_list = ['Single_Left_Click','Single_Right_Click','Double_Left_Click','Double_Right_Click',
                'Single_Back_Click','Left_Slide','Right_Slide']
threshold_list = ['threshold=10', 'threshold=20', 'threshold=30', 'threshold=40', 'threshold=50']
def u1_gesture_time_result(file_root):
    gesture_time_array = np.zeros(7)
    gesture_id_raw = np.loadtxt(file_root + '/' + 'ground_truth.txt')
    gesture_time_raw = np.loadtxt(file_root + '/' + 'gesture_time.txt')
    
    if len(gesture_id_raw) != len(gesture_time_raw) or len(gesture_id_raw) != 35:
        print(len(gesture_id_raw))
        print(len(gesture_time_raw))
        print(file_root,'wrong','*'*100)
        return 0

    for i in range(len(gesture_time_raw)):
        gesture_time_array[int(gesture_id_raw[i])] += gesture_time_raw[i]
    
    return gesture_time_array/5

def u1_gesture_accuracy_result(file_root):
    print('processing... ', file_root)
    gesture_accuracy_array = np.zeros(7)

    gesture_truth= np.loadtxt(file_root + '/' + 'ground_truth.txt')
    gesture_predict = np.loadtxt(file_root + '/' + 'predict_result.txt')

    if len(gesture_truth) != len(gesture_predict):
        print(len(gesture_truth))
        print(len(gesture_predict))
        print(file_root,'wrong','*'*100)
        return 0

    for i in range(len(gesture_truth)):
        gesture_truth_current = int(gesture_truth[i])
        gesture_predict_current = int(gesture_predict[i])
        if gesture_truth_current == gesture_predict_current:
            if gesture_truth_current in [0,1,2,3,4,5,6]:
                gesture_accuracy_array[int(gesture_truth_current)] += 1
            
            else:
                print('wrong...',gesture_accuracy_array[i])
    gesture_accuracy_array /= 5
    print('gesture_accuracy_array: ',gesture_accuracy_array)
    return gesture_accuracy_array

folders = os.listdir(all_root_folder)
folders.sort()
u1_gesture_time_result_array = np.zeros((5,10,7))
u1_gesture_accuracy_result_array = np.zeros((5,10,7))
for user_each_root in folders:
    if user_each_root[0] == 'p':
        user_each_root_path = all_root_folder + '/' + user_each_root
        user_id = int(user_each_root.split('_')[0][1:]) - 1
        threshold_id = int(int(user_each_root.split('_')[1]) / 10 - 1)
        u1_gesture_time_result_array[threshold_id][user_id] = u1_gesture_time_result(user_each_root_path)
        u1_gesture_accuracy_result_array[threshold_id][user_id] = u1_gesture_accuracy_result(user_each_root_path)

time_result = np.mean(u1_gesture_time_result_array, axis=1)
accuracy_result = np.mean(u1_gesture_accuracy_result_array, axis=1)
df_time = pd.DataFrame(np.mean(u1_gesture_time_result_array, axis=1), index=threshold_list, columns=gesture_list)
df_accuracy = pd.DataFrame(np.mean(u1_gesture_accuracy_result_array, axis=1), index=threshold_list, columns=gesture_list)
writer_time = pd.ExcelWriter('u1_time_result.xlsx')
writer_accuracy = pd.ExcelWriter('u1_accuracy_result.xlsx')
df_time.to_excel(writer_time, "Sheet1")
df_accuracy.to_excel(writer_accuracy, "Sheet1")
writer_time.save()
writer_accuracy.save()
print(time_result)
print(accuracy_result)