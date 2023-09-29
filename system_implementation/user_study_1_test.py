import matplotlib.pyplot as plt
import utils_module as tm
import numpy as np
from math import *
import serial, time, os
from multiprocessing import Process, Array, Value, Queue
import multiprocessing
import joblib

def process_1_collect_imu(class_num,sample_per_class,raw_imu,count_train,data_root_folder):
    ser = serial.Serial('arduino_port', 9600)
    gyro_value = np.zeros(3)
    try:
        while True:
            try:
                line = ser.readline().decode("utf-8")[:-1]
                if line[-1] == '*':
                    line = line[:-1]
                    gyro = line.split(',')
                    if len(gyro) == 3:
                        gyro_value[0] = float(gyro[0])
                        gyro_value[1] = float(gyro[1])
                        gyro_value[2] = float(gyro[2])
                        with open(data_root_folder+'/'+"raw_imu.txt","a+") as f:
                            f.write(str(gyro_value)[1:-1]+'\n')
                    else:
                        continue
            except:
                continue
            raw_imu.put(gyro_value)
    except KeyboardInterrupt:
        pass
def process_2_monitor(class_num,sample_per_class,order_array,gesture_list,raw_imu,count_test,data_root_folder):
    time_base = time.time()
    random_time_array = np.array([1,2,3])
    np.random.shuffle(random_time_array)
    time_interval = random_time_array[0]
    all_imu = []
    gesture_unit = []
    
    threshold_gyro_energy = 30
    threshold_imu_length = 20
    
    theshold_width = 15
    buffer_size = 20

    count_point = 0
    count_segment = 0
    count_recognize = 0
    flag_pos = 0
    flag_finish = 1
    extend_data_length = 100
    threshold_imu_length_max = extend_data_length
    current_gesture_id = int(order_array[int(count_test.value)])

    new_clf = joblib.load('gesture_classify_model.pkl') 

    def imu_point_energy(imu_point_3):
        return sqrt(imu_point_3[0]*imu_point_3[0]+imu_point_3[1]*imu_point_3[1]+imu_point_3[2]*imu_point_3[2])
    tn = time.time()
    while count_test.value < class_num * sample_per_class or flag_finish == 0:
        if flag_pos == 0 and time.time() - time_base > time_interval and flag_finish == 1:
            current_gesture_id = int(order_array[int(count_test.value)])
            print('NO.%d: ' % int(count_test.value), gesture_list[current_gesture_id])
            count_test.value += 1
            flag_finish = 0
            random_time_array = np.array([1,2,3])
            np.random.shuffle(random_time_array)
            time_interval = random_time_array[0]
        if raw_imu.empty() == False:
            count_point += 1
            imu_point = raw_imu.get()
            all_imu.append(imu_point)
            if imu_point_energy(imu_point[0:3])>threshold_gyro_energy:
                if flag_pos == 0:
                    tn = time.time()
                    if len(all_imu) > buffer_size:
                        gesture_unit = all_imu[-buffer_size:]
                    else:
                        gesture_unit = all_imu
                    flag_pos = 1
                    start_pos = count_point
                else:
                    gesture_unit.append(imu_point)
                    flag_pos = count_point
            elif count_point - flag_pos < theshold_width and count_point - flag_pos > 0 and flag_pos > 0:
                gesture_unit.append(imu_point)
            elif count_point - flag_pos > 0 and flag_pos > 0:
                if len(gesture_unit) > threshold_imu_length and len(gesture_unit) < threshold_imu_length_max:
                    t1 = time.time()
                    gesture_unit_data = np.array(gesture_unit)
                    np.savetxt(data_root_folder+'/segment_data/test/'+'%d_%d_%d.txt' % 
                        (count_segment, start_pos, current_gesture_id), gesture_unit_data, fmt='%.2f')
                    count_segment += 1
                    print('detected! Wait for next one............')
                    # print(raw_imu.qsize())
                    # print('-'*80)
                    gesture_unit_data_extend = tm.data_extend(gesture_unit_data,max_length=extend_data_length)
                    # data_line = norm_data(data_line,1)
                    gesture_unit_data_extend = gesture_unit_data_extend.reshape((1, gesture_unit_data_extend.shape[1], gesture_unit_data_extend.shape[0]))
                    predicted_gesture = new_clf.predict(gesture_unit_data_extend)[0]
                    with open(data_root_folder+'/'+"ground_truth.txt","a+") as f3:
                        f3.write(str(int(current_gesture_id))+'\n')
                    with open(data_root_folder+'/'+"predict_result.txt","a+") as f4:
                        f4.write(str(int(predicted_gesture))+'\n')
                    with open(data_root_folder+'/'+"gesture_time.txt","a+") as f5:
                        f5.write(str(t1 - tn)+'\n')
                    if predicted_gesture == current_gesture_id:
                        print('Right Gesture! ',gesture_list[int(predicted_gesture)])
                        count_recognize += 1
                    else:
                        print('Wrong! Predict: ',gesture_list[int(predicted_gesture)],'Truth: ',gesture_list[current_gesture_id])
                        ttt = time.time()
                    print('\n')
                    flag_finish = 1
                    t2 = time.time()
                    # print('calculation time: ', t2-t1)
                    time_base = time.time()

                else:
                    print('invalid teeth gesture')
                gesture_unit = []
                flag_pos = 0
                
    print('Final Accuracy: ',count_recognize/len(order_array))
                



if __name__ == '__main__':
   
    user_name = input("please input your name: ")
    date = time.strftime("%m_%d_%H_%M", time.localtime())
    data_root_folder = 'paper_figure_user_data/'+user_name+'_'+date
    os.mkdir(data_root_folder)
    os.mkdir(data_root_folder+'/segment_data')
    os.mkdir(data_root_folder+'/segment_data/test')
    gesture_list = ['Single_Left_Click','Single_Right_Click',
                    'Double_Left_Click','Double_Right_Click',
                    'Single_Back_Click',
                    'Left_Slide','Right_Slide']
    class_num = len(gesture_list)
    sample_per_class = 3
    order_array = tm.generate_id(class_num,sample_per_class)
    np.savetxt(data_root_folder+'/'+'order_array.txt',order_array.reshape(-1,1),fmt='%d')
    print(order_array)

    raw_imu = Queue()
    count_test = Value('d',0)

    p1 = Process(target=process_1_collect_imu,args=(class_num,sample_per_class,raw_imu,count_test,
        data_root_folder,))
    p2 = Process(target=process_2_monitor,args=(class_num,sample_per_class,order_array,gesture_list,
        raw_imu,count_test,data_root_folder,))
    # start process
    p1.start()    
    p2.start()
    # finish process
    p2.join()
    p1.terminate()
