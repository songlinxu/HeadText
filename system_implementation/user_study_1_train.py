import matplotlib.pyplot as plt
import utils_module as tm
import numpy as np
from math import *
import serial, time, os
from multiprocessing import Process, Array, Value, Queue
import multiprocessing

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
def process_2_monitor(class_num,sample_per_class,order_array,gesture_list,raw_imu,count_train,data_root_folder,user_name):
    time_base = time.time()
    random_time_array = np.array([1,2,3])
    np.random.shuffle(random_time_array)
    time_interval = random_time_array[0]
    all_imu = []
    gesture_unit = []
    # best 30
    threshold_gyro_energy = 30
    threshold_imu_length = 20
    theshold_width = 15
    buffer_size = 20

    count_point = 0
    count_segment = 0
    flag_pos = 0
    flag_finish = 1
    current_gesture_id = int(order_array[int(count_train.value)])
    def imu_point_energy(imu_point_3):
        return sqrt(imu_point_3[0]*imu_point_3[0]+imu_point_3[1]*imu_point_3[1]+imu_point_3[2]*imu_point_3[2])

    while count_train.value < class_num * sample_per_class or flag_finish == 0:
        if flag_pos == 0 and time.time() - time_base > time_interval and flag_finish == 1:
            current_gesture_id = int(order_array[int(count_train.value)])
            print('NO.%d: ' % int(count_train.value), gesture_list[current_gesture_id])
            count_train.value += 1
            flag_finish = 0
            random_time_array = np.array([1,2,3])
            np.random.shuffle(random_time_array)
            time_interval = random_time_array[0]
        if raw_imu.empty() == False:
            count_point += 1
            # print(raw_imu.qsize())
            imu_point = raw_imu.get()
            all_imu.append(imu_point)
            if imu_point_energy(imu_point[0:3])>threshold_gyro_energy:
                if flag_pos == 0:
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
                if len(gesture_unit) > threshold_imu_length:
                    t1 = time.time()
                    np.savetxt(data_root_folder+'/segment_data/train/'+'%d_%d_%d.txt' % 
                        (count_segment, start_pos, current_gesture_id), np.array(gesture_unit), fmt='%.2f')
                    count_segment += 1
                    print('detected! Wait for next one............')
                    print('-'*80)
                    # print(raw_imu.qsize())
                    t2 = time.time()
                    # print('processing time: ',t2-t1)
                    flag_finish = 1
                    time_base = time.time()
                else:
                    print('invalid teeth gesture')
                gesture_unit = []
                flag_pos = 0
                
    tm.process_file_train(data_root_folder+'/segment_data/train',100)
    tm.train_model('train_data.npy','train_label.npy',user_name)
                


if __name__ == '__main__':
   
    user_name = input("please input your name: ")
    date = time.strftime("%m_%d_%H_%M", time.localtime())
    data_root_folder = 'user_data/'+user_name+'_'+date
    os.mkdir(data_root_folder)
    os.mkdir(data_root_folder+'/segment_data')
    os.mkdir(data_root_folder+'/segment_data/train')
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
    count_train = Value('d',0)

    p1 = Process(target=process_1_collect_imu,args=(class_num,sample_per_class,raw_imu,count_train,
        data_root_folder,))
    p2 = Process(target=process_2_monitor,args=(class_num,sample_per_class,order_array,gesture_list,
        raw_imu,count_train,data_root_folder,user_name,))
    # start process
    p1.start()
    p2.start()
    # finish process
    p2.join()
    p1.terminate()
