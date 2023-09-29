import serial, os, time, random, ctypes, sys, json
import numpy as np
from multiprocessing import Process, Array, Value, Queue
import multiprocessing
import pandas as pd
import matplotlib as mp 
mp.use('TkAgg')
import matplotlib.pyplot as plt
import utils_module as tm
from math import *
import joblib

def process_0_collect_imu(raw_imu,data_root_folder):
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
            

def process_1_gesture_detection(raw_imu,data_root_folder,user_text_queue,refresh_flag,next_phrase_flag,auto_text_queue,target_text_queue,input_number_flag,input_number_num,input_gesture_id):
    all_imu = []
    gesture_unit = []

    threshold_gyro_energy = 30
    threshold_imu_length = 20
    theshold_width = 15
    buffer_size = 20

    count_point = 0
    count_segment = 0
    flag_pos = 0
    extend_data_length = 100

    new_clf = joblib.load('gesture_classify_model.pkl') 
    gesture_list = ['Single_Left_Click','Single_Right_Click',
                    'Double_Left_Click','Double_Right_Click',
                    'Single_Back_Click',
                    'Left_Slide','Right_Slide']
    Teeth_Gesture_Select_Set = gesture_list[0:4]
    Teeth_Gesture_Delete_Action = 'Left_Slide'
    Teeth_Gesture_Jump_Action = 'Single_Back_Click'
    Teeth_Gesture_Next_Action = 'Right_Slide'

    ### load word prob dictionary
    with open('word_prob_15000.json', 'r') as f1:
        word_prob_dict = json.load(fp=f1)
    with open('spatial_model_4_group_dict.json', 'r') as f2:
        emission_matrix_dict = json.load(fp=f2)

    jump_flag = 0
    phrase_lock = 0
    delete_lock = 0
    gesture_num = 0
    word_num_per_phrase = 0
    user_self_input_num_per_word = 0

    auto_word_set = []
    user_word_set = []
    input_number_set = []
    print('initializing ...')
    time.sleep(3)
    target_word_set = target_text_queue.get()
    target_text_queue.put(target_word_set)

    timer_1 = time.time()
    timer_2 = time.time()

    word_time_1 = time.time()
    word_time_2 = time.time()

    t1 = time.time()
    t2 = time.time()
    t3 = time.time()

    auto_predict_time_1 = time.time()
    auto_predict_time_2 = time.time()

    def gesture_to_number(teeth_gesture):
        if teeth_gesture in Teeth_Gesture_Select_Set:
            return str(Teeth_Gesture_Select_Set.index(teeth_gesture) + 1)
        else:
            print('invalid teeth gesture...')
            return '0'

    def imu_point_energy(imu_point_3):
        return sqrt(imu_point_3[0]*imu_point_3[0]+imu_point_3[1]*imu_point_3[1]+imu_point_3[2]*imu_point_3[2])

    while True:
        if raw_imu.empty() == False:
            count_point += 1
            imu_point = raw_imu.get()
            all_imu.append(imu_point)
            if imu_point_energy(imu_point[0:3])>threshold_gyro_energy:
                if flag_pos == 0:
                    t3 = time.time()
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
                    print('*'*100)
                    print('gesture collection time: ', t1-t3)
                    gesture_unit_data = np.array(gesture_unit)
                    np.savetxt(data_root_folder+'/segment_data/test/'+'%d_%d.txt' % (count_segment, start_pos), gesture_unit_data, fmt='%.2f')
                    count_segment += 1
                    print('detected! Wait for next one............')
                    gesture_unit_data_extend = tm.data_extend(gesture_unit_data,max_length=extend_data_length)
                    gesture_unit_data_extend = gesture_unit_data_extend.reshape((1, gesture_unit_data_extend.shape[1], gesture_unit_data_extend.shape[0]))
                    predicted_gesture_index = new_clf.predict(gesture_unit_data_extend)
                    predicted_gesture = gesture_list[int(predicted_gesture_index)]
                    print('Predict: ', predicted_gesture)
                    t2 = time.time()
                    print('gesture calculation time: ',t2-t1)
                    gesture_num += 1
                    input_gesture_id.value = int(predicted_gesture_index)
                    if gesture_num == 1:
                        word_time_1 = time.time()
                        timer_1 = time.time()
                        print('start now !!!!!','.'*80)
                    if predicted_gesture in Teeth_Gesture_Select_Set and gesture_num > 1 and phrase_lock == 0:
                        if jump_flag != 0 and len(user_word_set) != 0:
                            target_word_set = target_text_queue.get()
                            target_text_queue.put(target_word_set)
                            if delete_lock == 0:
                                with open(data_root_folder+'/'+"user_entry_word.txt","a+") as f3:
                                    f3.write(str(user_word_set[-2])+'\n')
                                with open(data_root_folder+'/'+"user_entry_word_time.txt","a+") as f4:
                                    f4.write(str(word_time_2-word_time_1)+'\n')
                                with open(data_root_folder+'/'+"truth_text.txt","a+") as f5:
                                    f5.write(str(target_word_set.split()[word_num_per_phrase])+'\n')
                                with open(data_root_folder+'/'+"user_self_input_word_length.txt","a+") as f6:
                                    f6.write(str(user_self_input_num_per_word)+'\n')
                                with open(data_root_folder+'/'+"user_input_gesture_set.txt","a+") as f7:
                                    f7.write(''.join(input_number_set_temp)+'\n')
                                print('User Input Word: ',user_word_set[-2])
                                print('Truth Word: ',target_word_set.split()[word_num_per_phrase])
                                print('Single Word Input Time: ',word_time_2-word_time_1)
                                word_time_1 = word_time_2
                            word_num_per_phrase += 1
                            print('word_num_per_phrase: ', word_num_per_phrase)
                            jump_flag = 0
                            user_self_input_num_per_word = 0
                        elif jump_flag != 0:
                            word_num_per_phrase += 1
                            jump_flag = 0
                            user_self_input_num_per_word = 0
                        input_number = gesture_to_number(predicted_gesture)
                        input_number_set.append(input_number)
                        input_number_set_temp = input_number_set
                        input_number_num.value += 1
                        user_self_input_num_per_word += 1
                        print('input_number_set: ',input_number_set)
                        auto_predict_time_1 = time.time()
                        auto_word_set = tm.calculate_prob(input_number_set, word_prob_dict, emission_matrix_dict)
                        auto_predict_time_2 = time.time()
                        print('auto predict word time: ', auto_predict_time_2 - auto_predict_time_1)
                        candidate_top_three = ' '.join(auto_word_set)
                        temp = auto_text_queue.get()
                        auto_text_queue.put(candidate_top_three)
                        input_number_flag.value = int(input_number)
                        refresh_flag.value = 1
                        delete_lock = 0
                        print('target_word_set: ', target_word_set)
                        print('auto_word_set: ', auto_word_set)
                        print('user_word_set: ', user_word_set)

                    elif predicted_gesture == Teeth_Gesture_Delete_Action and gesture_num > 1:
                        if jump_flag == 0:
                            input_number_set = input_number_set[0:-1]
                            input_number_set_temp = input_number_set
                            input_number_num.value -= 1
                            if len(input_number_set) == 0:
                                auto_word_set = []
                            else:
                                auto_predict_time_1 = time.time()
                                auto_word_set = tm.calculate_prob(input_number_set, word_prob_dict, emission_matrix_dict)
                                auto_predict_time_2 = time.time()
                                ('auto predict word time: ', auto_predict_time_2 - auto_predict_time_1)
                            candidate_top_three = ' '.join(auto_word_set)
                            temp = auto_text_queue.get()
                            auto_text_queue.put(candidate_top_three)
                            refresh_flag.value = 1
                            print('target_word_set: ', target_word_set)
                            print('auto_word_set: ', auto_word_set)
                            print('user_word_set: ', user_word_set)
                            print('input_number_set: ',input_number_set)
                            print('cancel the last gesture')
                        elif len(user_word_set) > 0:
                            del user_word_set[-1]
                            del user_word_set[-1]
                            phrase_lock = 0
                            word_num_per_phrase -= 1
                            input_number_set = []
                            input_number_num.value = 0
                            auto_word_set = []
                            candidate_top_three = ' '
                            temp = auto_text_queue.get()
                            auto_text_queue.put(candidate_top_three)
                            temp = user_text_queue.get()
                            user_text_queue.put(''.join(user_word_set))
                            refresh_flag.value = 1
                            print('target_word_set: ', target_word_set)
                            print('auto_word_set: ', auto_word_set)
                            print('user_word_set: ', user_word_set)
                            print('input_number_set: ',input_number_set)
                            print('cancel the last word')
                            jump_flag = 1
                            delete_lock = 1

                    elif predicted_gesture == Teeth_Gesture_Jump_Action and gesture_num > 1 and len(auto_word_set) != 0:
                        
                        timer_2 = time.time()
                        word_time_2 = time.time()
                        jump_flag += 1
                        if jump_flag == 1:
                            user_word_set.append(auto_word_set[jump_flag-1])
                            user_word_set.append(' ')
                            input_number_set = []
                            input_number_num.value = 0
                            
                        elif jump_flag > 1:
                            user_word_set[-2] = auto_word_set[int((jump_flag-1)%3)]
                            input_number_set = []
                            input_number_num.value = 0
                        
                        if len(user_word_set) == 2 * len(target_word_set.split()):
                            phrase_lock = 1

                        temp = user_text_queue.get()
                        user_text_queue.put(''.join(user_word_set))
                        refresh_flag.value = 1
                        print('target_word_set: ', target_word_set)
                        print('auto_word_set: ', auto_word_set)
                        print('user_word_set: ', user_word_set)
                        print('input_number_set: ', input_number_set)

                    elif predicted_gesture == Teeth_Gesture_Next_Action and gesture_num > 1:
                        phrase_lock = 0
                        print('word_num_per_phrase: ',word_num_per_phrase)
                        if word_num_per_phrase + 1 == len(target_word_set.split()):
                            with open(data_root_folder+'/'+"user_entry_word.txt","a+") as f3:
                                f3.write(str(user_word_set[-2])+'\n')
                            with open(data_root_folder+'/'+"user_entry_word_time.txt","a+") as f4:
                                f4.write(str(word_time_2-word_time_1)+'\n')
                            with open(data_root_folder+'/'+"truth_text.txt","a+") as f5:
                                f5.write(str(target_word_set.split()[word_num_per_phrase])+'\n')
                            with open(data_root_folder+'/'+"user_self_input_word_length.txt","a+") as f6:
                                f6.write(str(user_self_input_num_per_word)+'\n')
                            with open(data_root_folder+'/'+"user_input_gesture_set.txt","a+") as f7:
                                f7.write(''.join(input_number_set_temp)+'\n')
                            with open(data_root_folder+'/'+"user_entry_phrase_time.txt","a+") as f8:
                                f8.write(str(timer_2-timer_1)+'\n')
                            print('User Input Word: ',user_word_set[-2])
                        
                            print('Truth Word: ',target_word_set.split()[word_num_per_phrase])
                            print('Single Word Input Time: ',word_time_2-word_time_1)
                            print('phrase input time: ',timer_2-timer_1)
                            user_self_input_num_per_word = 0
                            next_phrase_flag.value = 1
                            word_num_per_phrase = 0
                            user_word_set = []
                            auto_word_set = []
                            input_number_set = []
                            input_number_num.value = 0
                            target_word_set = target_text_queue.get()
                            target_text_queue.put(target_word_set)
                            word_time_1 = time.time()
                            print('start counting time...','.'*80)
                            print('word_num_per_phrase: ', word_num_per_phrase)
                            jump_flag = 0
                            timer_1 = time.time()
                            refresh_flag.value = 1
                            gesture_unit = []
                            flag_pos = 0
                            print('start counting time...','.'*80)
                        else:
                            gesture_unit = []
                            flag_pos = 0
                else:
                    print('invalid teeth gesture')
                gesture_unit = []
                flag_pos = 0

def process_2_GUI(user_text_queue,refresh_flag,next_phrase_flag,auto_text_queue,target_text_queue,input_number_flag,input_number_num,input_gesture_id):
    def get_phrase(dir):
        phrase_set = pd.read_csv(dir)
        phrase_set = np.array(phrase_set)
        return phrase_set
    
    gesture_list = ['Single_Left_Click','Single_Right_Click',
                    'Double_Left_Click','Double_Right_Click',
                    'Single_Back_Click',
                    'Left_Slide','Right_Slide']
    
    x0 = np.linspace(0.2,0.5,10)
    y0 = np.ones(10) * 0.17

    count_phrase = 0
    next_phrase_flag.value = 0
    phrase_set = get_phrase('phrases_set_session_6.txt')

    target_text = phrase_set[0][0]
    temp = target_text_queue.get()
    target_text_queue.put(target_text)
    auto_text = ''
    user_text = ''
    plt.ion()
    plt.rcParams['figure.figsize'] = (6, 5) 
    while True:
        plt.clf()
        ax=plt.subplot(1, 1, 1)
        
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.draw_artist(ax.plot(x0+0.4*((input_number_flag.value + 1) % 2),y0-0.1*(int((input_number_flag.value + 1) / 2) - 1))[0])
        ax.draw_artist(ax.text(0.1,0.5,'Target: '+'\n\n\n\n'+'Auto-Complete: '+
                                                            '\n\n'+'Input Number: '+'\n\n'+'Writing: '))
        ax.draw_artist(ax.text(0.4,0.35,gesture_list[int(input_gesture_id.value)],fontweight='light'))
        ax.draw_artist(ax.text(0.4,0.6,str(int(input_number_num.value)),fontweight='light'))
        ax.draw_artist(ax.text(0.12,0.1,'A    B    C    D    E    F    G   ||   H    I      J     K    L    M    N' +
                                '\n\n' +  '      O    P    Q    R    S    T   ||   U    V    W    X    Y    Z'))
        ax.draw_artist(ax.text(0.4,0.5,target_text+'\n\n\n\n'+'\n\n\n\n'+user_text))
        ax.draw_artist(ax.text(0.4,0.7,auto_text))
        if len(auto_text) != 0 and input_number_num.value != 0 and auto_text != ' ':
            ax.draw_artist(ax.text(0.4,0.7,auto_text.split()[0][0:int(input_number_num.value)]))
            ax.draw_artist(ax.text(0.4,0.7,auto_text.split()[0][0:int(input_number_num.value)]))
            ax.draw_artist(ax.text(0.4,0.7,auto_text.split()[0][0:int(input_number_num.value)]))
        plt.pause(0.001)
        if refresh_flag.value == 1:
            if next_phrase_flag.value != 1:
                user_text = user_text_queue.get()
                user_text_queue.put(user_text)
                auto_text = auto_text_queue.get()
                auto_text_queue.put(auto_text)
                refresh_flag.value = 0
            else:
                count_phrase = count_phrase + 1
                target_text = phrase_set[count_phrase][0]
                print('next phrase ---------------------------------------- next one : ',target_text)
                temp = target_text_queue.get()
                target_text_queue.put(target_text)
                auto_text = ''
                user_text = ''
                temp = user_text_queue.get()
                user_text_queue.put(user_text)
                temp = auto_text_queue.get()
                auto_text_queue.put(auto_text)
                refresh_flag.value = 0
                next_phrase_flag.value = 0
    plt.ioff()
    plt.show()

if __name__ ==  '__main__':
    raw_imu = Queue()
    user_text_queue = Queue()
    user_text_queue.put('user~text~queue~init')
    auto_text_queue = Queue()
    auto_text_queue.put('auto~text~queue~init')
    target_text_queue = Queue()
    target_text_queue.put('target~text~queue~init')
    refresh_flag = Value('d',0)
    next_phrase_flag = Value('d',0)
    input_number_flag = Value('d',0)
    input_number_num = Value('d',0)
    input_gesture_id = Value('d',0)
    user_name = input("please input your name: ")
    date = time.strftime("%m_%d_%H_%M", time.localtime())
    data_root_folder = 'user_data/'+user_name+'_'+date
    os.mkdir(data_root_folder)
    os.mkdir(data_root_folder+'/segment_data')
    os.mkdir(data_root_folder+'/segment_data/test')
    P0 = Process(target=process_0_collect_imu,args=(raw_imu,data_root_folder,))
    P1 = Process(target=process_1_gesture_detection,args=(raw_imu,data_root_folder,user_text_queue,refresh_flag,
        next_phrase_flag,auto_text_queue,target_text_queue,input_number_flag,input_number_num,input_gesture_id,))
    P2 = Process(target=process_2_GUI,args=(user_text_queue,refresh_flag,next_phrase_flag,auto_text_queue,
        target_text_queue,input_number_flag,input_number_num,input_gesture_id,))
    P0.start()
    P1.start()
    P2.start()
    P1.join()
    P0.terminate()
    P2.terminate()


