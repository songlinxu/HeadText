import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import csv
#plt.style.use('seaborn-whitegrid')
palette = pyplot.get_cmap('Set1')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 32,
}
x = np.array([10,20,30,40,50])
s = np.array([1,2,3,4,5,6])
fig=plt.figure(figsize=(8,6))
z_accuracy = np.array([[0.94,0.98,0.9,0.8,0.98,0.94,0.88],
            [0.96,0.94,0.94,0.86,0.98,0.9,0.92],
            [0.98,0.92,0.94,0.92,0.96,0.94,0.94],
            [1,0.9,0.9,0.86,1,0.88,0.86],
            [0.98,0.98,0.74,0.78,1,0.82,0.9]])
z_time= np.array(
[[0.792796669,0.790626502,1.196427937,1.18796662,0.739529119,1.154203682,1.139073195],
[0.725759373,0.717327437,1.076911573,1.090640399,0.566505575,1.040935998,1.007734571],
[0.684752865,0.695052867,1.044154363,1.053745861,0.48189291,0.968091373,0.950198049],
[0.654529448,0.630395517,1.003669696,0.969088422,0.45117867,0.903840971,0.916166954],
[0.630163283,0.617475753,0.931585792,0.963149161,0.388842592,0.885152574,0.912127981]]
)
y = np.array([[7.8,7.8,6.9,7.8,8.1], [5.4,5.4,5.1,7.2,8.7], [8.1,7.5,6.3,6.9,6.9], [7.8,6.9,6.9,8.1,8.4], [7.8,6.6,5.4,6.9,7.8],[6.9,5.7,5.1,7.2,8.1],])
def select_u2(column_to_read):# select the column you want options:0,4,1,3
    file_path = 'u2_result_details_simplied.csv'  # change it to path of u2_result_details_simplied.csv
    u2_per = np.zeros((10, 6))
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for i, row in enumerate(csvreader):
            u2_per[i//6, i%6] = float(row[column_to_read])
    return u2_per
def draw_line(color_index,datas,color_l):
    color=palette(color_index)
    avg=np.mean(datas,axis=0)
    std=np.std(datas,axis=0)
    r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))#上方差
    r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))#下方差
    if color_l==1:
        plt.plot(x, avg, color="#FF0000", linewidth=3.5)  # 红色
        plt.plot(x, avg, 'o', color="#800000", markersize=8, linewidth=3.5)  # 深红色
        plt.fill_between(x, r1, r2, color="#FF9999", alpha=0.2)  # 浅红色
    elif color_l ==2:
        plt.plot(x, avg, color="#00FF00", linewidth=3.5)  # 浅绿色
        plt.plot(x, avg, 'o', color="#008000", markersize=8, linewidth=3.5)  # 深绿色
        plt.fill_between(x, r1, r2, color="#99FF99", alpha=0.2)
    else:
        plt.plot(x, avg, color="#0066CC",linewidth=3.5)
        plt.plot(x, avg, 'o', color="#004080", markersize=8,linewidth=3.5)
        plt.fill_between(x, r1, r2, color="#99CCFF", alpha=0.2)
def draw_line_1(color_index,datas,color_l):
    color=palette(color_index)
    avg=np.mean(datas,axis=0)
    std=np.std(datas,axis=0)
    r1 = list(map(lambda s: s[0]-s[1], zip(avg, std)))#上方差
    r2 = list(map(lambda s: s[0]+s[1], zip(avg, std)))#下方差
    if color_l==1:
        # 在每个点周围标注实际数值
        for i, txt in enumerate(avg):
            plt.annotate(f'{txt:.2f}', (s[i], avg[i]),fontsize=16, textcoords="offset points", xytext=(0, 10), ha='center')
        plt.plot(s, avg, color="#FF0000", linewidth=3.5)  # 红色
        plt.plot(s, avg, 'o', color="#800000", markersize=8, linewidth=3.5)  # 深红色
        plt.fill_between(s, r1, r2, color="#FF9999", alpha=0.2)  # 浅红色
    elif color_l ==2:
        for i, txt in enumerate(avg):
            plt.annotate(f'{txt:.2f}', (s[i], avg[i]),fontsize=16, textcoords="offset points", xytext=(0, 10), ha='center')
        plt.plot(s, avg, color="#00FF00", linewidth=3.5)  # 浅绿色
        plt.plot(s, avg, 'o', color="#008000", markersize=8, linewidth=3.5)  # 深绿色
        plt.fill_between(s, r1, r2, color="#99FF99", alpha=0.2)
    elif color_l == 3:
        for i, txt in enumerate(avg):
            plt.annotate(f'{txt:.2f}', (s[i], avg[i]),fontsize=16, textcoords="offset points", xytext=(0, 10), ha='center')
        plt.plot(s, avg, color="#0066CC",linewidth=3.5)
        plt.plot(s, avg, 'o', color="#004080", markersize=8,linewidth=3.5)
        plt.fill_between(s, r1, r2, color="#99CCFF", alpha=0.2)
    else:
        for i, txt in enumerate(avg):
            plt.annotate(f'{txt:.2f}', (s[i], avg[i]),fontsize=16, textcoords="offset points", xytext=(0, 10), ha='center')
        plt.plot(s, avg, color="#FFA500", linewidth=3.5)
        plt.plot(s, avg, 'o', color="#FF8C00", markersize=8, linewidth=3.5)
        plt.fill_between(s, r1, r2, color="#FFD700", alpha=0.2)

def accuracy():
    draw_line(1,z_accuracy.T,1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(x,fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Threshold',fontsize=16)
    plt.legend(loc='upper left',prop=font1)
    plt.title("Average Accuracy",fontsize=22)
    plt.legend([x[i] for i in range(1)], loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.show()
def time():
    draw_line(1,z_time.T,2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    #draw_line("alg2",2,z_time)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(x,fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Threshold',fontsize=16)
    plt.legend(loc='upper left',prop=font1)
    plt.title("Average Gesture Time(s)",fontsize=22)
    plt.legend([x[i] for i in range(1)], loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.show()
def nasa():
    draw_line(1,y,3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    #draw_line("alg2",2,z_time)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(x,fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Threshold',fontsize=16)
    plt.legend(loc='upper left',prop=font1)
    plt.title("NASA TLX Score",fontsize=22)
    plt.legend([x[i] for i in range(1)], loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.show()
def speed():
    u2_per = select_u2(0)
    draw_line_1(1,u2_per,1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    #draw_line("alg2",2,z_time)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(s,fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Session',fontsize=16)
    plt.legend(loc='upper left',prop=font1)
    plt.title("Mean Text Entry Speed(WMP)",fontsize=22)
    plt.legend([s[i] for i in range(1)], loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.show()
def auto_complete():
    u2_per = select_u2(4)
    draw_line_1(1,u2_per,2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    #draw_line("alg2",2,z_time)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(s,fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Session',fontsize=16)
    plt.legend(loc='upper left',prop=font1)
    plt.title("Mean Auto-complete Rate",fontsize=22)
    plt.legend([s[i] for i in range(1)], loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.show()
def UER():
    u2_per = select_u2(1)
    draw_line_1(1,u2_per,3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    #draw_line("alg2",2,z_time)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(s,fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Session',fontsize=16)
    plt.legend(loc='upper left',prop=font1)
    plt.title("Mean Uncorrected Error Rate",fontsize=22)
    plt.legend([s[i] for i in range(1)], loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.show()
def TER():
    u2_per = select_u2(3)
    draw_line_1(1,u2_per,4)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    #draw_line("alg2",2,z_time)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(s,fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Session',fontsize=16)
    plt.legend(loc='upper left',prop=font1)
    plt.title("Mean Total Error Rate",fontsize=22)
    plt.legend([s[i] for i in range(1)], loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    plt.show()
#TER()
#UER()
auto_complete()
#speed()

#time()
#accuracy()
#nasa()