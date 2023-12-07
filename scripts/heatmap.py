import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# 假设x、y和z的值
x = np.array(['SLC','SRC','DLC','DRC','SBC','LS','RS'])
    #(['Single_Left_Click','Single_Right_Click','Double_Left_Click','Double_Right_Click','Single_Back_Click','Left_Slide','Right_Slide'])
y = np.array([10, 20, 30, 40, 50])
x_nasa = np.array(['Mental','Physical','Temporal','Effort','Performance','Frustration'])
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
z_nasa = np.array([[7.8,7.8,6.9,7.8,8.1], [5.4,5.4,5.1,7.2,8.7], [8.1,7.5,6.3,6.9,6.9], [7.8,6.9,6.9,8.1,8.4], [7.8,6.6,5.4,6.9,7.8],[6.9,5.7,5.1,7.2,8.1],])
z_accuracy_normalized = np.array([
    [0,1,0.8,0.142857143,0.5,1,0.25],
    [0.333333333,0.5,1,0.571428571,0.5,0.666667,0.75],
    [0.666666667,0.25,1,1,0,1,1],
    [1,0,0.8,0.571428571,1,0.5,0.5],
    [0.666666667,1,0,0,1,0,0],
])
x_u2 = np.array(['UER','CER','TER','Auto-Complete-Rate'])
y_u2 = np.array(['s1', 's2', 's3', 's4', 's5','s6'])
z_u2 = np.array(
[[0.206871036,0.039534884,0.24640592,0.087336719],
[0.22826087,0.032608696,0.260869565,0.150355418],
[0.169767442,0.034883721,0.204651163,0.115598007],
[0.170123701,0.022360767,0.192484469,0.165991426],
[0.156870748,0.014829932,0.17170068,0.143827772],
[0.152083333,0.016666667,0.16875,0.204409722]]
)
# 读取CSV文件
def select_u2(column_to_read):# select the column you want options:0,4,1,3
    file_path = 'u2_result_details_simplied.csv'  # change it to path of u2_result_details_simplied.csv
    u2_per = np.zeros((10, 6))
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for i, row in enumerate(csvreader):
            u2_per[i//6, i%6] = float(row[column_to_read])
    return u2_per
# 绘制热力图
def accuracy():
    plt.figure(figsize=(11, 6))
    ax = sns.heatmap(z_accuracy_normalized, cmap="Reds",annot=True,fmt=".2f", cbar=True, xticklabels=x, yticklabels=y)
    # adjust the size of word in cube
    text_objs = ax.get_children()
    for text_obj in text_objs:
        if isinstance(text_obj, plt.Text):
            text_obj.set_fontsize(16)
    # adjust the size of cbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # plt.xlabel('user')
    # adjust the size of xticks and yticks
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.ylabel('Threshold', fontsize=16)
    plt.title('Accuracy', fontsize=22)
    plt.show()
def time():
    plt.figure(figsize=(11, 6))
    ax = sns.heatmap(z_time, annot=True,cmap="Greens", cbar=True, fmt=".2f", xticklabels=x, yticklabels=y)#figure 5 gesture time
    # adjust the size of word in cube
    text_objs = ax.get_children()
    for text_obj in text_objs:
        if isinstance(text_obj, plt.Text):
            text_obj.set_fontsize(16)
    # adjust the size of cbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # adjust the size of xticks and yticks
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.ylabel('Threshold', fontsize=16)
    plt.title('Time', fontsize=22)
    plt.show()
def nasa_tlx():
    plt.figure(figsize=(11, 6))
    ax = sns.heatmap(z_nasa.T, annot=True,cmap="Blues", cbar=True, fmt=".2f", xticklabels=x_nasa, yticklabels=y)
    # adjust the size of word in cube
    text_objs = ax.get_children()
    for text_obj in text_objs:
        if isinstance(text_obj, plt.Text):
            text_obj.set_fontsize(16)
    # adjust the size of cbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    # adjust the size of xticks and yticks
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.ylabel('Threshold', fontsize=16)
    plt.title('NASA TLX Score', fontsize=22)
    plt.show()
def speed():
    u2_per =  select_u2(0)
    ax = sns.heatmap(u2_per.T, annot=True, cmap="Reds",cbar=True, fmt=".2f", xticklabels=p_u2, yticklabels=y_u2)
    plt.figure(figsize=(11, 6))
    # adjust the size of word in cube
    text_objs = ax.get_children()
    for text_obj in text_objs:
        if isinstance(text_obj, plt.Text):
            text_obj.set_fontsize(16)
    # adjust the size of cbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('user')
    # adjust the size of xticks and yticks
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.ylabel('Session', fontsize=16)
    plt.title('Text Entry Speed(WPM)', fontsize=22)
    plt.show()

def auto_complete():
    u2_per = select_u2(4)
    ax = sns.heatmap(u2_per.T, annot=True, cmap="Greens", cbar=True, fmt=".2f", xticklabels=p_u2, yticklabels=y_u2)
    plt.figure(figsize=(11, 6))
    # adjust the size of word in cube
    text_objs = ax.get_children()
    for text_obj in text_objs:
        if isinstance(text_obj, plt.Text):
            text_obj.set_fontsize(16)
    # adjust the size of cbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('user')
    # adjust the size of xticks and yticks
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.ylabel('Session', fontsize=16)
    plt.title('Auto-complete Rate', fontsize=22)
    plt.show()

def UER():
    u2_per = select_u2(1)
    ax = sns.heatmap(u2_per.T, annot=True, cmap="Blues", cbar=True, fmt=".2f", xticklabels=p_u2, yticklabels=y_u2)
    plt.figure(figsize=(11, 6))
    # adjust the size of word in cube
    text_objs = ax.get_children()
    for text_obj in text_objs:
        if isinstance(text_obj, plt.Text):
            text_obj.set_fontsize(16)
    # adjust the size of cbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('user')
    # adjust the size of xticks and yticks
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.ylabel('Session', fontsize=16)
    plt.title('Uncorrected Error Rate', fontsize=22)
    plt.show()
def TER():
    u2_per = select_u2(3)
    ax = sns.heatmap(u2_per.T, annot=True, cmap="Blues", cbar=True, fmt=".2f", xticklabels=p_u2, yticklabels=y_u2)
    plt.figure(figsize=(11, 6))
    # adjust the size of word in cube
    text_objs = ax.get_children()
    for text_obj in text_objs:
        if isinstance(text_obj, plt.Text):
            text_obj.set_fontsize(16)
    # adjust the size of cbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel('user')
    # adjust the size of xticks and yticks
    plt.xticks(fontsize=18, rotation=0)
    plt.yticks(fontsize=18, rotation=0)
    plt.ylabel('Session', fontsize=16)
    plt.title('Total Error Rate', fontsize=22)
    plt.show()
