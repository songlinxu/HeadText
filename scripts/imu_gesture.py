import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# path to original data
file_path = './u1_data/Test_Session/p1_10/segment_data/test/0_215_0.txt'
#file_path = './u1_data/Test_Session/p1_10/segment_data/test/7_2147_1.txt'
#file_path = './u1_data/Test_Session/p1_10/segment_data/test/12_3292_2.txt'
#file_path = './u1_data/Test_Session/p1_10/segment_data/test/5_1584_3.txt'
#file_path = './u1_data/Test_Session/p1_10/segment_data/test/35_8690_4.txt'
#file_path = './u1_data/Test_Session/p1_10/segment_data/test/28_6974_5.txt'
#file_path = './u1_data/Test_Session/p1_10/segment_data/test/15_3785_6.txt'
data = np.loadtxt(file_path)

#transfer data to dataframe
df = pd.DataFrame(data, columns=['gyro1', 'gyro2', 'gyro3'])

#draw the figure
plt.figure(figsize=(16, 4))
#plt.ylim(-150, 150)
sns.lineplot(data=df,dashes=False)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.legend(fontsize=20,loc='lower left')
plt.show()
