import numpy as np
import pandas as pd
# 如何将金属离子建模是一个问题

ionic_list = ['ZN', 'MG', 'CA', 'MN', 'FE', 'CU', 'FE2', 'NA', 'K', 'NI', 'G']

def normalize(dict):
    array = np.array(list(dict.values()))
    return (array-np.min(array))/(np.max(array)-np.min(array)) 
def getIonicEmb(ionic):
    # 元素周期表序
    period_dict = {'ZN': 30,'MG': 12,'CA': 20,'MN': 25,'FE': 26,'CU': 29,'FE2': 26,'NA': 11,'K': 19,'NI': 28,'G': 31}
    # 最外围电子数
    charge_dict = {'ZN': 2,'MG': 2,'CA': 2,'MN': 7,'FE': 8,'CU': 1,'FE2': 8,'NA': 1,'K': 1,'NI': 10,'G': 3}
    # 按照导电性分
    ec_dict = {'ZN': 16.6,'MG': 22.6,'CA': 29.7,'MN': 7.8,'FE': 10,'CU': 59.6,'FE2': 9.71,'NA': 21.5,'K': 13.3,'NI': 14.7,'G': 7}
    # 金属质量
    mass_dict = {'ZN': 65.38,'MG': 24.31,'CA': 40.08,'MN': 54.94,'FE': 55.85,'CU': 63.55,'FE2': 55.85,'NA': 22.99,'K': 39.10,'NI': 58.69,'G': 196.97}
    # 金属半径
    radius_dict = {'ZN': 1.39,'MG': 1.60,'CA': 1.97,'MN': 1.73,'FE': 1.72,'CU': 1.58,'FE2': 1.72,'NA': 1.66,'K': 2.03,'NI': 1.63,'G': 1.37}
    # 金属密度
    density_dict = {'ZN': 7.14,'MG': 1.74,'CA': 1.55,'MN': 7.43,'FE': 7.87,'CU': 8.96,'FE2': 7.87,'NA': 0.97,'K': 0.86,'NI': 8.91,'G': 19.32}
    # 金属熔点 Ceclsius
    melting_dict = {'ZN': 692.68,'MG': 923.15,'CA': 1115,'MN': 1519,'FE': 1808,'CU': 1357.77,'FE2': 1808,'NA': 371.15,'K': 336.53,'NI': 1728,'G': 933.47}
    # 金属沸点 Ceclsius
    boiling_dict = {'ZN': 1180,'MG': 1363,'CA': 1757,'MN': 2334,'FE': 3023,'CU': 2835,'FE2': 3023,'NA': 1156,'K': 1032,'NI': 3186,'G': 2807}
    # 金属热导率
    thermal_dict = {'ZN': 116,'MG': 156,'CA': 201,'MN': 7.81,'FE': 80.4,'CU': 401,'FE2': 80.4,'NA': 142,'K': 102,'NI': 90.9,'G': 317}
    # 金属热膨胀系数
    expansion_dict = {'ZN': 30.2,'MG': 24.8,'CA': 22.3,'MN': 21.7,'FE': 11.8,'CU': 16.5,'FE2': 11.8,'NA': 71,'K': 83,'NI': 13.4,'G': 14.2}
    # 金属电负性
    electronegativity_dict = {'ZN': 1.65,'MG': 1.31,'CA': 1.00,'MN': 1.55,'FE': 1.83,'CU': 1.90,'FE2': 1.83,'NA': 0.93,'K': 0.82,'NI': 1.91,'G': 2.04}
    # 金属电离能
    ionization_dict = {'ZN': 906.4,'MG': 737.7,'CA': 589.8,'MN': 717.3,'FE': 762.5,'CU': 745.5,'FE2': 762.5,'NA': 495.8,'K': 418.8,'NI': 737.1,'G': 906.4}
    # 金属电子亲和能
    affinity_dict = {'ZN': 0,'MG': 0,'CA': 0,'MN': 0,'FE': 0,'CU': 0,'FE2': 0,'NA': 52.8,'K': 48.4,'NI': 112,'G': 0}
    # 金属晶体结构
    structure_dict = {'ZN': 0,'MG': 1,'CA': 1,'MN': 3,'FE': 3,'CU': 2,'FE2': 3,'NA': 4,'K': 4,'NI': 3,'G': 1}
    # 金属反应性
    reactivity_dict = {'ZN': 1,'MG': 2,'CA': 1,'MN': 1,'FE': 2,'CU': 1,'FE2': 1,'NA': 2,'K': 2,'NI': 1,'G': 0}
    # 金属色彩
    color_dict = {'ZN': 1,'MG': 0,'CA': 0,'MN': 3,'FE': 0,'CU': 2,'FE2': 3,'NA': 0,'K': 0,'NI': 0,'G': 0}
    # 金属多价性
    valence_dict = {'ZN': 2,'MG': 1,'CA': 2,'MN': 5,'FE': 5,'CU': 2,'FE2': 1,'NA': 1,'K': 1,'NI': 2,'G': 1}
    # 金属催化性
    catalysis_dict = {'ZN': 1,'MG': 0,'CA': 1,'MN': 1,'FE': 1,'CU': 1,'FE2': 1,'NA': 0,'K': 1,'NI': 1,'G': 0}
    # 范德华力半径
    vander_dict = {'ZN': 139,'MG': 173,'CA': 231,'MN': 197,'FE': 156,'CU': 140,'FE2': 194,'NA': 227,'K': 275,'NI': 163,'G': 187}
    
    # 一共有19个特征
    # 合并
    merge_df = pd.DataFrame(columns=['ZN', 'MG', 'CA', 'MN', 'FE', 'CU', 'FE2', 'NA', 'K', 'NI', 'G'])
    # for i in range(19):
    #     merge_df = pd.concat([merge_df, pd.DataFrame(period_dict)], ignore_index=True)
    print(type(period_dict))
    df = pd.DataFrame(period_dict)
    merge_df = pd.concat([merge_df, df], ignore_index=True)
    # 进行归一化操作
    print(merge_df)
    
# 将金属离子的特征嵌入保存到文件中
if __name__ == '__main__':
    getIonicEmb('ZN')