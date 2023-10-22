# 因为存在配体序列号的问题，所以需要重新划分一下fasta序列文件。
# 将ligand serial number不为1的加入到为1的01序列中而不要添加新序列
# 同时缩短文件长度，方便于准确的训练
import os
root_path = os.getcwd().replace('\\', '/').split('/')
root_path = root_path[0: len(root_path) - 2]
root_path = '/'.join(root_path)
fasta_ligand_nr = root_path + '/data/class/nr/fasta/'
fasta_ligand_re = root_path + '/data/class/re/fasta/'
cluster_nr = root_path + '/data/fasta/ionic/nr/'
cluster_re = root_path + '/data/fasta/ionic/re/'

def orStr(str1,str2):
    if len(str1)!=len(str2):
        raise ValueError
    l1 =list(str1)
    l2 = list(str2)
    res = []
    for i in range(len(str1)):
        i1 = str1[i]
        i2 = str2[i]
        if i1=='1' or i2=='1':
            res.append('1')
        elif i1 !='\n':
            res.append('0')
    return ''.join(res)+'\n'

# 冗余集
# 读取文件
ionic_list = ['ZN', 'MG', 'CA', 'MN', 'FE', 'CU', 'FE2', 'NA', 'K', 'NI', 'G']
# 分别读取对应的文件
# 算法：
# 创建一个新的文件保存聚类的结果
# 将第一次出现的序列保存到文件中
# 将第二次出现的序列对聚类的最后一条进行修改。。。
for ionic in ionic_list:
    with open(fasta_ligand_re+ionic+'.fa', 'r') as source:
        with open(cluster_re+ionic+'.fa', 'a') as target:
            temp = ''
            # 每次读取两行
            while True:
                name = source.readline()
                label = source.readline()
                item = name + label
                # item就是一个条目
                if item == '':
                    break
                if item != temp:  # 如果item非空同时不与上一个相同就加入的cluster中
                    target.write(item)
                    temp = item
print('dataset finished!')                   

for ionic in ionic_list:
    with open(fasta_ligand_re+ionic+'_label.fa', 'r') as source:
        with open(cluster_re+ionic+'_label.fa', 'a') as target:
            #temp保存同名的最后一行，如果与target的最后一行同名，就将1加上去
            temp_name = source.readline()
            temp_label = source.readline()
            # 每次读取两行
            while True:
                name = source.readline()
                label = source.readline()
                item = name + label
                # item就是一个条目
                if name == '':
                    target.write(temp_name+temp_label)
                    break
                if name != temp_name:  # 如果item非空同时不与上一个相同就加入的cluster中
                    target.write(temp_name+temp_label)
                    temp_name=name
                    temp_label=label
                elif name == temp_name:  # 如果名称一样则修改label
                    temp_label = orStr(temp_label,label)
print('nr dataset finished!')        
