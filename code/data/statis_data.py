import os
import re
import shutil

import tqdm

root_path = os.getcwd().replace('\\', '/').split('/')
root_path = root_path[0: len(root_path) - 2]
root_path = '/'.join(root_path)
pdb_path = root_path + '/data/BioLiP/all_PDB/Receptor/'
pdb_nr_path = root_path + '/data/BioLiP/all_PDB_nr/receptor_nr/'
# with open(root_path + '/data/BioLiP/BioLiP.txt', 'r') as bio_file:
#     dic = {}
#     # 读取文件
#     for line in bio_file.readlines():
#         item = line.split('\t')[4]
#         # 如果在字典中不存在该类别，则添加，并设置值为1. 如果字典中存在该类别，则值加1
#         if item not in dic.keys():
#             dic[item] = 1
#         else:
#             dic[item] += 1
#     # 打印value大于100的字典
#     v_dic = {k: v for k, v in dic.items() if v > 100}
#     # 按照值降序排序
#     sorted_dict = dict(sorted(v_dic.items(), key=lambda item: item[1], reverse=True))
#     for k, v in sorted_dict.items():
#         print(f"{k}:{v}")
# with open(root_path + '/data/BioLiP/BioLiP_nr.txt', 'r') as bio_file:
#     dic = {}
#     # 读取文件
#     for line in bio_file.readlines():
#         item = line.split('\t')[4]
#         # 如果在字典中不存在该类别，则添加，并设置值为1. 如果字典中存在该类别，则值加1
#         if item not in dic.keys():
#             dic[item] = 1
#         else:
#             dic[item] += 1
#     # 打印value大于100的字典
#     v_dic = {k: v for k, v in dic.items() if v > 100}
#     # 按照值降序排序
#     sorted_dict = dict(sorted(v_dic.items(), key=lambda item: item[1], reverse=True))
#     for k, v in sorted_dict.items():
#         print(f"{k}:{v}")

# 创建对应金属离子的文件夹
# path为/data/class/文件夹
# 在数据集中选择对应的金属离子的数据
# Zn\Mg\Ca\Mn\Fe\Cu\Fe2\Na\K\Ni\G
# 创建文件夹
ionic_list = ['ZN', 'MG', 'CA', 'MN', 'FE', 'CU', 'FE2', 'NA', 'K', 'NI', 'G']

for ionic in ionic_list:
    # 判断文件夹是否存在
    if not os.path.exists(root_path + '/data/class/nr/pdb/' + ionic):
        # 创建文件夹
        os.mkdir(root_path + '/data/class/nr/pdb/' + ionic)
for ionic in ionic_list:
    # 判断文件夹是否存在
    if not os.path.exists(root_path + '/data/class/re/pdb/' + ionic):
        # 创建文件夹
        os.mkdir(root_path + '/data/class/re/pdb/' + ionic)


def write2ionic(ionic_file, res):
    with open(ionic_file, 'a') as f:
        f.write(res)


def getIonicLabel(name, site, seq):
    pos_num = re.sub(r'[A-Z]|[a-z]', '', site).split('\40')
    # 生成一个与字符串等长的01序列
    seq = '0' * len(seq.replace('\n', ''))
    # 使用pos_num修改对应位置为1
    seq = list(seq)
    for idx in pos_num:
        seq[int(idx) - 1] = '1'
    seq = ''.join(seq)
    return '>' + name + '\n' + seq + '\n'


def getIonicPath(ligand, nre=True, pdb=False, label=''):
    """
        nre = True 表示为re冗余集
        nre = False 表示为nr非冗余集
        pdb = True 表示蛋白质文件
        pdb = False 表示fasta文件
    """
    if pdb:
        if nre:
            return root_path + '/data/class/re/pdb/' + ligand + '/'
        else:
            return root_path + '/data/class/nr/pdb/' + ligand + '/'
    else:
        if nre:
            return root_path + '/data/class/re/fasta/' + ligand + label + '.fa'
        else:
            return root_path + '/data/class/nr/fasta/' + ligand + label + '.fa'


# 读取BioLiP.txt文件,并将对应的金属离子的数据写入到对应的文件夹中
# with open(root_path + '/data/BioLiP/BioLiP.txt', 'r') as bio_file:
#     # 读取文件
#     for line in tqdm.tqdm(bio_file.readlines(), desc='classifing ligand to file fold', ncols=100, unit='lines'):
#         item = line.split('\t')
#         # 需要获取的信息，蛋白质名，配体名，位置和序列。
#         # 判断该数据是否为金属离子
#         # 需要名称01 配体4 位点8 序列-1
#         ligand = item[4]
#         name = item[0] + item[1]
#         if ligand in ionic_list:
#             # 整理数据 保存到对应的fasta文件，并且复制pdb文件到对应配体文件夹
#             data = '>' + name + '\n' + item[-1]  # 蛋白质序列的fasta条目
#             write2ionic(getIonicPath(ligand), data)
#             # 保存01序列
#             write2ionic(getIonicPath(ligand, label='_label'), getIonicLabel(name=name, site=item[8], seq=item[-1]))
#             # 复制蛋白质结构
#             shutil.copyfile(pdb_path + name + '.pdb', getIonicPath(ligand, pdb=True) + name + '.pdb')
#     print('================All File Classify Finished!================')
# 读取BioLiP.txt文件,并将对应的金属离子的数据写入到对应的文件夹中
with open(root_path + '/data/BioLiP/BioLiP_nr.txt', 'r') as bio_file:
    # 读取文件
    for line in tqdm.tqdm(bio_file.readlines(), desc='classifing ligand to file fold', ncols=100, unit='lines'):
        item = line.split('\t')
        # 需要获取的信息，蛋白质名，配体名，位置和序列。
        # 判断该数据是否为金属离子
        # 需要名称01 配体4 位点8 序列-1
        ligand = item[4]
        name = item[0] + item[1]
        if ligand in ionic_list:
            # 整理数据 保存到对应的fasta文件，并且复制pdb文件到对应配体文件夹
            data = '>' + name + '\n' + item[-1]  # 蛋白质序列的fasta条目
            write2ionic(getIonicPath(ligand, nre=False), data)
            # 保存01序列
            write2ionic(getIonicPath(ligand, nre=False, label='_label'),
                        getIonicLabel(name=name, site=item[8], seq=item[-1]))
            # 复制蛋白质结构
            shutil.copyfile(pdb_nr_path + name + '.pdb', getIonicPath(ligand, nre=False, pdb=True) + name + '.pdb')
    print('================All File Classify Finished!================')


