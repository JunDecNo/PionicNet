# 导入自己的工具包
import os
import re

import tqdm

root_path = os.getcwd().replace('\\', '/').split('/')
root_path = root_path[0: len(root_path) - 2]
root_path = '/'.join(root_path)


def save2fasta(file, fasta_file):
    # 定义字典
    dic = {}  # 用于存储类别
    for content in tqdm.tqdm(file.readlines(), desc='Reading file', ncols=100, unit='lines'):
        # 按行分割
        items = content.split('\t')
        class_index = items[4]
        # 如果字典中不存在该类别，则添加，并设置值为1. 如果字典中存在该类别，则值加1
        dic[class_index] = 1 if class_index not in dic.keys() else dic[class_index] + 1
        # 添加fasta
        out_lines = '>' + items[0] + items[1] + '\n' + items[-1]
        fasta_file.write(out_lines)
    # 打印value大于100的字典
    print({k: v for k, v in dic.items() if v > 100})


# 查询BioLiP.txt文件中配体的类别
with open(root_path + '/data/fasta/BioLiP.fa', 'w') as bio_fa:
    with open(root_path + '/data/BioLiP/BioLiP.txt', 'r') as bio_file:
        save2fasta(bio_file, bio_fa)
with open(root_path + '/data/fasta/BioLiP_nr.fa', 'w') as bio_fa_nr:
    with open(root_path + '/data/BioLiP/BioLiP_nr.txt', 'r') as bio_file_nr:
        save2fasta(bio_file_nr, bio_fa_nr)


# 生成label_fasta文件
def save2label(file, fasta_file, all_file):
    for content in tqdm.tqdm(file.readlines(), desc='Reading file', ncols=100, unit='lines'):
        # 按行分割
        item = content.split('\t')
        # 使用空格划分
        pos = item[8]
        pos_num = re.sub(r'[A-Z]|[a-z]', '', pos).split('\40')
        # 生成一个与字符串等长的01序列
        seq = '0' * len(item[-1].replace('\n', ''))
        # 使用pos_num修改对应位置为1
        seq = list(seq)
        for idx in pos_num:
            seq[int(idx)-1] = '1'
        seq = ''.join(seq)
        # 保存蛋白质名称，蛋白质序列，01标签序列
        out_line = '>' + item[0] + item[1] + '\n' + seq + '\n'
        all_line = '>' + item[0] + item[1] + '\n' + item[-1] + seq + '\n'
        fasta_file.write(out_line)
        all_file.write(all_line)


with open(root_path + '/data/fasta/BioLiP_all.fa', 'w') as all_fa:
    with open(root_path + '/data/fasta/BioLiP_label.fa', 'w') as label_fa:
        with open(root_path + '/data/BioLiP/BioLiP.txt', 'r') as bio_file:
            save2label(bio_file, label_fa, all_fa)

with open(root_path + '/data/fasta/BioLiP_all_nr.fa', 'w') as all_fa_nr:
    with open(root_path + '/data/fasta/BioLiP_label_nr.fa', 'w') as label_fa_nr:
        with open(root_path + '/data/BioLiP/BioLiP_nr.txt', 'r') as bio_file_nr:
            save2label(bio_file_nr, label_fa_nr, all_fa_nr)
