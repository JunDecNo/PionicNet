# 负责处理序列信息和结构信息的模块

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from Bio import SeqIO
from pyfastx import Fasta

root_path = os.getcwd().replace('\\', '/').split('/')
root_path = root_path[0: len(root_path) - 2]
root_path = '/'.join(root_path)


def getSeqInfo(protein, ligand, nr):
    """
    Args:
        name (str): 蛋白质的名字
        ionic (str or int): ligand的名字
        nr (bool): 来自于哪个数据集
        return:
            str: 表示蛋白质序列
            str: 表示蛋白质结合位点标签
    """
    # 获取序列信息
    # 根据输入的参数决定选择的文件夹。
    if nr:
        fa_file = root_path + '/data/fasta/ionic/nr/' + ligand + '.fa'
        la_file = root_path + '/data/fasta/ionic/nr/' + ligand + '_label.fa'
    else:
        fa_file = root_path + '/data/fasta/ionic/re/' + ligand + '.fa'
        la_file = root_path + '/data/fasta/ionic/re/' + ligand + '_label.fa'

    sequence = Fasta(fa_file)
    label = Fasta(la_file)
    assert protein in sequence.keys(), '不存在这个蛋白质，请检查代码'
    return sequence[protein],label[protein]

def getStruInfo(protein, ligand, nr):
    """
    Args:
        name (str): 蛋白质的名字
        ionic (str or int): ligand的名字
        nr (bool): 来自于哪个数据集
        return:
            str: 表示蛋白质结构信息
            str: 表示蛋白质文件路径
    """
    if nr:
        pdb_file = root_path + '/data/class/nr/pdb/'+ ligand + '/'
    else:
        pdb_file = root_path + '/data/class/re/pdb/'+ ligand + '/'
    with open(pdb_file + protein + '.pdb', 'r') as f:
        stru = f.read()
    return stru, pdb_file + protein + '.pdb'
        
if __name__ == '__main__':
    # 用于测试代码是否正确
    input,label = getStruInfo('1pa2A', 'CA', True)
    print(input)
    print(label)