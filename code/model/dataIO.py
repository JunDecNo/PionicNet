#负责处理序列信息和结构信息的模块

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

def getSeqInfo(protein,ligand,nr):
    """
    Args:
        name (str): 蛋白质的名字
        ionic (str or int): ligand的名字
        nr (bool): 来自于哪个数据集
    """
    # 获取序列信息
    # 根据输入的参数决定选择的文件夹。
    if nr:
        fa_file = root_path + '/data/fasta/ionic/nr/'+ligand+'.fa'
        la_file = root_path + '/data/fasta/ionic/nr/'+ligand+'_label.fa'
    else:
        fa_file = root_path + '/data/fasta/ionic/re/'+ligand+'.fa'
        la_file = root_path + '/data/fasta/ionic/re/'+ligand+'_label.fa'
    print(fa_file)
    # seq = SeqIO.parse(fa_file,'fasta')
    # # if protein in seq.id:
    # #     print(seq[protein])
    # seq.
    # for record in seq:
    #     print(record.id)
    #     print(record.seq)
    #     break
    fasta = Fasta(fa_file)
    print(fasta[protein])
    
# if __name__ == 'main':
getSeqInfo('1pa2A','CA',True)
        