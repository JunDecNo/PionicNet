# 负责处理序列信息和结构信息的模块
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from Bio import PDB
from pyfastx import Fasta
from tqdm import tqdm
from scipy.spatial import distance
from utils import pd2list

root_path = os.getcwd().replace('\\', '/').split('/')
root_path = root_path[0: len(root_path) - 2]
root_path = '/'.join(root_path)
save_path = root_path + '/data/sets'

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

def getResAbbr(protein):
    res_dict = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
            'GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F',
            'PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
    return res_dict[protein]

def getVander(atom):
    vender_dict = {
    'H': 1.20,'He': 1.40,'Li': 1.82,'Be': 1.53,'B': 1.92,'C': 1.70,'N': 1.55,'O': 1.52,
    'F': 1.47,'Ne': 1.54,'Na': 2.27,'Mg': 1.73,'Al': 1.84,'Si': 2.10,'P': 1.80,'S': 1.80,
    'Cl': 1.75,'K': 2.75,'Ar': 1.88,'Ca': 2.31,'Sc': 2.16,'Ti': 1.87,'V': 1.79,'Cr': 1.89,
    'Mn': 1.97,'Fe': 1.94,'Co': 1.92,'Ni': 1.84,'Cu': 1.86,'Zn': 2.10,'Ga': 1.87,'Ge': 2.11,
    'As': 1.85,'Se': 1.90,'Br': 1.85,'Kr': 2.02,'Rb': 3.03,'Sr': 2.49,'Y': 2.19,'Zr': 2.12,
    'Nb': 2.20,'Mo': 2.16,'Tc': 2.10,'Ru': 2.10,'Rh': 2.05,'Pd': 2.02,'Ag': 2.02,'Cd': 2.23,
    'In': 2.17,'Sn': 2.17,'Sb': 2.06,'Te': 2.06,'I': 1.98,'Xe': 2.16,'Cs': 3.43,'Ba': 2.68,
    'La': 2.43,'Ce': 2.42,'Pr': 2.40,'Nd': 2.39,'Pm': 2.38,'Sm': 2.36,'Eu': 2.35,'Gd': 2.34,
    'Tb': 2.33,'Dy': 2.32,'Ho': 2.32,'Er': 2.30,'Tm': 2.30,'Yb': 2.29,'Lu': 2.28,'Hf': 2.23,
    'Ta': 2.22,'W': 2.21,'Re': 2.17,'Os': 2.16,'Ir': 2.13,'Pt': 2.10,'Au': 2.20,'Hg': 2.18,
    'Tl': 1.96,'Pb': 2.02,'Bi': 2.07,'Po': 1.97,'At': 2.02,'Rn': 2.20,'Fr': 3.48,'Ra': 2.83,
    'Ac': 2.53,'Th': 2.45,'Pa': 2.47,'U': 2.43,'Np': 2.46,'Pu': 2.44,'Am': 2.44,'Cm': 2.45,
    'Bk': 2.44,'Cf': 2.45,'Es': 2.45,'Fm': 2.45,'Md': 2.46,'No': 2.46,'Lr': 2.46,'Rf': 2.46,
    'Db': 2.46,'Sg': 2.46,'Bh': 2.46,'Hs': 2.46,'Mt': 2.46,'Ds': 2.46,'Rg': 2.46,'Cn': 2.46,
    'Nh': 2.46,'Fl': 2.46,'Mc': 2.46,'Lv': 2.46,'Ts': 2.46,'Og': 2.46,}
    return vender_dict[atom]

def atom_lack_feature(residue,atom):
    """
    Args:
        residue (str): 残基名
        atom (str): 原子名

    Returns:
        list: charge num_H ring
    """
    residue = getResAbbr(residue)
    A = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,3,0]}
    V = {'N':[0,1,0], 'CA':[0,1,0], 'C':[0,0,0], 'O':[0,0,0], 'CB':[0,1,0], 'CG1':[0,3,0], 'CG2':[0,3,0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,1,1] }
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0],'CB':[0,2,1], 'CG':[0,2,1], 'CD':[0,2,1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,1,0], 'CD1':[0,3,0], 'CD2':[0,3,0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'CG1':[0,2,0], 'CG2':[0,3,0], 'CD1':[0,3,0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,2,0], 'CD':[0,2,0], 'NE':[0,1,0], 'CZ':[1,0,0], 'NH1':[0,2,0], 'NH2':[0,2,0] }
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[-1,0,0], 'OD1':[-1,0,0], 'OD2':[-1,0,0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[-1,0,0], 'OE1':[-1,0,0], 'OE2':[-1,0,0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'OG':[0,1,0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,1,0], 'OG1':[0,1,0], 'CG2':[0,3,0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'SG':[-1,1,0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,0,0], 'OD1':[0,0,0], 'ND2':[0,2,0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,0,0], 'OE1':[0,0,0], 'NE2':[0,2,0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'ND1':[-1,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'NE2':[-1,1,1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'CD':[0,2,0], 'CE':[0,2,0], 'NZ':[0,3,1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,1,1], 'CE1':[0,1,1], 'CE2':[0,1,1], 'CZ':[0,0,1], 'OH':[-1,1,0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0], 'CG':[0,2,0], 'SD':[0,0,0], 'CE':[0,3,0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB':[0,2,0],
         'CG':[0,0,1], 'CD1':[0,1,1], 'CD2':[0,0,1], 'NE1':[0,1,1], 'CE2':[0,0,1], 'CE3':[0,1,1], 'CZ2':[0,1,1], 'CZ3':[0,1,1], 'CH2':[0,1,1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                   'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0]/2+0.5,i_fea[1]/3,i_fea[2]] # 归一化
    # atom_features就是包含有charge H ring且归一化后的特征了，value是dict
    try:
        res = atom_features[residue][atom]
    except KeyError:
        print(residue,atom)
    return res

# 定义原子特征    
def atom_features(residue,atom):
    """
        传递atom信息
        原子类型名称  链标识  原子质量  B因子  范德华半径  占有率  坐标  电子电荷  氢原子数  是否处于环中
    """
    # 定义原子特征字典
    atom_dict = {'residue':residue}
    # 原子类型
    atom_name = atom.get_id() # 原子名称，包含侧链信息
    atom_type = atom.element  # 元素名称
    atom_dict['atom_name'] = atom_name
    atom_dict['atom_type'] = atom_type
    # ATOM中侧链标识
    atom_dict['chain'] = 0 if atom.get_id() in ['H','C','O','N','CA'] else 1
    # 原子质量
    atom_dict['mass'] = atom.mass
    # B因子 使用biopython中的Bfactor
    atom_dict['bfactor'] = atom.get_bfactor()
    # 范德华力半径
    atom_dict['vander'] = getVander(atom_type)
    # 占用率
    atom_dict['occupancy'] = atom.get_occupancy()
    # 原子的坐标
    atom_dict['coord'] = [atom.get_coord()]
    # 电荷信息 PDB没有包含，需要手动定义
    # 氢原子数
    # 是否在环内
    atom_dict['charge'],atom_dict['numH'],atom_dict['ring'] = atom_lack_feature(residue,atom_name)
    return atom_dict
    
    
def getPostion(atom_pos,atom_mass,atom_chain):
    """
    Args:
        residue (str): 残基名
        atom (str): 原子名
        type (str): 表示选择哪些原子的位置作为残基的位置
    Returns:
        residue的位置
        需要传入，残基中所有原子的位置和质量
    """
    # C表示主链原子0，C表示侧链原子1，CA表示所有原子2
    C = np.where(atom_chain==0)
    SC = np.where(atom_chain==1)
    if len(SC[0])==0:
        SC = C
    CA = np.where(atom_chain!=2)
    pos_CA = np.dot(atom_mass[CA],atom_pos[CA])/np.sum(atom_mass[CA])
    pos_C = np.dot(atom_mass[C],atom_pos[C])/np.sum(atom_mass[C])
    pos_SC = np.dot(atom_mass[SC],atom_pos[SC])/np.sum(atom_mass[SC])
    return np.round(pos_CA,8),np.round(pos_C,8),np.round(pos_SC,8)
    
def getNeighbor(ns,residue,position,radius):
    """
    Args:
        ns (NeighborSearch): 用于搜索的结构
        residue (str): 残基名
        position(np.array): 残基的位置
        radius (int): 搜索半径
    Returns:
        邻居的信息
    """
    # 然后获取邻居残基的信息
    neighbor = ns.search(position, radius, level='R')
    print(neighbor)
    print(type(neighbor))
    # 获取邻居的信息

    return #neighbor_df

def savePosition(path,nr,ligand,name):
    pdb_path = path + '/' + nr + '/pdb/' + ligand + '/' + name 
    parser = PDB.PDBParser(QUIET=True)
    pdb = parser.get_structure('pdb', pdb_path)
    assert len(pdb[0]) == 1 and len(pdb)==1 , '多链蛋白质或者多模'
    for chain in pdb[0]:
        chainA = chain  # 就是一个蛋白质链
    res_pos_df = pd.DataFrame(columns=['residue','id','pos_CA','pos_C','pos_SC'])
    atom_df = pd.DataFrame(columns=['chain','mass','coord'])
    for residue in chainA:
        # 每次取一个残基
        res_name = residue.get_resname()
        # 清空df
        atom_df = atom_df.drop(atom_df.index,axis=0)
        for atom in residue.get_unpacked_list():
            atom_mass = atom.mass
            atom_chain = 0 if atom.get_id() in ['H','C','O','N','CA'] else 1
            atom_coord = [atom.get_coord()]
            atom_df = pd.concat([atom_df,pd.DataFrame({'chain':atom_chain,'mass':atom_mass,'coord':atom_coord})],ignore_index=True)
        # 获取残基的位置信息
        pos_CA,pos_C,pos_SC = getPostion(np.vstack(atom_df['coord'].values),atom_df['mass'].values,atom_df['chain'].values)
        res_pos_df = pd.concat([res_pos_df,pd.DataFrame({'residue':res_name,'id':residue.get_id()[1],'pos_CA':[pos_CA],'pos_C':[pos_C],'pos_SC':[pos_SC]})],ignore_index=True)   
    res_pos_df.to_csv(save_path+'/'+nr+'/'+ligand+'/'+os.path.splitext(name)[0]+'_pos'+'.csv',index=False)
    return


  
def calAdjacency(pdb_path): 
    pos_df = pd.read_csv(pdb_path,converters={'pos_CA':pd2list,'pos_C':pd2list,'pos_SC':pd2list})
    pos_CA = np.vstack(pos_df['pos_CA'].values)
    pos_C = np.vstack(pos_df['pos_C'].values)
    pos_SC = np.vstack(pos_df['pos_SC'].values)
    # 都是二维列表,计算出邻接矩阵
    dis_dict = {
        'dis_CA':np.sqrt(np.sum((pos_CA[:,np.newaxis]-pos_CA)**2,axis=-1)),
        'dis_C':np.sqrt(np.sum((pos_C[:,np.newaxis]-pos_C)**2,axis=-1)),
        'dis_SC':np.sqrt(np.sum((pos_SC[:,np.newaxis]-pos_SC)**2,axis=-1))}
    return dis_dict
   
def residue_feature(pdb_file):
    """
    Args:
        pdb_file (str): pdb_file path
    Output:
        residue_feature (dict): residue feature
            atom_feature + evolution_feature + residue_feature
    """
    parser = PDB.PDBParser(QUIET=True)
    pdb = parser.get_structure('pdb', pdb_file)
    res_pos_df = pd.DataFrame(columns=['residue','id','pos_CA','pos_C','pos_SC'])
    # ns = PDB.NeighborSearch(list(pdb.get_atoms()))
    # radius = 5
    # neighbor = ns.search(pdb[0]['A'][10].get_coord(), radius, level='R')
    # 传入的模型往往是Model0，ChainA 所以只需要获取这个信息即可
    assert len(pdb[0]) == 1 and len(pdb)==1 , '多链蛋白质或者多模'
    for chain in pdb[0]:
        chainA = chain  # 就是一个蛋白质链
    # 定义一个dataframe用于存储信息
    atom_df = pd.DataFrame(columns=['residue','atom_name','atom_type','chain','mass','bfactor','vander','occupancy','coord','charge','numH','ring'])
    for residue in chainA:
        # 每次取一个残基
        res_name = residue.get_resname()
        # 清空df
        atom_df = atom_df.drop(atom_df.index,axis=0)
        for atom in residue.get_unpacked_list():
            if atom.get_id() == 'OXT':
                continue
            atom_df = pd.concat([atom_df,pd.DataFrame(atom_features(res_name,atom))],ignore_index=True)
            print(type(atom_features(res_name,atom)))
            print(pd.DataFrame(atom_features(res_name,atom)))
            break
        break
        # 获取残基的位置信息
        # print(atom_df)
        pos_CA,pos_C,pos_SC = getPostion(np.vstack(atom_df['coord'].values),atom_df['mass'].values,atom_df['chain'].values)
        res_pos_df = pd.concat([res_pos_df,pd.DataFrame({'residue':residue.get_resname(),'id':residue.get_id()[1],'pos_CA':[pos_CA],'pos_C':[pos_C],'pos_SC':[pos_SC]})],ignore_index=True)   
        # 获取残基的进化信息
        # 获取残基的信息
        
        # 获取残基的标签信息
    print(res_pos_df)
    # res_pos_df.to_csv()
    


# 获取了信息后，就需要将pdb文件转换成图的结构。
def pdb2graph(pdb):
    """
    
    Args:
        pdb (str): 蛋白质的结构信息
    需要读取的信息有：
        ATOM    原子信息
        HETATM  
        HELIX   alpha螺旋
        SHEET   beta折叠
        SSBOND  二硫键
        CONECT  原子连接信息
        ANISOU  原子热振动信息
        LINK    原子连接信息
    """
    # 首先图结构需要包含所有的结构信息
    # ATOM标签格式
    # ATOM SerialNumber AtomNumber ResidueName ChainIdentifier ResidueSeqNum  Position               Occupancy  B-Factor ElemSymbol
    # ATOM      7          N           PRO           B           8            29.888  40.853  73.985  1.00       35.19      N
    # 定义残基对应的ResidueName,将序列和结构对应
    
    # 定义节点信息，包含原子信息和残基信息
    
    # 计算邻接矩阵
    
    # 计算边的信息
    
    # 保存标签信息
    
    #保存图结构信息

if __name__ == '__main__':
    # 用于测试代码是否正确
    # input,label = getStruInfo('1pa2A', 'CA', True)
    # print(input)
    # print(label)
    # atom = def_atom_features()
    # print('A',atom['A'])
    # with open('E:\PaperCode\对标GraphBind\Datasets\PMG\PDB_DF\\6HMZ_X.csv.pkl', 'rb') as file:
    #     loaded_data = pickle.load(file)
    #     print(loaded_data)
    # 整体流程应该是：
    # 先获取序列信息fasta
    # 再获取结构信息各个ligand的pdb
    # 将所有的pdb文件转换成图的结构
    # - 读取pdb文件，获取结构信息
    # - 生成原子特征
    # - 由原子特征得到一部分残基特征
    # - 计算残基位置信息
    # - 计算残基进化信息
    # - 由位置信息计算邻接矩阵--> 将所有的残基都计算出位置，并保存到文件中
    # residue_feature('E:\\OwnCode\\PionicNet\\data\\class\\nr\\pdb\\K\\1d7rA.pdb')
    # savePosition('E:\\OwnCode\\PionicNet\\data\\class','nr','K','1d7rA.pdb')
    #防止重复处理设置两个log文件
    # for i in ['nr']:
    #     for lig in ['ZN', 'MG', 'CA', 'MN', 'FE', 'CU', 'FE2', 'NA', 'K', 'NI', 'G']:
    #         # 使用进度条
    #         if not os.path.exists(save_path+'/Position/'+i+'/save_pos_'+i+lig+'.log'):
    #             with open(save_path+'/Position/'+i+'/save_pos_'+i+lig+'.log','w') as log:
    #                 log.write('')
    #         with open(save_path+'/Position/'+i+'/save_pos_'+i+lig+'.log','r') as log:
    #             logs = log.readlines()
    #         for name in tqdm(os.listdir(root_path + '/data/class/'+i+'/pdb/'+lig+'/'),ncols=100, unit='files'):
    #             if not lig+'_'+name+'\n' in logs:  
    #                 savePosition(root_path+'/data/class',i,lig,name)
    #                 with open(save_path+'/Position/'+i+'/save_pos_'+i+lig+'.log','a') as log:
    #                     log.write(lig+'_'+name+'\n')
    # residue_feature('E:/OwnCode/PionicNet/data/class/nr/pdb/NI/1a5oC.pdb') # 根据输出，pdb存在一些极端的情况，后续需要处理
    # 计算邻接矩阵
    data = calAdjacency('E:/OwnCode/PionicNet/data/sets/nr/Position/CA/1a0jA_pos.csv')
    print(data) `  `
    with open('array.pkl','wb') as file:
        array = pickle.dump(data,file)
                   