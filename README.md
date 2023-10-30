# PionicNet

Save my paper code through this repository

## dataSet

BioLiP: <http://zhanglab.ccmb.med.umich.edu/BioLiP/> 2023/10/7 version
PDB: <https://www.rcsb.org/>

## environment

- python 3.8.10
- pytorch 2.0.1
- cuda 11.8
- cudnn 8.9.5.29

## code

### data

- download_data.py 将BioLiP提供文件下载下来并解压
- classify_data.py 将BioLiP的文件转换为fasta格式并且进行多种不同方式的分类
- statis_data.py 将BioLiP中每个结合配体进行统计
  - Zn 61064 7516
  - Mg 48642 3774
  - Ca 42190 5723
  - Mn 12363 1090
  - Fe 7009 677
  - Cu 6294 537
  - Fe2 3613 202
  - Na 962 <100
  - K 631 <100
  - Ni 404 <100
  - G 257 <100
- fasta_lig.py 将fasta文件中金属离子按类别分开便于相互训练

### model

金属离子主要有以下特征：

- 正电荷
- 金属性
- 固态状态
- 金属晶体结构
- 反应性
- 色彩
- 多价性
- 催化性

氨基酸主要包含以下特征：

- 化学结构
- 侧链基团
- 极性
- 电荷
- 亲水性
- 分子量
- 吸收特性
- 生化反应
- 转化特性
- 酸碱性

原子主要包含以下特征：

- 链标识
- 原子质量
- B因子
- 电子电荷
- 氢原子数
- 范德华半径
- 电子信息 云 壳 数 分布
- 半径
- 辐射性
- 离子化能
- 是否处于环中

### dataset

- BioLiP fold 保存了下载文件log和从BioLiP中下载的所有pdb文件
- class fold 保存了没有经过配体序列合并的的fasta文件和pdb文件
- fasta fold 保存了经过配体序列合并后的fasta文件
- test fold 保存了测试集
- train fold 保存了训练集
- valid fold 保存了验证集
  