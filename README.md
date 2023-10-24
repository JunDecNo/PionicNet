# PionicNet

Save my paper code through this repository

### dataSet

BioLiP: <http://zhanglab.ccmb.med.umich.edu/BioLiP/> 2023/10/7 version
PDB: <https://www.rcsb.org/>

### environment

- python 3.8.10
- pytorch 2.0.1
- cuda 11.8
- cudnn 8.9.5.29

### code

**data**

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

 - 正电荷。
 - 金属性
 - 固态状态
 - 金属晶体结构
 - 反应性
 - 色彩
 - 多价性
 - 催化性
