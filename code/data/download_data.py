import threading
import urllib.request
from lxml import etree
import tqdm
import os
import tarfile

# 如果系统是windows
if os.name == 'nt':
    root_path = 'E:/OwnCode/PionicNet'
else:
    root_path = '/mnt/sdd/user/zzjun/PionicNet'
# 检查所有的文件夹是否存在, 不存在则创建
if not os.path.exists(root_path + '/data/BioLiP/all_PDB'):
    os.mkdir(root_path + '/data/BioLiP/all_PDB')
    os.mkdir(root_path + '/data/BioLiP/all_PDB/Tar_BZ')
    os.mkdir(root_path + '/data/BioLiP/all_PDB/Tar_BZ_Li')
    os.mkdir(root_path + '/data/BioLiP/all_PDB/Ligand')
    os.mkdir(root_path + '/data/BioLiP/all_PDB/Receptor')
if not os.path.exists(root_path + '/data/BioLiP/all_PDB_nr'):
    os.mkdir(root_path + '/data/BioLiP/all_PDB_nr')
    os.mkdir(root_path + '/data/BioLiP/all_PDB_nr/Tar_BZ')
    os.mkdir(root_path + '/data/BioLiP/all_PDB_nr/Tar_BZ_Li')
    os.mkdir(root_path + '/data/BioLiP/all_PDB_nr/Ligand')
    os.mkdir(root_path + '/data/BioLiP/all_PDB_nr/Receptor')

url = 'https://zhanggroup.org/BioLiP2/weekly.html'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) ' +
                  'Chrome/104.0.5112.81 Safari/537.36 Edg/104.0.1293.54'
}

# Download the HTML page
response = urllib.request.Request(url=url, headers=headers)
response = urllib.request.urlopen(response)

html_save_path = root_path + '/code/temp/weekly.html'
domain = 'https://zhanggroup.org/BioLiP2/'

if response is not None:
    with open(html_save_path, 'w') as file:
        content = response.read().decode('utf-8')
        file.write(content)
else:
    print('Failed to download the HTML page')
    exit(1)

# 保存所有的下载链接
hrefs = []
# 下载文件并解压保存到指定的文件夹

with open(html_save_path, 'r') as file:  # 打开下载的html文件
    content = file.read()
    # 打开下载的html文件
    html = etree.HTML(content)
    # 获取所有的a标签的href属性
    ah = html.xpath('//td/a/@href')
    if type(ah) == list:
        hrefs.extend(ah)  # 将所有的a标签的href属性添加到hrefs列表中
    else:
        print('Failed to get the hrefs')
        exit(1)
urls = [domain + href for href in hrefs]

# 使用多线程下载文件
def downloadTar(idx, start, end, urls):
    for url in tqdm.tqdm(urls[start:end], desc='Downloading files and extracting', ncols=100, unit='files', leave=False):
        # 下载链接
        path = ''
        # 根据链接选择保存路径
        if 'nr' in url and 'receptor' in url:  # 如果链接中包含nr和receptor, 蛋白质非冗余结构
            path = root_path + '/data/BioLiP/all_PDB_nr/Tar_BZ/'
        elif 'nr' not in url and 'receptor' in url:  # 如果链接中不包含nr, 但包含receptor, 蛋白质冗余结构
            path = root_path + '/data/BioLiP/all_PDB/Tar_BZ/'
        elif 'nr' in url and 'ligand' in url:  # 如果链接中包含nr和ligand, 配体非冗余结构
            path = root_path + '/data/BioLiP/all_PDB_nr/Tar_BZ_Li/'
        elif 'nr' not in url and 'ligand' in url:  # 如果链接中不包含nr, 但包含ligand, 配体冗余结构
            path = root_path + '/data/BioLiP/all_PDB/Tar_BZ_Li/'
        # 文件名
        file_name = url.split('/')[-1]  # 获取链接中的文件名
        save_path = path + file_name  # 压缩包保存路径
        nr_path = root_path + '/data/BioLiP/all_PDB_nr/'  # 蛋白质非冗余结构保存路径
        r_path = root_path + '/data/BioLiP/all_PDB/'  # 蛋白质冗余结构保存路径
        # 下载文件
        urllib.request.urlretrieve(url=url, filename=save_path)
        # 解压文件
        with tarfile.open(save_path) as tar:
            tar.extractall(path=nr_path if 'nr' in url else r_path)
        # 保存下载条目
        with open(root_path + f'/data/BioLiP/download_{idx}.txt', 'a') as file_name:
            file_name.write(url + '\t' + save_path + '\n')


threads = []
num_threads = 16  # 线程数, 可自己根据情况设置，因为是16核cpu，所以设置为16
length = len(urls)
print(length)
# 每个线程下载的文件数
num = length // num_threads
# 创建线程
for i in range(num_threads):
    if i == num_threads - 1:
        t = threading.Thread(target=downloadTar, args=(i, i * num, length, urls))
    else:
        t = threading.Thread(target=downloadTar, args=(i, i * num, (i + 1) * num, urls))
    threads.append(t)
# 启动线程
for thread in threads:
    thread.start()

# 等待所有线程结束
for thread in threads:
    thread.join()
print('================All File Download Finished!================')
