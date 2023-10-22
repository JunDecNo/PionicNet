# 用'rb'模式打开.tar.bz2文件
# with tarfile.open('E:\OwnCode\PionicNet\data\BioLiP\\all_PDB\\receptor_08.tar.bz2', 'r:bz2') as tar:
#     # 列出文件列表
#     file_list = tar.getnames()
#     print(file_list)
#     tar.extractall(path='E:\OwnCode\PionicNet\data\BioLiP\\all_PDB\\')
#     print("Files in the archive:")
#     for file in file_list:
#         print(file)


# # 创建一个包含多个值的列表
# my_list = [1, 2, 3, 4, 5]
#
# # 使用for循环一次性取多个值
# for value1, value2 in zip(my_list, my_list[1:]):
#     print(f"Value 1: {value1}, Value 2: {value2}")

# def downloadTar(idx, start, end):
#     for url in tqdm.tqdm(range(start,end), desc='Downloading files and extracting', ncols=100, unit='files', leave=True):
#         sleep(0.1)
#
#
# length = 3956
# num = length // 16
#
# threads = []
# num_threads = 16  # 线程数, 可自己根据情况设置，因为是16核cpu，所以设置为16
# print(length)
# # 每个线程下载的文件数
# num = length // num_threads
# # 创建线程
# for i in range(num_threads):
#     if i == num_threads - 1:
#         t = threading.Thread(target=downloadTar, args=(i, i * num, length))
#     else:
#         t = threading.Thread(target=downloadTar, args=(i, i * num, (i + 1) * num))
#     threads.append(t)
# # 启动线程
# for thread in threads:
#     thread.start()
#
# # 等待所有线程结束
# for thread in threads:
#     thread.join()
# print('================All File Download Finished!================')

# import threading
# from tqdm import tqdm
# import time
#
# # 创建一个锁，用于控制多线程访问进度条
# progress_lock = threading.Lock()
#
# # 定义一个简单的函数，用于模拟任务
# def task(thread_id):
#     for i in tqdm(range(10), desc=f"Thread {thread_id}", leave=False, ncols=100, progress_lock=progress_lock):
#         time.sleep(0.3)
#
# # 创建多个线程来执行任务
# threads = []
#
# num_threads = 3  # 假设创建3个线程
#
# for i in range(num_threads):
#     thread = threading.Thread(target=task, args=(i,))
#     threads.append(thread)
#
# # 启动所有线程
# for thread in threads:
#     thread.start()
#
# # 等待所有线程完成
# for thread in threads:
#     thread.join()
#
# print("All threads have finished.")

# from time import sleep
# from tqdm import trange, tqdm
# from multiprocessing import Pool, Lock

# L = list(range(9))

# def progresser(n):
#     interval = 0.001 / (n + 2)
#     total = 5000
#     text = "#{}, est. {:<04.2}s".format(n, interval * total)
#     for _ in trange(total, desc=text, position=n):
#         sleep(interval)


# if __name__ == "__main__":
#     p = Pool(
#         len(L),
#         initializer=tqdm.set_lock,
#         initargs=(Lock(),),
#     )
#     p.map(progresser, L)
# # import os
# root_path = 'E:\OwnCode\PionicNet'
# # idx = 0
# # start = 0 
# # with open(root_path + f'/data/BioLiP/download_{idx}.txt', 'r') as file_name:
# #     lines = file_name.readlines()
# #     start += len(lines)
# # print(start)
# # print(3956//16)
# # print(os.getcwd())
# for idx in range(16):
#     with open(root_path + f'\code/temp/download_{idx}.txt', 'a') as file_name:
#         pass

str1 = '10000000000000001000000000000010000000000000000000000000000000000000000000'
str2 = '01000000000000000100000000000100000000000000000000000000000000000000000000'
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
        else:
            res.append('0')
    return ''.join(res)
print(orStr(str1,str2))