from LinkedList import *
from datawithll import get_training_data
from datawithll import get_photo_paths
import time
import os
import psutil
import numpy as np
import pickle
if __name__ == '__main__':
    n = 12
    lltime1 = []
    lltime2 = []
    lltime3 = []
    llmemory1 = []
    llmemory2 = []
    llmemory3 = []
    for i in range(n):
        train = LinkedList()
        train_data_label = LinkedList()
        # 获取照片地址列表
        NORMAL_train_photo_path = get_photo_paths(r'D:\dshm\chest_xray\train\NORMAL')
        PNEUMONIA_train_photo_path = get_photo_paths(r'D:\dshm\chest_xray\train\PNEUMONIA')
        train_photo_path = NORMAL_train_photo_path + PNEUMONIA_train_photo_path

        # 获取当前进程的内存占用情况
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB

        # 将图片信息加载到数据结构中
        start_time = time.perf_counter()
        for path in train_photo_path:
            print('path')
            train.add_image(path)
            print('over')
        end_time = time.perf_counter()
        print('over1')
        elapsed_time = end_time - start_time
        print(f"将图片信息加载到数据结构中的加载时间: {elapsed_time} 秒")

        # 循环结束后再次获取内存占用情况
        end_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        # 计算内存占用增加量
        memory_increase = end_memory - start_memory
        print(f"添加数据后内存占用增加：{memory_increase} MB")
        lltime1.append(elapsed_time)
        llmemory1.append((memory_increase))

        # 获取当前进程的内存占用情况
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB

        start_time = time.perf_counter()
        train_data = get_training_data(train, train_data_label)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"将训练数据和训练标签匹配的加载时间: {elapsed_time} 秒")

        # 循环结束后再次获取内存占用情况
        end_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        # 计算内存占用增加量
        memory_increase = end_memory - start_memory
        print(f"添加数据后内存占用增加：{memory_increase} MB")
        lltime2.append(elapsed_time)
        llmemory2.append((memory_increase))

        # 获取当前进程的内存占用情况
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        start_time = time.perf_counter()

        train_label = np.array(train_data_label.to_list(), dtype=np.int32)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"使用链表中的内置函数将链表转换成列表后转为array数组的时间: {elapsed_time} 秒")

        # 循环结束后再次获取内存占用情况
        end_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        # 计算内存占用增加量
        memory_increase = end_memory - start_memory
        print(f"添加数据后内存占用增加：{memory_increase} MB")
        lltime3.append(elapsed_time)
        llmemory3.append((memory_increase))

    lltime1 = [0.0020850000000791624, 0.002318400001968257, 0.0019806999916909263, 0.0019991999870399013
               , 0.0021370000031311065, 0.00202980000176467, 0.001931699996930547, 0.0020440000080270693
               , 0.0021360999962780625, 0.0024514999968232587, 0.0021930000075371936, 0.002380299993092194]
    lltime2 = [20.474980900005903, 20.67972660000669, 20.854642799997237, 20.824155399997835
               , 20.935724299997673, 20.81376750000345, 20.863444100003107, 20.90902589999314
               , 20.755620399999316, 20.671048799995333, 20.996215599996503, 20.606208000011975]
    lltime3 = [0.0010006427764892578, 0.0010495185852050781, 0.0, 0.0005087852478027344
               , 0.0005593299865722656, 0.0010004043579101562, 0.0005121231079101562, 0.0
               , 0.0, 0.0005233287811279297, 0.0013263225555419922, 0.0005106925964355469]
    llmemory1 = [0.328125, 0.3203125, 0.3203125, 0.31640625
                 , 0.3203125, 0.31640625, 0.31640625, 0.3203125
                 , 0.3203125, 0.3203125, 0.31640625, 0.3203125]
    llmemory2 = [115.78125,115.44921875, 115.46484375, 115.328125
                 , 115.44921875, 115.22265625, 115.3671875, 115.421875
                 , 115.4765625, 115.3984375, 115.38671875, 115.453125]
    llmemory3 = [0.0625, 0.109375, 0.0625, 0.1015625
                 , 0.140625, 0.11328125, 0.21875, 0.109375
                 , 0.07421875, 0.1015625, 0.19921875, 0.1953125]

    # 指定文件夹路径
    folder_path = r'D:\pythonstarter\dshwwithpytorch\time&memory'
    # 将列表保存到文件
    file_path1 = os.path.join(folder_path, 'lltime1.pkl')
    with open(file_path1, 'wb') as file:
        pickle.dump(lltime1, file)
    file_path2 = os.path.join(folder_path, 'lltime2.pkl')
    with open(file_path2, 'wb') as file:
        pickle.dump(lltime2, file)
    file_path3 = os.path.join(folder_path, 'lltime3.pkl')
    with open(file_path3, 'wb') as file:
        pickle.dump(lltime3, file)
    file_path4 = os.path.join(folder_path, 'llmemory1.pkl')
    with open(file_path4, 'wb') as file:
        pickle.dump(llmemory1, file)
    file_path5 = os.path.join(folder_path, 'llmemory2.pkl')
    with open(file_path5, 'wb') as file:
        pickle.dump(llmemory2, file)
    file_path6 = os.path.join(folder_path, 'llmemory3.pkl')
    with open(file_path6, 'wb') as file:
        pickle.dump(llmemory3, file)

