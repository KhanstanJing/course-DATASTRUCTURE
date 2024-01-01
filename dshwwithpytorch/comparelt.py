from LinearTable import *
from datawithlt import get_training_data
from datawithlt import get_photo_paths
import time
import os
import psutil
import numpy as np
import pickle
if __name__ == '__main__':
    n = 12
    lttime1 = []
    lttime2 = []
    lttime3 = []
    ltmemory1 = []
    ltmemory2 = []
    ltmemory3 = []
    for i in range(n):
        train = LinearTable()
        train_data_label = LinearTable()
        # 获取照片地址列表
        NORMAL_train_photo_path = get_photo_paths(r'D:\dshm\chest_xray\train\NORMAL')
        PNEUMONIA_train_photo_path = get_photo_paths(r'D:\dshm\chest_xray\train\PNEUMONIA')
        train_photo_path = NORMAL_train_photo_path + PNEUMONIA_train_photo_path

        # 获取当前进程的内存占用情况
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB

        # 将图片信息加载到数据结构中
        start_time = time.perf_counter()
        formatted_start_time = "{:.6f}".format(start_time)
        for path in train_photo_path:
            train.add_data(path)
        end_time = time.perf_counter()
        formatted_end_time = "{:.6f}".format(end_time)
        elapsed_time = end_time - start_time
        print(f"将图片信息加载到数据结构中的加载时间: {elapsed_time} 秒")

        # 循环结束后再次获取内存占用情况
        end_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        # 计算内存占用增加量
        memory_increase = end_memory - start_memory
        print(f"添加数据后内存占用增加：{memory_increase} MB")
        lttime1.append(elapsed_time)
        ltmemory1.append((memory_increase))

        # 获取当前进程的内存占用情况
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB

        start_time = time.time()
        formatted_start_time2 = "{:.6f}".format(start_time)
        train_data = get_training_data(train, train_data_label)
        end_time = time.time()
        formatted_end_time2 = "{:.6f}".format(end_time)
        elapsed_time = end_time - start_time
        print(f"将训练数据和训练标签匹配的加载时间: {elapsed_time} 秒")

        # 循环结束后再次获取内存占用情况
        end_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        # 计算内存占用增加量
        memory_increase = end_memory - start_memory
        print(f"添加数据后内存占用增加：{memory_increase} MB")
        lttime2.append(elapsed_time)
        ltmemory2.append((memory_increase))

        # 获取当前进程的内存占用情况
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        start_time = time.perf_counter()

        train_label = np.array(train_data_label.datalist, dtype=np.int32)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(start_time, end_time)
        print(f"列表转为array数组的时间: {elapsed_time} 秒")

        # 循环结束后再次获取内存占用情况
        end_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
        # 计算内存占用增加量
        memory_increase = end_memory - start_memory
        print(f"添加数据后内存占用增加：{memory_increase} MB")
        lttime3.append(elapsed_time)
        ltmemory3.append((memory_increase))

    lttime1 = [0.0005118000117363408, 0.0005742000066675246, 0.0005168999923625961, 0.0005616999987978488
               , 0.00049560000479687, 0.0004760000010719523, 0.0005355000030249357, 0.00046109998947940767
               , 0.0005249999958323315, 0.0005347000114852563, 0.0005021999968448654, 0.0005476000078488141]
    lttime2 = [25.875924825668335, 23.436539888381958, 23.29699945449829, 23.38879084587097
               , 23.295969009399414, 23.42807674407959, 22.52750515937805, 22.91178011894226
               , 23.119014501571655, 23.39899182319641, 23.381412267684937, 23.683305025100708]
    lttime3 = [0.00013269999180920422, 0.00011779999476857483, 0.00013240000407677144, 0.0001235999952768907
               , 0.00011770000855904073, 0.0001247000036528334, 0.00012909999350085855, 0.0001171000039903447
               , 0.00011269999959040433, 0.00012660000356845558, 0.00011979999544564635, 0.00012789999891538173]
    ltmemory1 = [0.0546875, 0.16015625, 0.078125, 0.0546875
                , 0.09765625, 0.05859375, 0.05859375, 0.1640625
                , 0.0703125, 0.06640625, 0.1171875, 0.1171875]
    ltmemory2 = [114.4765625, 114.6015625, 114.890625, 114.69921875
                 , 114.98828125, 115.01171875, 114.68359375, 114.796875
                 , 114.65625, 115.09375, 114.85546875, 114.71484375]
    ltmemory3 = [0.01953125, 0.015625, 0.015625, 0.015625
                 , 0.01953125, 0.01953125, 0.015625, 0.015625
                 , 0.015625, 0.015625, 0.015625, 0.01953125]
    # 指定文件夹路径
    folder_path = r'D:\pythonstarter\dshwwithpytorch\time&memory'
    # 将列表保存到文件
    file_path1 = os.path.join(folder_path, 'lttime1.pkl')
    with open(file_path1, 'wb') as file:
        pickle.dump(lttime1, file)
    file_path2 = os.path.join(folder_path, 'lttime2.pkl')
    with open(file_path2, 'wb') as file:
        pickle.dump(lttime2, file)
    file_path3 = os.path.join(folder_path, 'lttime3.pkl')
    with open(file_path3, 'wb') as file:
        pickle.dump(lttime3, file)
    file_path4 = os.path.join(folder_path, 'ltmemory1.pkl')
    with open(file_path4, 'wb') as file:
        pickle.dump(ltmemory1, file)
    file_path5 = os.path.join(folder_path, 'ltmemory2.pkl')
    with open(file_path5, 'wb') as file:
        pickle.dump(ltmemory2, file)
    file_path6 = os.path.join(folder_path, 'ltmemory3.pkl')
    with open(file_path6, 'wb') as file:
        pickle.dump(ltmemory3, file)

