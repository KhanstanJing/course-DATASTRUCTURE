from LinkedList import *
import numpy as np
import time
import os
import cv2
import psutil
def get_photo_paths(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 '{folder_path}' 不存在。")
        return []

    # 获取文件夹中的所有文件
    file_list = os.listdir(folder_path)

    # 得到照片地址
    photo_paths = [os.path.join(folder_path, file) for file in file_list]

    return photo_paths

# 将数据加载到数据结构中
def get_training_data(LinkedList1, LinkedList2):
    labels = ['bacteria', 'virus', 'NORMAL']
    img_size = 150
    data = []
    for node in LinkedList1:
        for label in labels:
            if label in node:
                LinkedList2.add_image(labels.index(label))
                try:
                    img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))
                    data.append(resized_arr)
                except Exception as e:
                    print(e)
    return np.array(data)

if __name__ == '__main__':
    # 初始化线性表
    train = LinkedList()
    test = LinkedList()
    val = LinkedList()

    # 获取照片地址列表
    NORMAL_train_photo_path = get_photo_paths(r'D:\dshm\chest_xray\train\NORMAL')
    PNEUMONIA_train_photo_path = get_photo_paths(r'D:\dshm\chest_xray\train\PNEUMONIA')
    train_photo_path = NORMAL_train_photo_path + PNEUMONIA_train_photo_path

    NORMAL_test_photo_path = get_photo_paths(r'D:\dshm\chest_xray\test\NORMAL')
    PNEUMONIA_test_photo_path = get_photo_paths(r'D:\dshm\chest_xray\test\PNEUMONIA')
    test_photo_path = NORMAL_test_photo_path + PNEUMONIA_test_photo_path

    NORMAL_val_photo_path = get_photo_paths(r'D:\dshm\chest_xray\val\NORMAL')
    PNEUMONIA_val_photo_path = get_photo_paths(r'D:\dshm\chest_xray\val\PNEUMONIA')
    val_photo_path = NORMAL_val_photo_path + PNEUMONIA_val_photo_path


    # 获取当前进程的内存占用情况
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB

    # 将图片信息加载到数据结构中
    start_time = time.perf_counter()
    for path in train_photo_path:
        train.add_image(path)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"将图片信息加载到数据结构中加载时间: {elapsed_time} 秒")

    # 循环结束后再次获取内存占用情况
    end_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
    # 计算内存占用增加量
    memory_increase = end_memory - start_memory
    print(f"添加数据后内存占用增加：{memory_increase} MB")

    for path in test_photo_path:
        test.add_image(path)

    for path in val_photo_path:
        val.add_image(path)

    # 建立用来存放标签的线性表
    train_data_label = LinkedList()
    test_data_label = LinkedList()
    val_data_label = LinkedList()

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

    test_data = get_training_data(test, test_data_label)
    val_data = get_training_data(val, val_data_label)

    # 归一化
    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255
    val_data = np.array(val_data, dtype=np.float32) / 255

    # 预处理
    img_size = 150
    train_data = train_data.reshape(-1, 1, img_size, img_size)
    test_data = test_data.reshape(-1, 1, img_size, img_size)
    val_data = val_data.reshape(-1, 1, img_size, img_size)

    # 获取当前进程的内存占用情况
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB

    # 在MindSpore中，Conv2d 层的输入数据类型要求与权重（weight）的数据类型相同。输入数据的类型为 Float64，而期望的是 Float32。使用 astype 来转换数据类型。
    start_time = time.time()
    formatted_start_time3 = "{:.6f}".format(start_time)
    train_label = np.array(train_data_label.to_list(), dtype=np.int32)
    end_time = time.time()
    formatted_end_time3 = "{:.6f}".format(end_time)
    elapsed_time = end_time - start_time
    print(f"使用链表中的内置函数将链表转换成列表后转为array数组的时间: {elapsed_time} 秒")

    # 循环结束后再次获取内存占用情况
    end_memory = process.memory_info().rss / 1024 / 1024  # 转换为 MB
    # 计算内存占用增加量
    memory_increase = end_memory - start_memory
    print(f"添加数据后内存占用增加：{memory_increase} MB")

    test_label = np.array(test_data_label.to_list(), dtype=np.int32)
    val_label = np.array(val_data_label.to_list(), dtype=np.int32)

    # 修改标签的形状
    train_label = np.squeeze(train_label)
    test_label = np.squeeze(test_label)
    val_label = np.squeeze(val_label)

    # 保存文件
    save_dir = r'D:\pythonstarter\dshwnotbroken\mindspore\data'

    # 保存数据和标签到指定路径
    np.save(os.path.join(save_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(save_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(save_dir, 'val_data.npy'), val_data)
    np.save(os.path.join(save_dir, 'train_label.npy'), train_label)
    np.save(os.path.join(save_dir, 'test_label.npy'), test_label)
    np.save(os.path.join(save_dir, 'val_label.npy'), val_label)
    print('保存完毕')
