import numpy as np

class LinearTable:
    def __init__(self):
        self.datalist = []

    def __iter__(self):
        return iter(self.datalist)

    def __len__(self):
        return len(self.datalist)

    def index(self, data):
        # 返回第一个匹配项的下标，如果不存在则返回 -1
        try:
            return self.datalist.index(data)
        except ValueError:
            return -1

    def add_data(self, data):
        self.datalist.append(data)

    def find_data(self, data):
        result = []
        for index in range(len(self.datalist)):
            if self.datalist[index] == data:
                result.append(index)
        return result

    def insert_data(self, location, data):
        self.datalist.insert(location, data)

    def delete_data(self, data):
        self.datalist.remove(data)

if __name__ == '__main__':
    ls = LinearTable()
    # print(ls)
    # print(ls)则返回'<__main__.LinearTable object at 0x00000217BAB38430>'地址
    print(ls.datalist)
    for i in range(100):
        ls.add_data(np.zeros(i))
    print(ls.datalist)