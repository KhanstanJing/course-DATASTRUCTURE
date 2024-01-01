import os

class ImageNode:
    def __init__(self, image_path=None):
        self.image_path = image_path
        self.next_node = None
        self.prev_node = None

class ImageLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def __getitem__(self, index):
        current = self.head
        for _ in range(index):
            if current is None:
                raise IndexError("Index out of range")
            current = current.next_node
        return current.image_path

    def add_image(self, image_path):
        new_node = ImageNode(image_path)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev_node = self.tail
            self.tail.next_node = new_node
            self.tail = new_node

    def delete_image(self, image_path):
        current = self.head
        while current:
            if current.image_path == image_path:
                if current.prev_node:
                    current.prev_node.next_node = current.next_node
                else:
                    self.head = current.next_node

                if current.next_node:
                    current.next_node.prev_node = current.prev_node
                else:
                    self.tail = current.prev_node

                return True  # Image deleted
            current = current.next_node

        return False  # Image not found

    def search(self, image_path):
        current = self.head
        while current:
            if current.image_path == image_path:
                return current
            current = current.next_node
        return None  # Image not found

    def is_empty(self):
        return self.head is None

    def length(self):
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next_node
        return count

    def display_images(self):
        current = self.head
        while current:
            print(current.image_path)
            current = current.next_node

if __name__ == '__main__':
    # 遍历文件夹中的图像文件并添加到链表中
    image_folder = r"D:\dshm\chest_xray\train\NORMAL"
    image_list = ImageLinkedList()

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            image_list.add_image(image_path)

    # 显示链表中的图像地址
    image_list.display_images()
    print(image_list[0])
