
import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MyData(Dataset):
    def __init__(self, csv_path, root_path, transform=None, target_transform=None):
        """
        tex_path : txt文本路径，该文本包含了图像的路径信息，以及标签信息
        transform：数据处理，对图像进行随机剪裁，以及转换成tensor
        """
        imgs = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                img = os.path.splitext(row[0])[0]+'.jpg'
                label = row[1:]
                label = [int(x) for x in label]
                imgs.append((os.path.join(root_path, img), label))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            label = torch.Tensor(label)
        return img, label

    def __len__(self):
        return len(self.imgs)
