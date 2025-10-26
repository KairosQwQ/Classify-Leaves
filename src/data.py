import pandas as pd 
from torch.utils.data import Dataset
import os 
from PIL import Image


class LeaveDataset(Dataset):
    """
        继承Dataset类，该类主要记录数据集的地址以及获取方法，获取后的数据预处理。
        
        当dataLoader类变量调用Dataset时会调用__getitem__()这个函数.读取存在磁盘上的数据到内存
    """
    def __init__(self, dir, csv_file, transform=None):
        self.transform = transform
        self.data = pd.read_csv(csv_file) 
        self.dir = dir
        self.has_label = 'label' in self.data.columns

        # 如果有标签列，建立类别映射表
        if self.has_label:
            #.unique() 会返回所有不同的数据，按出现数据返回
            classes = sorted(self.data['label'].unique()) 
            #字典映射，softmax需要将标签名称转换为类别数字label。
            self.label2idx = {cls_name: i for i, cls_name in enumerate(classes)}#enumerate返回数据的同时返回其标号

        #必须写的 返回数据量
    def __len__(self):
        
        return len(self.data)

    def __getitem__(self, index):
        
        # 获取数据的方法写在这里，dataloader会调用这个方法
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.dir, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        #通过字典将label转换为数字label
        if self.has_label:
            label_name = self.data.iloc[index]['label']
            label = self.label2idx[label_name]
            # 返回数据 （已打开转换后的图片，label）
            return image, label
        else:
            return image, image_name
        
        

