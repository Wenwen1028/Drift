import  torch
import numpy as np

class Mydataset(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)

def loadData(fileName):
    dataSetList = []
    fr = open(fileName)
    for row in fr.readlines():
        cur_line = row.strip().split(' ') # 该字符串将从原始字符串的开头和结尾删除给定的字符
        proce_line = np.array(list(map(float, cur_line)))
        dataSetList.append(proce_line)
    dataSetList = np.array(dataSetList)
    return dataSetList

def loadLab(fileName):
    dataSetList = []
    fr = open(fileName)
    for row in fr.readlines():
        cur_line = row.strip().split()
        proce_line = np.array(list(map(int, cur_line)))
        #默认删除单位为1的维度，也就是降维的作用
        dataSetList.append(np.squeeze(proce_line))
    dataSetList = np.array(dataSetList)
    return dataSetList

def load_dataset(root_path, dir, batch_size, kwargs):
    DataPath = root_path + 'Dataset_ext/' + dir + '_ext.txt'
    LabPath = root_path + 'Dataset_lab/' + dir + '_lab.txt'
    x = loadData(DataPath)
    y = loadLab(LabPath)
    dataset = Mydataset(x, y)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                               **kwargs)                  # 数据加载器
    return dataset_loader


