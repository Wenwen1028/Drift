import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd

src_names = ["batch1", "batch2", "batch3", "batch4", "batch5", "batch6", "batch7", "batch8", "batch9", "batch10", "batch11", "batch12", "batch13", "batch14"]

def loadDataSet(fileName):
    dataSetList = []
    fr = open(fileName)
    for row in fr.readlines():
        cur_line = row.strip().split(' ')
        proce_line = list(map(float, cur_line))
        dataSetList.append(proce_line)
    dataSetList = np.array(dataSetList)
    return dataSetList

def loadLabels(fileName):
    labels = []
    fr = open(fileName)
    for row in fr.readlines():
        label = int(row.strip())
        labels.append(label)
    labels = np.array(labels)
    return labels

def _PCA():
    for src_name in src_names:
        data_path = f'../Dataset/Dataset_ext/{src_name}_ext.txt'
        label_path = f'../Dataset/Dataset_lab/{src_name}_lab.txt'
        data = loadDataSet(data_path)
        labels = loadLabels(label_path)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)
        plt.figure()
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', marker='o')
        plt.title(f'{src_name}_PCA')
        plt.legend(handles=scatter.legend_elements()[0], labels=set(labels))
        plt.savefig(f'../pca_plots/{src_name}_pca.png')
        plt.show()

        df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
        df['label'] = labels
        df.to_excel(f'../pca_plots/{src_name}_pca.xlsx', index=False)
        print(f'{src_name} PCA data and labels saved to Excel.')

if __name__ == '__main__':
    _PCA()



