import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from sklearn.manifold import TSNE
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

def _TSNE():
    for src_name in src_names:
        data_path = f'../Dataset/Dataset_ext/{src_name}_ext.txt'
        label_path = f'../Dataset/Dataset_lab/{src_name}_lab.txt'

        data = loadDataSet(data_path)
        labels = loadLabels(label_path)

        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(data)

        plt.figure()
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', marker='o')
        plt.title(f'{src_name}_tSNE')
        plt.legend(handles=scatter.legend_elements()[0], labels=set(labels))
        plt.savefig(f'./tsne_plots/{src_name}_tsne.png')
        plt.show()

        # Save to Excel
        df = pd.DataFrame(data=tsne_result, columns=['tSNE1', 'tSNE2'])
        df['label'] = labels
        df.to_excel(f'./tsne_plots/{src_name}_tsne.xlsx', index=False)
        print(f'{src_name} tSNE data and labels saved to Excel.')

if __name__ == '__main__':
    _TSNE()




