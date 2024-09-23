# Unsupervised-Attention-Based-Multi-Source-Domain-Adaptation-Framework
![示例图片](image/framework.jpg)
*Fig. 1 Overview of the proposed unsupervised AMDS-PFFA model framework for drift compensation.
# Introduction
The continuous, long-term monitoring of hazardous, noxious, explosive, and flammable gases in industrial environments using electronic nose (E-nose) systems is hindered by a significant challenge: the decline in gas identification accuracy caused by time-varying sensor drift. To tackle this issue, we propose a novel unsupervised attention-based multi-source domain shared-private feature fusion adaptation (AMDS-PFFA) framework for drift-compensated gas identification in E-nose systems. By leveraging labeled data from multiple source domains collected at the initial stage, the AMDS-PFFA model accurately identifies gases from unlabeled gas sensor array drift signals in the target domain. We validated the model's effectiveness through extensive experiments using the University of California, Irvine (UCI) standard drift gas dataset, collected over 36 months, and data from our self-developed E-nose system, collected over 30 months. The AMDS-PFFA model consistently outperformed recent drift compensation methods, achieving the highest average gas recognition accuracy with strong convergence—83.20% on the UCI dataset and 93.96% on our E-nose system across all target domain batches. These results highlight the superior capability of the AMDS-PFFA model in gas identification with drift compensation, significantly outperforming existing approaches.
# Getting Started
## Installation
### Configure virtual (Anaconda) environment
    
    conda create -n env_name python=3.9
    conda activate env_name
### Install Packages   
    conda install torch==2.4.0
    conda install torchvision==0.19.0
    conda install torchaudio==2.4.0
    conda install scikit-learn==1.5.1
    conda install pandas==2.2.2
    conda install matplotlib==3.9.2
    conda install matplotlib-inline==0.1.7
    conda install numpy==1.26.4
    conda install openpyxl==3.1.5
    or You can directly use the uploaded requirement.txt file by entering the following command：
    pip install -r requirements.txt
### Program Overview
- **folder{PCA_batch1_batch14}**：***1) pca_2d_label.py***: code for reducing batches 1 to 14 to a 2D labeled display using PCA. ***2) folder-pca_plots***: the folder designated for storing data after PCA dimensionality reduction.
- **folder{Preprocessing_UCIdrift_dataset}**: ***1) maxmin_normalization.py***: maximum and minimum normalization for UCI Batches 1 to 10. ***2) maxmin_normalization.py***: Zero mean normalization for UCI batches 1 to 10. ***3) batch1 to batch 10.py***: UCI Dataset can be download from https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset. In this work, we employ zero mean normalization method ()  
- **folder{data_01}**：Record the accuracy and loss curve values throughout the experiment for convergence analysis.
- **folder{data_loader}**：***data_loader.py***: data loading code.
- **folder{ema}**：***1) ema.py***: This demonstration extracts dynamic features from drift data of the TGS2610 sensor in a gas mixture of 180 ppm CO and 180 ppm H₂. For detailed information, see Section V, subsection on experimental validation of the model using drift data from the self-developed E-nose system, and Fig. 8. ***2) 180ppmH2_CO2.text***: Dynamic response signal of a mixed gas containing 180 ppm H₂ and 180 ppm CO, recorded using a self-developed electronic nose.  ***3) ema_data***: Save the EMA signal at $alpha$=0.1, $\alpha$=0.01, $\alpha$=0.001.
- **folder{hsh}**： ***1) Dataset_ext***: batch1_ext ~ batch10_ext: UCI drift feature data;  batch1_ext ~ batch10_ext: UCI drift feature data; 
  








    
    
    

  
    
   

