# Unsupervised-Attention-Based-Multi-Source-Domain-Adaptation-Framework
Unsupervised Attention-Based Multi-Source Domain Adaptation Framework for Drift Compensation in Electronic Nose Systems
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



    
    
    

  
    
   

