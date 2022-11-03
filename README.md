# COVID-19 Drug Screening Using ML

## About
This project focuses on developing Machine Learning models using both the SMILES molecular descriptors and 3D atomic representations of drug-like compounds targeting SARS-CoV-2 for rapid screening of binding affinity. Work was developed as part of the Lawrence Livermore National Laboratory Data Science Summer Institute 2022 Challenge Problem led by Dr. Hyojin Kim.

## Getting started

To clone this repository:
```
$ git clone https://github.com/marinadunn/covid19-drug-screening-ML.git COVID19-ML
$ cd COVID19-ML
```

It's recommended to first create a virtual Python environment, then install dependencies from the file `requirements.txt`:
```
$ python3 -m venv my_env
$ source my_env/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Project Steps

- Exploratory Data Analysis

- Task 1: Use SMILES molecular descriptors of ligand data to predict binding affinity (Binary Classification)

- Task 2: Use 3D atomic representation of ligand data to predict binding affinity (Binary Classification)

## Usage

All development was done with Python 3.10. Some development was done using Jupyter notebooks, while others were done using Python scripts. Model training for Task 2 was performed on Livermore Compute HPCs.

Code for development & testing is located in the respective folders for team members.

## Data

Task 1 original data is located in the file `mpro_exp_data2_rdkit_feat.csv`, with cleaned datasets located in their respective data folders for each individual.

Task 2 original train, test, and validation datasets can be found in the files `postera_protease2_pos_neg_train.hdf5`, `postera_protease2_pos_neg_test.hdf5`, and `postera_protease2_pos_neg_val.hdf5` respectively, with cleaned datasets once again located in their respective data folders for each individual.

## Initial Results

Task 1:
    - Marina: 11 classifier models tested initially using scikit-learn. Tested for both dataset where outliers greater than 4-sigma were removed, as well as dataset where they were not removed. Best performing models: 
        - KNN: 89% Accuracy, 90% Precision (wgtd.), 89% Recall (wgtd.), 89% F1 (wgtd.)
        - Gaussian Process: 88% Accuracy, 89% Precision (wgtd.), 88% Recall (wgtd.), 88% F1 (wgtd.)
        - Neural Network: 89% Accuracy, 89% Precision (wgtd.), 89% Recall (wgtd.), 89% F1 (wgtd.)
    - Robert:
        - Logistic Regression Classifier: 60% Accuracy (poor generalization)
        - Feature Subset with Multi-layer Perceptron: >75% for some models (good generaization), 60%> for others (poor)
        - Ensemble Approach: 80.5% Accuracy, 74% Precision (wgtd.), 74% Recall (wgtd.), 81.2% F1 (wgtd.)
        
    Ultimately, could not explore further given time constraints. Future considerations: look at additional metrics (i.e. ROC AUC), try model fusion, additional feature analysis

Task 2:
    - Robert & Lance:
        - Graph Neural Network: ~80% Accuracy
        - Graph Neural Network Ensemble: 81.25% Accuracy
    - Marina: 3D Convolutional Neural Network
    
    Ultimately, could not explore further given time constraints. Future considerations: test more 3D CNN models, explore using uncertainty quantification, do comprehensive performance & behavior analysis, weight "binding" examples more than "non-binding" in GNN, test other kinds of pooling with GNN, try additional ensemble methods, explore additional methods of translating ligand data to graph in GNN, implement fusion model with GNN & 3D CNN

## Authors and Acknowledgments

All code was developed as part of the Lawrence Livermore National Laboratory Data Science Summer Institute 2022 Challenge Problem, testing various ML approaches for screening molecular inhibitors for SARS-CoV-2 protein targets. More information can be found [here.](https://data-science.llnl.gov/dssi/class/2022)

Authors: Marina M. Dunn, Lance Fletcher, Robert Stephany, Moises Santiago Cardenas

Please send us an email for any support questions or inquiries:

* Marina M. Dunn (<mdunn014@ucr.edu>)

* Lance Fletcher (<lance.g.fletcher@gmail.com>)

* Moises Santiago Cardenas (<santiagomoises@berkeley.edu>)

* Robert Stephany (<rrs254@cornell.edu>)
