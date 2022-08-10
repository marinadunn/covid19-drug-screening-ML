# Developing ML models using molecular descriptors and 3D atomic representation for rapid screening of drug-like compounds targeting SARS-CoV-2.

## Getting started

To clone this repository:
```
$ git clone https://github.com/marinadunn/covid19-drug-screening-ML.git
$ cd covid19-drug-screening-ML
```

It's recommended to install dependencies from the file `requirements.txt` in a virtual environment:
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

## Project Steps

- Exploratory Data Analysis

- Task 1: Use SMILES molecular descriptors of ligand data to predict binding affinity (Binary Classification)

- Task 2: Use 3D atomic representation of ligand data to predict binding affinity (Binary Classification)

## Usage

All development was done with Python 3.10. Some development was done using Jupyter notebooks and can be accessed and run using LC Jupyterhub, while others were done using Python scripts. 

Task 1 classifiers exploratory data analysis notebook can be found `Task 1/classifiers/EDA.ipynb`, and model development in `Task 1/classifiers/classifiers.ipynb`, while ensemble development can be found in `Task 1/ensemble/`.

Task 2 development for the 3D CNN can be found in `Task 2/3D_CNN/3D_CNN.ipynb` and `Task 2/3D_CNN/3DCNN.py`, and development for the Graph CNN can be found in `Task 2/GNN`.

## Data

Task 1 original data is located in `Task 1/data/mpro_exp_data2_rdkit_feat.csv`. Cleaned datasets for the classifiers approach can be found in the folder `Task 1/classifiers/data` (with the full cleaned dataset and no outliers removed in `cleaned_full_dataset.csv` and cleaned dataset with removed rows outside 4+ standard deviations away from mean located in `cleaned_data.csv`). Task 1 ensemble data can be found in `Task 1/ensemble/data`.

Task 2 data can be found in `Task 2/data`, with original train, test, and validation datasets found in `postera_protease2_pos_neg_train.hdf5`, `postera_protease2_pos_neg_test.hdf5`, and `postera_protease2_pos_neg_val.hdf5` respectively.

## Authors and Acknowledgments

All code was developed as part of the Lawrence Livermore National Laboratory Data Science Summer Institute 2022 Challenge Problem, testing various ML approaches for screening molecular inhibitors for SARS-CoV-2 protein targets. More information can be found [here.](https://myconfluence.llnl.gov/display/DSSI/2022+Challenge+Problem)

Authors: Marina M. Dunn, Lance Fletcher, Robert Stephany, Moises Santiago Cardenas

Please send us an email for any support questions or inquiries:

* Marina M. Dunn (mdunn014 at ucr dot edu)

* Lance Fletcher (lance dot g dot fletcher at gmail dot com)

* Moises Santiago Cardenas (santiagomoises at berkeley dot edu)

* Robert Stephany (rrs254 at cornell dot edu)
