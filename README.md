# SpinAI


This repository contains a magnetic/nonagnetic classifier developed using deep neural network algorithm. We use this classifier along with CubicGAN model for screening new spintronic materials. Our work is reported in detail in the following publication.

## Prerequisites
- python 3.7
- pandas 1.3.0
- numpy 1.21.0
- sklearn 1.0.0
- scipy 1.5.1

## Training the Model

The data files must be in the DATA folder. In the DATA folder, we provide a file with ternary materials' data (train_data.csv), which was used in the paper.  <br />  <br />


To train the model for the ternary materials with cubic crystal system, run the following command. <br />
```bash
python train.py --file_name train_data.csv --test_size 0.1
```

## Predicting New Metals or Non-metals 
Mention all the chemical formulas in a .csv data file with the above format. In order to keep the file strucutre, you can state 1 or 0 in the Target column. For clarity, a sample file named predict_data.csv is in DATA folder. We also provided a trained model in the TRAINED forlder for ternary cubic materials. <br /> <br />

As an example, to predict metals/non-metals, run the following command. <br /> 
```bash
python predict.py --file_name predict_data.csv --model_name model-2023_02_15_00_28_17.sav
```

