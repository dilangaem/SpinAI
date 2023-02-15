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
