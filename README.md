# MSCRS-main
## Framework
![image](https://github.com/user-attachments/assets/83d58ca2-833c-4859-93d6-abc6d07f193b)
## Requirements
- python == 3.8.13
- pytorch == 1.8.1
- cudatoolkit == 11.1.1
- transformers == 4.15.0
- pyg == 2.0.1
- accelerate == 0.8.0

## Start
We run all experiments and tune hyperparameters on a GPU with 32GB memory, you can adjust per_device_train_batch_size and per_device_eval_batch_size according to your GPU.

## Datasets
The dataset can be found here (https://drive.google.com/drive/folders/1M2cIDFD_a-o1HCgGmN8pEeGgT21xo5oe?usp=drive_link)

## Recommendation Task

### Pretrain
#### ReDial
python rec/src/train_pre_redial.py
#### INSPIRED
python rec/src/train_pre_inspired.py

### Train
#### ReDial
python rec/src/train_rec_redial.py
#### INSPIRED
python rec/src/train_rec_inspired.py

## Conversation Train
python conv/src/train_conv.py
















