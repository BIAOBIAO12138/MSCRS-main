# MSCRS: Multi-modal Semantic Graph Prompt Learning Framework for Conversational Recommender Systems
## Framework
![image](https://github.com/user-attachments/assets/83d58ca2-833c-4859-93d6-abc6d07f193b)
🔥 2025.4: Our paper is accepted by SIGIR'25! Thank all Collaborators! 🎉🎉

## Requirements
python == 3.8.13

pytorch == 1.8.1

cudatoolkit == 11.1.1

transformers == 4.15.0

pyg == 2.0.1

accelerate == 0.8.0

## Datasets
🌹🌹 We have supplemented the REDIAL and INSPIRED datasets with additional multimodal data. If you use the multimodal conversational recommendation dataset, please cite our paper~ ❤️
### Processed dataset
The processed dataset can be found here (https://drive.google.com/drive/folders/1M2cIDFD_a-o1HCgGmN8pEeGgT21xo5oe?usp=drive_link)
### Image
We have released the visual information obtained through web crawling. (The following image shows the posters and stills acquired by web crawling using item information from the INSPIRED dataset.)
![image](https://github.com/user-attachments/assets/e6ce02cc-23b1-4455-b376-202361af73e1)

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

## Conversation Task
python conv/src/train_conv.py


## Acknowledgement
We thank [VRICR](https://github.com/zxd-octopus/VRICR/tree/master), [UNICRS](https://github.com/wxl1999/UniCRS/tree/main) and [DCRS](https://github.com/huyquangdao/DCRS?tab=readme-ov-file) for providing the useful source code for the data preprocessing and prompt learning steps.












