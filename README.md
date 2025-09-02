This solution is based on BoT (Bag of Tricks), but with several modifications. You can clone this repository to reproduce our approach. The main file of this GitHub project is PIPELINE.py, which was used to train the different models we tested. It provides a simple interface to select the dataset (including the different variants of the original dataset, further details are provided in the published paper) and the models (initialized with publicly available pretrained weights).

The pipeline covers all the steps of the training and evaluation process, except for the creation of the dataset variants. To make our work fully reproducible, we have also uploaded in this repository the code required to implement the dataset modifications we introduced.
 
In addition, we provide [https://drive.google.com/drive/folders/190rlARYUD5_4RPh_33xMUzoil7Oq0A8k?usp=drive_link](https://drive.google.com/drive/folders/190rlARYUD5_4RPh_33xMUzoil7Oq0A8k?usp=drive_link):

- A link to the trained weights of the best-performing model.

- A link to the dataset variant that yielded the best results.

- The final results obtained with this configuration.

This way, the entire workflow, from dataset preparation to model training and evaluation—can be reproduced and extended by other researchers.
