This solution is based on BoT (Bag of Tricks), but with several modifications. You can clone this repository to reproduce our approach. The main file of this GitHub project is PIPELINE.py, which was used to train the different models we tested. It provides a simple interface to select the dataset (including the different variants of the original dataset, further details are provided in the published paper) and the models (initialized with publicly available pretrained weights).

The pipeline covers all the steps of the training and evaluation process, except for the creation of the dataset variants. To make our work fully reproducible, we have also uploaded in this repository the code required to implement the dataset modifications we introduced.
 
In addition, we provide [https://zenodo.org/records/17039479?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM3MzZkOTFhLWFkNTUtNDgyOC1hN2EwLWRjNTg2OTgzNTAxZCIsImRhdGEiOnt9LCJyYW5kb20iOiI1ZjY5ZDRmZjkyMjQzYmY5MTI2MGFkYTk0ZWE5NDEzOSJ9.e4S2EI6SAPckc_zqAv8lIlIoTPjGYQaHA9goHwwdFFMxzCEqy2tOoM_o9Hb-SKsJQ5X1lV7xt4IkCYCP_f-hWA](https://zenodo.org/records/17039479?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQ1ZGYxNzkzLTNlNzktNDYxOC05ZThjLTA5NjdlMzJiZjI4NiIsImRhdGEiOnt9LCJyYW5kb20iOiI0YzgzMDNhMjE5NGE4MGRmMmJkOGVjZTcxMzllZGYzMiJ9.8ZxuHiQGPniaDUz0RvV8RV9cUOyYr3t-ovfSxoal-qVHCd-dPQkjKzSddMav3B-M06KpdOi3-ZC48TY_7R1OuA):

- A link to the trained weights of the best-performing model.

- A link to the dataset variant that yielded the best results.

- The final results obtained with this configuration.

This way, the entire workflow, from dataset preparation to model training and evaluation—can be reproduced and extended by other researchers.
