## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This project conducts an experiment on feature selection for SVM classification algorithms 
with a given dataset.
	
## Technologies
Project is created with:
* Python 3.9
	
## Setup
* Download data in the folder of case study 2 and extract it to ``data`` folder
* Run ``pip install -r requirements.txt`` to install required libraries
* Run ``python main.py`` to run feature selection and train SVM model. The results of this process will be saved in 
``results/evaluation.json`` file 
* Run ``python show_results.py`` to print out average accuracy, sensitivity and specificity scores for each feature 
selection scheme