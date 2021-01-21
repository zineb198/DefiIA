# DefiIA

This repository is the code used during the Kaggle Challenge : [IA and Ethics](https://www.kaggle.com/c/defi-ia-insa-toulouse/overview). The aim of this competition is to assign the correct job category to a job description.
The data is therefore representative of what can be found on the English speaking part of the Internet, and thus contains a certain amount of bias. One of the goals of this competition is to design a solution that is both accurate as well as fair.

## Tutorial

We have two ways to solve this problem :
- Classical NLP methods.
- Transformers : Bert.
For the first one, you can either train locally or on the google cloud computing platform. For the second one, it is recommended to use google colaboratory. 

### 1. Classical NLP methods

##### Installing environement dependencies
> pip install -r requirements.txt

##### Load data
> Download from the [link](https://www.kaggle.com/c/defi-ia-insa-toulouse/data) the data.


There are multiple ways to run the code :

#### a. Training locally 
First change the path in the three scripts (`cleaning.py`, `embedding.py`, `classification.py`). <br />
From the command line, go into the script folder and run the following lines : <br />
> python cleaning.py <br />
> python embedding.py <br />
> python classification.py <br />
From the code you can modify the parameters to change the embedding method for example. <br />


#### b. Training on google cloud platform
In the 'instance.py' file, modify the parameters for your virtual machine. <br />
Change the path in the four scripts (`project.py`, `cleaning.py`, `embedding.py`, `classification.py`). <br />
From the command line run the following line :
> python main.py

In this architecture, you will find the models in the 'model' folder and the submission files in the 'result' folder. <br />


### 2. Transformers : Bert
Upload the 'bert.ipynb' notebook to colab. <br />
Choose the GPU. <br />
Install the packages and import the data. <br /> 
Run the cells and download the submission file. <br />
