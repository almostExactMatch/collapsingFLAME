# Collapsing-Fast-Large-Almost-Matching-Exactly: A Matching Method for Causal Inference [[Paper]](https://arxiv.org/list/stat.ML/recent). 
Awa Dieng, Yameng Liu, Sudeepa Roy, Cynthia Rudin, Alexander Volsfovsky 

Duke University

## Abstract
We aim to create the highest possible quality of treatment-control matches for cate- gorical data in the potential outcomes framework. Matching methods are heavily used in the social sciences due to their interpretability, but most matching methods in the past do not pass basic sanity checks in that they fail when irrelevant variables are introduced. Also, past methods tend to be either computationally slow or produce poor matches. The method proposed in this work aims to match units on a weighted Hamming distance, tak- ing into account the relative importance of the covariates; the algorithm aims to match units on as many relevant variables as possible. To do this, the algorithm creates a hi- erarchy of covariate combinations on which to match (similar to downward closure), in the process solving an optimization problem for each unit in order to construct the opti- mal matches. The algorithm uses a single dynamic program to solve all of optimization problems simultaneously. Notable advantages of our method over existing matching pro- cedures are its high-quality matches, versatility in handling different data distributions that may have irrelevant variables, and ability to handle missing data by matching on as many available covariates as possible.

## Run
* **Step1**: clone this repository

      git clone https://github.com/almostExactMatch/collapsingFLAME.git
      
* **Step2**: Make sure all required packages are up-to-date (see requirements.txt);
      
      pip install -r requirements.txt

* **Step3**: To reproduce <name-of-experiment>: 
 
      open the experiments/<name-of-experiement> notebook and run all cells;
      
The "experiments" folder contains jupyter notebooks to reproduce the results from the paper.
Please follow the instructions in the notebooks to run the FLAME algorithms and obtain the dataframe for the matched groups and estimated CATEs.

**All scripts are in Python**.

Contact adieng@cs.duke.edu for inquiries.
