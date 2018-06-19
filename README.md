# Collapsing-Fast-Large-Almost-Matching-Exactly: A Matching Method for Causal Inference [[Paper]](https://arxiv.org/abs/1806.06802). 
Awa Dieng, Yameng Liu, Sudeepa Roy, Cynthia Rudin, Alexander Volsfovsky <br>
Duke University

## Abstract
We aim to create the highest possible quality of treatment-control matches for categorical data in the potential outcomes framework. Matching methods are heavily used in the social sciences due to their interpretability, but most matching methods in the past do not pass basic sanity checks in that they fail when irrelevant variables are introduced. Also, past methods tend to be either computationally slow or produce poor matches. The method proposed in this work aims to match units on a weighted Hamming distance, taking into account the relative importance of the covariates; the algorithm aims to match units on as many relevant variables as possible. To do this, the algorithm creates a hierarchy of covariate combinations on which to match (similar to downward closure), in the process solving an optimization problem for each unit in order to construct the optimal matches. The algorithm uses a single dynamic program to solve all of optimization problems simultaneously. Notable advantages of our method over existing matching procedures are its high-quality matches, versatility in handling different data distributions that may have irrelevant variables, and ability to handle missing data by matching on as many available covariates as possible.

## Run
* **Step1**: clone this repository

      git clone https://github.com/almostExactMatch/collapsingFLAME.git
      
* **Step2**: Make sure all required packages are up-to-date (see requirements.txt);

* **Step3**: To reproduce name-of-experiment: 
 
      open the experiments/<name-of-experiement> notebook and run all cells;
      
The "experiments" folder contains jupyter notebooks to reproduce the results from the paper.
Please follow the instructions in the notebooks to run the FLAME algorithms and obtain the dataframe for the matched groups and estimated CATEs.

**All scripts are in Python**.

## Citation
If you use this code, please cite:

      @article{2018arXiv180606802D,
          author = {{Dieng}, A. and {Liu}, Y. and {Roy}, S. and {Rudin}, C. and {Volfovsky}, A.},
          title = "{Collapsing-Fast-Large-Almost-Matching-Exactly: A Matching Method for Causal Inference}",
          journal = {ArXiv e-prints},
          archivePrefix = "arXiv",
          eprint = {1806.06802},
          year = 2018,
          month = jun,
          }

Contact adieng@cs.duke.edu for inquiries.
