# Track2Vec - Fairness Music Recommendation with a GPU-Free Customizable-Driven Framework

## Introduction

This is the submission of team wwweiwei to the [EvalRS Data Challenge](https://github.com/RecList/evalRS-CIKM-2022). 
* Proposed Framework: Track2Vec
![Track2Vec Framework](images/Track2Vec_framework.jpg)
* Proposed Fairness Metric: Miss Rate - Inverse Ground Truth Frequency (MR-ITF)
![MR-ITF Equation](images/MR_ITF_equation.png)

## Instructions
### Setup
- Build environment
    ```
    conda env create -f environment.yml
    ```
### Run script
```
python submission.py
```
* Notes: Our proposed metric MR-ITF will automatically report in the corresponding json file with other standard metric.
