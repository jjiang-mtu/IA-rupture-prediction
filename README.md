# Can We Explain Machine Learning-based Prediction for Rupture Status Assessments of Intracranial Aneurysm?

## Usage

### Datasets
From an internal database, patient-specific IA models were created from medical imaging data (DICOM images of 3D rotational angiographies) acquired from three sources: the University of Michigan, Changhai Hospital (Shanghai), and the Aneurisk open-source repository (http://ecm2.mathcs.emory.edu/aneuriskweb/index).  

### Machine Learning (ML) Algorithms
Six selected ML algorithms include multivariate logistic regression (LR), multiple layer perceptron neural network (MLPNN), support vector machine (SVM), random forest (RF), extreme gradient boosting (XGBoost), and Bayesian additive regressions trees (BART). Python codes of BART can be downloaded from https://github.com/JakeColtman/bartpy. 

### Global and Local Model-agnostic Methods
Permutation feature importance, local interpretable model-agnostic explanations (LIME), and SHapley Additive exPlanations (SHAP) algorithms are adopted to explain and analyze each ML method, respectively. 

### Training and Testing
Run ML-based models to get evaluation results (accuracy, AUC, precision, recall, and F1-score) and permutation feature importances.

```shell
$ python test_ML.py
```
If you want to get the LIME and SHAP feature importances, please use:

```shell
$ python test_LIME_SHAP.py
```

## Cite
If you find our code useful for your research, please cite our paper:

N. Mu, M. Rezaeitaleshmahalleh, Z. Lyu, M. Wang, J. Tang, C. M. Strother, J.J. Gemmete, A.S. Pandey, and J. Jiang, "Can We Explain Machine Learning-based Prediction for Rupture Status Assessments of Intracranial Aneurysm?", Biomedical Physics & Engineering Express, under review, 2023. 

In case of any questions, please contact the corresponding author J. Jiang at jjiang1@mtu.edu
