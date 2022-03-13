# ML_Health1_Sepsis_AKI
2019 PhysioNet Challenge for Sepsis and AKI Prediction
Subphenotyping using EHR data

## Setup
You need to have Python 3.8 or later and R 4.1.3 installed
```
pip install -r requirements.txt
```
You need to download the Sepsis and AKI prediction data and put it into
a folder called 'dataset'.

Likewise, you need a folder called 'results'. This is where the scripts will
output the files.

## Contribuitors
You should install the additional requirements file for contribuitors:
```
pip install -r requirements.dev.txt
```

Before commiting make sure to follow the style format:
```
black .
flake8
```

## Programs to run in order for disease phenotyping of AKI / Sepsis
```
python construct_cohort.py
python get_distributions.py
python subphenotyping.py
```
The following files will be created in `./results` after running the three scripts
1. aki_cluster_features.csv
2. aki_clusters.csv
3. aki_clusters.jpeg
4. aki_cohort.pickle
5. aki_features.csv
6. aki_index.csv
7. aki_patients.pickle
8. features.txt
9. sepsis_cluster_features.csv
10. sepsis_custers.csv
11. sepsis_cohort.pickle
12. sepsis_features.csv
13. sepsis_index.csv
14. sepsis_patients.pickle
15. sepsis_trajectories.jpeg