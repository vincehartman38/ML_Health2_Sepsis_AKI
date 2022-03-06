# ML_Health1_Sepsis_AKI
2019 PhysioNet Challenge for Sepsis and AKI Prediction
Subphenotyping using EHR data

## Setup
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