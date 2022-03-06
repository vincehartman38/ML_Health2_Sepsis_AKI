# Libraries to Load
import load_and_save
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", r"All-NaN slice encountered")


def replace_outliers(group, stds):
    group[np.abs(group - group.mean()) > stds * group.std()] = np.nan
    return group


def aki_extraction(dataset: list) -> list:
    features_extracted = np.empty((len(dataset), 35))
    for i, patient_data in enumerate(dataset):
        np_patient_data = np.array(patient_data, dtype=np.float64)
        np_patient_data = np_patient_data[:35, :]
        features_extracted[i][0] = np.nanmax(np_patient_data[0])  # Max of HR
        features_extracted[i][1] = np.nanmin(np_patient_data[1])  # Min of 02Sat
        features_extracted[i][2] = np.nanmax(np_patient_data[2])  # Max of Temp
        features_extracted[i][3] = np.nanmin(np_patient_data[3])  # Min of SBP
        features_extracted[i][4] = np.nanmin(np_patient_data[4])  # Min of MAP
        features_extracted[i][5] = np.nanmin(np_patient_data[5])  # Min of DBP
        features_extracted[i][6] = np.nanmax(np_patient_data[6])  # Max of Resp
        features_extracted[i][7] = np.nanmin(np_patient_data[7])  # Min of Etco2
        features_extracted[i][8] = np.nanmin(np_patient_data[8])  # Min of BaseExcess
        features_extracted[i][9] = np.nanmin(np_patient_data[9])  # Min of Hco3
        features_extracted[i][10] = np.nanmax(np_patient_data[10])  # Max of FIO2
        features_extracted[i][11] = np.nanmin(np_patient_data[11])  # Min of pH
        features_extracted[i][12] = np.nanmin(np_patient_data[12])  # Min of Paco2
        features_extracted[i][13] = np.nanmin(np_patient_data[13])  # Min of Sao2
        features_extracted[i][14] = np.nanmax(np_patient_data[14])  # Max of AST
        features_extracted[i][15] = np.nanmax(np_patient_data[15])  # Max of BUN
        features_extracted[i][16] = np.nanmax(np_patient_data[16])  # Max of Alka
        features_extracted[i][17] = np.nanmax(np_patient_data[17])  # Max of Calcium
        features_extracted[i][18] = np.nanmax(np_patient_data[18])  # Max of Chloride
        features_extracted[i][19] = np.nanmax(np_patient_data[19])  # Max of Creatinine
        features_extracted[i][20] = np.nanmax(np_patient_data[20])  # Max of Bil dir
        features_extracted[i][21] = np.nanmax(np_patient_data[21])  # Max of Glucose
        features_extracted[i][22] = np.nanmax(np_patient_data[22])  # Max of Lactate
        features_extracted[i][23] = np.nanmax(np_patient_data[23])  # Max of Magnesium
        features_extracted[i][24] = np.nanmax(np_patient_data[24])  # Max of Phosphate
        features_extracted[i][25] = np.nanmin(np_patient_data[25])  # Min of Potassium
        features_extracted[i][26] = np.nanmax(np_patient_data[26])  # Max of Bil tot
        features_extracted[i][27] = np.nanmax(np_patient_data[27])  # Max of Troponinl
        features_extracted[i][28] = np.nanmin(np_patient_data[28])  # Min of Hct
        features_extracted[i][29] = np.nanmin(np_patient_data[29])  # Min of Hgb
        features_extracted[i][30] = np.nanmax(np_patient_data[30])  # Max of PTT
        features_extracted[i][31] = np.nanmax(np_patient_data[31])  # Max of WBC
        features_extracted[i][32] = np.nanmin(np_patient_data[32])  # Min of Fibrinogen
        features_extracted[i][33] = np.nanmin(np_patient_data[33])  # Min of Platelets
        features_extracted[i][34] = np.nanmax(np_patient_data[34])  # Max of Age
    # Fill missing values using the median over all patients
    col_median = np.nanmedian(features_extracted, axis=0)
    inds = np.where(np.isnan(features_extracted))
    features_extracted[inds] = np.take(col_median, inds[1])
    # Remove outliers that are 3 std. above or below mean
    features_outliers_removed = replace_outliers(features_extracted, 3)
    # Normalize data
    features_scaled = MinMaxScaler().fit_transform(features_outliers_removed)
    return features_scaled


def sepsis_extraction(dataset: list) -> list:
    features_extracted = np.empty((len(dataset), 72))
    features_extracted[:] = np.nan
    for i, patient_data in enumerate(dataset):
        temp_data = patient_data[2][:72]
        for j, ele in enumerate(temp_data):
            features_extracted[i, j] = ele
    # # LOCF
    # # https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    # mask = np.isnan(features_extracted)
    # idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    # np.maximum.accumulate(idx, axis=1, out=idx)
    # features_extracted[mask] = features_extracted[np.nonzero(mask)[0], idx[mask]]
    # LOCF
    df = pd.DataFrame(features_extracted)
    df.fillna(method="ffill", axis=1, inplace=True)
    df.fillna(method="bfill", axis=1, inplace=True)
    features_extracted = df.to_numpy()
    return features_extracted


def main():
    print("Loading cohorts...")
    aki_data = load_and_save.open_pickle("./results/aki_cohort.pickle")
    sepsis_data = load_and_save.open_pickle("./results/sepsis_cohort.pickle")
    features = load_and_save.open_txt("./results/features.txt")
    print("Performing feature extraction...")
    aki_features = aki_extraction(aki_data)
    sepsis_features = sepsis_extraction(sepsis_data)
    print("Saving to CSV...")
    aki_save = np.vstack((features[:35], aki_features))
    sepsis_save = np.vstack((list(range(1, 73)), sepsis_features))
    load_and_save.create_csv("./results/aki_extraction.csv", aki_save)
    load_and_save.create_csv("./results/sepsis_extraction.csv", sepsis_save)
    print("Done. File is saved in results directory.")


if __name__ == "__main__":
    main()
