# Libraries to Load
import load_and_save


def develops_aki(dataset: list, patients: list) -> list:
    """
    Def of AKI:
    ● an increase in creatinine of 0.3 mg/dl (26.5 μmol/l) within 48 hours; or
    ● an increase in creatinine of 1.5 times the baseline creatinine level of
    a patient (the lowest creatinine value found in the data),
    known or presumed to have occurred within the prior 7 days.

    Note: if there are no creatine records, the patient did not suffer from AKI
    """
    aki_patients = []
    aki_dataset = []
    for i, patient_data in enumerate(dataset):
        # Get the list of creatinine for the patient
        Creatinine = patient_data[19]
        if Creatinine and len(Creatinine) > 2:
            n_hours = len(Creatinine)
            for j in range(1, n_hours):
                if Creatinine[j] and Creatinine[:j]:
                    prev_48 = 0 if j < 48 else (j - 48)
                    prev_168 = 0 if j < 168 else (j - 168)
                    min_48 = min(
                        [x for x in Creatinine[prev_48:j] if x is not None],
                        default=None,
                    )
                    min_168 = min(
                        [x for x in Creatinine[prev_168:j] if x is not None],
                        default=None,
                    )
                    if (min_48 and (Creatinine[j] >= round(min_48 + 0.3, 2))) or (
                        min_168 and (Creatinine[j] >= round(min_168 * 1.5, 2))
                    ):
                        # save the values
                        aki_patients.append(patients[i])
                        aki_data = [row[: (j + 1)] for row in patient_data]
                        aki_dataset.append(aki_data)
                        break
    return aki_patients, aki_dataset


def develops_sepsis(dataset: list, patients: list) -> list:
    """
    If patient went into septic shock, it is provided in the data
    in the column 'Septic_Shock' with the last value set to 1
    """
    sepsis_patients = []
    sepsis_dataset = []
    for i, patient_data in enumerate(dataset):
        # Get the index of when patient develops Sepsis
        SepsisLabel = patient_data[40]
        sepsis = False
        if any(SepsisLabel):
            sepsis = True
        if sepsis:
            ICULOS = patient_data[39]
            sepsis_onset = ICULOS[SepsisLabel.index(1)]
            if sepsis_onset <= 72:
                sepsis_patients.append(patients[i])
                sepsis_dataset.append(patient_data)
    return sepsis_patients, sepsis_dataset


def main():
    print("Loading dataset...")
    patients, features, data = load_and_save.data_transpose("./dataset/")
    # create aki cohort
    print("Creating aki cohort...")
    aki_patients, aki_cohort = develops_aki(data, patients)
    # create sepsis cohort
    print("Creating sepsis cohort...")
    sepsis_patients, sepsis_cohort = develops_sepsis(data, patients)
    # save cohorts
    print("Saving aki and sepsis cohorts")
    load_and_save.create_txt("./results/features.txt", features)
    load_and_save.create_txt("./results/aki_patients.txt", aki_patients)
    load_and_save.create_pickle("./results/aki_cohort.pickle", aki_cohort)
    load_and_save.create_txt("./results/sepsis_patients.txt", sepsis_patients)
    load_and_save.create_pickle("./results/sepsis_cohort.pickle", sepsis_cohort)
    print("Done. Files are saved in results directory.")


if __name__ == "__main__":
    main()
