# Libraries to Load
import os
import csv
import pickle
from pathlib import Path

# Make data and results directories if they do not exist
Path("./results").mkdir(parents=True, exist_ok=True)

# load dataset
def data_transpose(f_path: str) -> list:
    files = os.listdir(f_path)
    data = []
    patients = []
    first_line = True
    for infile in files:
        with open(f_path + infile, newline="") as f:
            if first_line:
                features = next(f).rstrip().split("|")
                first_line = False
            else:
                next(f)
            reader = csv.reader(f, delimiter="|")
            dataset = [
                [float(i) if i != "NaN" else None for i in list(i)]
                for i in zip(*list(reader))
            ]
        data.append(dataset)
        patient_id, _ = os.path.splitext(infile)
        patients.append(patient_id)
    return patients, features, data


def read_csv(f_path: str, skip_header: bool = True) -> list:
    with open(f_path, newline="") as f:
        if skip_header:
            next(f)
        reader = csv.reader(f)
        dataset = list(reader)
    return dataset


# Save CSV file
def create_csv(filepath: str, data: list):
    with open(filepath, "w", newline="") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerows(data)


# Create pickle file
def create_pickle(filepath: str, data: list):
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


# Open pickle file
def open_pickle(filepath: str):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


# Create txt file
def create_txt(filepath: str, data: list):
    with open(filepath, "w") as f:
        for item in data:
            f.write("%s\n" % item)


# Create txt file
def open_txt(filepath: str):
    file = open(filepath, "r")
    file_lines = file.read()
    list_of_lines = file_lines.split("\n")
    return list_of_lines
