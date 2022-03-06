# Libraries to Load
import statistics
import load_and_save


def gen_characteristics(dataset: list, features: list) -> list:
    data = [[] for _ in range(35)]
    for patient in dataset:
        # limit to only the 35 continuous features
        for i, row in enumerate(patient[:35]):
            data[i] += row
    distr = [["feature", "min", "max", "mean", "median", "std", "missing"]]
    for i, feature in enumerate(data):
        exc_empty = [x for x in feature if x is not None]
        # calculate the minimum
        min_v = min(exc_empty)
        # calculate the maximum
        max_v = max(exc_empty)
        # calculate the mean
        mean_v = statistics.mean(exc_empty)
        # calculcate the median
        median_v = statistics.median(exc_empty)
        # get std
        std_v = statistics.stdev(exc_empty)
        # calculate the number of missing values
        missing_v = (len(feature) - len(exc_empty)) / len(feature) * 100
        # save the values
        distr.append([features[i], min_v, max_v, mean_v, median_v, std_v, missing_v])
    return distr


def main():
    print("Loading cohorts...")
    aki_data = load_and_save.open_pickle("./results/aki_cohort.pickle")
    sepsis_data = load_and_save.open_pickle("./results/sepsis_cohort.pickle")
    features = load_and_save.open_txt("./results/features.txt")
    print("Getting feature distributions...")
    aki_distribution = gen_characteristics(aki_data, features)
    sepsis_distribution = gen_characteristics(sepsis_data, features)
    print("Saving to CSV...")
    load_and_save.create_csv("./results/aki_features.csv", aki_distribution)
    load_and_save.create_csv("./results/sepsis_features.csv", sepsis_distribution)
    print("Done. File is saved in results directory.")


if __name__ == "__main__":
    main()
