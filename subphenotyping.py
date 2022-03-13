# Libraries to Load
import load_and_save
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from tslearn.metrics import dtw
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

warnings.filterwarnings("ignore", r"All-NaN slice encountered")
warnings.filterwarnings("ignore", r"Mean of empty slice")

rpy2.robjects.numpy2ri.activate()
NbClust = importr("NbClust")
rstats = importr("stats")


def convert_rvector(vec):
    d = dict(zip(vec.names, map(list, list(vec))))
    return d


def replace_outliers(group, stds):
    # set to np.nan for zero values.
    group[np.abs(group - group.mean(axis=0)) > stds * group.std(axis=0)] = np.nan
    return group


def fill_nan_median(dataset):
    # Fill missing values using the median over all patients
    col_median = np.nanmedian(dataset, axis=0)
    inds = np.where(np.isnan(dataset))
    dataset[inds] = np.take(col_median, inds[1])
    return dataset


def get_feature_stats(dataset: np.ndarray, kmeans_labels: list, total_clusters: int):
    # test to see if the data is not normally distribuited
    _, f = dataset.shape
    values = np.empty((f, total_clusters * 2 + 1))
    for i in range(f):
        normal = True
        shapiro_test = stats.shapiro(dataset[:, i])
        if shapiro_test.pvalue < 0.05:
            normal = False
        all_cluster_data = []
        feature_stats = []
        for c in range(total_clusters):
            row_indexes = np.where(kmeans_labels == c)[0]
            cluster_data = dataset[row_indexes, i].flatten()
            feature_stats.append(np.nanmean(cluster_data))  # mean
            feature_stats.append(np.nanstd(cluster_data))  # std
            all_cluster_data.append(cluster_data)
        if total_clusters == 2:  # only two clusters:
            if normal:
                t_test = stats.ttest_ind(*all_cluster_data)
                feature_stats.append(t_test.pvalue)
            else:
                mwu_test = stats.mannwhitneyu(*all_cluster_data)
                feature_stats.append(mwu_test.pvalue)
        else:  # more than two clusters
            if normal:
                anova_test = stats.f_oneway(*all_cluster_data)
                feature_stats.append(anova_test.pvalue)
            else:
                kruskal_test = stats.kruskal(*all_cluster_data)
                feature_stats.append(kruskal_test.pvalue)
        values[i] = feature_stats
    return values


def visualize_aki_clusters(dataset: list, labels: list, total_clusters: list):
    np_data = np.array(dataset, dtype=np.float64)
    # Normalize data
    np_data_scaled = MinMaxScaler().fit_transform(np_data)
    result = TSNE(n_components=2, init="pca", random_state=42).fit_transform(
        np_data_scaled
    )
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "cyan"]
    for c in range(total_clusters):
        cluster_data = result[np.where(labels == c)]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[c])
        # plt.plot(center[c, 0], center[c, 1], "kx")
    plt.savefig("./results/aki_clusters.jpeg")
    plt.clf()


def visualize_sepsis_clusters(dataset: list, labels: list, total_clusters: int):
    np_data = np.array(dataset)
    # maximum of 8 possible clusters, so possible colors
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "cyan"]
    for c in range(total_clusters):
        cluster_data = np_data[np.where(labels == c)]
        n = len(cluster_data)
        m, se = np.mean(cluster_data, axis=0), stats.sem(cluster_data, axis=0)
        h = se * stats.t.ppf((1 + 0.95) / 2.0, n - 1)
        plt.plot(range(1, 73), m, color=colors[c])
        plt.fill_between(range(72), m - h, m + h, color=colors[c], alpha=0.25)
    plt.savefig("./results/sepsis_trajectories.jpeg")
    plt.clf()


def aki_feature_extraction(dataset: list, patients: list):
    features_extracted = np.empty((len(dataset), 35))
    features_mean = np.empty((len(dataset), 35))
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
        features_mean[i] = np.nanmean(np_patient_data, axis=1)
    # Fill missing values using the median over all patients
    features_extracted = fill_nan_median(features_extracted)
    features_mean = fill_nan_median(features_mean)
    # Set nan outliers that are 3 std. above or below mean
    features_outliers_nan = replace_outliers(features_extracted, 3)
    row_indexes = np.isnan(features_outliers_nan).any(axis=1)
    np_patients = np.array(patients)
    features_outliers_removed = features_outliers_nan[~row_indexes]
    features_mean_removed = features_mean[~row_indexes]
    patients_outliers_removed = np_patients[~row_indexes]
    patients_outliers_removed = patients_outliers_removed.tolist()
    print("Total AKI Patients: ", len(patients_outliers_removed))
    return features_outliers_removed, features_mean_removed, patients_outliers_removed


def aki_identify_clusters(dataset: list):
    indexes = [
        "kl",
        "ch",
        "db",
        "silhouette",
        "duda",
        "hartigan",
        "beale",
        "cindex",
        "ratkowsky",
        "mcclain",
    ]
    num_clusters = []
    f = ro.r(
        """
        f<-function(data, index){
            NbClust(data,distance="euclidean", min.nc=2, max.nc=8, method="average", index=index)
            }
        """
    )
    np_data = np.array(dataset, dtype=np.float64)
    nr, nc = np_data.shape
    # Normalize data
    np_data_scaled = MinMaxScaler().fit_transform(np_data)
    # R code Matrix
    r_vec = ro.FloatVector(np_data_scaled.transpose().reshape((np_data_scaled.size)))
    r_data = ro.r.matrix(r_vec, nrow=nr, ncol=nc)
    for ind in indexes:
        num_clusters.append(int(convert_rvector(f(r_data, ind))["Best.nc"][0]))
    indexes.append("majority")
    majority = stats.mode(num_clusters)[0][0]
    num_clusters.append(majority)
    values = np.hstack((np.c_[indexes], np.c_[num_clusters]))
    table_index = np.vstack((["index", "num_clusters"], values))
    print("AKI Majority Vote Clusters: ", majority)
    return majority, table_index


def aki_kmeans_clusters(dataset: list, patients: list, optimal_majority: int):
    np_data = np.array(dataset, dtype=np.float64)
    # Normalize data
    np_data_scaled = MinMaxScaler().fit_transform(np_data)
    # k means configuration
    kmeans = KMeans(n_clusters=optimal_majority, init="k-means++", random_state=42).fit(
        np_data_scaled
    )
    kmeans_labels = np.array(kmeans.labels_)
    table_clusters = [["id"] + ["cluster_" + str(x) for x in range(optimal_majority)]]
    for i, pid in enumerate(patients):
        row = [pid]
        for j in range(optimal_majority):
            if kmeans_labels[i] == j:
                row.append(1)
            else:
                row.append(0)
        table_clusters.append(row)
    table_clusters = np.array(table_clusters)
    totals = np.sum(table_clusters[1:, 1:].astype(int), axis=0)
    print("Cluster Totals: ", totals)
    table_clusters = np.vstack((table_clusters, np.concatenate((["total"], totals))))
    return kmeans_labels, kmeans.cluster_centers_, table_clusters


def aki_statistical_analysis(
    dataset: list,
    features: list,
    kmeans_labels: list,
    optimal_cluster: int,
):
    table_header = (
        ["feature"]
        + [
            "cluster_" + str(x) + "_" + y
            for x in range(optimal_cluster)
            for y in ["mean", "std"]
        ]
        + ["pvalue"]
    )
    np_data = np.array(dataset)
    values = get_feature_stats(np_data, kmeans_labels, optimal_cluster)
    f_values = np.hstack((np.c_[features[:35]], values))
    # return only features that have a significant pvalue
    sig_only = f_values[f_values[:, -1].astype(np.float64) < 0.05]
    table = np.vstack((table_header, sig_only))
    return table


def sepsis_feature_extraction(dataset: list, patients: list):
    features_temp = np.empty((len(dataset), 72))
    features_mean = np.empty((len(dataset), 35))
    features_temp[:] = np.nan
    for i, patient_data in enumerate(dataset):
        np_patient_data = np.array(patient_data, dtype=np.float64)
        ICULOS = np_patient_data[39]
        np_patient_data = np_patient_data[:35, :]
        temp_data = np_patient_data[2]
        for j, ele in enumerate(temp_data):
            hour = int(ICULOS[j])
            features_temp[i, hour - 1] = ele
            if hour >= 72:
                break
        features_mean[i] = np.nanmean(np_patient_data, axis=1)
    features_mean = fill_nan_median(features_mean)
    df = pd.DataFrame(features_temp)
    df.fillna(method="ffill", axis=1, inplace=True)
    df.fillna(method="bfill", axis=1, inplace=True)
    features_temp = df.to_numpy()
    row_indexes = np.isnan(features_temp).any(axis=1)
    np_patients = np.array(patients)
    features_temp_removed = features_temp[~row_indexes]
    features_mean_removed = features_mean[~row_indexes]
    patients_novalues_removed = np_patients[~row_indexes]
    patients_novalues_removed = patients_novalues_removed.tolist()
    print("Total Sepsis Patients: ", len(patients_novalues_removed))
    return features_temp_removed, features_mean_removed, patients_novalues_removed


def sepsis_identify_clusters(dataset: list):
    patients_n = len(dataset)
    dtw_matrix = np.empty((patients_n, patients_n))
    for i, temps_i in enumerate(dataset):
        for j, temps_j in enumerate(dataset):
            dtw_matrix[i, j] = dtw(temps_i, temps_j)
    load_and_save.create_csv("./results/sepsis_distances.csv", dtw_matrix)

    # dtw_matrix = load_and_save.read_csv("./results/sepsis_distances.csv", False)
    # dtw_matrix = np.array(dtw_matrix, dtype=np.float64)
    indexes = [
        "ptbiserial",
        "frey",
        "kl",
        "ch",
        "db",
        "silhouette",
        "hartigan",
        "cindex",
        "ratkowsky",
        "mcclain",
    ]
    num_clusters = []
    f = ro.r(
        """
        f<-function(data, d, index){
            NbClust(data, diss=as.dist(d), distance=NULL, min.nc=2, max.nc=8, method="complete", index=index)
            }
        """
    )
    # Normalize data
    np_data = np.array(dataset)
    nr, nc = np_data.shape
    np_data_scaled = MinMaxScaler().fit_transform(np_data)
    # convert dataset to r matrix
    r_data_vec = ro.FloatVector(
        np_data_scaled.transpose().reshape((np_data_scaled.size))
    )
    r_data = ro.r.matrix(r_data_vec, nrow=nr, ncol=nc)
    # convert diss to r matrix
    dr, dc = dtw_matrix.shape
    r_diss_vec = ro.FloatVector(dtw_matrix.transpose().reshape((dtw_matrix.size)))
    r_diss = ro.r.matrix(r_diss_vec, nrow=dr, ncol=dc)
    for ind in indexes:
        num_clusters.append(int(convert_rvector(f(r_data, r_diss, ind))["Best.nc"][0]))
    indexes.append("majority")
    majority = stats.mode(num_clusters)[0][0]
    num_clusters.append(majority)
    values = np.hstack((np.c_[indexes], np.c_[num_clusters]))
    table = np.vstack((["index", "num_clusters"], values))
    print("Sepsis Majority Vote Clusters: ", majority)
    return (
        majority,
        dtw_matrix,
        table,
    )


def sepsis_agglomerative_clusters(
    distances: list, patients: list, optimal_majority: int
):
    # AgglomerativeClustering
    agglomerative = AgglomerativeClustering(
        n_clusters=optimal_majority, linkage="complete", affinity="precomputed"
    ).fit(distances)
    agglomerative_labels = agglomerative.labels_
    table_clusters = [["id"] + ["cluster_" + str(x) for x in range(optimal_majority)]]
    for i, pid in enumerate(patients):
        row = [pid]
        for j in range(optimal_majority):
            if agglomerative_labels[i] == j:
                row.append(1)
            else:
                row.append(0)
        table_clusters.append(row)
    table_clusters = np.array(table_clusters)
    totals = np.sum(table_clusters[1:, 1:].astype(int), axis=0)
    print("Cluster Totals: ", totals)
    table_clusters = np.vstack((table_clusters, np.concatenate((["total"], totals))))
    return agglomerative_labels, table_clusters


def sepsis_statistical_analysis(
    dataset: list,
    features: list,
    kmeans_labels: list,
    optimal_cluster: int,
):
    table_header = (
        ["feature"]
        + [
            "cluster_" + str(x) + "_" + y
            for x in range(optimal_cluster)
            for y in ["mean", "std"]
        ]
        + ["pvalue"]
    )
    cluster_values = np.c_[features[:35]]
    cluster_values = np.delete(cluster_values, 2, axis=0)  # delete TEMP
    np_data = np.array(dataset)
    np_data = np.delete(np_data, 2, axis=1)  # delete TEMP
    values = get_feature_stats(np_data, kmeans_labels, optimal_cluster)
    f_names = np.c_[features[:35]]
    f_names = np.delete(f_names, 2, axis=0)  # delete TEMP
    f_values = np.hstack((f_names, values))
    # return only features that have a significant pvalue
    sig_only = f_values[f_values[:, -1].astype(np.float64) < 0.05]
    table = np.vstack((table_header, sig_only))
    return table


def main():
    print("Loading cohorts...")
    aki_data = load_and_save.open_pickle("./results/aki_cohort.pickle")
    aki_patients = load_and_save.open_pickle("./results/aki_patients.pickle")
    sepsis_data = load_and_save.open_pickle("./results/sepsis_cohort.pickle")
    sepsis_patients = load_and_save.open_pickle("./results/sepsis_patients.pickle")
    feature_names = load_and_save.open_txt("./results/features.txt")
    print("Performing feature extraction for aki and sepsis...")
    aki_extraction, aki_mean, aki_patients = aki_feature_extraction(
        aki_data, aki_patients
    )
    sepsis_extraction, sepsis_mean, sepsis_patients = sepsis_feature_extraction(
        sepsis_data, sepsis_patients
    )
    # load_and_save.create_csv("./results/aki_extraction.csv", aki_extraction)
    # load_and_save.create_csv("./results/aki_mean.csv", aki_mean)
    # load_and_save.create_csv("./results/sepsis_extraction.csv", sepsis_extraction)
    # load_and_save.create_csv("./results/sepsis_mean.csv", sepsis_mean)
    print("Identifying optimal number of clusters for aki...")
    (
        aki_cluster_optimal,
        aki_index_table,
    ) = aki_identify_clusters(aki_extraction)
    load_and_save.create_csv("./results/aki_index.csv", aki_index_table)
    print("Identifying optimal number of clusters for sepsis...")
    (
        sepsis_cluster_optimal,
        sepsis_distance_matrix,
        sespis_index_table,
    ) = sepsis_identify_clusters(sepsis_extraction)
    load_and_save.create_csv("./results/sepsis_index.csv", sespis_index_table)
    print("Performing kmeans on aki clusters...")
    (
        aki_kmeans_labels,
        aki_kmeans_centroids,
        aki_cluster_table,
    ) = aki_kmeans_clusters(aki_extraction, aki_patients, aki_cluster_optimal)
    load_and_save.create_csv("./results/aki_clusters.csv", aki_cluster_table)
    visualize_aki_clusters(aki_extraction, aki_kmeans_labels, aki_cluster_optimal)
    print("Performing agglomerative clustering on sepsis clusters...")
    (
        sepsis_agglomerative_labels,
        sepsis_cluster_table,
    ) = sepsis_agglomerative_clusters(
        sepsis_distance_matrix, sepsis_patients, sepsis_cluster_optimal
    )
    load_and_save.create_csv("./results/sepsis_clusters.csv", sepsis_cluster_table)
    visualize_sepsis_clusters(
        sepsis_extraction, sepsis_agglomerative_labels, sepsis_cluster_optimal
    )
    print("Performing statistical analysis on clusters...")
    aki_cluster_features = aki_statistical_analysis(
        aki_mean, feature_names, aki_kmeans_labels, aki_cluster_optimal
    )
    load_and_save.create_csv("./results/aki_cluster_features.csv", aki_cluster_features)
    sepsis_cluster_features = sepsis_statistical_analysis(
        sepsis_mean, feature_names, sepsis_agglomerative_labels, sepsis_cluster_optimal
    )
    load_and_save.create_csv(
        "./results/sepsis_cluster_features.csv", sepsis_cluster_features
    )
    print("Done. File is saved in results directory.")


if __name__ == "__main__":
    main()
