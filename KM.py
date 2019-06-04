# Author: Nunzio Gatti / England aaaa

# Date: 07th of June, 2018
import sys
import os
import numpy as np


def update_centroids(labelled_data, centroids, id_to_name_mapping):
    """
    Updates the centroids after re-allocation of
    the data to the new clusters
    """
    for idx, cluster_name in id_to_name_mapping.items():
        single_cluster_datapoints = labelled_data[labelled_data[:, 2] == idx]
        new_centroid = single_cluster_datapoints[:, :2].mean(axis=0)
        centroids[cluster_name] = new_centroid
    return centroids


def get_distance_to_cluster(data, centre):
    return (((data - np.array(centre)) ** 2).sum(axis=1)) ** 0.5


def update_distances(data, distance_matrix, centroids):
    """This functions updates the matrix of distances
    from each point to each centre
    """
    for idx, centre in enumerate(centroids.values()):
        distance_matrix[:, idx] = get_distance_to_cluster(data, centre)
    return distance_matrix


def fit(data, distance_matrix, centroids):
    """
    k-means implementation
    :returns tuple: labelled_data - a matrix Nx(K+1) - k-cols of data and 1 col with labels
                    error: Nx1 - the distance to the closest centre
    """
    distance_matrix = update_distances(data, distance_matrix, centroids)
    error = distance_matrix.min(axis=1).sum()  # get the distance to the closest centre
    temporary_vector = distance_matrix.argmin(axis=1)  # get its index
    labelled_data = np.concatenate((data, temporary_vector.reshape(data.shape[0], 1)), axis=1)
    return labelled_data, error


def cluster(data):
    """
    Clusters the data points.
    Returns error and the cluster names
    """
    iter_count = 50  # number of iterations to find the best centroids
    # initial centroids
    centroids = {
        "Adam": [-0.357, -0.253],
        "Bob": [-0.055, 4.392],
        "Charley": [2.674, -0.001],
        "David": [-1.044, -1.251],
        "Edward": [-1.495, -0.090]
    }

    # mapping of the Cluster IDs to the Cluster Names
    id_to_name_mapping = dict(zip(range(0, len(centroids)), centroids.keys()))

    # initialization of an empty matrix for the distances
    # per each cluster. the shape of the matrix is Nx5 (5 clusters)
    distance_matrix = np.zeros((data.shape[0], len(centroids)))
    # initialization of a temporary vector to store
    # the clusters (to be used in the following loop)
    error = sys.maxsize  # the error variable
    for _ in range(iter_count):
        previous_error = error
        labelled_data, error = fit(data, distance_matrix, centroids)
        # update the centroids
        centroids = update_centroids(labelled_data, centroids, id_to_name_mapping)
        # if converged
        if previous_error == error:
            return error, get_names(labelled_data[:, 2], id_to_name_mapping)


def get_names(data, id_to_name_mapping):
    """ gets names of the clusters by the ids """
    return [id_to_name_mapping[clutser] for clutser in data]


def save_to_file(filename, output, error):
    error = "error = {}\n".format(error)
    with open(filename, 'w') as f:
        f.write(error)
        for line in output:
            f.write(line + '\n')


def main():
    """ Main functions that clusters the data and saves the result into a file """
    path = os.getcwd() + '/input.csv'
    data = np.genfromtxt(path, delimiter=',')  # used genfromtxt due to its speed
    filename = "OUTPUT.TXT"
    error, clustered_data = cluster(data)
    save_to_file(filename, clustered_data, error)


if __name__ == "__main__":
    main()
    print("Finished!")
