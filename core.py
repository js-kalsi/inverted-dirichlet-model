"""
/*
*       Coded by : Jaspreet Singh Kalsi.
*
*       "Thesis  Chapter-2 Part A
*       (Image Fragmentation using Inverted Dirichlet Distribution using Markov Random Field as a Prior).
*
*       ```python core.py <Image-Name>```
*
*
*   EM Algorithm Inverted Dirichlet Mixture Model.
*       1) Convert image pixels into array.
*       2) Normalize it.
*       3) Assume Number of Cluster(K) =  10 & apply KMeans clustering algorithm
*          to obtain K clusters for Initialization purposes.
*       4) Use `Method of Moments` for obtaining the initial values for Mixing Parameters.
*       5) Expectation Step:
*                           => Compute the Posterior Probability.
*       6) Maximization Step:
*                           => Update the Mixing Parameter.
*                           => Update the Alpha Parameter using `Newton Raphson` Method.
*       7) If Mixing Parameter of any Cluster < Cluster-Skipping Threshold:
*                           => Skip that particular Cluster.
*       8) Compute the Log Likelihood and check for Convergence by comparing the difference
*          between last two likelihood values with the Convergence Threshold.
*       9) If algorithm(converge) :terminate
*       10) Go-to Step 5(Expectation Step).
*/

"""

import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from helpers import cluster_density_evaluation, method_of_moment, g_estimation, \
    load_dataset, posterior_estimator, split_data_by_label, hessian, hessian_inverse, alpha_updater
from inverted_dirichlet import inverted_dirichlet
from parameters import DATASET_PATH, K, THRESHOLD


def initial_algorithm():
    """

    @return:
    """
    data_set, original_labels = load_dataset(DATASET_PATH)
    data_set = normalize(data_set) + sys.float_info.epsilon
    data_set = ((data_set - np.amin(data_set, axis=0)) / (np.max(data_set, axis=0) - np.amin(data_set, axis=0)))
    data_set = data_set + sys.float_info.epsilon
    labels = np.asarray(KMeans(n_clusters=K).fit(data_set).predict(data_set)).reshape(1, len(data_set))[0]
    clusters, unique_clusters, dimension = split_data_by_label(labels, data_set)
    initial_py = np.asarray([len(clusters[k]) / len(data_set) for k in clusters]).reshape(1, K)
    initial_alpha = method_of_moment(K, clusters, dimension)
    return initial_alpha, data_set, dimension, len(data_set), initial_py, original_labels


def estimation_step(no_of_clusters, mix, alpha, data, dim):
    """
    This function Contains the Estimation Step logic.

    @param no_of_clusters:
    @param mix:
    @param alpha:
    @param data:
    @param dim:
    @return:
    """
    pdf = inverted_dirichlet(no_of_clusters, alpha, data, dim).pdf_fetcher()
    posterior = posterior_estimator(pdf, mix)
    return pdf, posterior


def maximization_step(no_of_clusters, alpha, data, dim, posterior, size):
    """
    This function Contains the Maximization Step logic.

    @param K:
    @param alpha:
    @param data:
    @param dim:
    @param posterior:
    @param size:
    @return:
    """
    mix_m_step = (np.sum(posterior, axis=0) / size).reshape(1, no_of_clusters)
    G = g_estimation(data, size, alpha, posterior, dim)
    h_diagonal, h_constant, h_a = hessian(dim, posterior, alpha)
    h_inverse = hessian_inverse(no_of_clusters, h_diagonal, h_constant, h_a)
    alpha_m_step = alpha_updater(alpha, h_inverse, G, no_of_clusters, dim)
    return mix_m_step, alpha_m_step


if __name__ == '__main__':
    alpha, data, dim, size, mix, original_labels = initial_algorithm()
    counter = 1
    obj = {'alpha': []}

    while True:
        pdf, posterior = estimation_step(K, mix, alpha, data, dim)
        mix, alpha = maximization_step(K, alpha, data, dim, posterior, size)
        obj['alpha'].append(alpha)
        # mix, alpha, K = cluster_drop_test(mix, alpha, cluster_drop_val, K, dim)
        # converge = convergence_test(obj['alpha'], CONST["algConverge"])
        labels = posterior.argmax(axis=1)
        accuracy = accuracy_score(original_labels, labels)
        counter = counter + 1
        print("ORIGINAL  :>", cluster_density_evaluation(original_labels))
        print("PREDICTED :>", cluster_density_evaluation(labels))
        if counter == THRESHOLD:
            print("################### Final Parameters ###################")
            print("K : ", K)
            print("Mix : ", mix, np.sum(mix))
            print("Alpha : ", alpha)
            print("Counter : ", counter)
            print("ORIGINAL  :>", cluster_density_evaluation(original_labels))
            print("PREDICTED :>", cluster_density_evaluation(labels))
            exit()
