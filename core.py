import warnings

from sklearn.cluster import KMeans  # contains KNN related functionality.

from dataset import load_dataset  # Importing DataSet
from inverted_dirichlet import inverted_dirichlet as inverted_dirichlet
from lib.constants import CONST  # contains the constant values.
from lib.helpers import *  # class contains the logic like performanceMeasure, Precision etc.

warnings.filterwarnings("error")


def initial_algorithm(no_of_clusters):
    data_set, size, original_labels = load_dataset(CONST['CM1'])
    data_set = normalize(data_set) + sys.float_info.epsilon
    kmeans = KMeans(n_clusters=CONST['K'], random_state=0, n_init="auto").fit(data_set)
    labels = ASARRAY(kmeans.predict(data_set)).reshape(1, size)[0]
    clusters, unique_clusters, dimension = split_data_by_label(labels, data_set)
    initial_py = ASARRAY(mixer_estimator(clusters, size)).reshape(1, no_of_clusters)
    initial_alpha = method_of_moment(no_of_clusters, clusters, dimension)
    return initial_alpha, data_set, dimension, size, initial_py, original_labels


"""
/**
 * This function Contains the Estimation Step logic.
 * @param  {Integer} K.
 * @param  {Integer} mix.
 * @param  {Integer} alphaSet.
 * @param  {Integer} imgPixels.
 * @param  {Integer} pixelSize.
 * @return {String} pdfMatrix.
 * @return {String} posteriorProbability.
 */
"""


def estimation_step(K, mix, alpha, data, dim):
    pdf = inverted_dirichlet(K, alpha, data, dim).pdf_fetcher()
    posterior = posterior_estimator(pdf, mix)
    return pdf, posterior


"""
/**
 * This function Contains the Maximization Step logic.
 * @param  {Integer} K.
 * @param  {Integer} alphaSet.
 * @param  {Integer} imgPixels.
 * @param  {Integer} dim.
 * @param  {Integer} posteriorProb.
 * @param  {Integer} pixelSize.
 * @param  {Integer} imageH.
 * @param  {Integer} imageW.
 * @param  {Integer} mix.
 * @return {String} mix.
 * @return {String} alpha.
 */
"""


def maximization_step(K, alpha, data, dim, posterior, size):
    mix_m_step = mix_updater(posterior, size, K)  # Checked: Working Fine!
    G = g_estimation(data, size, alpha, posterior, dim, K)  # Checked: Working Fine!
    h_diagonal, h_constant, h_a = hessian(dim, posterior, alpha)  # Checked: Working Fine!
    h_inverse = hessian_inverse(K, h_diagonal, h_constant, h_a)  # Checked: Working Fine!
    alpha_m_step = alpha_updater(alpha, h_inverse, G, K, dim)  # Checked: Working Fine!
    return mix_m_step, alpha_m_step


"""
/**
 * This function add the array's element and return them in the form of a String.
 * @param  {Integer} a.
 * @return {String} which contains the Sum of Array.
 */
"""

if __name__ == '__main__':
    K = CONST['K']
    cluster_drop_val = CONST['cluster_drop_val']
    alpha, data, dim, size, mix, original_labels = initial_algorithm(K)
    counter = 1
    obj = {'alpha': []}

    while True:
        pdf, posterior = estimation_step(K, mix, alpha, data, dim)
        mix, alpha = maximization_step(K, alpha, data, dim, posterior, size)
        obj['alpha'].append(alpha)
        labels = predict_labels(posterior)
        accuracy = predict_accuracy(labels, original_labels)
        counter = counter + 1
        if counter == CONST['THRESHOLD']:
            print("################### Final Parameters ###################")
            print("K : ", K)
            print("Mix : ", mix, SUM(mix))
            print("Alpha : ", alpha)
            print("Counter : ", counter)
            print("ORIGINAL  :>", cluster_density_evaluation(original_labels))
            print("PREDICTED :>", cluster_density_evaluation(labels))
            exit()
